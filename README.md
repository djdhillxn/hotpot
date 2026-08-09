# ReAct Multi-Hop Question Answering Agent (HotpotQA FullWiki)

A retrieval-and-reasoning project that compares a **single-pass RAG baseline** against a **ReAct (Reason + Act) multi-hop agent** on HotpotQA FullWiki. Both systems use the same frozen `Qwen/Qwen2.5-7B-Instruct` model and the same global Wikipedia retrieval backend; the experimental difference is whether retrieval happens once from the original question or adaptively after each observation.

---

## Final Benchmark Architecture

```text
Official HotpotQA Wikipedia (Oct. 1, 2017)
        ~5M introductory paragraphs
                    |
        +-----------+-----------+
        |                       |
  Lucene BM25             BGE dense vectors
   (Pyserini)          (bge-base-en-v1.5 + FAISS)
        |                       |
        +-----------+-----------+
                    |
          Reciprocal Rank Fusion
                    |
           shared FullWiki backend
             /                \
            /                  \
Single-Pass RAG                ReAct Multi-Hop Agent
1 query -> top 7 docs     up to 7 adaptive tool turns
1 Qwen generation          top 6 docs/search -> Qwen
                           max 15 unique docs in working context
            \                  /
             HotpotQA official metrics
```

### Why the HotpotQA Wikipedia abstracts corpus?

HotpotQA defines the FullWiki setting over the first paragraphs of all Wikipedia articles. The project uses the official October 1, 2017 introductory-paragraph release (`enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2`, MD5 `01edf64cd120ecc03a2745352779514c`) rather than current live Wikipedia. The source sentence segmentation is preserved exactly, so retrieved evidence remains compatible with HotpotQA `[title, sentence_id]` supporting-fact evaluation.

---

## Core Components & Engineering Safeguards

- **Frozen LLM:** `Qwen/Qwen2.5-7B-Instruct`, served through local vLLM on an NVIDIA L4. No LLM fine-tuning or post-training.
- **Sparse retrieval:** Lucene BM25 through Pyserini/Anserini (pins `pyserini==1.6.0` requiring Java 21).
- **Dense retrieval:** `BAAI/bge-base-en-v1.5`, L2-normalized embeddings, persisted in a memory-efficient FAISS IVF-PQ index (`IVF4096,PQ96x8`, `nprobe=32`). Corpus encoding is performed once during index construction; benchmark-time query encoding runs on CPU so it does not compete with vLLM for GPU VRAM.
- **Hybrid retrieval:** Reciprocal Rank Fusion (RRF) over BM25 and dense rankings.
- **Retrieval protocol:**
  - **Single-Pass RAG Baseline:** Retrieves top 7 passages once from the original question (1 generation pass).
  - **ReAct Multi-Hop Agent:** Retrieves top 20 passages per `search[...]` turn (`candidate_k: 60`). Every unique retrieved document enters a per-question archive, is scored once against the original question by `BAAI/bge-reranker-base` (Sentence-Level Max-Scoring), and the 10 highest-scoring documents form the recurrent Active Evidence Memory across up to 7 adaptive turns. Later strong evidence can evict weaker early evidence.
  - **Shared Index:** Both systems query the exact same pre-built hybrid index (`indexes/fullwiki/`). Zero index rebuild required.
- **Qwen Prompting & ChatML System Structuring:**
  - Formatted as structured `[SystemMessage(...), HumanMessage(...)]` objects passed to `llm.invoke()`, forcing vLLM to format context using Qwen's native `<|im_start|>system...` ChatML template.
  - Includes standard HotpotQA Few-Shot exemplars demonstrating `Action: finish[canonical answer]`, `lookup[keyword]`, AND sentence-level citations `Support: [["Title", sentence_id], ...]`, enabling high Joint F1 scores.
- **Hard Generation Stop Sequences:** `stop=["\nObservation:", "Observation:"]` is explicitly bound on model invocations in `agent/engine.py`. This guarantees vLLM cuts off generation immediately after `Action: ...`, preventing Qwen from self-hallucinating Wikipedia observations.
- **Generation Constraints & Repetition Guard:** `max_tokens = 150` prevents runaway monologues. Consecutive duplicate searches are guarded, while repeated `lookup[keyword]` calls are allowed because classic ReAct uses them to advance through successive matches on the current page.
- **Classic Current-Page Lookup:** `lookup[keyword]` searches only the current rank-1 page selected by the latest search. Repeating the same lookup advances to the next matching sentence on that same page, mirroring the original ReAct Wikipedia environment.
- **Evidence Memory Reranker (`BAAI/bge-reranker-base`):** Uses BAAI's 110M parameter Cross-Encoder for high-precision semantic evidence ranking with Sentence-Level Max-Scoring. To run the reranker on GPU alongside vLLM, launch vLLM with `--gpu-memory-utilization 0.85` to reserve headroom for PyTorch CUDA context.
- **ReAct controller:** LangGraph state machine enforcing Thought -> Action -> Observation, with delimiter-safe action parsing, markdown codeblock stripping (`replace("```", "")`), and a mandatory final synthesis call at the hop budget.
- **Official-compatible evaluation:** Answer EM/F1, Supporting Fact EM/F1, Joint EM/F1, and evaluator-format `official_predictions.json`.
- **Structured experiment artifacts:** complete trajectories, raw model outputs, sentence-level evidence, sparse/dense/fused ranks and scores, retrieval latency, run manifests, failures, and evidence graphs.

---

## End-to-End Execution Guide on GCE L4

Target machine: `g2-standard-8`, NVIDIA L4 (24 GB), 32 GB system RAM, Ubuntu 22.04.

### Step 1: Create or Update the Environment

Fresh environment:

```bash
git clone https://github.com/your-username/hotpot.git
cd hotpot
conda env create -f environment.yml
conda activate hotpot
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

If updating an existing environment:

```bash
conda activate hotpot
conda env update -f environment.yml --prune
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

Verify runtime:

```bash
python --version
java -version
```

### Step 2: Build Global FullWiki Retrieval Indexes (Done Once)

Do this **before launching vLLM**, because the one-time BGE corpus encoding uses the L4 GPU.

```bash
python retrieval/build_fullwiki_index.py
```

This single command:
1. downloads the official HotpotQA Wikipedia intro archive if needed;
2. verifies its official MD5 checksum (`01edf64cd120ecc03a2745352779514c`);
3. streams the archive into sharded JSONL without materializing an expanded Wikipedia dump;
4. preserves every source sentence and its 0-based sentence ID;
5. builds a Lucene BM25 index with stored raw documents;
6. encodes the corpus with `BAAI/bge-base-en-v1.5`;
7. builds a FAISS IVF-PQ dense index;
8. verifies Lucene/FAISS document counts; and
9. writes `indexes/fullwiki/manifest.json` with the exact corpus/index configuration.

The build is idempotent. Existing verified stages are reused unless `--force-*` is supplied.

### Step 3: Verify FullWiki Retrieval Before Spending LLM Compute

Run a first-stage retrieval sanity check over validation questions:

```bash
python retrieval/evaluate_retrieval.py \
    --source official_json \
    --modes bm25 dense hybrid \
    --ks 1 5 6 10 20
```

Reports mean gold-document Recall@K and full recovery rates for BM25, dense, and hybrid retrieval, saving outputs to `eval_results/retrieval/retrieval_results.json`.

For a quick 100-sample smoke run:

```bash
python retrieval/evaluate_retrieval.py --source official_json --samples 100
```

### Step 4: Launch Local Qwen/vLLM Server

```bash
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192
```

Wait for `Application startup complete` on `http://localhost:8000/v1`.

### Step 5: Run Unit Tests

```bash
PYTHONPATH=. pytest tests/
```

### Step 6: Pre-Cache Evidence Reranker & Pre-Flight Verification

> [!NOTE]
> **Baseline Confirmation**: If you already ran the baseline evaluation on GCE with `top_k=7`, **you do NOT need to re-run the baseline**. The baseline code is completely unchanged and your previous baseline `results.json` remains 100% valid.

Before launching the full 64-worker ReAct benchmark, pre-download and cache the BAAI Cross-Encoder weights locally to avoid thread race conditions during HuggingFace download:

```bash
# 1. Pre-cache BAAI Cross-Encoder Reranker model:
python3 -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base'); print('BGE Cross-Encoder model successfully cached!')"

# 2. Run 5-Sample ReAct Smoke Test with YAML config:
python eval/run_eval.py \
    --config config/fullwiki.yaml \
    --source official_json \
    --samples 5 \
    --concurrency 4 \
    --output-dir eval_results/test_reranker
```

Inspect `eval_results/test_reranker/trajectories.json` to confirm:
- `memory_reranker` model is loaded and scoring passages cleanly.
- `active_memory_documents` retains top-10 cross-encoder scored passages.
- `rank1_in_active_memory` is `true` for rank-1 search pages, maintaining `lookup` functionality.
- `finish[answer]` extracts canonical short answers and valid `Support: [...]` citations.

### Step 7: Execute Full 7,405 Evaluation Benchmark

```bash
# ReAct Multi-Hop Agent (Concurrent 64 Workers):
python eval/run_eval.py \
    --config config/fullwiki.yaml \
    --source official_json \
    --concurrency 64 \
    --output-dir eval_results/react
```

### Step 8: Generate Baseline vs. ReAct Comparative Analysis

```bash
python eval/compare_results.py \
    --baseline eval_results/baseline/results.json \
    --react eval_results/react/results.json \
    --output-dir eval_results/comparison
```

Each benchmark directory contains:

```text
results.json
trajectories.json
official_predictions.json
official_gold.json
run_manifest.json
run.log
```

The ReAct trajectory for every retrieval step retains:
- generated search query;
- BM25 rank/score;
- dense rank/score;
- fused rank/score;
- complete raw top-6 retrieval candidates, the full unique-document archive count, cross-encoder scores, the active top-15 memory, and documents added/evicted after each search;
- sparse/dense/fused ranks and scores for every retrieved candidate;
- duplicate-query status, query/title-match diagnostic, evidence-memory/archive counts, rank-1 retention status, and memory/observation omissions;
- exact exposed Wikipedia titles and sentence IDs;
- retrieval latency by sparse/dense/fusion component;
- raw Qwen response;
- parsed Thought/Action;
- Observation;
- predicted/invalid supporting-fact citations; and
- final answer/evidence graph.

Check predictions with HotpotQA's official evaluator:

```bash
python hotpot_evaluate_v1.py \
    eval_results/react/official_predictions.json \
    eval_results/react/official_gold.json
```

### Step 9: Launch Interactive Web Dashboard UI

```bash
streamlit run app/web_ui.py --server.port 8501 --server.address 0.0.0.0
```

---

## Retrieval Modes

The benchmark runners support three modes:

- `offline`: searches only the 10 candidate paragraphs bundled with each FullWiki dev example; useful for tests/debugging, not the final global-retrieval experiment.
- `fullwiki`: searches the global HotpotQA Wikipedia index; this is the primary benchmark mode.
- `live`: current Wikipedia API; qualitative only because current Wikipedia sentence boundaries do not match HotpotQA's 2017 supporting-fact IDs.

Within `fullwiki`, `--retriever` can be `bm25`, `dense`, or `hybrid`. The primary reported experiment uses `hybrid`.

---

## Repository Structure

```text
hotpot/
├── agent/
│   ├── baseline_rag.py           # Single-pass RAG pipeline
│   ├── engine.py                 # LangGraph state machine with stop sequence & repetition guards
│   ├── parser.py                 # Flexible action parser (whitespace/case/markdown resilient)
│   ├── prompt.py                 # ChatML system prompt & HotpotQA exemplars with Support lines
│   └── state.py                  # Trajectory state definitions
├── retrieval/
│   ├── __init__.py
│   ├── corpus.py                 # Official archive download & streaming conversion
│   ├── build_fullwiki_index.py   # BM25 + BGE/FAISS index construction
│   ├── fullwiki_retriever.py     # Hybrid BM25+BGE search backend & multi-hit lookup
│   └── evaluate_retrieval.py     # Recall@K sanity benchmark
├── tools/
│   ├── wikipedia.py              # Live API qualitative fallback
│   └── local_retriever.py        # Candidate-pool retriever for offline mode
├── eval/
│   ├── artifacts.py              # Trajectory logging & manifest generation
│   ├── compare_results.py        # Baseline vs ReAct delta evaluation
│   ├── dataset.py                # HuggingFace & official JSON dataset loader
│   ├── metrics.py                # Official HotpotQA tokenization & Joint F1 score computation
│   ├── plot_results.py           # Performance & hop-distribution visualization
│   ├── run_baseline.py           # Single-pass RAG concurrent runner
│   └── run_eval.py               # ReAct agent concurrent runner
├── app/
│   └── web_ui.py                 # Interactive Streamlit dashboard
├── portfolio/                    # Saved analysis reports & visualizations
├── tests/
│   ├── test_agent.py
│   └── test_retrieval.py
├── config.py                     # Environment variables & default hyperparameters
├── environment.yml
├── requirements.txt
└── README.md
```

---

## Troubleshooting

### Pyserini / Java error

```bash
java -version
```

The project expects Java 21. If the current environment was created before the retrieval upgrade:

```bash
conda env update -f environment.yml --prune
conda activate hotpot
```

### `RuntimeError: Could not find nvcc` from vLLM

The optimized non-eager vLLM path may JIT-compile CUDA kernels. Install a CUDA toolkit/compiler compatible with the PyTorch CUDA build, then ensure `nvcc` is discoverable via `PATH`/`CUDA_HOME` before starting vLLM. `--enforce-eager` remains a fallback for debugging.

### `CXXABI` / `libstdc++` error

```bash
conda install -n hotpot -c conda-forge libstdcxx-ng sysroot_linux-64 -y
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

---

## License

MIT License. The HotpotQA Wikipedia corpus is distributed separately under the license stated by the HotpotQA authors; generated corpus/index files are not committed to this repository.
