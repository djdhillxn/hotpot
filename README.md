# ReAct Multi-Hop Question Answering Agent (HotpotQA FullWiki)

A retrieval-and-reasoning project that compares a **single-pass RAG baseline** against a **ReAct (Reason + Act) multi-hop agent** on HotpotQA FullWiki. Both systems use the same frozen `Qwen/Qwen2.5-7B-Instruct` model and the same global Wikipedia retrieval backend; the experimental difference is whether retrieval happens once from the original question or adaptively after each observation.

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
Single-Pass RAG                ReAct
1 query -> top 7 docs     up to 7 adaptive queries
1 Qwen generation          top 1 doc/query -> Qwen
            \                  /
             HotpotQA official metrics
```

### Why the HotpotQA Wikipedia abstracts corpus?

HotpotQA defines the FullWiki setting over the first paragraphs of all Wikipedia articles. The project uses the official October 1, 2017 introductory-paragraph release (`enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2`, MD5 `01edf64cd120ecc03a2745352779514c`) rather than current live Wikipedia. The source sentence segmentation is preserved exactly, so retrieved evidence remains compatible with HotpotQA `[title, sentence_id]` supporting-fact evaluation.

## Core Components

- **Frozen LLM:** `Qwen/Qwen2.5-7B-Instruct`, served through local vLLM on an NVIDIA L4. No LLM fine-tuning or post-training.
- **Sparse retrieval:** Lucene BM25 through Pyserini/Anserini.
- **Dense retrieval:** `BAAI/bge-base-en-v1.5`, L2-normalized embeddings, persisted in a memory-efficient FAISS IVF-PQ index (`IVF4096,PQ96x8`). Corpus encoding is performed once during index construction; benchmark-time query encoding defaults to CPU so it does not compete with vLLM for the L4.
- **Hybrid retrieval:** Reciprocal Rank Fusion (RRF) over BM25 and dense rankings. No dataset-specific fusion-weight sweep.
- **Fair retrieval budget:** Single-pass RAG sees up to 7 documents from one retrieval. ReAct sees one document per adaptive search for at most 7 searches.
- **ReAct controller:** LangGraph state machine enforcing Thought -> Action -> Observation, with delimiter-safe action parsing and a mandatory final synthesis call at the hop budget.
- **Official-compatible evaluation:** Answer EM/F1, Supporting Fact EM/F1, Joint EM/F1, and evaluator-format `official_predictions.json`.
- **Structured experiment artifacts:** complete trajectories, raw model outputs, sentence-level evidence, sparse/dense/fused ranks and scores, retrieval latency, run manifests, failures, and evidence graphs.

---

# End-to-End Execution Guide on GCE L4

Target machine: `g2-standard-8`, NVIDIA L4 (24 GB), 32 GB system RAM, Ubuntu 22.04.

## Step 1: Create or Update the Environment

Fresh environment:

```bash
git clone https://github.com/your-username/hotpot.git
cd hotpot
conda env create -f environment.yml
conda activate hotpot
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

If the existing `hotpot` environment predates FullWiki retrieval support:

```bash
conda activate hotpot
conda env update -f environment.yml --prune
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

Verify the retrieval runtime:

```bash
python --version
java -version
```

The repository intentionally pins `pyserini==1.6.0`, the Python 3.11-compatible release line. Pyserini 1.6.0 requires Java 21, which is installed by `environment.yml`.

## Step 2: Build the Global FullWiki Retrieval Indexes

Do this **before launching vLLM**, because the one-time BGE corpus encoding uses the L4.

```bash
python retrieval/build_fullwiki_index.py
```

This single command:

1. downloads the official HotpotQA Wikipedia intro archive if needed;
2. verifies its official MD5 checksum;
3. streams the archive into sharded JSONL without materializing an expanded Wikipedia dump;
4. preserves every source sentence and its 0-based sentence ID;
5. builds a Lucene BM25 index with stored raw documents;
6. encodes the same corpus with `BAAI/bge-base-en-v1.5`;
7. builds a FAISS IVF-PQ dense index;
8. verifies Lucene/FAISS document counts; and
9. writes `indexes/fullwiki/manifest.json` with the exact corpus/index configuration.

Generated data are deliberately ignored by Git:

```text
data/fullwiki/
indexes/fullwiki/
```

The build is idempotent. Existing verified stages are reused unless an explicit `--force-*` flag is supplied.

Useful overrides:

```bash
python retrieval/build_fullwiki_index.py --help
```

## Step 3: Verify FullWiki Retrieval Before Spending LLM Compute

Run a first-stage retrieval sanity check over the validation questions:

```bash
python retrieval/evaluate_retrieval.py \
    --source official_json \
    --modes bm25 dense hybrid \
    --ks 1 5 10 20
```

This does **not** call Qwen. It reports mean gold-document Recall@K and the rate at which all gold documents are recovered for BM25, dense, and hybrid retrieval, and saves the per-question results to:

```text
eval_results/retrieval/retrieval_results.json
```

For an initial smoke run:

```bash
python retrieval/evaluate_retrieval.py --source official_json --samples 100
```

## Step 4: Launch the Local Qwen/vLLM Server

The dense corpus is already encoded, so benchmark-time dense query embeddings run on CPU by default. The L4 can therefore be dedicated to Qwen.

```bash
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --enforce-eager
```

Wait for `Application startup complete` on `http://localhost:8000/v1`.

## Step 5: Run Tests

```bash
PYTHONPATH=. pytest tests/
```

The retrieval tests do not require the full Wikipedia indexes and exercise corpus parsing, sentence-ID preservation, RRF, and the per-question FullWiki retriever contract.

## Step 6: Run the Budget-Matched Single-Pass FullWiki RAG Baseline

One hybrid retrieval from the original question, top 7 Wikipedia paragraphs, one Qwen generation:

```bash
python eval/run_baseline.py \
    --mode fullwiki \
    --retriever hybrid \
    --top-k 7 \
    --source official_json \
    --output-dir eval_results/baseline
```

## Step 7: Run the FullWiki ReAct Agent

Each ReAct search uses the **same hybrid backend** but exposes only its top-1 paragraph. The agent may make at most seven adaptive searches, so its maximum retrieved-document budget matches the baseline's seven documents.

```bash
python eval/run_eval.py \
    --mode fullwiki \
    --retriever hybrid \
    --top-k 1 \
    --source official_json \
    --concurrency 16 \
    --max-hops 7 \
    --output-dir eval_results/react
```

For debugging, the old candidate-pool mode remains available:

```bash
python eval/run_eval.py --mode offline --source official_json --samples 20 --max-hops 7
```

`--mode live` remains a qualitative current-Wikipedia demo and should not be used for official HotpotQA supporting-fact comparisons.

## Step 8: Generate the Baseline-vs-ReAct Comparison

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
- exact retrieved Wikipedia title and sentence IDs;
- retrieval latency by sparse/dense/fusion component;
- raw Qwen response;
- parsed Thought/Action;
- Observation;
- predicted/invalid supporting-fact citations; and
- final answer/evidence graph.

Official-format predictions can be checked with HotpotQA's evaluator:

```bash
python hotpot_evaluate_v1.py \
    eval_results/react/official_predictions.json \
    eval_results/react/official_gold.json
```

## Step 9: Launch the Existing Interactive UI

```bash
streamlit run app/web_ui.py --server.port 8501 --server.address 0.0.0.0
```

---

## Retrieval Modes

The benchmark runners support three modes:

- `offline`: searches only the 10 candidate paragraphs bundled with each FullWiki dev example; useful for tests/debugging, not the final global-retrieval experiment.
- `fullwiki`: searches the global HotpotQA Wikipedia index; this is the primary benchmark mode.
- `live`: current Wikipedia API; qualitative only because current Wikipedia sentence boundaries do not match HotpotQA's 2017 supporting-fact IDs.

Within `fullwiki`, `--retriever` can be `bm25`, `dense`, or `hybrid`. The primary reported experiment should use `hybrid`; the other modes exist mainly for retrieval sanity analysis.

---

## Repository Structure

```text
hotpot/
├── agent/
│   ├── baseline_rag.py
│   ├── engine.py
│   ├── parser.py
│   ├── prompt.py
│   └── state.py
├── retrieval/
│   ├── __init__.py
│   ├── corpus.py                 # official archive download/streaming conversion
│   ├── build_fullwiki_index.py   # BM25 + BGE/FAISS build pipeline
│   ├── fullwiki_retriever.py     # shared global backend + per-question sessions
│   └── evaluate_retrieval.py     # Recall@K retrieval sanity benchmark
├── tools/
│   ├── wikipedia.py
│   └── local_retriever.py
├── eval/
│   ├── artifacts.py
│   ├── compare_results.py
│   ├── dataset.py
│   ├── metrics.py
│   ├── plot_results.py
│   ├── run_baseline.py
│   └── run_eval.py
├── app/
├── portfolio/
├── tests/
│   ├── test_agent.py
│   └── test_retrieval.py
├── config.py
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

Use the `--enforce-eager` vLLM flag shown above. If needed:

```bash
conda install -n hotpot -c nvidia cuda-toolkit -y
```

### `CXXABI` / `libstdc++` error

```bash
conda install -n hotpot -c conda-forge libstdcxx-ng sysroot_linux-64 -y
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

---

## License

MIT License. The HotpotQA Wikipedia corpus is distributed separately under the license stated by the HotpotQA authors; generated corpus/index files are not committed to this repository.
