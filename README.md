# ReAct Multi-Hop Question Answering Agent (HotpotQA FullWiki)

A ReAct (Reasoning + Acting) Agent built using Python, LangGraph, and local vLLM inference on Google Compute Engine (GCE) NVIDIA L4 GPU (24GB VRAM) to solve the HotpotQA FullWiki multi-hop benchmark.

---

## Model Selection & Hardware Optimization

- **Finalized LLM**: `Qwen/Qwen2.5-7B-Instruct`
- **Target Hardware**: Google Compute Engine (GCE) g2-standard-8 VM with 1x NVIDIA L4 GPU (24GB VRAM) and 32GB RAM.
- **Why Qwen-2.5-7B-Instruct?**
  - Leading open-source 7B model for instruction following, structured tool calling, and multi-step ReAct reasoning loops.
  - Fits within ~14GB VRAM in bfloat16 precision, leaving 10GB VRAM for vLLM KV-cache memory, enabling 100+ tokens/sec local inference throughput.

---

## Key Architecture & Features

- Strict Agentic Control Loop: Built using LangGraph StateGraph enforcing an explicit Thought -> Action -> Observation cycle.
- Dual Wikipedia Tool Suite:
  - search[entity]: Searches Wikipedia API and retrieves lead section summaries.
  - lookup[keyword]: Searches paragraphs within loaded Wikipedia pages for exact keyword matches.
- Interactive Trajectory & Knowledge Graph Visualizer: Streamlit dashboard visualizing step-by-step reasoning traces and an interactive PyVis Knowledge Bridge Graph showing entity transitions.
- Official HotpotQA Benchmark & Plotting Engine: Automated evaluation measuring Answer Exact Match (EM), Answer F1, Supporting Facts F1, Joint Exact Match (Joint EM), and Joint F1 on full HotpotQA validation sets, automatically generating metric bar charts and markdown evaluation reports.

---

## End-to-End Execution Guide on GCE L4 GPU

### Step 1: Clone Repository & Create Conda Environment

```bash
git clone https://github.com/your-username/hotpot.git
cd hotpot

# Create and activate Conda environment
conda env create -f environment.yml
conda activate hotpot
```

### Step 2: Launch vLLM Local Model Server on L4 GPU

Start the OpenAI-compatible vLLM inference server using local GPU memory:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192
```

Wait until vLLM outputs `Application startup complete` on `http://localhost:8000/v1`.

### Step 3: Run Unit Tests

Verify the ReAct parser, retrieval tools, and LangGraph engine state machine:

```bash
PYTHONPATH=. pytest tests/
```

### Step 4: Run Benchmark & Generate Metric Plots + Evaluation Report

Execute evaluation on sample questions or the full official HotpotQA validation dataset:

```bash
# 1. Quick test on sample multi-hop questions
python eval/run_eval.py --samples 4 --mode offline --source sample

# 2. Full benchmark evaluation on official validation set
python eval/run_eval.py --samples 100 --mode offline --source official_json --output-dir eval_results
```

Generated Output Artifacts (`eval_results/`):
- `results.json`: Raw prediction JSON with per-question EM/F1 metrics.
- `benchmark_metrics.png`: Bar chart of Answer EM, Answer F1, Supporting Facts F1, Joint EM, and Joint F1.
- `hop_distribution.png`: Histogram showing trajectory hop counts per question.
- `evaluation_report.md`: Markdown summary report with metrics table, latency, and sample question predictions.

### Step 5: Launch Interactive Web UI

```bash
streamlit run app/web_ui.py --server.port 8501 --server.address 0.0.0.0
```

Access the visual dashboard at `http://<GCE_EXTERNAL_IP>:8501`.

---

## Benchmark Metrics & Evaluation Output Format

```text
=== OFFICIAL HOTPOTQA LEADERBOARD METRICS ===
Answer Exact Match (EM):      75.0%
Answer F1 Score:              87.5%
Supporting Facts F1:          100.0%
Joint Exact Match (Joint EM): 75.0%
Joint F1 Score (Joint F1):    87.5%
Avg Hops / Question:          2.50
Total Evaluation Time:        4.12s
```

---

## Repository Structure

```
hotpot/
├── agent/
│   ├── engine.py          # LangGraph StateGraph agent loop
│   ├── parser.py          # Regex ReAct output parser & error handling
│   ├── prompt.py          # System prompt & multi-hop ReAct examples
│   └── state.py           # Agent state schema constructor
├── tools/
│   ├── wikipedia.py       # Live Wikipedia API (Search & Lookup)
│   └── local_retriever.py # Local HotpotQA corpus retriever
├── eval/
│   ├── dataset.py         # HotpotQA dataset loader
│   ├── metrics.py         # Official Joint EM & Joint F1 evaluation metrics
│   ├── plot_results.py    # Metric plotting & markdown report generator
│   └── run_eval.py        # CLI benchmark runner
├── app/
│   ├── web_ui.py          # Streamlit portfolio dashboard
│   └── graph_view.py      # PyVis knowledge graph generator
├── tests/
│   └── test_agent.py      # Pytest test suite
├── config.py              # Environment & model configurations
├── environment.yml        # Conda environment definition
├── requirements.txt       # Pip requirements file
└── README.md
```

---

## License

MIT License.