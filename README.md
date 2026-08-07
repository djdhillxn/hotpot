# ReAct Multi-Hop Question Answering Agent (HotpotQA FullWiki)

A ReAct (Reasoning + Acting) Agent built using Python, LangGraph, and local vLLM inference on Google Compute Engine (GCE) NVIDIA L4 GPU (24GB VRAM) to solve the HotpotQA FullWiki multi-hop benchmark, featuring a side-by-side **Single-Pass RAG Baseline Comparison Study**.

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
- Single-Pass RAG Baseline Engine: Dedicated direct prompting engine (`agent/baseline_rag.py`) for comparative baseline study.
- Dual Wikipedia Tool Suite:
  - search[entity]: Searches Wikipedia API and retrieves lead section summaries.
  - lookup[keyword]: Searches paragraphs within loaded Wikipedia pages for exact keyword matches.
- Interactive Trajectory & Knowledge Graph Visualizer: Streamlit dashboard visualizing step-by-step reasoning traces and an interactive PyVis Knowledge Bridge Graph showing entity transitions.
- Official HotpotQA Benchmark & Comparison Engine: Automated evaluation measuring Answer Exact Match (EM), Answer F1, Supporting Facts F1, Joint Exact Match (Joint EM), and Joint F1 on full HotpotQA validation sets, automatically generating comparative metric bar charts and markdown evaluation reports.

---

## End-to-End Execution Guide on GCE L4 GPU

### Step 1: Clone Repository & Create Conda Environment

```bash
git clone https://github.com/your-username/hotpot.git
cd hotpot

# Create and activate Conda environment
conda env create -f environment.yml
conda activate hotpot

# Fix for Linux CXXABI libstdc++ compatibility on GCE
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### Step 2: Launch vLLM Local Model Server on L4 GPU

Start the OpenAI-compatible vLLM inference server using local GPU memory (including `--enforce-eager` to bypass JIT compilation requiring `nvcc`):

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

Wait until vLLM outputs `Application startup complete` on `http://localhost:8000/v1`.

### Step 3: Run Unit Tests

Verify the ReAct parser, baseline engine, retrieval tools, and LangGraph engine state machine:

```bash
PYTHONPATH=. pytest tests/
```

### Step 4: Run Single-Pass RAG Baseline Benchmark

Execute Single-Pass RAG evaluation to establish the non-agentic baseline:

```bash
python eval/run_baseline.py --samples 100 --mode offline --source official_json --output-dir eval_results/baseline
```

### Step 5: Run ReAct Multi-Hop Agent Benchmark

Execute the ReAct Agent evaluation:

```bash
python eval/run_eval.py --samples 100 --mode offline --source official_json --output-dir eval_results/react
```

### Step 6: Generate Comparative Analysis & Side-by-Side Plots

Generate comparative bar charts (`comparison_metrics.png`) and performance report (`comparison_report.md`):

```bash
python eval/compare_results.py --baseline eval_results/baseline/results.json --react eval_results/react/results.json --output-dir eval_results/comparison
```

Generated Output Artifacts (`eval_results/comparison/`):
- `comparison_metrics.png`: Side-by-side bar chart comparing Single-Pass RAG vs ReAct Multi-Hop Agent for EM, F1, and Joint F1.
- `comparison_report.md`: Markdown summary report demonstrating the exact accuracy gain achieved by Agentic AI over static RAG.

### Step 7: Launch Interactive Web UI

```bash
streamlit run app/web_ui.py --server.port 8501 --server.address 0.0.0.0
```

Access the visual dashboard at `http://<GCE_EXTERNAL_IP>:8501`.

---

## Troubleshooting GCE Errors

### 1. `RuntimeError: Could not find nvcc`
Add `--enforce-eager` flag to your `vllm` launch command (as shown in Step 2 above). Alternatively, install CUDA toolkit into Conda:
```bash
conda install -n hotpot -c nvidia cuda-toolkit -y
```

### 2. `ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version CXXABI_1.3.15 not found`
Run:
```bash
conda install -n hotpot -c conda-forge libstdcxx-ng sysroot_linux-64 -y
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

---

## Repository Structure

```
hotpot/
├── agent/
│   ├── baseline_rag.py    # Single-Pass RAG direct prompting baseline engine
│   ├── engine.py          # LangGraph StateGraph agent loop
│   ├── parser.py          # Regex ReAct output parser & error handling
│   ├── prompt.py          # System prompt & multi-hop ReAct examples
│   └── state.py           # Agent state schema constructor
├── tools/
│   ├── wikipedia.py       # Live Wikipedia API (Search & Lookup)
│   └── local_retriever.py # Local HotpotQA corpus retriever
├── eval/
│   ├── compare_results.py # Baseline vs ReAct comparative analysis script
│   ├── dataset.py         # HotpotQA dataset loader
│   ├── metrics.py         # Official Joint EM & Joint F1 evaluation metrics
│   ├── plot_results.py    # Metric plotting & markdown report generator
│   ├── run_baseline.py    # CLI runner for Single-Pass RAG baseline
│   └── run_eval.py        # CLI runner for ReAct Agent
├── portfolio/
│   ├── app.js             # Portfolio JavaScript trajectory renderer
│   ├── index.html         # Portfolio trajectory visualizer HTML component
│   ├── portfolio_trajectories.json # Saved trajectory JSON dataset
│   ├── style.css          # Portfolio CSS stylesheet
│   └── README.md          # Portfolio embedding guide
├── app/
│   ├── web_ui.py          # Streamlit portfolio dashboard
│   └── graph_view.py      # PyVis knowledge graph generator
├── tests/
│   └── test_agent.py      # Pytest test suite
├── config.py              # Environment & model configurations
├── environment.yml        # Conda environment definition
├── project_proposal.md    # Comprehensive project proposal
├── requirements.txt       # Pip requirements file
└── README.md
```

---

## License

MIT License.