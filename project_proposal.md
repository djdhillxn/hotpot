# Project Proposal: ReAct Multi-Hop Question Answering Agent (HotpotQA FullWiki)

## 1. Executive Summary

This project delivers an end-to-end, production-grade ReAct (Reasoning + Acting) Agent framework from scratch to solve the HotpotQA FullWiki multi-hop question-answering benchmark. Powered by Python, LangGraph, and open-source Large Language Models (`Qwen/Qwen2.5-7B-Instruct`), the system dynamically alternates between explicit reasoning steps (Thought), Wikipedia search and paragraph lookup operations (Action), and environment feedback (Observation).

Designed specifically for execution on a Google Compute Engine (GCE) instance equipped with a single NVIDIA L4 GPU (24GB VRAM) and 32GB System RAM, the agent utilizes `vLLM` for high-throughput local inference without relying on external cloud APIs.

---

## 2. Problem Statement & Motivation

Traditional Single-Pass Retrieval-Augmented Generation (RAG) models struggle with complex multi-hop questions because the information required to produce an answer is distributed across multiple distinct documents. For example, answering:

> *"Were Scott Derrickson and Ed Wood born in the same state?"*

requires:
1. Identifying who Scott Derrickson is and searching his biography to discover his birthplace (Denver, Colorado).
2. Identifying who Ed Wood is and searching his biography to discover his birthplace (Poughkeepsie, New York).
3. Comparing the two extracted locations to formulate the final answer ("no").

Static retrieval fails on such tasks because the query for step 2 depends directly on the entity discovered in step 1. An iterative agentic loop capable of dynamic search, lookup, reasoning, and anti-hallucination guardrails is necessary.

---

## 3. Technical Objectives

1. **Custom Agent Control Loop**: Implement a ReAct state machine using `LangGraph` `StateGraph` without black-box wrappers, ensuring total transparency over state transitions, scratchpad formatting, and hop counts.
2. **Local High-Performance Inference**: Serve `Qwen/Qwen2.5-7B-Instruct` locally on a single NVIDIA L4 GPU using `vLLM`, achieving 100+ tokens/second with PagedAttention optimization.
3. **Dual Tool Suite**: Provide both a live Wikipedia API wrapper and an offline HotpotQA paragraph context retriever for fast, reproducible benchmarking.
4. **Official Leaderboard Metrics**: Implement standard HotpotQA leaderboard metrics:
   - Answer Exact Match (EM) & Answer F1
   - Supporting Facts Exact Match (SP EM) & Supporting Facts F1
   - **Joint Exact Match (Joint EM)** & **Joint F1 Score (Joint F1)**
5. **Visual Portfolio Interface**: Build an interactive Streamlit visualizer accompanied by a PyVis entity graph renderer that maps bridged Wikipedia articles in real-time.

---

## 4. System Architecture & Component Breakdown

```
[User Question] -> [LangGraph ReAct Agent Node]
                          |
             (Parse Thought & Action)
                          |
             [Is Tool Call or Finish?]
            /                         \
   (Action: Search / Lookup)       (Action: Finish)
          /                             \
[Wikipedia Tool Executor]        [Final Answer + Evidence Graph]
          |
     (Observation)
          |
   [LangGraph ReAct Agent Node]
```

### Module Responsibilities

- `agent/engine.py`: Defines the `LangGraph` workflow, node execution, conditional edges, and execution loop (`run_react_agent`).
- `agent/parser.py`: Robust regular expression parser converting raw LLM text into structured actions (`search[query]`, `lookup[keyword]`, `finish[answer]`), recovering automatically from LLM formatting syntax drifts.
- `agent/prompt.py`: Strict system prompt template with multi-hop ReAct examples and grounding rules.
- `agent/state.py`: Initializes clean dictionary state schemas tracking hop counts, scratchpad history, visited pages, and knowledge graph edges.
- `tools/wikipedia.py`: Real-time Wikipedia REST API client implementing `search` and `lookup`.
- `tools/local_retriever.py`: Local HotpotQA corpus retriever for offline evaluation.
- `eval/metrics.py`: Evaluates predictions against ground truth using standard Joint EM and Joint F1 formulas.
- `eval/dataset.py`: Streamlined dataset loader supporting curated samples, HuggingFace datasets, and official HotpotQA S3 JSON benchmarks.
- `eval/plot_results.py`: Automatically generates metric bar charts (`benchmark_metrics.png`), trajectory step distribution histograms (`hop_distribution.png`), and markdown summary reports (`evaluation_report.md`).
- `eval/run_eval.py`: CLI benchmark entry point orchestrating dataset loading, agent execution, metric computation, and report rendering.
- `app/web_ui.py`: Streamlit dashboard offering an interactive playground, step-by-step trajectory inspector, and automated evaluation tab.
- `app/graph_view.py`: PyVis network visualization generator mapping entity transition graphs.
- `tests/test_agent.py`: Pytest suite verifying ReAct output parsing, retrieval tools, metric functions, and end-to-end agent execution with mock chat models.

---

## 5. Hardware & Deployment Specification

- **Cloud Platform**: Google Compute Engine (GCE)
- **Machine Type**: `g2-standard-8` (8 vCPUs, 32GB RAM)
- **Accelerator**: 1x NVIDIA L4 GPU (24GB VRAM)
- **Inference Engine**: `vLLM` v0.6+ serving an OpenAI-compatible API on `http://localhost:8000/v1`
- **Environment**: Conda environment (`hotpot`) with Python 3.11

### vLLM Local Launch Command

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192
```

---

## 6. End-to-End Execution Workflow

1. **Environment Setup**:
   ```bash
   conda env create -f environment.yml
   conda activate hotpot
   ```
2. **Local Model Serving**: Launch `vLLM` server in background or separate tmux/screen session.
3. **Verification**: Run `PYTHONPATH=. pytest tests/` to confirm 100% test pass rate.
4. **Evaluation Benchmark**:
   ```bash
   python eval/run_eval.py --samples 100 --mode offline --source official_json --output-dir eval_results
   ```
5. **Visualization**: Launch Streamlit interface (`streamlit run app/web_ui.py`).

---

## 7. Final Readiness Assessment

The repository has undergone a comprehensive audit:
- All typing imports have been stripped for clean, readable Python code.
- Conda environment management is fully configured via `environment.yml`.
- All unit tests pass with zero errors.
- Official HotpotQA leaderboard metrics (Joint EM, Joint F1) and automated plotting reports are operational.
- The project is fully self-contained and ready to be frozen, committed, and pulled into Google Compute Engine for execution.
