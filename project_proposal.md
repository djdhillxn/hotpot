# Project Proposal: ReAct Multi-Hop Question Answering Agent (HotpotQA FullWiki)

## 1. Executive Summary

This project delivers an end-to-end, production-grade ReAct (Reasoning + Acting) Agent framework from scratch to solve the HotpotQA FullWiki multi-hop question-answering benchmark. Powered by Python, LangGraph, and open-source Large Language Models (`Qwen/Qwen2.5-7B-Instruct`), the system dynamically alternates between explicit reasoning steps (Thought), Wikipedia search and paragraph lookup operations (Action), and environment feedback (Observation).

To rigorously prove the superiority of agentic control loops over traditional architectures, the project includes a dedicated **Single-Pass RAG Baseline Engine** and an automated comparative benchmark suite.

Designed specifically for execution on a Google Compute Engine (GCE) instance equipped with a single NVIDIA L4 GPU (24GB VRAM) and 32GB System RAM, the agent utilizes `vLLM` for high-throughput local inference without relying on external cloud APIs.

---

## 2. Problem Statement & Motivation

Traditional Single-Pass Retrieval-Augmented Generation (RAG) models struggle with complex multi-hop questions because the information required to produce an answer is distributed across multiple distinct documents. For example, answering:

> *"Were Scott Derrickson and Ed Wood born in the same state?"*

requires:
1. Identifying who Scott Derrickson is and searching his biography to discover his birthplace (Denver, Colorado).
2. Identifying who Ed Wood is and searching his biography to discover his birthplace (Poughkeepsie, New York).
3. Comparing the two extracted locations to formulate the final answer ("no").

Static single-pass retrieval fails on such tasks because the query for step 2 depends directly on the entity discovered in step 1. An iterative agentic loop capable of dynamic search, lookup, reasoning, and anti-hallucination guardrails is necessary.

---

## 3. Technical Objectives

1. **Custom Agent Control Loop**: Implement a ReAct state machine using `LangGraph` `StateGraph` without black-box wrappers, ensuring total transparency over state transitions, scratchpad formatting, and hop counts.
2. **Single-Pass RAG Comparative Baseline**: Implement a dedicated direct prompting baseline engine (`agent/baseline_rag.py`) and comparative analysis runner (`eval/compare_results.py`) to quantify the exact performance gap between static RAG and ReAct Agentic AI.
3. **Local High-Performance Inference**: Serve `Qwen/Qwen2.5-7B-Instruct` locally on a single NVIDIA L4 GPU using `vLLM`, achieving 100+ tokens/second with PagedAttention optimization.
4. **Dual Tool Suite**: Provide both a live Wikipedia API wrapper and an offline HotpotQA paragraph context retriever for fast, reproducible benchmarking.
5. **Official Leaderboard Metrics**: Implement standard HotpotQA leaderboard metrics:
   - Answer Exact Match (EM) & Answer F1
   - Supporting Facts Exact Match (SP EM) & Supporting Facts F1
   - **Joint Exact Match (Joint EM)** & **Joint F1 Score (Joint F1)**
6. **Visual Portfolio Interface**: Build an interactive Streamlit visualizer accompanied by a PyVis entity graph renderer and an embedded Vanilla HTML/JS trajectory inspector (`portfolio/`).

---

## 4. System Architecture & Component Breakdown

```
                  [User Question]
                         |
           [Select Architecture Engine]
          /                            \
(ReAct Multi-Hop Agent)      (Single-Pass RAG Baseline)
        |                                   |
[LangGraph ReAct Node]             [Single Retrieval Query]
        |                                   |
(Thought -> Action -> Obs)         [Direct LLM Generation]
        |                                   |
[Final ReAct Trajectory]           [Baseline Prediction]
```

### Module Responsibilities

- `agent/engine.py`: Defines the `LangGraph` workflow, node execution, conditional edges, and execution loop (`run_react_agent`).
- `agent/baseline_rag.py`: Implements the Single-Pass RAG direct prompting baseline (`run_single_pass_rag`).
- `agent/parser.py`: Robust regular expression parser converting raw LLM text into structured actions (`search[query]`, `lookup[keyword]`, `finish[answer]`).
- `agent/prompt.py`: Strict system prompt template with multi-hop ReAct examples and grounding rules.
- `agent/state.py`: Initializes clean dictionary state schemas tracking hop counts, scratchpad history, visited pages, and knowledge graph edges.
- `tools/wikipedia.py`: Real-time Wikipedia REST API client implementing `search` and `lookup`.
- `tools/local_retriever.py`: Local HotpotQA corpus retriever for offline evaluation.
- `eval/metrics.py`: Evaluates predictions against ground truth using standard Joint EM and Joint F1 formulas.
- `eval/dataset.py`: Streamlined dataset loader supporting curated samples, HuggingFace datasets, and official HotpotQA S3 JSON benchmarks.
- `eval/run_eval.py`: CLI runner for ReAct Agent evaluation.
- `eval/run_baseline.py`: CLI runner for Single-Pass RAG baseline evaluation.
- `eval/compare_results.py`: Comparative analysis script generating side-by-side metric charts (`comparison_metrics.png`) and performance report (`comparison_report.md`).
- `portfolio/`: Standalone HTML/JS/CSS trajectory visualizer component for GitHub Pages embedding.
- `app/web_ui.py`: Streamlit dashboard offering an interactive playground, step-by-step trajectory inspector, and automated evaluation tab.
- `app/graph_view.py`: PyVis network visualization generator mapping entity transition graphs.
- `tests/test_agent.py`: Pytest suite verifying ReAct output parsing, baseline engine, retrieval tools, metric functions, and end-to-end agent execution.

---

## 5. End-to-End Execution Workflow

1. **Environment Setup**:
   ```bash
   conda env create -f environment.yml
   conda activate hotpot
   ```
2. **Local Model Serving**: Launch `vLLM` server in background or separate tmux/screen session.
3. **Verification**: Run `PYTHONPATH=. pytest tests/` to confirm 100% test pass rate.
4. **Single-Pass Baseline Evaluation**:
   ```bash
   python eval/run_baseline.py --samples 100 --mode offline --source official_json --output-dir eval_results/baseline
   ```
5. **ReAct Agent Evaluation**:
   ```bash
   python eval/run_eval.py --samples 100 --mode offline --source official_json --output-dir eval_results/react
   ```
6. **Side-by-Side Comparative Analysis**:
   ```bash
   python eval/compare_results.py --baseline eval_results/baseline/results.json --react eval_results/react/results.json --output-dir eval_results/comparison
   ```
7. **Visualization**: Launch Streamlit interface (`streamlit run app/web_ui.py`).

---

## 6. Final Readiness Assessment

The repository is fully updated:
- Single-Pass RAG Baseline engine and comparative analysis scripts are operational.
- Separate modular CLI entry points (`run_baseline.py`, `run_eval.py`, `compare_results.py`) ensure maximum clarity.
- All unit tests pass with zero errors.
- The project is fully self-contained and ready to be frozen, committed, and pulled into Google Compute Engine for execution.
