import sys
import os
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import LLM_MODEL_NAME, OPENAI_API_KEY, OPENAI_API_BASE, MAX_AGENT_HOPS
from agent.engine import run_react_agent
from tools.wikipedia import WikipediaToolSet
from tools.local_retriever import LocalHotpotRetriever
from eval.dataset import SAMPLE_HOTPOT_QUESTIONS, load_hotpot_dataset
from eval.metrics import evaluate_prediction
from app.graph_view import render_evidence_graph

st.set_page_config(
    page_title="ReAct Multi-Hop QA Agent | HotpotQA FullWiki",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .stAppHeader {
        background-color: rgba(14, 17, 23, 0.8);
    }
    .thought-box {
        background-color: #1e222d;
        border-left: 4px solid #3b82f6;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    .action-box {
        background-color: #262b3a;
        border-left: 4px solid #10b981;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    .obs-box {
        background-color: #1a1e29;
        border-left: 4px solid #f59e0b;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 12px;
        font-family: monospace;
        font-size: 0.9em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_llm(model_name, api_key, api_base):
    from langchain_openai import ChatOpenAI
    kwargs = {
        "model_name": model_name,
        "temperature": 0.0,
        "base_url": api_base if api_base else "http://localhost:8000/v1",
        "api_key": api_key if api_key else "EMPTY",
    }
    return ChatOpenAI(**kwargs)


st.sidebar.title("ReAct Agent Configuration")
st.sidebar.markdown("FullWiki Multi-Hop QA Engine built with LangGraph.")

retrieval_mode = st.sidebar.radio(
    "Retrieval Backend",
    options=["Live Wikipedia API", "Offline HotpotQA Context"],
    index=0,
)

model_choice = st.sidebar.text_input("LLM Model Name", value=LLM_MODEL_NAME)
api_base_input = st.sidebar.text_input("Local vLLM Base URL", value=OPENAI_API_BASE, placeholder="http://localhost:8000/v1")
api_key_input = st.sidebar.text_input("API Key (Optional for vLLM)", value=OPENAI_API_KEY if OPENAI_API_KEY != "EMPTY" else "", type="password")
max_hops = st.sidebar.slider("Max Agent Hops", min_value=3, max_value=10, value=MAX_AGENT_HOPS)

st.sidebar.divider()
st.sidebar.markdown("### System Pipeline")
st.sidebar.markdown("- ReAct Reasoning Loop: Thought -> Action -> Observation")
st.sidebar.markdown("- Entity Knowledge Graph Visualizer")
st.sidebar.markdown("- Official HotpotQA Leaderboard Metrics")

st.title("ReAct Multi-Hop Question Answering Agent")
st.caption("Implementation for HotpotQA FullWiki benchmark using LangGraph and Local vLLM Inference")

tab1, tab2 = st.tabs(["Interactive Playground and Trajectory Visualizer", "Benchmark Evaluation"])

with tab1:
    st.subheader("Select or Enter a Multi-Hop Question")

    preset_questions = ["Custom Question..."] + [q["question"] for q in SAMPLE_HOTPOT_QUESTIONS]
    selected_preset = st.selectbox("Preset HotpotQA Questions", options=preset_questions)

    if selected_preset != "Custom Question...":
        default_q = selected_preset
    else:
        default_q = "Were Scott Derrickson and Ed Wood born in the same state?"

    user_question = st.text_area("Question", value=default_q, height=70)

    if st.button("Run ReAct Agent", type="primary"):
        with st.spinner("Executing ReAct Reasoning Loop..."):
            try:
                llm = get_llm(model_choice, api_key_input, api_base_input)

                if retrieval_mode == "Live Wikipedia API":
                    toolset = WikipediaToolSet()
                else:
                    matched = next((q for q in SAMPLE_HOTPOT_QUESTIONS if q["question"] == user_question), None)
                    context = matched.get("context", []) if matched else []
                    toolset = LocalHotpotRetriever(context_paragraphs=context)

                final_state = run_react_agent(
                    question=user_question, llm=llm, toolset=toolset, max_hops=max_hops
                )

                st.success("ReAct Agent Execution Complete.")

                col1, col2 = st.columns([3, 2])

                with col1:
                    st.markdown("### Final Answer")
                    final_ans = final_state.get("final_answer") or "No Answer Found"
                    st.info(f"**{final_ans}**")

                    preset_match = next((q for q in SAMPLE_HOTPOT_QUESTIONS if q["question"] == user_question), None)
                    if preset_match:
                        gold_ans = preset_match["answer"]
                        metrics = evaluate_prediction(
                            prediction=final_ans,
                            ground_truth=gold_ans,
                            visited_pages=final_state.get("visited_pages", []),
                            gold_titles=[f[0] for f in preset_match.get("supporting_facts", [])],
                            step_count=final_state.get("step_count", 0),
                        )
                        st.caption(f"Ground Truth: **{gold_ans}** | Exact Match: **{metrics['exact_match']}** | Joint F1: **{metrics['joint_f1']:.2f}**")

                    st.markdown("### Step-by-Step ReAct Trajectory")
                    steps = final_state.get("steps", [])

                    for step in steps:
                        with st.expander(f"Step {step['step']}: {step['action']}", expanded=True):
                            st.markdown(f"<div class='thought-box'><b>Thought:</b> {step['thought']}</div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='action-box'><b>Action:</b> <code>{step['action']}</code></div>", unsafe_allow_html=True)
                            if step['observation']:
                                st.markdown(f"<div class='obs-box'><b>Observation:</b><br>{step['observation']}</div>", unsafe_allow_html=True)

                with col2:
                    st.markdown("### Wikipedia Evidence Graph")
                    evidence_graph = final_state.get("evidence_graph", [])
                    if evidence_graph:
                        html_file = render_evidence_graph(evidence_graph)
                        with open(html_file, "r") as f:
                            graph_html = f.read()
                        components.html(graph_html, height=420)
                    else:
                        st.caption("No entity transitions recorded.")

                    st.markdown("### Retrived Entity Pages")
                    for page in final_state.get("visited_pages", []):
                        st.markdown(f"- `{page}`")

            except Exception as e:
                st.error(f"Execution Error: {str(e)}")

with tab2:
    st.subheader("Automated HotpotQA Benchmark Suite")
    st.markdown("Run quantitative evaluation over HotpotQA validation questions.")

    eval_source = st.selectbox("Dataset Source", options=["sample", "huggingface", "official_json"])
    num_eval_samples = st.number_input("Evaluation Samples Limit (0 = Full Dataset)", min_value=0, max_value=7405, value=4)

    if st.button("Run Evaluation Suite"):
        with st.spinner("Running evaluation suite..."):
            try:
                llm = get_llm(model_choice, api_key_input, api_base_input)
                limit = num_eval_samples if num_eval_samples > 0 else None
                samples = load_hotpot_dataset(num_samples=limit, source=eval_source)

                results = []
                progress_bar = st.progress(0)

                for idx, s in enumerate(samples):
                    toolset = LocalHotpotRetriever(context_paragraphs=s.get("context", []))
                    state = run_react_agent(question=s["question"], llm=llm, toolset=toolset)

                    pred = state.get("final_answer") or ""
                    eval_m = evaluate_prediction(
                        prediction=pred,
                        ground_truth=s["answer"],
                        visited_pages=state.get("visited_pages", []),
                        gold_titles=[f[0] for f in s.get("supporting_facts", [])],
                        step_count=state.get("step_count", 0),
                    )
                    eval_m["question"] = s["question"]
                    eval_m["pred"] = pred
                    eval_m["gold"] = s["answer"]
                    results.append(eval_m)

                    progress_bar.progress((idx + 1) / len(samples))

                avg_em = sum(r["exact_match"] for r in results) / len(results)
                avg_f1 = sum(r["f1"] for r in results) / len(results)
                avg_joint_f1 = sum(r["joint_f1"] for r in results) / len(results)
                avg_steps = sum(r["step_count"] for r in results) / len(results)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Answer EM", f"{avg_em * 100:.1f}%")
                m2.metric("Answer F1", f"{avg_f1 * 100:.1f}%")
                m3.metric("Joint F1", f"{avg_joint_f1 * 100:.1f}%")
                m4.metric("Avg Hops", f"{avg_steps:.2f}")

                st.markdown("### Per-Question Breakdown")
                table_data = []
                for r in results:
                    table_data.append({
                        "Question": r["question"],
                        "Prediction": r["pred"],
                        "Ground Truth": r["gold"],
                        "EM": "YES" if r["exact_match"] else "NO",
                        "F1": f"{r['f1']:.2f}",
                        "Joint F1": f"{r['joint_f1']:.2f}",
                        "Steps": r["step_count"],
                    })
                st.dataframe(table_data, use_container_width=True)

            except Exception as e:
                st.error(f"Benchmark Error: {str(e)}")
