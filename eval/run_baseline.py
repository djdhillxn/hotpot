import sys
import os
import time
import json
import logging
import argparse
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import LLM_MODEL_NAME, OPENAI_API_KEY, OPENAI_API_BASE
from agent.baseline_rag import run_single_pass_rag
from tools.wikipedia import WikipediaToolSet
from tools.local_retriever import LocalHotpotRetriever
from eval.dataset import load_hotpot_dataset
from eval.metrics import evaluate_prediction
from eval.plot_results import generate_eval_plots_and_report


def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "run.log")

    logger = logging.getLogger("baseline_runner")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode="w")
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, log_file


def get_llm_model(model_name=LLM_MODEL_NAME, api_base=OPENAI_API_BASE, api_key=OPENAI_API_KEY):
    from langchain_openai import ChatOpenAI

    kwargs = {
        "model_name": model_name,
        "temperature": 0.0,
        "base_url": api_base,
        "api_key": api_key if api_key else "EMPTY",
    }
    return ChatOpenAI(**kwargs)


def run_baseline_benchmark(num_samples=None, mode="offline", model_name=LLM_MODEL_NAME, source="sample", api_base=OPENAI_API_BASE, output_dir="eval_results/baseline"):
    logger, log_file = setup_logger(output_dir)

    info_msg = (
        f"=== Single-Pass RAG (Direct Prompting Baseline) Benchmark ===\n"
        f"Timestamp: {datetime.now().isoformat()}\n"
        f"Model: {model_name}\n"
        f"Inference Base URL: {api_base}\n"
        f"Retrieval Mode: {mode.upper()}\n"
        f"Dataset Source: {source}\n"
        f"Samples Limit: {num_samples if num_samples else 'Full Set'}\n"
    )
    print(info_msg)
    logger.info(info_msg)

    samples = load_hotpot_dataset(num_samples=num_samples, source=source)
    llm = get_llm_model(model_name=model_name, api_base=api_base)

    results = []
    full_trajectories = []
    total_start_time = time.time()

    pbar = tqdm(samples, desc="Single-Pass RAG Evaluation", unit="question", dynamic_ncols=True)

    for idx, sample in enumerate(pbar, 1):
        question = sample["question"]
        gold_answer = sample["answer"]
        gold_titles = [f[0] for f in sample.get("supporting_facts", [])]

        logger.info(f"Question [{idx}/{len(samples)}]: {question}")
        logger.info(f"Ground Truth: {gold_answer}")

        if mode == "offline":
            toolset = LocalHotpotRetriever(context_paragraphs=sample.get("context", []))
        else:
            toolset = WikipediaToolSet()

        t0 = time.time()
        agent_state = run_single_pass_rag(question=question, llm=llm, toolset=toolset)
        latency = time.time() - t0

        pred_answer = agent_state.get("final_answer") or "No Answer"
        visited_pages = agent_state.get("visited_pages", [])

        eval_metrics = evaluate_prediction(
            prediction=pred_answer,
            ground_truth=gold_answer,
            visited_pages=visited_pages,
            gold_titles=gold_titles,
            step_count=1,
        )
        eval_metrics["latency"] = round(latency, 3)
        eval_metrics["question"] = question
        eval_metrics["pred_answer"] = pred_answer
        eval_metrics["gold_answer"] = gold_answer
        eval_metrics["timestamp"] = datetime.now().isoformat()

        results.append(eval_metrics)

        current_em = sum(r["exact_match"] for r in results) / len(results)
        current_joint_f1 = sum(r["joint_f1"] for r in results) / len(results)
        pbar.set_postfix({
            "EM": f"{current_em * 100:.1f}%",
            "Joint_F1": f"{current_joint_f1 * 100:.1f}%",
        })

        logger.info(
            f"Prediction: '{pred_answer}' | EM: {eval_metrics['exact_match']} | "
            f"F1: {eval_metrics['f1']:.3f} | Joint F1: {eval_metrics['joint_f1']:.3f} | "
            f"Latency: {latency:.2f}s"
        )

        trajectory_entry = {
            "id": sample.get("id", f"sample_{idx}"),
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "ground_truth": gold_answer,
            "predicted_answer": pred_answer,
            "exact_match": eval_metrics["exact_match"],
            "joint_f1": round(eval_metrics["joint_f1"], 3),
            "step_count": 1,
            "latency_seconds": round(latency, 3),
            "visited_pages": visited_pages,
            "evidence_graph": agent_state.get("evidence_graph", []),
            "steps": agent_state.get("steps", []),
        }
        full_trajectories.append(trajectory_entry)

    total_time = time.time() - total_start_time
    total_count = len(results)
    avg_em = sum(r["exact_match"] for r in results) / total_count
    avg_f1 = sum(r["f1"] for r in results) / total_count
    avg_sp_f1 = sum(r["sp_f1"] for r in results) / total_count
    avg_joint_em = sum(r["joint_em"] for r in results) / total_count
    avg_joint_f1 = sum(r["joint_f1"] for r in results) / total_count
    avg_lat = sum(r["latency"] for r in results) / total_count

    summary_text = (
        "\n=== SINGLE-PASS RAG BASELINE METRICS ===\n"
        f"Answer Exact Match (EM):   {avg_em * 100:.1f}%\n"
        f"Answer F1 Score:           {avg_f1 * 100:.1f}%\n"
        f"Supporting Facts F1:       {avg_sp_f1 * 100:.1f}%\n"
        f"Joint Exact Match (Joint EM): {avg_joint_em * 100:.1f}%\n"
        f"Joint F1 Score (Joint F1):   {avg_joint_f1 * 100:.1f}%\n"
        f"Avg Latency / Question:    {avg_lat:.2f}s\n"
        f"Total Evaluation Time:     {total_time:.2f}s\n"
    )
    print(summary_text)
    logger.info(summary_text)

    results_json_path = os.path.join(output_dir, "results.json")
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2)

    trajectories_json_path = os.path.join(output_dir, "trajectories.json")
    with open(trajectories_json_path, "w") as f:
        json.dump(full_trajectories, f, indent=2)

    print(f"Saved Single-Pass RAG raw JSON to: {results_json_path}")
    print(f"Saved Single-Pass RAG trajectories JSON to: {trajectories_json_path}")
    print(f"Saved execution log file to: {log_file}")

    generate_eval_plots_and_report(results, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Single-Pass RAG Baseline Benchmark")
    parser.add_argument("--samples", type=int, default=None, help="Number of questions to test (default: all)")
    parser.add_argument("--mode", choices=["offline", "live"], default="offline", help="Retrieval mode")
    parser.add_argument("--source", choices=["sample", "huggingface", "official_json"], default="sample", help="Dataset source")
    parser.add_argument("--model", type=str, default=LLM_MODEL_NAME, help="LLM model name")
    parser.add_argument("--api-base", type=str, default=OPENAI_API_BASE, help="Local vLLM / OpenAI server URL")
    parser.add_argument("--output-dir", type=str, default="eval_results/baseline", help="Directory for baseline outputs")

    args = parser.parse_args()
    run_baseline_benchmark(
        num_samples=args.samples,
        mode=args.mode,
        model_name=args.model,
        source=args.source,
        api_base=args.api_base,
        output_dir=args.output_dir,
    )
