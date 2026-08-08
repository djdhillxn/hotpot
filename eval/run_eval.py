import sys
import os
import time
import json
import logging
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import LLM_MODEL_NAME, OPENAI_API_KEY, OPENAI_API_BASE, MAX_AGENT_HOPS
from agent.engine import run_react_agent
from tools.wikipedia import WikipediaToolSet
from tools.local_retriever import LocalHotpotRetriever
from eval.dataset import load_hotpot_dataset
from eval.metrics import evaluate_prediction
from eval.plot_results import generate_eval_plots_and_report


def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "run.log")

    logger = logging.getLogger("eval_runner")
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


def process_single_question(sample, idx, total, mode, llm, max_hops, logger):
    question = sample["question"]
    gold_answer = sample["answer"]
    gold_titles = [f[0] for f in sample.get("supporting_facts", [])]

    if mode == "offline":
        toolset = LocalHotpotRetriever(context_paragraphs=sample.get("context", []))
    else:
        toolset = WikipediaToolSet()

    t0 = time.time()
    agent_state = run_react_agent(question=question, llm=llm, toolset=toolset, max_hops=max_hops)
    latency = time.time() - t0

    pred_answer = agent_state.get("final_answer") or "No Answer"
    visited_pages = agent_state.get("visited_pages", [])
    step_count = agent_state.get("step_count", 0)

    eval_metrics = evaluate_prediction(
        prediction=pred_answer,
        ground_truth=gold_answer,
        visited_pages=visited_pages,
        gold_titles=gold_titles,
        step_count=step_count,
    )
    eval_metrics["latency"] = round(latency, 3)
    eval_metrics["question"] = question
    eval_metrics["pred_answer"] = pred_answer
    eval_metrics["gold_answer"] = gold_answer
    eval_metrics["timestamp"] = datetime.now().isoformat()
    eval_metrics["idx"] = idx

    logger.info(
        f"Question [{idx}/{total}]: '{question}' | Pred: '{pred_answer}' | Gold: '{gold_answer}' | "
        f"EM: {eval_metrics['exact_match']} | Joint F1: {eval_metrics['joint_f1']:.3f} | "
        f"Steps: {step_count} | Latency: {latency:.2f}s"
    )

    trajectory_entry = {
        "id": sample.get("id", f"sample_{idx}"),
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "ground_truth": gold_answer,
        "predicted_answer": pred_answer,
        "exact_match": eval_metrics["exact_match"],
        "joint_f1": round(eval_metrics["joint_f1"], 3),
        "step_count": step_count,
        "latency_seconds": round(latency, 3),
        "visited_pages": visited_pages,
        "evidence_graph": agent_state.get("evidence_graph", []),
        "steps": agent_state.get("steps", []),
        "idx": idx,
    }

    return eval_metrics, trajectory_entry


def run_benchmark(num_samples=None, mode="offline", model_name=LLM_MODEL_NAME, source="sample", api_base=OPENAI_API_BASE, output_dir="eval_results/react", concurrency=16, max_hops=MAX_AGENT_HOPS):
    logger, log_file = setup_logger(output_dir)

    info_msg = (
        f"=== ReAct Agent HotpotQA Benchmark ===\n"
        f"Timestamp: {datetime.now().isoformat()}\n"
        f"Model: {model_name}\n"
        f"Inference Base URL: {api_base}\n"
        f"Retrieval Mode: {mode.upper()}\n"
        f"Dataset Source: {source}\n"
        f"Samples Limit: {num_samples if num_samples else 'Full Set'}\n"
        f"Concurrency Workers: {concurrency}\n"
        f"Max Hops Limit: {max_hops}\n"
    )
    print(info_msg)
    logger.info(info_msg)

    samples = load_hotpot_dataset(num_samples=num_samples, source=source)
    llm = get_llm_model(model_name=model_name, api_base=api_base)

    results = []
    full_trajectories = []
    total_start_time = time.time()
    total = len(samples)

    pbar = tqdm(total=total, desc="ReAct Agent Evaluation", unit="question", dynamic_ncols=True)

    if concurrency <= 1:
        for idx, sample in enumerate(samples, 1):
            eval_metrics, trajectory_entry = process_single_question(sample, idx, total, mode, llm, max_hops, logger)
            results.append(eval_metrics)
            full_trajectories.append(trajectory_entry)

            current_em = sum(r["exact_match"] for r in results) / len(results)
            current_joint_f1 = sum(r["joint_f1"] for r in results) / len(results)
            pbar.set_postfix({
                "EM": f"{current_em * 100:.1f}%",
                "Joint_F1": f"{current_joint_f1 * 100:.1f}%",
                "Steps": eval_metrics["step_count"],
            })
            pbar.update(1)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_idx = {
                executor.submit(process_single_question, sample, idx, total, mode, llm, max_hops, logger): idx
                for idx, sample in enumerate(samples, 1)
            }

            for future in as_completed(future_to_idx):
                try:
                    eval_metrics, trajectory_entry = future.result()
                    results.append(eval_metrics)
                    full_trajectories.append(trajectory_entry)

                    current_em = sum(r["exact_match"] for r in results) / len(results)
                    current_joint_f1 = sum(r["joint_f1"] for r in results) / len(results)
                    pbar.set_postfix({
                        "EM": f"{current_em * 100:.1f}%",
                        "Joint_F1": f"{current_joint_f1 * 100:.1f}%",
                        "Steps": eval_metrics["step_count"],
                    })
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Error processing question: {str(e)}")
                    pbar.update(1)

    results.sort(key=lambda r: r.get("idx", 0))
    full_trajectories.sort(key=lambda t: t.get("idx", 0))

    total_time = time.time() - total_start_time
    total_count = len(results)
    avg_em = sum(r["exact_match"] for r in results) / total_count if total_count else 0
    avg_f1 = sum(r["f1"] for r in results) / total_count if total_count else 0
    avg_sp_f1 = sum(r["sp_f1"] for r in results) / total_count if total_count else 0
    avg_joint_em = sum(r["joint_em"] for r in results) / total_count if total_count else 0
    avg_joint_f1 = sum(r["joint_f1"] for r in results) / total_count if total_count else 0
    avg_steps = sum(r["step_count"] for r in results) / total_count if total_count else 0
    avg_lat = sum(r["latency"] for r in results) / total_count if total_count else 0

    summary_text = (
        "\n=== OFFICIAL HOTPOTQA LEADERBOARD METRICS ===\n"
        f"Answer Exact Match (EM):   {avg_em * 100:.1f}%\n"
        f"Answer F1 Score:           {avg_f1 * 100:.1f}%\n"
        f"Supporting Facts F1:       {avg_sp_f1 * 100:.1f}%\n"
        f"Joint Exact Match (Joint EM): {avg_joint_em * 100:.1f}%\n"
        f"Joint F1 Score (Joint F1):   {avg_joint_f1 * 100:.1f}%\n"
        f"Avg Hops / Question:       {avg_steps:.2f}\n"
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

    portfolio_dir = os.path.join(os.path.dirname(__file__), "..", "portfolio")
    os.makedirs(portfolio_dir, exist_ok=True)
    portfolio_json_path = os.path.join(portfolio_dir, "portfolio_trajectories.json")
    with open(portfolio_json_path, "w") as f:
        json.dump(full_trajectories, f, indent=2)

    print(f"Saved evaluation raw JSON to: {results_json_path}")
    print(f"Saved ReAct trajectories JSON to: {trajectories_json_path}")
    print(f"Saved execution log file to: {log_file}")
    print(f"Exported portfolio trajectories to: {portfolio_json_path}")

    generate_eval_plots_and_report(results, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ReAct HotpotQA Benchmark")
    parser.add_argument("--samples", type=int, default=None, help="Number of questions to test (default: all)")
    parser.add_argument("--mode", choices=["offline", "live"], default="offline", help="Retrieval mode")
    parser.add_argument("--source", choices=["sample", "huggingface", "official_json"], default="sample", help="Dataset source")
    parser.add_argument("--model", type=str, default=LLM_MODEL_NAME, help="LLM model name")
    parser.add_argument("--api-base", type=str, default=OPENAI_API_BASE, help="Local vLLM / OpenAI server URL")
    parser.add_argument("--output-dir", type=str, default="eval_results/react", help="Directory for outputs")
    parser.add_argument("--concurrency", type=int, default=16, help="Number of concurrent worker threads")
    parser.add_argument("--max-hops", type=int, default=MAX_AGENT_HOPS, help="Maximum hops per question")

    args = parser.parse_args()
    run_benchmark(
        num_samples=args.samples,
        mode=args.mode,
        model_name=args.model,
        source=args.source,
        api_base=args.api_base,
        output_dir=args.output_dir,
        concurrency=args.concurrency,
        max_hops=args.max_hops,
    )
