import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.engine import run_react_agent
from config import (
    FULLWIKI_INDEX_DIR,
    LLM_MODEL_NAME,
    MAX_AGENT_HOPS,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    REACT_MAX_EVIDENCE_DOCUMENTS,
    REACT_MAX_OBSERVATION_CHARS,
    REACT_SEARCH_TOP_K,
    REACT_MEMORY_RERANKER_BATCH_SIZE,
    REACT_MEMORY_RERANKER_DEVICE,
    REACT_MEMORY_RERANKER_MAX_LENGTH,
    REACT_MEMORY_RERANKER_MODEL,
)
from eval.artifacts import (
    context_diagnostics,
    observed_evidence_diagnostics,
    save_portfolio_trajectories,
    write_official_files,
    write_run_manifest,
)
from eval.dataset import load_hotpot_dataset
from eval.metrics import (
    compute_metrics_by_type,
    evaluate_prediction,
    print_segmented_report,
)
from eval.plot_results import generate_eval_plots_and_report
from tools.local_retriever import LocalHotpotRetriever
from tools.wikipedia import WikipediaToolSet
from retrieval.fullwiki_retriever import FullWikiSearchBackend


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
        "extra_body": {
            "temperature": 0.0,
            "top_p": 1.0,
        },
    }
    return ChatOpenAI(**kwargs)


def _sample_metadata(sample):
    gold_supporting_facts = sample.get("supporting_facts", []) or []
    gold_titles = list(dict.fromkeys(
        fact[0]
        for fact in gold_supporting_facts
        if isinstance(fact, (list, tuple)) and len(fact) == 2
    ))
    return gold_supporting_facts, gold_titles, context_diagnostics(sample)



def process_single_question(
    sample, idx, total, mode, llm, max_hops, logger, fullwiki_backend=None,
    search_top_k=REACT_SEARCH_TOP_K, max_evidence_documents=REACT_MAX_EVIDENCE_DOCUMENTS,
    max_observation_chars=REACT_MAX_OBSERVATION_CHARS,
):
    question = sample["question"]
    gold_answer = sample["answer"]
    gold_supporting_facts, gold_titles, context_info = _sample_metadata(sample)

    if mode == "offline":
        toolset = LocalHotpotRetriever(context_paragraphs=sample.get("context", []))
    elif mode == "fullwiki":
        if fullwiki_backend is None:
            raise RuntimeError("FullWiki backend was not initialized.")
        toolset = fullwiki_backend.create_session(
            search_top_k=search_top_k,
            max_observation_chars=max_observation_chars,
            max_evidence_documents=max_evidence_documents,
            duplicate_search_guard=True,
            question=question,
        )
    else:
        toolset = WikipediaToolSet()

    t0 = time.time()
    agent_state = run_react_agent(question=question, llm=llm, toolset=toolset, max_hops=max_hops)
    latency = time.time() - t0

    pred_answer = agent_state.get("final_answer") or "No Answer"
    predicted_supporting_facts = agent_state.get("predicted_supporting_facts", []) or []
    observed_supporting_facts = agent_state.get("observed_supporting_facts", []) or []
    invalid_supporting_facts = agent_state.get("invalid_supporting_facts", []) or []
    visited_pages = agent_state.get("visited_pages", [])
    step_count = agent_state.get("step_count", 0)
    active_memory_documents = (
        toolset.active_memory_snapshot() if hasattr(toolset, "active_memory_snapshot") else []
    )
    evidence_archive_count = (
        toolset.evidence_archive_count if hasattr(toolset, "evidence_archive_count") else None
    )
    observed_info = observed_evidence_diagnostics(
        gold_supporting_facts, gold_titles, observed_supporting_facts, visited_pages
    )

    eval_metrics = evaluate_prediction(
        prediction=pred_answer,
        ground_truth=gold_answer,
        predicted_supporting_facts=predicted_supporting_facts,
        gold_supporting_facts=gold_supporting_facts,
        visited_pages=visited_pages,
        gold_titles=gold_titles,
        step_count=step_count,
    )
    eval_metrics.update({
        "id": str(sample.get("id", f"sample_{idx}")),
        "idx": idx,
        "question": question,
        "question_type": sample.get("type", "unknown"),
        "difficulty_level": sample.get("level", "unknown"),
        "pred_answer": pred_answer,
        "gold_answer": gold_answer,
        "predicted_supporting_facts": predicted_supporting_facts,
        "observed_supporting_facts": observed_supporting_facts,
        "invalid_supporting_facts": invalid_supporting_facts,
        "invalid_supporting_fact_count": len(invalid_supporting_facts),
        "gold_supporting_facts": gold_supporting_facts,
        "visited_pages": visited_pages,
        "active_memory_documents": active_memory_documents,
        "evidence_archive_count": evidence_archive_count,
        "latency": round(latency, 3),
        "timestamp": datetime.now().isoformat(),
        **observed_info,
        **context_info,
    })

    logger.info(
        f"Question [{idx}/{total}]: '{question}' | Pred: '{pred_answer}' | Gold: '{gold_answer}' | "
        f"EM: {eval_metrics['exact_match']} | SP F1: {eval_metrics['sp_f1']:.3f} | "
        f"Joint F1: {eval_metrics['joint_f1']:.3f} | Steps: {step_count} | Latency: {latency:.2f}s"
    )

    trajectory_entry = {
        "id": eval_metrics["id"],
        "idx": idx,
        "timestamp": eval_metrics["timestamp"],
        "question": question,
        "question_type": eval_metrics["question_type"],
        "difficulty_level": eval_metrics["difficulty_level"],
        "ground_truth": gold_answer,
        "predicted_answer": pred_answer,
        "gold_supporting_facts": gold_supporting_facts,
        "predicted_supporting_facts": predicted_supporting_facts,
        "observed_supporting_facts": observed_supporting_facts,
        "invalid_supporting_facts": invalid_supporting_facts,
        "exact_match": eval_metrics["exact_match"],
        "answer_f1": round(eval_metrics["f1"], 6),
        "supporting_fact_em": eval_metrics["sp_em"],
        "supporting_fact_f1": round(eval_metrics["sp_f1"], 6),
        "joint_em": eval_metrics["joint_em"],
        "joint_f1": round(eval_metrics["joint_f1"], 6),
        "supporting_document_f1": round(eval_metrics["doc_f1"], 6),
        "step_count": step_count,
        "latency_seconds": round(latency, 3),
        "visited_pages": visited_pages,
        "active_memory_documents": active_memory_documents,
        "evidence_archive_count": evidence_archive_count,
        "observed_gold_document_recall": observed_info["observed_gold_document_recall"],
        "observed_gold_supporting_fact_recall": observed_info["observed_gold_supporting_fact_recall"],
        "all_gold_supporting_facts_observed": observed_info["all_gold_supporting_facts_observed"],
        "hotpot_supplied_context_titles": context_info["hotpot_supplied_context_titles"],
        "gold_titles_in_hotpot_supplied_context": context_info["gold_titles_in_hotpot_supplied_context"],
        "hotpot_supplied_context_gold_document_recall": context_info["hotpot_supplied_context_gold_document_recall"],
        "hotpot_supplied_context_gold_supporting_fact_recall": context_info["hotpot_supplied_context_gold_supporting_fact_recall"],
        "hotpot_supplied_context_has_all_gold_supporting_facts": context_info["hotpot_supplied_context_has_all_gold_supporting_facts"],
        "evidence_graph": agent_state.get("evidence_graph", []),
        "steps": agent_state.get("steps", []),
        "error": agent_state.get("error"),
    }

    return eval_metrics, trajectory_entry


def build_failure_record(sample, idx, error):
    gold_answer = sample.get("answer", "")
    gold_supporting_facts, gold_titles, context_info = _sample_metadata(sample)
    metrics = evaluate_prediction(
        prediction="No Answer",
        ground_truth=gold_answer,
        predicted_supporting_facts=[],
        gold_supporting_facts=gold_supporting_facts,
        visited_pages=[],
        gold_titles=gold_titles,
        step_count=0,
    )
    timestamp = datetime.now().isoformat()
    qid = str(sample.get("id", f"sample_{idx}"))
    metrics.update({
        "id": qid,
        "idx": idx,
        "question": sample.get("question", ""),
        "question_type": sample.get("type", "unknown"),
        "difficulty_level": sample.get("level", "unknown"),
        "pred_answer": "No Answer",
        "gold_answer": gold_answer,
        "predicted_supporting_facts": [],
        "observed_supporting_facts": [],
        "invalid_supporting_facts": [],
        "invalid_supporting_fact_count": 0,
        "gold_supporting_facts": gold_supporting_facts,
        "visited_pages": [],
        "latency": 0.0,
        "timestamp": timestamp,
        "failed": True,
        "error": str(error),
        "observed_gold_document_recall": 0.0 if gold_titles else 1.0,
        "observed_gold_supporting_fact_recall": 0.0 if gold_supporting_facts else 1.0,
        "all_gold_supporting_facts_observed": not bool(gold_supporting_facts),
        **context_info,
    })
    trajectory = {
        "id": qid,
        "idx": idx,
        "timestamp": timestamp,
        "question": sample.get("question", ""),
        "question_type": sample.get("type", "unknown"),
        "difficulty_level": sample.get("level", "unknown"),
        "ground_truth": gold_answer,
        "predicted_answer": "No Answer",
        "gold_supporting_facts": gold_supporting_facts,
        "predicted_supporting_facts": [],
        "observed_supporting_facts": [],
        "invalid_supporting_facts": [],
        "exact_match": metrics["exact_match"],
        "answer_f1": metrics["f1"],
        "supporting_fact_em": metrics["sp_em"],
        "supporting_fact_f1": metrics["sp_f1"],
        "joint_em": metrics["joint_em"],
        "joint_f1": metrics["joint_f1"],
        "supporting_document_f1": metrics["doc_f1"],
        "step_count": 0,
        "latency_seconds": 0.0,
        "visited_pages": [],
        "observed_gold_document_recall": 0.0 if gold_titles else 1.0,
        "observed_gold_supporting_fact_recall": 0.0 if gold_supporting_facts else 1.0,
        "all_gold_supporting_facts_observed": not bool(gold_supporting_facts),
        "hotpot_supplied_context_titles": context_info["hotpot_supplied_context_titles"],
        "gold_titles_in_hotpot_supplied_context": context_info["gold_titles_in_hotpot_supplied_context"],
        "hotpot_supplied_context_gold_document_recall": context_info["hotpot_supplied_context_gold_document_recall"],
        "hotpot_supplied_context_gold_supporting_fact_recall": context_info["hotpot_supplied_context_gold_supporting_fact_recall"],
        "hotpot_supplied_context_has_all_gold_supporting_facts": context_info["hotpot_supplied_context_has_all_gold_supporting_facts"],
        "evidence_graph": [],
        "steps": [],
        "error": str(error),
    }
    return metrics, trajectory


from config import (
    FULLWIKI_INDEX_DIR,
    FULLWIKI_RRF_K,
    FULLWIKI_SEARCH_CANDIDATES,
    LLM_MODEL_NAME,
    MAX_AGENT_HOPS,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    REACT_MAX_EVIDENCE_DOCUMENTS,
    REACT_MAX_OBSERVATION_CHARS,
    REACT_SEARCH_TOP_K,
    REACT_MEMORY_RERANKER_BATCH_SIZE,
    REACT_MEMORY_RERANKER_DEVICE,
    REACT_MEMORY_RERANKER_MAX_LENGTH,
    REACT_MEMORY_RERANKER_MODEL,
)


def run_benchmark(
    num_samples=None,
    mode="fullwiki",
    model_name=LLM_MODEL_NAME,
    source="sample",
    api_base=OPENAI_API_BASE,
    output_dir="eval_results/react",
    concurrency=16,
    max_hops=MAX_AGENT_HOPS,
    retriever="hybrid",
    candidate_k=FULLWIKI_SEARCH_CANDIDATES,
    rrf_k=FULLWIKI_RRF_K,
    index_dir=FULLWIKI_INDEX_DIR,
    search_top_k=REACT_SEARCH_TOP_K,
    max_evidence_documents=REACT_MAX_EVIDENCE_DOCUMENTS,
    max_observation_chars=REACT_MAX_OBSERVATION_CHARS,
    reranker_model=None,
    reranker_device=None,
    reranker_max_length=None,
    reranker_batch_size=None,
):
    reranker_model = reranker_model or REACT_MEMORY_RERANKER_MODEL
    reranker_device = reranker_device or REACT_MEMORY_RERANKER_DEVICE
    reranker_max_length = reranker_max_length or REACT_MEMORY_RERANKER_MAX_LENGTH
    reranker_batch_size = reranker_batch_size or REACT_MEMORY_RERANKER_BATCH_SIZE

    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if search_top_k < 1:
        raise ValueError("search_top_k must be >= 1")
    if max_evidence_documents < 1:
        raise ValueError("max_evidence_documents must be >= 1")
    if max_observation_chars < 1:
        raise ValueError("max_observation_chars must be >= 1")

    logger, log_file = setup_logger(output_dir)
    run_started_at = datetime.now().isoformat()

    info_msg = (
        f"=== ReAct Agent HotpotQA Benchmark ===\n"
        f"Timestamp: {run_started_at}\n"
        f"Model: {model_name}\n"
        f"Inference Base URL: {api_base}\n"
        f"Retrieval Mode: {mode.upper()}\n"
        f"Dataset Source: {source}\n"
        f"Samples Limit: {num_samples if num_samples else 'Full Set'}\n"
        f"Concurrency Workers: {concurrency}\n"
        f"Max Hops Limit: {max_hops}\n"
        f"Retriever: {retriever if mode == 'fullwiki' else 'n/a'}\n"
        f"Documents per adaptive search: {search_top_k if mode == 'fullwiki' else 1}\n"
        f"Max unique working-evidence documents: {max_evidence_documents if mode == 'fullwiki' else 'n/a'}\n"
        f"Max characters per retrieval observation: {max_observation_chars if mode == 'fullwiki' else 'n/a'}\n"
    )
    if mode == "live":
        info_msg += (
            "NOTE: Official supporting-fact metrics are only benchmark-comparable when sentence IDs "
            "come from HotpotQA-aligned context/corpus. Live Wikipedia mode is qualitative.\n"
        )
    print(info_msg)
    logger.info(info_msg)

    samples = load_hotpot_dataset(num_samples=num_samples, source=source)
    llm = get_llm_model(model_name=model_name, api_base=api_base)
    fullwiki_backend = None
    if mode == "fullwiki":
        fullwiki_backend = FullWikiSearchBackend(
            bm25_index_dir=os.path.join(index_dir, "bm25"),
            dense_index_path=os.path.join(index_dir, "dense.faiss"),
            manifest_path=os.path.join(index_dir, "manifest.json"),
            mode=retriever,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            evidence_reranker_model=reranker_model,
            evidence_reranker_device=reranker_device,
            evidence_reranker_max_length=reranker_max_length,
            evidence_reranker_batch_size=reranker_batch_size,
        )

    results = []
    full_trajectories = []
    total_start_time = time.time()
    total = len(samples)

    pbar = tqdm(total=total, desc="ReAct Agent Evaluation", unit="question", dynamic_ncols=True)

    if concurrency <= 1:
        for idx, sample in enumerate(samples, 1):
            try:
                eval_metrics, trajectory_entry = process_single_question(
                    sample, idx, total, mode, llm, max_hops, logger, fullwiki_backend, search_top_k,
                    max_evidence_documents, max_observation_chars
                )
            except Exception as exc:
                logger.exception(f"Error processing question {idx}: {exc}")
                eval_metrics, trajectory_entry = build_failure_record(sample, idx, exc)
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
            future_to_meta = {
                executor.submit(
                    process_single_question, sample, idx, total, mode, llm, max_hops, logger,
                    fullwiki_backend, search_top_k, max_evidence_documents, max_observation_chars
                ): (idx, sample)
                for idx, sample in enumerate(samples, 1)
            }

            for future in as_completed(future_to_meta):
                idx, sample = future_to_meta[future]
                try:
                    eval_metrics, trajectory_entry = future.result()
                except Exception as exc:
                    logger.exception(f"Error processing question {idx}: {exc}")
                    eval_metrics, trajectory_entry = build_failure_record(sample, idx, exc)

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

    pbar.close()
    results.sort(key=lambda r: r.get("idx", 0))
    full_trajectories.sort(key=lambda t: t.get("idx", 0))

    total_time = time.time() - total_start_time
    total_count = len(results)
    avg_em = sum(r["exact_match"] for r in results) / total_count if total_count else 0
    avg_f1 = sum(r["f1"] for r in results) / total_count if total_count else 0
    avg_sp_em = sum(r["sp_em"] for r in results) / total_count if total_count else 0
    avg_sp_f1 = sum(r["sp_f1"] for r in results) / total_count if total_count else 0
    avg_joint_em = sum(r["joint_em"] for r in results) / total_count if total_count else 0
    avg_joint_f1 = sum(r["joint_f1"] for r in results) / total_count if total_count else 0
    avg_doc_f1 = sum(r["doc_f1"] for r in results) / total_count if total_count else 0
    avg_steps = sum(r["step_count"] for r in results) / total_count if total_count else 0
    avg_lat = sum(r["latency"] for r in results) / total_count if total_count else 0
    avg_observed_doc_recall = sum(r["observed_gold_document_recall"] for r in results) / total_count if total_count else 0
    avg_observed_sp_recall = sum(r["observed_gold_supporting_fact_recall"] for r in results) / total_count if total_count else 0
    full_observed_count = sum(r["all_gold_supporting_facts_observed"] for r in results)
    avg_supplied_context_sp_recall = sum(r["hotpot_supplied_context_gold_supporting_fact_recall"] for r in results) / total_count if total_count else 0
    full_supplied_context_count = sum(r["hotpot_supplied_context_has_all_gold_supporting_facts"] for r in results)
    failed_count = sum(bool(r.get("failed")) for r in results)

    summary_text = (
        "\n=== HOTPOTQA DEV METRICS (OFFICIAL FORMULAS) ===\n"
        f"Answer Exact Match (EM):      {avg_em * 100:.1f}%\n"
        f"Answer F1 Score:              {avg_f1 * 100:.1f}%\n"
        f"Supporting Facts EM:          {avg_sp_em * 100:.1f}%\n"
        f"Supporting Facts F1:          {avg_sp_f1 * 100:.1f}%\n"
        f"Joint Exact Match (Joint EM): {avg_joint_em * 100:.1f}%\n"
        f"Joint F1 Score (Joint F1):    {avg_joint_f1 * 100:.1f}%\n"
        f"Supporting Document F1*:      {avg_doc_f1 * 100:.1f}%\n"
        f"Observed Gold Document Recall*: {avg_observed_doc_recall * 100:.1f}%\n"
        f"Observed Gold SP Recall*:       {avg_observed_sp_recall * 100:.1f}%\n"
        f"All Gold SP Observed*:          {full_observed_count}/{total_count} ({(full_observed_count / total_count * 100) if total_count else 0:.1f}%)\n"
        f"Hotpot Supplied-Context Gold SP Recall*: {avg_supplied_context_sp_recall * 100:.1f}%\n"
        f"Hotpot Supplied Context Has All Gold SP*: {full_supplied_context_count}/{total_count} ({(full_supplied_context_count / total_count * 100) if total_count else 0:.1f}%)\n"
        f"Avg Hops / Question:           {avg_steps:.2f}\n"
        f"Avg Latency / Question:        {avg_lat:.2f}s\n"
        f"Failed Questions:              {failed_count}\n"
        f"Total Evaluation Time:         {total_time:.2f}s\n"
        "* Diagnostic metric, not an official HotpotQA leaderboard metric.\n"
    )
    print(summary_text)
    logger.info(summary_text)

    results_json_path = os.path.join(output_dir, "results.json")
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    trajectories_json_path = os.path.join(output_dir, "trajectories.json")
    with open(trajectories_json_path, "w") as f:
        json.dump(full_trajectories, f, indent=2, ensure_ascii=False)

    portfolio_dir = os.path.join(os.path.dirname(__file__), "..", "portfolio")
    os.makedirs(portfolio_dir, exist_ok=True)
    portfolio_json_path = os.path.join(portfolio_dir, "portfolio_trajectories.json")
    with open(portfolio_json_path, "w") as f:
        json.dump(full_trajectories, f, indent=2, ensure_ascii=False)

    segmented_summary = compute_metrics_by_type(results)
    print_segmented_report(segmented_summary, model_name=f"ReAct Agent ({model_name})")

    prediction_path, gold_path = write_official_files(samples, full_trajectories, output_dir)
    save_portfolio_trajectories(full_trajectories, segmented_summary, portfolio_json_path)

    manifest_path = write_run_manifest(
        output_dir,
        {
            "runner": "react",
            "started_at": run_started_at,
            "finished_at": datetime.now().isoformat(),
            "model": model_name,
            "inference_base_url": api_base,
            "retrieval_mode": mode,
            "dataset_source": source,
            "requested_samples": num_samples,
            "dataset_size": total,
            "completed_records": total_count,
            "failed_records": failed_count,
            "concurrency": concurrency,
            "max_hops": max_hops,
            "documents_per_search": search_top_k if mode == "fullwiki" else 1,
            "retrieval_top_k": search_top_k if mode == "fullwiki" else 1,
            "max_working_evidence_documents": max_evidence_documents if mode == "fullwiki" else None,
            "working_evidence_policy": "cross_encoder_top_k" if mode == "fullwiki" else None,
            "max_observation_characters": max_observation_chars if mode == "fullwiki" else None,
            "duplicate_search_guard": mode == "fullwiki",
            "exact_title_promotion": False,
            "retriever": retriever if mode == "fullwiki" else mode,
            "retrieval_backend": fullwiki_backend.describe() if fullwiki_backend is not None else None,
            "segmented_metrics": segmented_summary,
            "total_evaluation_seconds": round(total_time, 3),
            "official_prediction_file": os.path.basename(prediction_path),
            "official_gold_file": os.path.basename(gold_path),
        },
    )

    print(f"Saved evaluation raw JSON to: {results_json_path}")
    print(f"Saved ReAct trajectories JSON to: {trajectories_json_path}")
    print(f"Saved official-format predictions to: {prediction_path}")
    print(f"Saved evaluator-compatible gold file to: {gold_path}")
    print(f"Saved run manifest to: {manifest_path}")
    print(f"Saved execution log file to: {log_file}")
    print(f"Exported portfolio trajectories to: {portfolio_json_path}")

    generate_eval_plots_and_report(results, output_dir=output_dir)


from config import (
    FULLWIKI_INDEX_DIR,
    LLM_MODEL_NAME,
    MAX_AGENT_HOPS,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    REACT_MAX_EVIDENCE_DOCUMENTS,
    REACT_MAX_OBSERVATION_CHARS,
    REACT_SEARCH_TOP_K,
    REACT_MEMORY_RERANKER_BATCH_SIZE,
    REACT_MEMORY_RERANKER_DEVICE,
    REACT_MEMORY_RERANKER_MAX_LENGTH,
    REACT_MEMORY_RERANKER_MODEL,
    load_eval_config,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ReAct HotpotQA Benchmark")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file (e.g. config/fullwiki.yaml)")
    parser.add_argument("--samples", type=int, default=None, help="Number of questions to test (default: all)")
    parser.add_argument("--mode", choices=["offline", "fullwiki", "live"], default=None, help="Retrieval mode")
    parser.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default=None, help="FullWiki first-stage retriever")
    parser.add_argument("--index-dir", type=str, default=FULLWIKI_INDEX_DIR, help="FullWiki index directory")
    parser.add_argument("--top-k", type=int, default=None, help="Documents returned by each adaptive FullWiki search")
    parser.add_argument("--max-evidence-docs", type=int, default=None, help="Maximum unique FullWiki documents retained in ReAct working evidence")
    parser.add_argument("--max-observation-chars", type=int, default=None, help="Maximum characters rendered by each FullWiki search observation")
    parser.add_argument("--candidate-k", type=int, default=None, help="First-stage RRF candidate pool size")
    parser.add_argument("--rrf-k", type=int, default=None, help="Reciprocal Rank Fusion smoothing parameter")
    parser.add_argument("--source", choices=["sample", "huggingface", "official_json"], default=None, help="Dataset source")
    parser.add_argument("--model", type=str, default=None, help="LLM model name")
    parser.add_argument("--api-base", type=str, default=None, help="Local vLLM / OpenAI server URL")
    parser.add_argument("--output-dir", type=str, default="eval_results/react", help="Directory for outputs")
    parser.add_argument("--concurrency", type=int, default=16, help="Number of concurrent worker threads")
    parser.add_argument("--reranker-model", type=str, default=None, help="Evidence reranker model name")
    parser.add_argument("--reranker-device", type=str, default=None, help="Evidence reranker device (cpu/cuda)")
    parser.add_argument("--max-hops", type=int, default=None, help="Maximum hops per question")

    args = parser.parse_args()
    cfg = load_eval_config(args.config) if args.config else {}

    mode = args.mode or cfg.get("retrieval_mode") or "offline"
    retriever = args.retriever or cfg.get("retriever") or "hybrid"
    search_top_k = args.top_k or cfg.get("react_top_k") or REACT_SEARCH_TOP_K
    max_evidence_docs = args.max_evidence_docs or cfg.get("max_evidence_documents") or REACT_MAX_EVIDENCE_DOCUMENTS
    max_observation_chars = args.max_observation_chars or cfg.get("max_observation_chars") or REACT_MAX_OBSERVATION_CHARS
    source = args.source or cfg.get("dataset_source") or "sample"
    model = args.model or cfg.get("model_name") or LLM_MODEL_NAME
    api_base = args.api_base or cfg.get("api_base") or OPENAI_API_BASE
    max_hops = args.max_hops or cfg.get("max_tool_steps") or MAX_AGENT_HOPS

    candidate_k = args.candidate_k or cfg.get("candidate_k") or FULLWIKI_SEARCH_CANDIDATES
    rrf_k = args.rrf_k or cfg.get("rrf_k") or FULLWIKI_RRF_K
    reranker_model = args.reranker_model or cfg.get("memory_reranker_model") or REACT_MEMORY_RERANKER_MODEL
    reranker_device = args.reranker_device or cfg.get("memory_reranker_device") or REACT_MEMORY_RERANKER_DEVICE
    reranker_batch_size = cfg.get("memory_reranker_batch_size") or REACT_MEMORY_RERANKER_BATCH_SIZE
    reranker_max_length = cfg.get("memory_reranker_max_length") or REACT_MEMORY_RERANKER_MAX_LENGTH

    run_benchmark(
        num_samples=args.samples,
        mode=mode,
        model_name=model,
        source=source,
        api_base=api_base,
        output_dir=args.output_dir,
        concurrency=args.concurrency,
        max_hops=max_hops,
        retriever=retriever,
        candidate_k=candidate_k,
        rrf_k=rrf_k,
        index_dir=args.index_dir,
        search_top_k=search_top_k,
        max_evidence_documents=max_evidence_docs,
        max_observation_chars=max_observation_chars,
        reranker_model=reranker_model,
        reranker_device=reranker_device,
        reranker_batch_size=reranker_batch_size,
        reranker_max_length=reranker_max_length,
    )
