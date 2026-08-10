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

from agent.baseline_rag import run_single_pass_rag
from config import (
    BASELINE_RERANK_TOP_K,
    BASELINE_SEARCH_TOP_K,
    FULLWIKI_INDEX_DIR,
    FULLWIKI_RRF_K,
    FULLWIKI_SEARCH_CANDIDATES,
    LLM_MODEL_NAME,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    REACT_LOCAL_RERANKER_BATCH_SIZE,
    REACT_LOCAL_RERANKER_DEVICE,
    REACT_LOCAL_RERANKER_MAX_LENGTH,
    REACT_LOCAL_RERANKER_MODEL,
    load_eval_config,
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
    import httpx

    http_client = httpx.Client(
        limits=httpx.Limits(max_connections=250, max_keepalive_connections=150),
        timeout=180.0,
    )

    return ChatOpenAI(
        model_name=model_name,
        temperature=0.0,
        base_url=api_base,
        api_key=api_key if api_key else "EMPTY",
        http_client=http_client,
        extra_body={
            "temperature": 0.0,
            "top_p": 1.0,
        },
    )


def _sample_metadata(sample):
    gold_supporting_facts = sample.get("supporting_facts", []) or []
    gold_titles = list(dict.fromkeys(
        fact[0]
        for fact in gold_supporting_facts
        if isinstance(fact, (list, tuple)) and len(fact) == 2
    ))
    return gold_supporting_facts, gold_titles, context_diagnostics(sample)



def process_single_question(
    sample, idx, total, mode, llm, logger, fullwiki_backend=None,
    top_k=BASELINE_SEARCH_TOP_K, rerank_top_k=BASELINE_RERANK_TOP_K,
):
    question = sample["question"]
    gold_answer = sample["answer"]
    gold_supporting_facts, gold_titles, context_info = _sample_metadata(sample)

    logger.info(f"Question [{idx}/{total}]: {question}")
    logger.info(f"Ground Truth: {gold_answer}")

    if mode == "offline":
        toolset = LocalHotpotRetriever(context_paragraphs=sample.get("context", []))
    elif mode == "fullwiki":
        if fullwiki_backend is None:
            raise RuntimeError("FullWiki backend was not initialized.")
        toolset = fullwiki_backend.create_baseline_session(
            rerank_top_k=rerank_top_k,
            output_top_k=top_k,
        )
    else:
        toolset = WikipediaToolSet()

    t0 = time.perf_counter()
    try:
        agent_state = run_single_pass_rag(question=question, llm=llm, toolset=toolset)
    except Exception as exc:
        latency = time.perf_counter() - t0
        logger.exception(f"Error processing question {idx}: {exc}")
        return build_failure_record(sample, idx, exc, latency=latency)
    latency = time.perf_counter() - t0

    pred_answer = agent_state.get("final_answer") or "No Answer"
    predicted_supporting_facts = agent_state.get("predicted_supporting_facts", []) or []
    observed_supporting_facts = agent_state.get("observed_supporting_facts", []) or []
    invalid_supporting_facts = agent_state.get("invalid_supporting_facts", []) or []
    visited_pages = agent_state.get("visited_pages", [])
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
        step_count=1,
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
        "latency": round(latency, 3),
        "timestamp": datetime.now().isoformat(),
        "failed": False,
        **observed_info,
        "error": None,
        **context_info,
    })

    logger.info(
        f"Prediction [{idx}/{total}]: '{pred_answer}' | EM: {eval_metrics['exact_match']} | "
        f"F1: {eval_metrics['f1']:.3f} | SP F1: {eval_metrics['sp_f1']:.3f} | "
        f"Joint F1: {eval_metrics['joint_f1']:.3f} | Latency: {latency:.2f}s"
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
        "step_count": 1,
        "latency_seconds": round(latency, 3),
        "visited_pages": visited_pages,
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
        "error": None,
    }

    return eval_metrics, trajectory_entry


def build_failure_record(sample, idx, error, latency=0.0):
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
        "latency": round(latency, 3),
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
        "latency_seconds": round(latency, 3),
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


def run_baseline_benchmark(
    num_samples=None,
    mode="offline",
    model_name=LLM_MODEL_NAME,
    source="sample",
    api_base=OPENAI_API_BASE,
    output_dir="eval_results/baseline",
    retriever="hybrid",
    candidate_k=FULLWIKI_SEARCH_CANDIDATES,
    rrf_k=FULLWIKI_RRF_K,
    index_dir=FULLWIKI_INDEX_DIR,
    top_k=BASELINE_SEARCH_TOP_K,
    rerank_top_k=BASELINE_RERANK_TOP_K,
    concurrency=64,
    reranker_model=None,
    reranker_device=None,
    reranker_max_length=None,
    reranker_batch_size=None,
):
    reranker_model = reranker_model or REACT_LOCAL_RERANKER_MODEL
    reranker_device = reranker_device or REACT_LOCAL_RERANKER_DEVICE
    reranker_max_length = reranker_max_length or REACT_LOCAL_RERANKER_MAX_LENGTH
    reranker_batch_size = reranker_batch_size or REACT_LOCAL_RERANKER_BATCH_SIZE

    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if rerank_top_k < top_k:
        raise ValueError("rerank_top_k must be >= top_k")

    logger, log_file = setup_logger(output_dir)
    run_started_at = datetime.now().isoformat()

    info_msg = (
        f"=== Single-Pass RAG (Direct Prompting Baseline) Benchmark ===\n"
        f"Timestamp: {run_started_at}\n"
        f"Model: {model_name}\n"
        f"Inference Base URL: {api_base}\n"
        f"Retrieval Mode: {mode.upper()}\n"
        f"Dataset Source: {source}\n"
        f"Samples Limit: {num_samples if num_samples else 'Full Set'}\n"
        f"Concurrency Workers: {concurrency}\n"
        f"Retriever: {retriever if mode == 'fullwiki' else 'n/a'}\n"
        f"Documents hydrated for page reranking: {rerank_top_k if mode == 'fullwiki' else 1}\n"
        f"Documents exposed to the reader: {top_k if mode == 'fullwiki' else 1}\n"
        f"Page reranker: {reranker_model if mode == 'fullwiki' else 'n/a'}\n"
        f"Sentence reranker: disabled\n"
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
            local_reranker_model=reranker_model,
            local_reranker_device=reranker_device,
            local_reranker_max_length=reranker_max_length,
            local_reranker_batch_size=reranker_batch_size,
        )

    results = []
    full_trajectories = []
    total_start_time = time.time()
    total = len(samples)

    pbar = tqdm(total=total, desc="Single-Pass RAG Evaluation", unit="question", dynamic_ncols=True)
    completed_em_sum = 0.0
    completed_joint_f1_sum = 0.0

    def record_completed(eval_metrics, trajectory_entry):
        nonlocal completed_em_sum, completed_joint_f1_sum
        results.append(eval_metrics)
        full_trajectories.append(trajectory_entry)
        completed_em_sum += eval_metrics["exact_match"]
        completed_joint_f1_sum += eval_metrics["joint_f1"]
        completed = len(results)
        pbar.set_postfix({
            "EM": f"{(completed_em_sum / completed) * 100:.1f}%",
            "Joint_F1": f"{(completed_joint_f1_sum / completed) * 100:.1f}%",
        })
        pbar.update(1)

    if concurrency == 1:
        for idx, sample in enumerate(samples, 1):
            try:
                eval_metrics, trajectory_entry = process_single_question(
                    sample, idx, total, mode, llm, logger, fullwiki_backend, top_k, rerank_top_k
                )
            except Exception as exc:
                logger.exception(f"Unexpected worker failure for question {idx}: {exc}")
                eval_metrics, trajectory_entry = build_failure_record(sample, idx, exc)
            record_completed(eval_metrics, trajectory_entry)
    else:
        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="baseline") as executor:
            future_to_meta = {
                executor.submit(
                    process_single_question, sample, idx, total, mode, llm, logger,
                    fullwiki_backend, top_k, rerank_top_k
                ): (idx, sample)
                for idx, sample in enumerate(samples, 1)
            }

            for future in as_completed(future_to_meta):
                idx, sample = future_to_meta[future]
                try:
                    eval_metrics, trajectory_entry = future.result()
                except Exception as exc:
                    logger.exception(f"Unexpected worker failure for question {idx}: {exc}")
                    eval_metrics, trajectory_entry = build_failure_record(sample, idx, exc)
                record_completed(eval_metrics, trajectory_entry)

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
    avg_lat = sum(r["latency"] for r in results) / total_count if total_count else 0
    avg_observed_doc_recall = sum(r["observed_gold_document_recall"] for r in results) / total_count if total_count else 0
    avg_observed_sp_recall = sum(r["observed_gold_supporting_fact_recall"] for r in results) / total_count if total_count else 0
    full_observed_count = sum(r["all_gold_supporting_facts_observed"] for r in results)
    avg_supplied_context_sp_recall = sum(r["hotpot_supplied_context_gold_supporting_fact_recall"] for r in results) / total_count if total_count else 0
    full_supplied_context_count = sum(r["hotpot_supplied_context_has_all_gold_supporting_facts"] for r in results)
    failed_count = sum(bool(r.get("failed")) for r in results)

    summary_text = (
        "\n=== HOTPOTQA DEV METRICS (OFFICIAL FORMULAS) — SINGLE-PASS RAG ===\n"
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
        f"Avg Hops / Question:           {1.0 if total_count else 0.0:.2f}\n"
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

    segmented_summary = compute_metrics_by_type(results)
    print_segmented_report(segmented_summary, model_name=f"Single-Pass RAG ({model_name})")

    prediction_path, gold_path = write_official_files(samples, full_trajectories, output_dir)
    save_portfolio_trajectories(full_trajectories, segmented_summary, os.path.join(output_dir, "portfolio_trajectories.json"))

    manifest_path = write_run_manifest(
        output_dir,
        {
            "runner": "single_pass_rag",
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
            "retrieval_calls_per_question": 1,
            "generation_calls_per_question": 1,
            "documents_per_search": top_k if mode == "fullwiki" else 1,
            "documents_hydrated_for_page_reranking": rerank_top_k if mode == "fullwiki" else 1,
            "documents_in_reader_context": top_k if mode == "fullwiki" else 1,
            "retrieval_top_k": rerank_top_k if mode == "fullwiki" else 1,
            "page_reranker_enabled": mode == "fullwiki",
            "page_reranker_model": reranker_model if mode == "fullwiki" else None,
            "sentence_reranker_enabled": False,
            "exact_title_promotion": False,
            "retriever": retriever if mode == "fullwiki" else mode,
            "retrieval_backend": fullwiki_backend.describe() if fullwiki_backend is not None else None,
            "segmented_metrics": segmented_summary,
            "total_evaluation_seconds": round(total_time, 3),
            "official_prediction_file": os.path.basename(prediction_path),
            "official_gold_file": os.path.basename(gold_path),
        },
    )

    print(f"Saved Single-Pass RAG raw JSON to: {results_json_path}")
    print(f"Saved Single-Pass RAG trajectories JSON to: {trajectories_json_path}")
    print(f"Saved official-format predictions to: {prediction_path}")
    print(f"Saved evaluator-compatible gold file to: {gold_path}")
    print(f"Saved run manifest to: {manifest_path}")
    print(f"Saved execution log file to: {log_file}")

    generate_eval_plots_and_report(results, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Single-Pass RAG Baseline Benchmark")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file (e.g. config/fullwiki.yaml)")
    parser.add_argument("--samples", type=int, default=None, help="Number of questions to test (default: all)")
    parser.add_argument("--mode", choices=["offline", "fullwiki", "live"], default=None, help="Retrieval mode")
    parser.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default=None, help="FullWiki first-stage retriever")
    parser.add_argument("--index-dir", type=str, default=FULLWIKI_INDEX_DIR, help="FullWiki index directory")
    parser.add_argument("--top-k", type=int, default=None, help="Page-reranked documents exposed to the single-pass reader")
    parser.add_argument("--rerank-top-k", type=int, default=None, help="FullWiki documents hydrated and page-reranked before selecting top-k")
    parser.add_argument("--candidate-k", type=int, default=None, help="First-stage RRF candidate pool size")
    parser.add_argument("--rrf-k", type=int, default=None, help="Reciprocal Rank Fusion smoothing parameter")
    parser.add_argument("--source", choices=["sample", "huggingface", "official_json"], default=None, help="Dataset source")
    parser.add_argument("--model", type=str, default=None, help="LLM model name")
    parser.add_argument("--api-base", type=str, default=None, help="Local vLLM / OpenAI server URL")
    parser.add_argument("--output-dir", type=str, default="eval_results/baseline", help="Directory for outputs")
    parser.add_argument("--reranker-model", type=str, default=None, help="Page-level cross-encoder model")
    parser.add_argument("--reranker-device", type=str, default=None, help="Page-level cross-encoder device")
    parser.add_argument("--concurrency", type=int, default=None, help="Number of concurrent worker threads")

    args = parser.parse_args()
    cfg = load_eval_config(args.config) if args.config else {}

    mode = args.mode or cfg.get("retrieval_mode") or "offline"
    retriever = args.retriever or cfg.get("retriever") or "hybrid"
    top_k = args.top_k or cfg.get("baseline_top_k") or BASELINE_SEARCH_TOP_K
    rerank_top_k = args.rerank_top_k or cfg.get("baseline_rerank_top_k") or BASELINE_RERANK_TOP_K
    source = args.source or cfg.get("dataset_source") or "sample"
    model = args.model or cfg.get("model_name") or LLM_MODEL_NAME
    api_base = args.api_base or cfg.get("api_base") or OPENAI_API_BASE

    candidate_k = args.candidate_k or cfg.get("candidate_k") or FULLWIKI_SEARCH_CANDIDATES
    rrf_k = args.rrf_k or cfg.get("rrf_k") or FULLWIKI_RRF_K
    reranker_model = args.reranker_model or cfg.get("local_reranker_model") or cfg.get("memory_reranker_model") or REACT_LOCAL_RERANKER_MODEL
    reranker_device = args.reranker_device or cfg.get("local_reranker_device") or cfg.get("memory_reranker_device") or REACT_LOCAL_RERANKER_DEVICE
    reranker_batch_size = cfg.get("local_reranker_batch_size") or cfg.get("memory_reranker_batch_size") or REACT_LOCAL_RERANKER_BATCH_SIZE
    reranker_max_length = cfg.get("local_reranker_max_length") or cfg.get("memory_reranker_max_length") or REACT_LOCAL_RERANKER_MAX_LENGTH
    concurrency = args.concurrency or cfg.get("baseline_concurrency") or 64

    run_baseline_benchmark(
        num_samples=args.samples,
        mode=mode,
        model_name=model,
        source=source,
        api_base=api_base,
        output_dir=args.output_dir,
        retriever=retriever,
        candidate_k=candidate_k,
        rrf_k=rrf_k,
        index_dir=args.index_dir,
        top_k=top_k,
        rerank_top_k=rerank_top_k,
        concurrency=concurrency,
        reranker_model=reranker_model,
        reranker_device=reranker_device,
        reranker_max_length=reranker_max_length,
        reranker_batch_size=reranker_batch_size,
    )
