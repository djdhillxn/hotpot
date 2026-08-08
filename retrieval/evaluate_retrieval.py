import argparse
import json
import os
import sys
import time

from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import FULLWIKI_INDEX_DIR
from eval.dataset import load_hotpot_dataset
from eval.metrics import normalize_answer
from retrieval.fullwiki_retriever import FullWikiSearchBackend


def gold_titles(sample):
    return list(dict.fromkeys(
        fact[0]
        for fact in sample.get("supporting_facts", [])
        if isinstance(fact, (list, tuple)) and len(fact) == 2
    ))


def recall_at_k(retrieved_titles, gold, k):
    gold_set = {normalize_answer(title) for title in gold if title}
    if not gold_set:
        return 1.0
    retrieved = {normalize_answer(title) for title in retrieved_titles[:k] if title}
    return len(gold_set & retrieved) / len(gold_set)


def main():
    parser = argparse.ArgumentParser(description="Sanity-check FullWiki first-stage retrieval")
    parser.add_argument("--source", choices=["sample", "huggingface", "official_json"], default="official_json")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--index-dir", default=FULLWIKI_INDEX_DIR)
    parser.add_argument("--modes", nargs="+", choices=["bm25", "dense", "hybrid"], default=["bm25", "dense", "hybrid"])
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10, 20])
    parser.add_argument("--output", default="eval_results/retrieval/retrieval_results.json")
    args = parser.parse_args()

    samples = load_hotpot_dataset(num_samples=args.samples, source=args.source)
    max_k = max(args.ks)
    backend = FullWikiSearchBackend(
        bm25_index_dir=os.path.join(args.index_dir, "bm25"),
        dense_index_path=os.path.join(args.index_dir, "dense.faiss"),
        manifest_path=os.path.join(args.index_dir, "manifest.json"),
        mode="hybrid" if any(mode in {"dense", "hybrid"} for mode in args.modes) else "bm25",
    )

    totals = {mode: {k: 0.0 for k in args.ks} for mode in args.modes}
    exact_all_gold = {mode: {k: 0 for k in args.ks} for mode in args.modes}
    latency = {mode: [] for mode in args.modes}
    per_question = []

    for sample in tqdm(samples, desc="FullWiki retrieval evaluation", unit="question"):
        gold = gold_titles(sample)
        row = {"id": str(sample["id"]), "question": sample["question"], "gold_titles": gold, "modes": {}}
        for mode in args.modes:
            started = time.perf_counter()
            result = backend.search(sample["question"], top_k=max_k, mode=mode)
            elapsed = time.perf_counter() - started
            titles = [hit["title"] for hit in result["hits"]]
            recalls = {}
            for k in args.ks:
                value = recall_at_k(titles, gold, k)
                totals[mode][k] += value
                exact_all_gold[mode][k] += int(value == 1.0)
                recalls[str(k)] = value
            latency[mode].append(elapsed)
            row["modes"][mode] = {
                "retrieved_titles": titles,
                "hits": [
                    {
                        "doc_id": hit["doc_id"],
                        "title": hit["title"],
                        "rank": hit["rank"],
                        "bm25_rank": hit.get("bm25_rank"),
                        "bm25_score": hit.get("bm25_score"),
                        "dense_rank": hit.get("dense_rank"),
                        "dense_score": hit.get("dense_score"),
                        "fused_score": hit.get("fused_score"),
                    }
                    for hit in result["hits"]
                ],
                "recall_at_k": recalls,
                "latency_seconds": round(elapsed, 6),
            }
        per_question.append(row)

    count = len(samples)
    summary = {}
    for mode in args.modes:
        summary[mode] = {
            "mean_gold_document_recall": {str(k): totals[mode][k] / count if count else 0.0 for k in args.ks},
            "all_gold_documents_retrieved_rate": {str(k): exact_all_gold[mode][k] / count if count else 0.0 for k in args.ks},
            "mean_latency_seconds": sum(latency[mode]) / len(latency[mode]) if latency[mode] else 0.0,
        }

    payload = {
        "dataset_source": args.source,
        "dataset_size": count,
        "ks": args.ks,
        "summary": summary,
        "per_question": per_question,
        "backend": backend.describe(),
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n=== FullWiki Retrieval Sanity Check ===")
    for mode in args.modes:
        print(f"\n{mode.upper()}")
        for k in args.ks:
            mean_recall = summary[mode]["mean_gold_document_recall"][str(k)] * 100
            all_gold = summary[mode]["all_gold_documents_retrieved_rate"][str(k)] * 100
            print(f"  Recall@{k:<2}: {mean_recall:6.2f}% | both/all gold docs retrieved: {all_gold:6.2f}%")
        print(f"  Mean latency: {summary[mode]['mean_latency_seconds'] * 1000:.1f} ms")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
