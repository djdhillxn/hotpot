import json
import os
import tempfile
import unittest

from eval.compare_results import (
    ComparisonValidationError,
    compare_results,
    validate_record_alignment,
)


class TestCompareResults(unittest.TestCase):
    def _record(self, qid, qtype, answer_correct, joint_correct, joint_f1, latency, hops):
        return {
            "id": qid,
            "question": f"Question {qid}?",
            "question_type": qtype,
            "gold_answer": f"gold-{qid}",
            "gold_supporting_facts": [[f"Title {qid}", 0]],
            "exact_match": answer_correct,
            "f1": 1.0 if answer_correct else 0.25,
            "sp_em": joint_correct,
            "sp_f1": 1.0 if joint_correct else 0.5,
            "joint_em": joint_correct,
            "joint_f1": joint_f1,
            "doc_f1": 0.5,
            "observed_gold_document_recall": 1.0,
            "observed_gold_supporting_fact_recall": 0.75,
            "all_gold_supporting_facts_observed": joint_correct,
            "step_count": hops,
            "latency": latency,
        }

    def _manifest(self, runner, total_pairs):
        manifest = {
            "runner": runner,
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "dataset_source": "official_json",
            "dataset_size": 2,
            "completed_records": 2,
            "failed_records": 0,
            "retrieval_mode": "fullwiki",
            "retriever": "hybrid",
            "concurrency": 64,
            "retrieval_backend": {
                "candidate_k": 50,
                "rrf_k": 60,
                "dense_model": "BAAI/bge-base-en-v1.5",
                "dense_query_device": "cpu",
                "dense_nprobe": 32,
                "local_reranker": {
                    "model": "BAAI/bge-reranker-base",
                    "device": "cuda",
                    "max_length": 512,
                    "batch_size": 32,
                    "total_pairs_scored": total_pairs,
                },
                "index_manifest": {
                    "corpus": {"corpus_sha256": "abc123", "source_archive_md5": "md5"},
                    "bm25": {
                        "engine": "Lucene/Anserini via Pyserini",
                        "parameters": "default BM25",
                        "document_count": 5233235,
                    },
                    "dense": {
                        "factory": "IVF4096,PQ96x8",
                        "dimension": 768,
                        "document_count": 5233235,
                    },
                },
            },
        }
        if runner == "single_pass_rag":
            manifest.update({
                "retrieval_calls_per_question": 1,
                "generation_calls_per_question": 1,
                "documents_hydrated_for_page_reranking": 15,
                "documents_in_reader_context": 7,
                "page_reranker_enabled": True,
                "sentence_reranker_enabled": False,
            })
        else:
            manifest.update({
                "max_hops": 7,
                "documents_per_search": 15,
                "local_rerank_page_count": 4,
                "max_working_evidence_snippets": 12,
                "max_working_evidence_characters": 6000,
                "working_evidence_policy": "lookup_protected_then_within_search_sentence_rank",
            })
        return manifest

    def test_record_alignment_rejects_gold_mismatch(self):
        baseline = [self._record("q1", "bridge", False, False, 0.0, 1.0, 1)]
        react = [self._record("q1", "bridge", True, True, 1.0, 2.0, 2)]
        react[0]["gold_answer"] = "different"
        with self.assertRaises(ComparisonValidationError):
            validate_record_alignment(baseline, react)

    def test_final_comparison_generates_publication_artifacts(self):
        baseline = [
            self._record("q1", "bridge", False, False, 0.10, 10.0, 1),
            self._record("q2", "comparison", True, True, 0.80, 12.0, 1),
        ]
        react = [
            self._record("q1", "bridge", True, True, 0.90, 20.0, 2),
            self._record("q2", "comparison", True, True, 1.00, 24.0, 3),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            baseline_dir = os.path.join(tmp, "baseline")
            react_dir = os.path.join(tmp, "react")
            output_dir = os.path.join(tmp, "comparison")
            publish_dir = os.path.join(tmp, "publish")
            os.makedirs(baseline_dir)
            os.makedirs(react_dir)

            with open(os.path.join(baseline_dir, "results.json"), "w") as f:
                json.dump(baseline, f)
            with open(os.path.join(react_dir, "results.json"), "w") as f:
                json.dump(react, f)
            with open(os.path.join(baseline_dir, "run_manifest.json"), "w") as f:
                json.dump(self._manifest("single_pass_rag", 30), f)
            with open(os.path.join(react_dir, "run_manifest.json"), "w") as f:
                json.dump(self._manifest("react", 80), f)

            summary = compare_results(
                os.path.join(baseline_dir, "results.json"),
                os.path.join(react_dir, "results.json"),
                output_dir=output_dir,
                publish_dir=publish_dir,
            )

            self.assertEqual(summary["comparison_validation"]["status"], "PASSED")
            self.assertEqual(summary["comparison_validation"]["question_count"], 2)
            self.assertEqual(
                summary["paired_outcomes"]["answer_exact_match"]["react_only"], 1
            )
            self.assertAlmostEqual(
                summary["official_metrics"]["delta_pp"]["joint_f1"], 50.0
            )

            expected = [
                "comparison_summary.json",
                "comparison_report.md",
                "official_metrics_comparison.svg",
                "joint_f1_by_question_type.svg",
                "evidence_coverage_comparison.svg",
                "paired_outcome_transitions.svg",
                "quality_cost_tradeoff.svg",
                "react_quality_by_hops.svg",
            ]
            for filename in expected:
                self.assertTrue(os.path.exists(os.path.join(output_dir, filename)))
                self.assertTrue(os.path.exists(os.path.join(publish_dir, filename)))


if __name__ == "__main__":
    unittest.main()
