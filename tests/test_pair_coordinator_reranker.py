import unittest
import time
from unittest.mock import MagicMock

from retrieval.reranker import CrossEncoderEvidenceReranker


class TestPairCoordinatorReranker(unittest.TestCase):

    def test_pair_coordinator_and_telemetry(self):
        reranker = CrossEncoderEvidenceReranker.__new__(CrossEncoderEvidenceReranker)
        reranker.model_name = "test-model"
        reranker.device = "cpu"
        reranker.max_length = 512
        reranker.batch_size = 4
        reranker._lock = MagicMock()
        reranker._cache = {}
        reranker._cache_lock = MagicMock()
        reranker._max_cache_size = 1000

        mock_model = MagicMock()
        def fake_predict(pairs, batch_size=32, **kwargs):
            return [float(len(q) + len(p)) for q, p in pairs]
        mock_model.predict = fake_predict
        reranker.model = mock_model

        import threading, queue
        reranker._telemetry_lock = threading.Lock()
        reranker._cache_lock = threading.Lock()
        reranker.stats_total_requests = 0
        reranker.stats_total_pairs = 0
        reranker.stats_cache_hits = 0
        reranker.stats_total_enqueue_to_first_batch_s = 0.0
        reranker.stats_total_request_completion_s = 0.0
        reranker.stats_total_eval_time_s = 0.0
        reranker.stats_physical_batches = 0
        reranker.stats_total_evaluated_pairs = 0
        reranker.stats_total_tokens = 0
        reranker.stats_total_padding_tokens = 0

        reranker._queue = queue.Queue()
        reranker._worker_thread = threading.Thread(target=reranker._pair_coordinator_worker_loop, daemon=True)
        reranker._worker_thread.start()

        pairs = [("What is X?", f"Passage content number {i}") for i in range(10)]
        scores, latency = reranker.score_pairs(pairs)

        self.assertEqual(len(scores), 10)
        self.assertTrue(latency > 0)

        desc = reranker.describe()
        self.assertEqual(desc["total_requests"], 1)
        self.assertEqual(desc["total_pairs_scored"], 10)
        self.assertTrue("avg_enqueue_to_first_batch_ms" in desc)
        self.assertTrue("avg_request_completion_ms" in desc)
        self.assertTrue("avg_model_eval_ms" in desc)
        self.assertTrue(desc["total_physical_batches"] > 0)
        self.assertTrue("token_padding_efficiency" in desc)

        scores2, latency2 = reranker.score_pairs(pairs)
        self.assertEqual(scores, scores2)
        desc2 = reranker.describe()
        self.assertEqual(desc2["cache_hits"], 10)
        self.assertEqual(desc2["cache_hit_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
