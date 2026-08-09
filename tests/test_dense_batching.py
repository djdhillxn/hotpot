import unittest
import numpy as np
import threading
import time
from unittest.mock import MagicMock

from retrieval.fullwiki_retriever import FullWikiSearchBackend, BGE_QUERY_INSTRUCTION


class TestDenseBatching(unittest.TestCase):

    def test_dense_batching_and_caching(self):
        backend = FullWikiSearchBackend.__new__(FullWikiSearchBackend)
        backend._dense_lock = threading.Lock()
        backend._embed_cache_lock = threading.Lock()
        backend._query_embed_cache = {}
        backend.dense_model_name = "BAAI/bge-base-en-v1.5"

        mock_encoder = MagicMock()
        def fake_encode(formatted_queries, batch_size=32, **kwargs):
            res = []
            for q in formatted_queries:
                val = float(len(q))
                res.append([val] * 768)
            return np.array(res, dtype=np.float32)

        mock_encoder.encode = fake_encode
        backend.encoder = mock_encoder

        import queue
        backend._dense_queue = queue.Queue()
        backend._dense_worker_thread = threading.Thread(target=backend._dense_batch_worker_loop, daemon=True)
        backend._dense_worker_thread.start()

        q1 = "Who directed Inception?"
        emb1 = backend._encode_query_batched(q1)
        self.assertEqual(emb1.shape, (768,))
        self.assertEqual(len(backend._query_embed_cache), 1)
        self.assertTrue(q1.strip().lower() in backend._query_embed_cache)

        emb1_cached = backend._encode_query_batched(q1)
        np.testing.assert_array_equal(emb1, emb1_cached)

    def test_concurrent_hybrid_search_mock(self):
        backend = FullWikiSearchBackend.__new__(FullWikiSearchBackend)
        from concurrent.futures import ThreadPoolExecutor
        backend._hybrid_executor = ThreadPoolExecutor(max_workers=4)

        backend._bm25_search = MagicMock(return_value=([{"doc_id": "1", "score": 10.0}], 0.01))
        backend._dense_search = MagicMock(return_value=([{"doc_id": "2", "score": 0.9}], 0.01))
        backend._parse_lucene_doc = MagicMock(return_value={"id": "1", "title": "Test Title", "sentences": ["Lead sent."]})
        backend.mode = "hybrid"
        backend.candidate_k = 10
        backend.rrf_k = 60

        res = backend.search("Inception", top_k=1, mode="hybrid")
        self.assertEqual(len(res["hits"]), 1)
        backend._bm25_search.assert_called_once()
        backend._dense_search.assert_called_once()


if __name__ == "__main__":
    unittest.main()
