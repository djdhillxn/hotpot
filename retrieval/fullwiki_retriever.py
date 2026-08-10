import json
import math
import os
import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from config import (
    DENSE_QUERY_DEVICE,
    FULLWIKI_BM25_INDEX_DIR,
    FULLWIKI_DENSE_INDEX_PATH,
    FULLWIKI_INDEX_MANIFEST,
    FULLWIKI_RRF_K,
    FULLWIKI_SEARCH_CANDIDATES,
)

from retrieval.reranker import CrossEncoderReranker

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def reciprocal_rank_fusion(sparse_hits, dense_hits, rrf_k=60):
    scores = defaultdict(float)
    metadata = {}

    for rank, hit in enumerate(sparse_hits, 1):
        doc_id = str(hit["doc_id"])
        scores[doc_id] += 1.0 / (rrf_k + rank)
        metadata.setdefault(doc_id, {})
        metadata[doc_id].update({
            "bm25_rank": rank,
            "bm25_score": float(hit["score"]),
        })

    for rank, hit in enumerate(dense_hits, 1):
        doc_id = str(hit["doc_id"])
        scores[doc_id] += 1.0 / (rrf_k + rank)
        metadata.setdefault(doc_id, {})
        metadata[doc_id].update({
            "dense_rank": rank,
            "dense_score": float(hit["score"]),
        })

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    output = []
    for rank, (doc_id, fused_score) in enumerate(ranked, 1):
        row = {
            "doc_id": doc_id,
            "rank": rank,
            "fused_score": float(fused_score),
            "bm25_rank": None,
            "bm25_score": None,
            "dense_rank": None,
            "dense_score": None,
        }
        row.update(metadata[doc_id])
        output.append(row)
    return output


class FullWikiSearchBackend:
    """Shared read-only global retrieval backend.

    Load this once per benchmark process. Per-question state lives in lightweight
    FullWikiRetriever sessions created via create_session().
    """

    def __init__(
        self,
        bm25_index_dir=FULLWIKI_BM25_INDEX_DIR,
        dense_index_path=FULLWIKI_DENSE_INDEX_PATH,
        manifest_path=FULLWIKI_INDEX_MANIFEST,
        mode="hybrid",
        candidate_k=FULLWIKI_SEARCH_CANDIDATES,
        rrf_k=FULLWIKI_RRF_K,
        dense_query_device=DENSE_QUERY_DEVICE,
        local_reranker_model=None,
        local_reranker_device="cpu",
        local_reranker_max_length=512,
        local_reranker_batch_size=16,
        evidence_reranker_model=None,
        evidence_reranker_device=None,
        evidence_reranker_max_length=None,
        evidence_reranker_batch_size=None,
    ):
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError("mode must be one of: bm25, dense, hybrid")

        self.mode = mode
        self.candidate_k = int(candidate_k)
        self.rrf_k = int(rrf_k)
        self.dense_query_device = dense_query_device
        self.bm25_index_dir = str(bm25_index_dir)
        self.dense_index_path = str(dense_index_path)
        self.manifest_path = str(manifest_path)
        self.manifest = self._load_manifest()
        self._dense_lock = threading.Lock()
        self._hybrid_executor = ThreadPoolExecutor(max_workers=32, thread_name_prefix="hybrid_retrieval")

        # Thread-safe retrieval caches
        self._doc_id_cache = {}
        self._doc_cache_lock = threading.Lock()

        self._title_doc_cache = {}
        self._title_cache_lock = threading.Lock()

        self._query_embed_cache = {}
        self._embed_cache_lock = threading.Lock()
        self._dense_queue = None
        self._dense_worker_thread = None

        try:
            os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
            from pyserini.search.lucene import LuceneSearcher
        except ImportError as exc:
            raise RuntimeError(
                "Pyserini is required for FullWiki retrieval. Install the project environment first."
            ) from exc

        if not os.path.isdir(self.bm25_index_dir):
            raise FileNotFoundError(
                f"BM25 index not found at {self.bm25_index_dir}. Run retrieval/build_fullwiki_index.py first."
            )
        self.lucene = LuceneSearcher(self.bm25_index_dir)

        self.faiss = None
        self.dense_index = None
        self.encoder = None
        self.dense_model_name = None
        self.dense_nprobe = None
        if self.mode in {"dense", "hybrid"}:
            self._load_dense_backend()

        # Query-local reranker. Legacy evidence_reranker_* keyword arguments remain
        # accepted so older scripts do not fail, but the reranker is no longer a
        # global evidence-memory governor.
        if local_reranker_model is None:
            local_reranker_model = evidence_reranker_model
        if evidence_reranker_device is not None and local_reranker_device == "cpu":
            local_reranker_device = evidence_reranker_device
        if evidence_reranker_max_length is not None and local_reranker_max_length == 512:
            local_reranker_max_length = evidence_reranker_max_length
        if evidence_reranker_batch_size is not None and local_reranker_batch_size == 16:
            local_reranker_batch_size = evidence_reranker_batch_size

        self.local_reranker = None
        if local_reranker_model:
            self.local_reranker = CrossEncoderReranker(
                local_reranker_model,
                device=local_reranker_device,
                max_length=local_reranker_max_length,
                batch_size=local_reranker_batch_size,
            )
        # Compatibility alias only; new retrieval code uses local_reranker.
        self.evidence_reranker = self.local_reranker

        self.db_path = os.path.join(os.path.dirname(self.manifest_path), "hyperlink_graph.db")
        alt_db_path = os.path.join("data", "fullwiki", "hyperlink_graph.db")
        if not os.path.isfile(self.db_path) and os.path.isfile(alt_db_path):
            self.db_path = alt_db_path

        self.title_graph_path = os.path.join(os.path.dirname(self.manifest_path), "title_graph.json")
        alt_graph_path = os.path.join("data", "fullwiki", "title_graph.json")
        if not os.path.isfile(self.title_graph_path) and os.path.isfile(alt_graph_path):
            self.title_graph_path = alt_graph_path
        self.title_graph = self._load_title_graph()

        self.title_to_doc_id_path = os.path.join(os.path.dirname(self.manifest_path), "title_to_doc_id.json")
        alt_title_path = os.path.join("data", "fullwiki", "title_to_doc_id.json")
        if not os.path.isfile(self.title_to_doc_id_path) and os.path.isfile(alt_title_path):
            self.title_to_doc_id_path = alt_title_path
        self.title_to_doc_id = self._load_title_to_doc_id()

    def _load_title_graph(self):
        import logging
        logger = logging.getLogger("fullwiki_retriever")
        if os.path.isfile(self.title_graph_path):
            try:
                with open(self.title_graph_path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning(f"Failed to load title_graph at {self.title_graph_path}: {exc}")
                return {}
        else:
            logger.warning(f"title_graph.json not found at {self.title_graph_path}")
        return {}

    def _load_title_to_doc_id(self):
        import logging
        logger = logging.getLogger("fullwiki_retriever")
        if os.path.isfile(self.title_to_doc_id_path):
            try:
                with open(self.title_to_doc_id_path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning(f"Failed to load title_to_doc_id at {self.title_to_doc_id_path}: {exc}")
                return {}
        return {}

    def get_outgoing_links(self, title):
        if not title or not self.title_graph:
            return []
        norm = str(title).strip().lower()
        if title in self.title_graph:
            return self.title_graph[title]
        for t, links in self.title_graph.items():
            if t.lower() == norm:
                return links
        return []

    def _get_db_conn(self):
        if not hasattr(self, "_db_conn") or self._db_conn is None:
            if hasattr(self, "db_path") and os.path.isfile(self.db_path):
                try:
                    import sqlite3
                    self._db_conn = sqlite3.connect(
                        f"file:{self.db_path}?mode=ro",
                        uri=True,
                        check_same_thread=False,
                    )
                except Exception:
                    self._db_conn = None
            else:
                self._db_conn = None
        return self._db_conn

    def get_outgoing_edges(self, title):
        if not title:
            return []
        conn = self._get_db_conn()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT source_title, source_doc_id, source_sent_id, anchor_text, target_title, target_doc_id "
                    "FROM edges WHERE source_title = ? COLLATE NOCASE",
                    (title,)
                )
                rows = cursor.fetchall()
                return [
                    {
                        "source_title": r[0],
                        "source_doc_id": r[1],
                        "source_sent_id": r[2],
                        "anchor_text": r[3],
                        "target_title": r[4],
                        "target_doc_id": r[5],
                    }
                    for r in rows
                ]
            except Exception:
                pass

        links = self.get_outgoing_links(title)
        return [
            {
                "source_title": title,
                "source_doc_id": "",
                "source_sent_id": 0,
                "anchor_text": target,
                "target_title": target,
                "target_doc_id": "",
            }
            for target in links
        ]

    def get_target_outdegree(self, target_title):
        if not target_title:
            return 1
        conn = self._get_db_conn()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT count FROM outdegree WHERE target_title = ?",
                    (target_title.strip().lower(),)
                )
                row = cursor.fetchone()
                if row:
                    return row[0]
            except Exception:
                pass
        return 1

    def _load_manifest(self):
        if not os.path.isfile(self.manifest_path):
            raise FileNotFoundError(
                f"FullWiki index manifest not found at {self.manifest_path}. "
                "Run retrieval/build_fullwiki_index.py first."
            )
        with open(self.manifest_path, encoding="utf-8") as f:
            return json.load(f)

    def _load_dense_backend(self):
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Dense FullWiki retrieval requires faiss-cpu and sentence-transformers."
            ) from exc

        if not os.path.isfile(self.dense_index_path):
            raise FileNotFoundError(
                f"Dense index not found at {self.dense_index_path}. Run retrieval/build_fullwiki_index.py first."
            )

        dense_manifest = self.manifest.get("dense", {})
        self.dense_model_name = dense_manifest.get("model", "BAAI/bge-base-en-v1.5")
        nprobe_default = dense_manifest.get("nprobe", 16)
        self.dense_nprobe = int(os.getenv("FULLWIKI_DENSE_NPROBE", str(nprobe_default)))
        self.faiss = faiss

        # Set FAISS OpenMP threads explicitly to prevent CPU oversubscription across 64 workers
        faiss_omp_threads = int(os.getenv("FAISS_OMP_NUM_THREADS", "1"))
        faiss.omp_set_num_threads(faiss_omp_threads)

        self.dense_index = faiss.read_index(self.dense_index_path)
        if hasattr(self.dense_index, "nprobe"):
            self.dense_index.nprobe = self.dense_nprobe
        self.encoder = SentenceTransformer(self.dense_model_name, device=self.dense_query_device)

        # Central Dense Query Batching Queue & Worker
        self._dense_queue = queue.Queue()
        self._dense_worker_thread = threading.Thread(target=self._dense_batch_worker_loop, daemon=True)
        self._dense_worker_thread.start()

    def _dense_batch_worker_loop(self):
        while True:
            try:
                first_item = self._dense_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            gathered_items = [first_item]
            max_batch = 32
            start_wait = time.perf_counter()

            while len(gathered_items) < max_batch and (time.perf_counter() - start_wait) < 0.002:
                try:
                    item = self._dense_queue.get_nowait()
                    gathered_items.append(item)
                except queue.Empty:
                    break

            if not gathered_items:
                continue

            unique_queries = []
            query_to_indices = {}
            for idx, item in enumerate(gathered_items):
                q = item["query"]
                if q not in query_to_indices:
                    query_to_indices[q] = []
                    unique_queries.append(q)
                query_to_indices[q].append(idx)

            formatted_queries = [BGE_QUERY_INSTRUCTION + q for q in unique_queries]

            try:
                with self._dense_lock:
                    embeddings = self.encoder.encode(
                        formatted_queries,
                        batch_size=min(32, len(formatted_queries)),
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    ).astype("float32")

                with self._embed_cache_lock:
                    for q, emb in zip(unique_queries, embeddings):
                        norm_q = q.strip().lower()
                        if len(self._query_embed_cache) < 50000:
                            self._query_embed_cache[norm_q] = emb

                for q, emb in zip(unique_queries, embeddings):
                    for idx in query_to_indices[q]:
                        item = gathered_items[idx]
                        item["embedding"] = emb
                        item["event"].set()

            except Exception as exc:
                for item in gathered_items:
                    item["exception"] = exc
                    item["event"].set()

            for _ in range(len(gathered_items)):
                self._dense_queue.task_done()

    def _encode_query_batched(self, query):
        norm_q = query.strip().lower()
        with self._embed_cache_lock:
            cached_encoded = self._query_embed_cache.get(norm_q)

        if cached_encoded is not None:
            return cached_encoded

        if self._dense_queue is None:
            with self._dense_lock:
                if self._dense_queue is None:
                    self._load_dense_backend()

        event = threading.Event()
        item = {
            "query": query,
            "event": event,
            "embedding": None,
            "exception": None,
        }
        self._dense_queue.put(item)
        signaled = event.wait(timeout=60.0)
        if not signaled:
            raise TimeoutError("Dense query batch worker timed out.")
        if item["exception"] is not None:
            raise item["exception"]

        return item["embedding"]

    def _parse_lucene_doc(self, doc_id):
        doc_id_str = str(doc_id)
        with self._doc_cache_lock:
            if doc_id_str in self._doc_id_cache:
                return dict(self._doc_id_cache[doc_id_str])

        doc = self.lucene.doc(doc_id_str)
        if doc is None:
            return None
        raw = doc.raw()
        if raw is None:
            return None
        data = json.loads(raw)
        data["id"] = str(data.get("id", doc_id))
        data["doc_id"] = data["id"]
        data["sentences"] = [str(sentence) for sentence in data.get("sentences", [])]

        with self._doc_cache_lock:
            if len(self._doc_id_cache) < 200000:
                self._doc_id_cache[doc_id_str] = data
        return dict(data)

    def get_doc_by_title(self, title):
        title = str(title or "").strip().strip("'\"")
        if not title:
            return None
        norm_target = title.lower()
        with self._title_cache_lock:
            if norm_target in self._title_doc_cache:
                res = self._title_doc_cache[norm_target]
                return dict(res) if res else None

        found_doc = None
        if self.title_to_doc_id and norm_target in self.title_to_doc_id:
            doc_id = self.title_to_doc_id[norm_target]
            found_doc = self._parse_lucene_doc(doc_id)

        if not found_doc:
            escaped_title = title.replace('"', '\\"')
            hits = self.lucene.search(f'"{escaped_title}"', k=10)
            for hit in hits:
                doc = self._parse_lucene_doc(str(hit.docid))
                if doc and str(doc.get("title", "")).strip().lower() == norm_target:
                    found_doc = doc
                    break

        with self._title_cache_lock:
            if len(self._title_doc_cache) < 100000:
                self._title_doc_cache[norm_target] = found_doc

        return dict(found_doc) if found_doc else None

    def _bm25_search(self, query, k):
        started = time.perf_counter()
        hits = self.lucene.search(query, k=k)
        rows = [{"doc_id": str(hit.docid), "score": float(hit.score)} for hit in hits]
        return rows, time.perf_counter() - started

    def _dense_search(self, query, k):
        started = time.perf_counter()
        encoded = self._encode_query_batched(query)

        if encoded.ndim == 1:
            encoded_mat = np.expand_dims(encoded, axis=0)
        else:
            encoded_mat = encoded

        scores, ids = self.dense_index.search(encoded_mat, int(k))

        rows = []
        for doc_id, score in zip(ids[0], scores[0]):
            if int(doc_id) < 0:
                continue
            rows.append({"doc_id": str(int(doc_id)), "score": float(score)})
        return rows, time.perf_counter() - started

    def search(self, query, top_k=1, mode=None):
        query = str(query).strip().strip("'\"")
        if not query:
            return {"query": query, "mode": mode or self.mode, "hits": [], "latency_ms": {}}

        mode = mode or self.mode
        candidate_k = max(int(top_k), self.candidate_k)
        sparse_hits = []
        dense_hits = []
        bm25_latency = 0.0
        dense_latency = 0.0

        if mode == "hybrid":
            f_bm25 = self._hybrid_executor.submit(self._bm25_search, query, candidate_k)
            dense_hits, dense_latency = self._dense_search(query, candidate_k)
            sparse_hits, bm25_latency = f_bm25.result()
        elif mode == "bm25":
            sparse_hits, bm25_latency = self._bm25_search(query, candidate_k)
        elif mode == "dense":
            dense_hits, dense_latency = self._dense_search(query, candidate_k)

        fuse_started = time.perf_counter()
        if mode == "bm25":
            ranking = [
                {
                    "doc_id": hit["doc_id"],
                    "rank": rank,
                    "fused_score": None,
                    "bm25_rank": rank,
                    "bm25_score": hit["score"],
                    "dense_rank": None,
                    "dense_score": None,
                }
                for rank, hit in enumerate(sparse_hits, 1)
            ]
        elif mode == "dense":
            ranking = [
                {
                    "doc_id": hit["doc_id"],
                    "rank": rank,
                    "fused_score": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "dense_rank": rank,
                    "dense_score": hit["score"],
                }
                for rank, hit in enumerate(dense_hits, 1)
            ]
        else:
            ranking = reciprocal_rank_fusion(sparse_hits, dense_hits, self.rrf_k)
        fusion_latency = time.perf_counter() - fuse_started

        hydrated = []
        for row in ranking[: int(top_k)]:
            document = self._parse_lucene_doc(row["doc_id"])
            if document is None:
                continue
            document.update(row)
            hydrated.append(document)

        search_wall_clock = max(bm25_latency, dense_latency) if mode == "hybrid" else (bm25_latency + dense_latency)
        total_search_latency = search_wall_clock + fusion_latency

        return {
            "query": query,
            "mode": mode,
            "candidate_k": candidate_k,
            "top_k": int(top_k),
            "hits": hydrated,
            "latency_ms": {
                "bm25": round(bm25_latency * 1000, 3),
                "dense": round(dense_latency * 1000, 3),
                "fusion": round(fusion_latency * 1000, 3),
                "total": round(total_search_latency * 1000, 3),
            },
        }

    @staticmethod
    def _page_reranker_passage(document):
        title = str(document.get("title", "")).strip()
        sentences = [
            str(text).strip()
            for text in document.get("sentences", [])
            if str(text).strip()
        ]
        body = " ".join(sentences).strip()
        return f"{title}\n{body}".strip() if body else title

    def score_page_documents(self, query_context, documents):
        """Score exactly one full-intro passage per document for the current search."""
        if self.local_reranker is None:
            raise RuntimeError("Local reranker is not configured on this FullWiki backend.")
        documents = list(documents or [])
        passages = [self._page_reranker_passage(doc) for doc in documents]
        pairs = [(str(query_context), passage) for passage in passages]
        if not pairs:
            return [], 0.0, 0

        # Cheap telemetry only. The model itself still owns exact tokenizer truncation.
        max_length = int(getattr(self.local_reranker, "max_length", 512))
        estimated_truncated = sum(
            1
            for q, p in pairs
            if len(q.split()) + len(p.split()) + 3 > max_length
        )
        scores, latency = self.local_reranker.score_pairs(pairs)
        return [float(score) for score in scores], latency, estimated_truncated

    def score_document_sentences(self, query_context, documents):
        """Score every non-empty sentence in the supplied pages, preserving sentence IDs."""
        if self.local_reranker is None:
            raise RuntimeError("Local reranker is not configured on this FullWiki backend.")

        all_pairs = []
        pair_meta = []
        for document in documents or []:
            doc_id = str(document.get("doc_id", document.get("id", "")))
            title = str(document.get("title", "")).strip()
            for sent_id, text in enumerate(document.get("sentences", [])):
                clean_text = str(text).strip()
                if not clean_text:
                    continue
                all_pairs.append((str(query_context), f"{title}: {clean_text}"))
                pair_meta.append((doc_id, int(sent_id)))

        scores_by_doc = {
            str(document.get("doc_id", document.get("id", ""))): {}
            for document in documents or []
        }
        if not all_pairs:
            return scores_by_doc, 0.0, 0

        scores, latency = self.local_reranker.score_pairs(all_pairs)
        for (doc_id, sent_id), score in zip(pair_meta, scores):
            scores_by_doc.setdefault(doc_id, {})[sent_id] = float(score)
        return scores_by_doc, latency, len(all_pairs)

    def create_session(
        self,
        search_top_k=15,
        local_rerank_page_count=4,
        max_evidence_snippets=12,
        max_evidence_chars=6000,
        max_observation_chars=6000,
        max_evidence_documents=None,
        duplicate_search_guard=False,
        question=None,
        use_graph_expansion=False,
        graph_focus_doc_count=2,
        graph_candidate_quota=10,
        graph_weight_source_sent_score=2.0,
        graph_weight_anchor_overlap=1.5,
        graph_weight_title_overlap=1.0,
        graph_weight_outdegree_penalty=0.3,
    ):
        # max_evidence_documents is a legacy compatibility alias. In the new
        # architecture the bounded recurrent state is sentence snippets, not docs.
        if max_evidence_documents is not None:
            max_evidence_snippets = int(max_evidence_documents)
        return FullWikiRetriever(
            self,
            search_top_k=search_top_k,
            local_rerank_page_count=local_rerank_page_count,
            max_evidence_snippets=max_evidence_snippets,
            max_evidence_chars=max_evidence_chars,
            max_observation_chars=max_observation_chars,
            duplicate_search_guard=duplicate_search_guard,
            question=question,
            use_graph_expansion=use_graph_expansion,
            graph_focus_doc_count=graph_focus_doc_count,
            graph_candidate_quota=graph_candidate_quota,
            graph_weight_source_sent_score=graph_weight_source_sent_score,
            graph_weight_anchor_overlap=graph_weight_anchor_overlap,
            graph_weight_title_overlap=graph_weight_title_overlap,
            graph_weight_outdegree_penalty=graph_weight_outdegree_penalty,
        )

    def create_baseline_session(self, rerank_top_k=15, output_top_k=7):
        """Create the one-search RAG + page-reranker baseline session.

        The baseline shares first-stage retrieval and the page-level cross-encoder
        with ReAct, but deliberately has no sentence reranker, lookup, or memory.
        """
        return FullWikiRerankedBaselineRetriever(
            self,
            rerank_top_k=rerank_top_k,
            output_top_k=output_top_k,
        )

    def describe(self):
        return {
            "backend": "fullwiki",
            "retriever": self.mode,
            "candidate_k": self.candidate_k,
            "rrf_k": self.rrf_k if self.mode == "hybrid" else None,
            "bm25_index_dir": os.path.abspath(self.bm25_index_dir),
            "dense_index_path": os.path.abspath(self.dense_index_path) if self.mode in {"dense", "hybrid"} else None,
            "dense_model": self.dense_model_name,
            "dense_query_device": self.dense_query_device if self.mode in {"dense", "hybrid"} else None,
            "dense_nprobe": self.dense_nprobe,
            "local_reranker": (
                self.local_reranker.describe() if self.local_reranker is not None else None
            ),
            "index_manifest": self.manifest,
        }


class FullWikiRerankedBaselineRetriever:
    """One-retrieval, one-generation FullWiki RAG + page-reranker session.

    Contract:
    - Retrieve/hydrate ``rerank_top_k`` documents from the shared FullWiki backend.
    - Score each hydrated document exactly once with the page-level cross-encoder.
    - Expose the top ``output_top_k`` reranked documents in full, preserving the
      original HotpotQA sentence IDs.
    - No sentence-level reranking, persistent memory, lookup, or second search.
    """

    def __init__(self, backend, rerank_top_k=15, output_top_k=7):
        self.backend = backend
        self.rerank_top_k = int(rerank_top_k)
        self.output_top_k = int(output_top_k)
        if self.rerank_top_k < 1:
            raise ValueError("rerank_top_k must be >= 1")
        if self.output_top_k < 1:
            raise ValueError("output_top_k must be >= 1")
        if self.output_top_k > self.rerank_top_k:
            raise ValueError("output_top_k cannot exceed rerank_top_k")
        if getattr(self.backend, "local_reranker", None) is None:
            raise RuntimeError(
                "The single-pass reranked baseline requires a configured page-level local reranker."
            )
        if not hasattr(self.backend, "score_page_documents"):
            raise RuntimeError("FullWiki backend does not provide page-level reranking.")
        self.reset()

    def reset(self):
        self.current_title = None
        self.current_document = None
        self.last_result = None
        self.visited_pages = []

    @property
    def current_page_title(self):
        return self.current_title

    @staticmethod
    def _logged_hit(hit):
        return {
            "doc_id": str(hit.get("doc_id", hit.get("id", ""))),
            "title": hit.get("title"),
            "rank": hit.get("rank"),
            "bm25_rank": hit.get("bm25_rank"),
            "bm25_score": hit.get("bm25_score"),
            "dense_rank": hit.get("dense_rank"),
            "dense_score": hit.get("dense_score"),
            "fused_score": hit.get("fused_score"),
            "sentences": [
                {"sent_id": sent_id, "text": str(text)}
                for sent_id, text in enumerate(hit.get("sentences", []))
            ],
        }

    @staticmethod
    def _visible_sentences(hit):
        return [
            {"sent_id": int(sent_id), "text": str(text).strip()}
            for sent_id, text in enumerate(hit.get("sentences", []))
            if str(text).strip()
        ]

    def search(self, query):
        query = str(query).strip().strip("'\"")
        if not query:
            return "Observation: Search query cannot be empty."

        result = self.backend.search(query, top_k=self.rerank_top_k)
        hits = [dict(hit) for hit in result.get("hits", [])]
        raw_logged_hits = [self._logged_hit(hit) for hit in hits]
        if not hits:
            self.last_result = {
                "action": "search",
                "query": query,
                "status": "not_found",
                "retriever": result.get("mode"),
                "candidate_k": result.get("candidate_k"),
                "rerank_top_k": self.rerank_top_k,
                "output_top_k": self.output_top_k,
                "hits": [],
                "retrieved_hits": raw_logged_hits,
                "local_reranked_hits": [],
                "local_page_pair_count": 0,
                "page_pair_estimated_truncations": 0,
                "latency_ms": result.get("latency_ms", {}),
                "title": None,
                "sentences": [],
            }
            return f"Observation: FullWiki retrieval found no document for '{query}'."

        page_scores, page_latency, estimated_truncated = self.backend.score_page_documents(
            query, hits
        )
        if len(page_scores) != len(hits):
            raise RuntimeError(
                f"Page reranker returned {len(page_scores)} scores for {len(hits)} pages."
            )

        ranked_pages = []
        for hit, score in zip(hits, page_scores):
            row = dict(hit)
            row["retrieval_rank"] = hit.get("rank")
            row["local_rerank_score"] = float(score)
            ranked_pages.append(row)
        ranked_pages.sort(key=lambda hit: (
            -float(hit["local_rerank_score"]),
            int(hit.get("retrieval_rank") or 10**9),
            str(hit.get("doc_id", "")),
        ))
        for local_rank, hit in enumerate(ranked_pages, 1):
            hit["local_rerank_rank"] = local_rank

        selected_pages = ranked_pages[: self.output_top_k]
        exposed_hits = []
        context_blocks = []
        for output_rank, page in enumerate(selected_pages, 1):
            sentences = self._visible_sentences(page)
            exposed_hits.append({
                "doc_id": str(page.get("doc_id", "")),
                "title": page.get("title"),
                "rank": page.get("retrieval_rank"),
                "local_rerank_rank": page.get("local_rerank_rank"),
                "local_rerank_score": page.get("local_rerank_score"),
                "sentences": sentences,
            })
            lines = [f"Retrieved document {output_rank}: {page.get('title')}"]
            lines.extend(
                f"[{page.get('title')} | sent {sentence['sent_id']}] {sentence['text']}"
                for sentence in sentences
            )
            context_blocks.append("\n".join(lines))

        self.visited_pages = [
            page.get("title") for page in selected_pages if page.get("title")
        ]
        self.current_document = dict(selected_pages[0]) if selected_pages else None
        self.current_title = selected_pages[0].get("title") if selected_pages else None

        local_reranked_hits = [
            {
                "doc_id": str(page.get("doc_id", "")),
                "title": page.get("title"),
                "retrieval_rank": page.get("retrieval_rank"),
                "local_rerank_rank": page.get("local_rerank_rank"),
                "local_rerank_score": page.get("local_rerank_score"),
                "bm25_rank": page.get("bm25_rank"),
                "dense_rank": page.get("dense_rank"),
                "fused_score": page.get("fused_score"),
            }
            for page in ranked_pages
        ]
        latency_ms = dict(result.get("latency_ms", {}))
        retrieval_total = float(latency_ms.get("total", 0.0) or 0.0)
        page_reranker_ms = round(page_latency * 1000, 3)
        latency_ms["local_page_reranker"] = page_reranker_ms
        latency_ms["total_with_page_reranker"] = round(retrieval_total + page_reranker_ms, 3)

        self.last_result = {
            "action": "search",
            "query": query,
            "status": "loaded",
            "retriever": result.get("mode"),
            "candidate_k": result.get("candidate_k"),
            "rerank_top_k": self.rerank_top_k,
            "output_top_k": self.output_top_k,
            "hits": exposed_hits,
            "retrieved_hits": raw_logged_hits,
            "local_reranked_hits": local_reranked_hits,
            "local_page_pair_count": len(hits),
            "local_sentence_pair_count": 0,
            "page_pair_estimated_truncations": int(estimated_truncated),
            "sentence_reranker_enabled": False,
            "latency_ms": latency_ms,
            "title": self.current_title,
            "sentences": exposed_hits[0]["sentences"] if exposed_hits else [],
        }

        prefix = (
            f"Observation: Single-pass FullWiki context with {len(exposed_hits)} "
            "page-reranked documents.\n"
        )
        return prefix + "\n\n".join(context_blocks)


class FullWikiRetriever:
    """Per-question ReAct session with query-local reranking and snippet memory.

    Contract:
    - Every search has its own local page ranking. Scores never carry across queries.
    - One current page is selected from that search and lookup always operates on it.
    - Persistent memory is a bounded set of sentence snippets, independent of current page.
    """

    CURRENT_PAGE_OBSERVATION_SENTENCES = 3

    def __init__(
        self,
        backend,
        search_top_k=15,
        local_rerank_page_count=4,
        max_evidence_snippets=12,
        max_evidence_chars=6000,
        max_observation_chars=6000,
        max_evidence_documents=None,
        duplicate_search_guard=False,
        question=None,
        use_graph_expansion=False,
        graph_focus_doc_count=2,
        graph_candidate_quota=10,
        graph_weight_source_sent_score=2.0,
        graph_weight_anchor_overlap=1.5,
        graph_weight_title_overlap=1.0,
        graph_weight_outdegree_penalty=0.3,
    ):
        self.backend = backend
        self.search_top_k = int(search_top_k)
        requested_local_rerank_pages = int(local_rerank_page_count)
        if max_evidence_documents is not None:
            max_evidence_snippets = int(max_evidence_documents)
        self.max_evidence_snippets = int(max_evidence_snippets)
        self.max_evidence_chars = int(max_evidence_chars)
        self.max_observation_chars = int(max_observation_chars)
        self.duplicate_search_guard = bool(duplicate_search_guard)
        self.question = str(question).strip() if question is not None else None
        self.use_graph_expansion = bool(use_graph_expansion)
        self.graph_focus_doc_count = int(graph_focus_doc_count)
        self.graph_candidate_quota = int(graph_candidate_quota)
        self.graph_w_src = float(graph_weight_source_sent_score)
        self.graph_w_anchor = float(graph_weight_anchor_overlap)
        self.graph_w_title = float(graph_weight_title_overlap)
        self.graph_w_outdegree = float(graph_weight_outdegree_penalty)

        if self.search_top_k < 1:
            raise ValueError("search_top_k must be >= 1")
        if requested_local_rerank_pages < 1:
            raise ValueError("local_rerank_page_count must be >= 1")
        # A small search_top_k is useful for tests/debugging; the production
        # configuration remains exactly 4 of 15. Never ask the sentence stage
        # to open more pages than this search actually hydrated.
        self.local_rerank_page_count = min(requested_local_rerank_pages, self.search_top_k)
        if self.max_evidence_snippets < 1:
            raise ValueError("max_evidence_snippets must be >= 1")
        if self.max_evidence_chars < 1:
            raise ValueError("max_evidence_chars must be >= 1")
        if self.max_observation_chars < 1:
            raise ValueError("max_observation_chars must be >= 1")

        reranker = getattr(self.backend, "local_reranker", None)
        if reranker is None:
            reranker = getattr(self.backend, "evidence_reranker", None)
        self.use_local_reranking = (
            reranker is not None
            and bool(self.question)
            and hasattr(self.backend, "score_page_documents")
            and hasattr(self.backend, "score_document_sentences")
        )

        self.current_title = None
        self.current_document = None
        self.last_result = None
        self.visited_pages = []

        # Cheap per-question metadata archive. Cross-query raw CE scores are never
        # used to rank this archive or to select the current page.
        self._evidence_archive = {}
        self._archive_order = 0

        # Sentence-level persistent evidence memory.
        self._snippet_archive = {}
        self._snippet_order = 0
        self._active_snippet_keys = []
        self._active_memory_context = ""
        self._active_memory_hits = []

        # Compatibility-only doc views derived from active snippets. They are not
        # a memory policy and never control current-page selection or lookup.
        self._evidence_doc_ids = []
        self._evidence_doc_id_set = set()

        self._seen_search_queries = {}
        self._lookup_keyword = None
        self._lookup_matches = []
        self._lookup_index = 0

    @property
    def current_page_title(self):
        return self.current_title

    @property
    def evidence_archive_count(self):
        return len(self._evidence_archive)

    @property
    def evidence_snippet_archive_count(self):
        return len(self._snippet_archive)

    @property
    def evidence_snippet_count(self):
        return len(self._active_snippet_keys)

    @property
    def evidence_document_count(self):
        return len(self._evidence_doc_ids)

    def reset(self):
        self.current_title = None
        self.current_document = None
        self.last_result = None
        self.visited_pages = []
        self._evidence_archive = {}
        self._archive_order = 0
        self._snippet_archive = {}
        self._snippet_order = 0
        self._active_snippet_keys = []
        self._active_memory_context = ""
        self._active_memory_hits = []
        self._evidence_doc_ids = []
        self._evidence_doc_id_set = set()
        self._seen_search_queries = {}
        self._lookup_keyword = None
        self._lookup_matches = []
        self._lookup_index = 0

    @staticmethod
    def _normalize_query(query):
        return " ".join(str(query).strip().strip("'\"").lower().split())

    @staticmethod
    def _normalize_title(title):
        return " ".join(str(title).strip().lower().split())

    @staticmethod
    def _raw_logged_hit(hit):
        return {
            "doc_id": str(hit.get("doc_id", hit.get("id", ""))),
            "title": hit.get("title"),
            "rank": hit.get("rank"),
            "bm25_rank": hit.get("bm25_rank"),
            "bm25_score": hit.get("bm25_score"),
            "dense_rank": hit.get("dense_rank"),
            "dense_score": hit.get("dense_score"),
            "fused_score": hit.get("fused_score"),
            "sentences": [
                {"sent_id": sent_id, "text": str(text)}
                for sent_id, text in enumerate(hit.get("sentences", []))
            ],
        }

    @staticmethod
    def _snippet_key(doc_id, sent_id):
        return (str(doc_id), int(sent_id))

    def _local_query_context(self, query):
        query = str(query).strip()
        if self._normalize_query(query) == self._normalize_query(self.question):
            return self.question
        return f"Question: {self.question}\nCurrent search: {query}"

    def _archive_search_document(self, hit, query, local_score=None, local_rank=None, sentence_scores=None):
        doc_id = str(hit.get("doc_id", hit.get("id", "")))
        entry = self._evidence_archive.get(doc_id)
        if entry is None:
            self._archive_order += 1
            entry = {
                "document": dict(hit),
                "first_seen_order": self._archive_order,
                "first_seen_query": query,
                "search_history": [],
                "latest_sentence_scores": {},
            }
            self._evidence_archive[doc_id] = entry
        else:
            entry["document"] = dict(hit)

        entry["search_history"].append({
            "query": query,
            "retrieval_rank": hit.get("retrieval_rank", hit.get("rank")),
            "local_page_rank": local_rank,
            "local_page_score": float(local_score) if local_score is not None else None,
        })
        if sentence_scores is not None:
            entry["latest_sentence_scores"] = {
                int(sent_id): float(score) for sent_id, score in sentence_scores.items()
            }

    def _upsert_search_snippet(self, candidate, query):
        key = self._snippet_key(candidate["doc_id"], candidate["sent_id"])
        self._snippet_order += 1
        entry = self._snippet_archive.get(key)
        if entry is None:
            entry = {
                "doc_id": str(candidate["doc_id"]),
                "title": candidate["title"],
                "sent_id": int(candidate["sent_id"]),
                "text": candidate["text"],
                "source": "search",
                "best_local_sentence_rank": int(candidate["local_sentence_rank"]),
                "best_local_page_rank": int(candidate["local_page_rank"]),
                "first_seen_order": self._snippet_order,
                "last_seen_order": self._snippet_order,
                "first_seen_query": query,
                "last_seen_query": query,
                "last_local_score": float(candidate["score"]),
                "search_count": 1,
                "lookup_count": 0,
            }
            self._snippet_archive[key] = entry
            return key

        entry["last_seen_order"] = self._snippet_order
        entry["last_seen_query"] = query
        entry["last_local_score"] = float(candidate["score"])
        entry["search_count"] = int(entry.get("search_count", 0)) + 1
        entry["best_local_sentence_rank"] = min(
            int(entry.get("best_local_sentence_rank", candidate["local_sentence_rank"])),
            int(candidate["local_sentence_rank"]),
        )
        entry["best_local_page_rank"] = min(
            int(entry.get("best_local_page_rank", candidate["local_page_rank"])),
            int(candidate["local_page_rank"]),
        )
        return key

    def _upsert_lookup_snippet(self, sent_id, text, keyword):
        doc_id = str(self.current_document.get("doc_id", self.current_document.get("id", "")))
        key = self._snippet_key(doc_id, sent_id)
        self._snippet_order += 1
        entry = self._snippet_archive.get(key)
        if entry is None:
            entry = {
                "doc_id": doc_id,
                "title": self.current_title,
                "sent_id": int(sent_id),
                "text": str(text),
                "source": "lookup",
                "best_local_sentence_rank": None,
                "best_local_page_rank": None,
                "first_seen_order": self._snippet_order,
                "last_seen_order": self._snippet_order,
                "first_seen_query": f"lookup[{keyword}]",
                "last_seen_query": f"lookup[{keyword}]",
                "last_local_score": None,
                "search_count": 0,
                "lookup_count": 1,
            }
            self._snippet_archive[key] = entry
        else:
            entry["source"] = "lookup"
            entry["last_seen_order"] = self._snippet_order
            entry["last_seen_query"] = f"lookup[{keyword}]"
            entry["lookup_count"] = int(entry.get("lookup_count", 0)) + 1
        return key

    @staticmethod
    def _memory_sort_key(item):
        key, entry = item
        if entry.get("source") == "lookup":
            return (
                0,
                -int(entry.get("lookup_count", 0)),
                -int(entry.get("last_seen_order", 0)),
                key,
            )
        sentence_rank = entry.get("best_local_sentence_rank")
        page_rank = entry.get("best_local_page_rank")
        return (
            1,
            int(sentence_rank) if sentence_rank is not None else 10**9,
            int(page_rank) if page_rank is not None else 10**9,
            -int(entry.get("last_seen_order", 0)),
            key,
        )

    def _refresh_snippet_memory(self):
        previous = list(self._active_snippet_keys)
        previous_set = set(previous)
        ranked = sorted(self._snippet_archive.items(), key=self._memory_sort_key)

        prefix = (
            "Active Evidence Memory (bounded sentence snippets; local CE scores are never "
            "compared across searches):\n"
        )
        remaining = max(0, self.max_evidence_chars - len(prefix))
        active_keys = []
        rendered_lines = []
        active_hits = []

        for key, entry in ranked:
            if len(active_keys) >= self.max_evidence_snippets:
                break
            label = f"[{entry['title']} | sent {entry['sent_id']}] "
            text = str(entry.get("text", ""))
            line = label + text
            needed = len(line) + (1 if rendered_lines else 0)
            if needed > remaining:
                if not active_keys:
                    available = max(0, remaining - len(label) - 1)
                    if available <= 0:
                        continue
                    text = text[:available].rstrip()
                    if len(text) < len(str(entry.get("text", ""))):
                        text = text.rstrip("…") + "…"
                    line = label + text
                    needed = len(line)
                else:
                    continue

            active_keys.append(key)
            rendered_lines.append(line)
            remaining = max(0, remaining - needed)
            active_hits.append({
                "doc_id": entry["doc_id"],
                "title": entry["title"],
                "memory_rank": len(active_keys),
                "memory_source": entry.get("source"),
                "local_sentence_rank": entry.get("best_local_sentence_rank"),
                "local_page_rank": entry.get("best_local_page_rank"),
                "sentences": [{"sent_id": entry["sent_id"], "text": text}],
            })

        self._active_snippet_keys = active_keys
        self._active_memory_context = prefix + "\n".join(rendered_lines) if rendered_lines else ""
        self._active_memory_hits = active_hits

        doc_ids = []
        for hit in active_hits:
            if hit["doc_id"] not in doc_ids:
                doc_ids.append(hit["doc_id"])
        self._evidence_doc_ids = doc_ids
        self._evidence_doc_id_set = set(doc_ids)

        for hit in active_hits:
            title = hit.get("title")
            if title and title not in self.visited_pages:
                self.visited_pages.append(title)

        active_set = set(active_keys)
        added = [key for key in active_keys if key not in previous_set]
        evicted = [key for key in previous if key not in active_set]
        return added, evicted, list(active_hits)

    def _snippet_summary(self, key):
        entry = self._snippet_archive.get(key, {})
        return {
            "doc_id": entry.get("doc_id"),
            "title": entry.get("title"),
            "sent_id": entry.get("sent_id"),
            "source": entry.get("source"),
            "best_local_sentence_rank": entry.get("best_local_sentence_rank"),
            "best_local_page_rank": entry.get("best_local_page_rank"),
        }

    def render_active_evidence(self):
        return self._active_memory_context

    def active_memory_snapshot(self):
        rows = []
        for memory_rank, key in enumerate(self._active_snippet_keys, 1):
            entry = self._snippet_archive.get(key, {})
            rows.append({
                "memory_rank": memory_rank,
                "doc_id": entry.get("doc_id"),
                "title": entry.get("title"),
                "sent_id": entry.get("sent_id"),
                "text": entry.get("text"),
                "source": entry.get("source"),
                "best_local_sentence_rank": entry.get("best_local_sentence_rank"),
                "best_local_page_rank": entry.get("best_local_page_rank"),
                "first_seen_query": entry.get("first_seen_query"),
                "last_seen_query": entry.get("last_seen_query"),
            })
        return rows

    @staticmethod
    def _merge_exposed_hits(hits):
        by_doc = {}
        order = []
        seen_sentences = set()
        for hit in hits:
            if not hit:
                continue
            doc_id = str(hit.get("doc_id", ""))
            if doc_id not in by_doc:
                by_doc[doc_id] = {
                    k: v for k, v in hit.items() if k != "sentences"
                }
                by_doc[doc_id]["sentences"] = []
                order.append(doc_id)
            for sentence in hit.get("sentences", []):
                sent_id = int(sentence["sent_id"])
                key = (doc_id, sent_id)
                if key in seen_sentences:
                    continue
                seen_sentences.add(key)
                by_doc[doc_id]["sentences"].append({
                    "sent_id": sent_id,
                    "text": str(sentence.get("text", "")),
                })
        return [by_doc[doc_id] for doc_id in order]

    def _current_page_observation(self, query, current_page, current_sentence_scores, selection_reason):
        scored = []
        for sent_id, score in current_sentence_scores.items():
            sentences = current_page.get("sentences", [])
            if 0 <= int(sent_id) < len(sentences):
                text = str(sentences[int(sent_id)]).strip()
                if text:
                    scored.append((float(score), int(sent_id), text))
        scored.sort(key=lambda row: (-row[0], row[1]))

        if not scored:
            for sent_id, text in enumerate(current_page.get("sentences", [])):
                clean = str(text).strip()
                if clean:
                    scored.append((0.0, sent_id, clean))

        selected = scored[: self.CURRENT_PAGE_OBSERVATION_SENTENCES]
        current_sentences = [
            {"sent_id": sent_id, "text": text}
            for _, sent_id, text in selected
        ]
        local_rank = current_page.get("local_rerank_rank")
        reason_text = "exact-title match" if selection_reason == "exact_title" else "local cross-encoder rank 1"
        prefix = (
            f"Observation: FullWiki search '{query}' selected current page [{current_page['title']}] "
            f"by {reason_text} (local page rank {local_rank}).\n"
        )
        lines = [
            f"[{current_page['title']} | sent {item['sent_id']}] {item['text']}"
            for item in current_sentences
        ]
        observation = prefix + "\n".join(lines)
        if len(observation) > self.max_observation_chars:
            observation = observation[: max(0, self.max_observation_chars - 1)].rstrip() + "…"
        return observation, current_sentences

    def _graph_expand_candidates(self, query, normalized_query, hits):
        if not (
            self.use_graph_expansion
            and hasattr(self.backend, "get_outgoing_edges")
            and hasattr(self.backend, "get_doc_by_title")
        ):
            return hits, []

        focus_titles = []
        if self.current_title:
            focus_titles.append(self.current_title)
        for doc_id in self._evidence_doc_ids[: self.graph_focus_doc_count]:
            entry = self._evidence_archive.get(doc_id, {})
            title = entry.get("document", {}).get("title")
            if title and title not in focus_titles:
                focus_titles.append(title)

        q_tokens = set(self._normalize_query(self.question).split() + normalized_query.split())
        candidate_edges = []
        seen_target_norms = {self._normalize_title(t) for t in self.visited_pages}

        for source_title in focus_titles:
            source_doc_id = None
            if hasattr(self.backend, "title_to_doc_id"):
                source_doc_id = self.backend.title_to_doc_id.get(source_title.lower())
            sent_scores = {}
            if source_doc_id and source_doc_id in self._evidence_archive:
                sent_scores = self._evidence_archive[source_doc_id].get("latest_sentence_scores", {})

            for edge in self.backend.get_outgoing_edges(source_title):
                target_title = edge["target_title"]
                if self._normalize_title(target_title) in seen_target_norms:
                    continue
                source_sent_id = int(edge.get("source_sent_id", 0))
                source_sent_score = float(sent_scores.get(source_sent_id, 0.0))
                anchor_text = edge.get("anchor_text", target_title)
                anchor_overlap = len(q_tokens & set(self._normalize_title(anchor_text).split()))
                title_overlap = len(q_tokens & set(self._normalize_title(target_title).split()))
                outdegree = self.backend.get_target_outdegree(target_title)
                score = (
                    self.graph_w_src * source_sent_score
                    + self.graph_w_anchor * anchor_overlap
                    + self.graph_w_title * title_overlap
                    - self.graph_w_outdegree * math.log1p(outdegree)
                )
                candidate_edges.append((score, edge))

        best_by_target = {}
        for score, edge in candidate_edges:
            target = edge["target_title"]
            if target not in best_by_target or score > best_by_target[target][0]:
                best_by_target[target] = (score, edge)

        graph_hits = []
        logs = []
        existing_ids = {str(hit.get("doc_id", "")) for hit in hits}
        for score, edge in sorted(best_by_target.values(), key=lambda row: row[0], reverse=True):
            if len(graph_hits) >= self.graph_candidate_quota:
                break
            graph_doc = self.backend.get_doc_by_title(edge["target_title"])
            if not graph_doc:
                continue
            doc_id = str(graph_doc.get("doc_id", graph_doc.get("id", "")))
            if doc_id in existing_ids:
                continue
            graph_hit = dict(graph_doc)
            graph_hit.update({
                "doc_id": doc_id,
                "rank": len(hits) + len(graph_hits) + 1,
                "fused_score": 0.5 + float(score),
                "graph_candidate": True,
            })
            graph_hits.append(graph_hit)
            existing_ids.add(doc_id)
            logs.append({
                "source_title": edge.get("source_title"),
                "source_sent_id": edge.get("source_sent_id"),
                "anchor_text": edge.get("anchor_text"),
                "target_title": edge.get("target_title"),
                "score": round(float(score), 4),
            })

        if not graph_hits:
            return hits, logs

        quota = min(len(graph_hits), self.graph_candidate_quota, self.search_top_k)
        keep = max(0, self.search_top_k - quota)
        return hits[:keep] + graph_hits[:quota], logs

    def _search_with_local_reranking(
        self,
        query,
        normalized_query,
        result,
        hits,
        raw_logged_hits,
        title_match_rank,
        exact_title_injected,
        graph_expansion_logs,
    ):
        local_query = self._local_query_context(query)
        page_scores, page_latency, estimated_truncated = self.backend.score_page_documents(
            local_query, hits
        )
        if len(page_scores) != len(hits):
            raise RuntimeError(
                f"Local reranker returned {len(page_scores)} page scores for {len(hits)} pages."
            )

        ranked_pages = []
        for hit, score in zip(hits, page_scores):
            row = dict(hit)
            row["retrieval_rank"] = hit.get("rank")
            row["local_rerank_score"] = float(score)
            ranked_pages.append(row)
        ranked_pages.sort(key=lambda hit: (
            -float(hit["local_rerank_score"]),
            int(hit.get("retrieval_rank") or 10**9),
            str(hit.get("doc_id", "")),
        ))
        for local_rank, hit in enumerate(ranked_pages, 1):
            hit["local_rerank_rank"] = local_rank

        exact_match_page = next(
            (hit for hit in ranked_pages if self._normalize_title(hit["title"]) == normalized_query),
            None,
        )
        if exact_match_page is not None:
            current_page = exact_match_page
            selection_reason = "exact_title"
        else:
            current_page = ranked_pages[0]
            selection_reason = "local_rerank"

        # Point 2 invariant: current page is fixed by THIS search only. Memory is
        # refreshed later and never gets a chance to change this identity.
        self.current_document = dict(current_page)
        self.current_title = current_page["title"]
        self._lookup_keyword = None
        self._lookup_matches = []
        self._lookup_index = 0
        if self.current_title not in self.visited_pages:
            self.visited_pages.append(self.current_title)

        sentence_pages = list(ranked_pages[: self.local_rerank_page_count])
        current_doc_id = str(current_page.get("doc_id", ""))
        if current_doc_id not in {str(page.get("doc_id", "")) for page in sentence_pages}:
            sentence_pages = [current_page] + sentence_pages[: self.local_rerank_page_count - 1]

        scores_by_doc, sentence_latency, sentence_pair_count = self.backend.score_document_sentences(
            local_query, sentence_pages
        )

        sentence_page_ids = {str(page.get("doc_id", "")) for page in sentence_pages}
        for page in ranked_pages:
            doc_id = str(page.get("doc_id", ""))
            self._archive_search_document(
                page,
                query=query,
                local_score=page.get("local_rerank_score"),
                local_rank=page.get("local_rerank_rank"),
                sentence_scores=scores_by_doc.get(doc_id) if doc_id in sentence_page_ids else None,
            )

        local_sentence_candidates = []
        for page in sentence_pages:
            doc_id = str(page.get("doc_id", ""))
            page_sentences = page.get("sentences", [])
            for sent_id, score in scores_by_doc.get(doc_id, {}).items():
                if not (0 <= int(sent_id) < len(page_sentences)):
                    continue
                text = str(page_sentences[int(sent_id)]).strip()
                if not text:
                    continue
                local_sentence_candidates.append({
                    "doc_id": doc_id,
                    "title": page["title"],
                    "sent_id": int(sent_id),
                    "text": text,
                    "score": float(score),
                    "local_page_rank": int(page["local_rerank_rank"]),
                })

        local_sentence_candidates.sort(key=lambda item: (
            -float(item["score"]),
            int(item["local_page_rank"]),
            int(item["sent_id"]),
            str(item["doc_id"]),
        ))
        for sentence_rank, candidate in enumerate(local_sentence_candidates, 1):
            candidate["local_sentence_rank"] = sentence_rank
            self._upsert_search_snippet(candidate, query)

        added_keys, evicted_keys, active_hits = self._refresh_snippet_memory()

        current_sentence_scores = scores_by_doc.get(current_doc_id, {})
        observation, current_observation_sentences = self._current_page_observation(
            query,
            current_page,
            current_sentence_scores,
            selection_reason,
        )
        current_observation_hit = {
            "doc_id": current_doc_id,
            "title": current_page["title"],
            "rank": current_page.get("retrieval_rank"),
            "local_rerank_rank": current_page.get("local_rerank_rank"),
            "local_rerank_score": current_page.get("local_rerank_score"),
            "sentences": current_observation_sentences,
        }
        exposed_hits = self._merge_exposed_hits([current_observation_hit] + active_hits)

        local_reranked_hits = [
            {
                "doc_id": str(page.get("doc_id", "")),
                "title": page.get("title"),
                "retrieval_rank": page.get("retrieval_rank"),
                "local_rerank_rank": page.get("local_rerank_rank"),
                "local_rerank_score": page.get("local_rerank_score"),
                "bm25_rank": page.get("bm25_rank"),
                "dense_rank": page.get("dense_rank"),
                "fused_score": page.get("fused_score"),
            }
            for page in ranked_pages
        ]
        sentence_rerank_pages = [
            {
                "doc_id": str(page.get("doc_id", "")),
                "title": page.get("title"),
                "local_rerank_rank": page.get("local_rerank_rank"),
                "sentence_pair_count": len(scores_by_doc.get(str(page.get("doc_id", "")), {})),
            }
            for page in sentence_pages
        ]

        reranker = getattr(self.backend, "local_reranker", None)
        if reranker is None:
            reranker = getattr(self.backend, "evidence_reranker", None)
        reranker_desc = reranker.describe() if reranker is not None and hasattr(reranker, "describe") else None

        self.last_result = {
            "action": "search",
            "query": query,
            "status": "loaded",
            "retriever": result.get("mode"),
            "duplicate_query": False,
            "query_matches_retrieved_title": exact_match_page is not None,
            "query_title_match_rank": title_match_rank,
            "exact_title_injected": bool(exact_title_injected),
            "candidate_k": result.get("candidate_k"),
            "top_k": result.get("top_k"),
            "hits": exposed_hits,
            "retrieved_hits": raw_logged_hits,
            "local_reranked_hits": local_reranked_hits,
            "sentence_rerank_pages": sentence_rerank_pages,
            "current_page": self.current_title,
            "current_page_doc_id": current_doc_id,
            "current_page_local_rank": current_page.get("local_rerank_rank"),
            "current_page_selection_reason": selection_reason,
            "new_evidence_snippets": [self._snippet_summary(key) for key in added_keys],
            "evicted_evidence_snippets": [self._snippet_summary(key) for key in evicted_keys],
            "evidence_snippet_count": self.evidence_snippet_count,
            "evidence_snippet_archive_count": self.evidence_snippet_archive_count,
            "evidence_document_count": self.evidence_document_count,
            "evidence_archive_count": self.evidence_archive_count,
            "max_evidence_snippets": self.max_evidence_snippets,
            "max_evidence_chars": self.max_evidence_chars,
            "memory_policy": "lookup_protected_then_within_search_sentence_rank",
            "active_memory_snippets": self.active_memory_snapshot(),
            "local_reranker": reranker_desc,
            "local_page_pair_count": len(hits),
            "local_sentence_pair_count": int(sentence_pair_count),
            "page_pair_estimated_truncations": int(estimated_truncated),
            "graph_expansion_info": graph_expansion_logs or [],
            "latency_ms": {
                **result.get("latency_ms", {}),
                "local_page_reranker": round(page_latency * 1000, 3),
                "local_sentence_reranker": round(sentence_latency * 1000, 3),
                "local_reranker_total": round((page_latency + sentence_latency) * 1000, 3),
            },
            "title": self.current_title,
            "sentences": current_observation_sentences,
        }
        self._seen_search_queries[normalized_query] = {
            "retriever": result.get("mode"),
            "candidate_k": result.get("candidate_k"),
            "retrieved_hits": raw_logged_hits,
            "retrieved_titles": [hit.get("title") for hit in hits],
            "query_matches_retrieved_title": exact_match_page is not None,
            "query_title_match_rank": title_match_rank,
            "exact_title_injected": bool(exact_title_injected),
        }
        return observation

    def _search_without_local_reranking(self, query, normalized_query, result, hits, raw_logged_hits):
        current = dict(hits[0])
        self.current_document = current
        self.current_title = current["title"]
        self._lookup_keyword = None
        self._lookup_matches = []
        self._lookup_index = 0
        if self.current_title not in self.visited_pages:
            self.visited_pages.append(self.current_title)

        visible = []
        for sent_id, text in enumerate(current.get("sentences", [])):
            clean = str(text).strip()
            if clean:
                visible.append({"sent_id": sent_id, "text": clean})
            if len(visible) >= 5:
                break
        prefix = f"Observation: FullWiki current page [{self.current_title}].\n"
        body = "\n".join(
            f"[{self.current_title} | sent {item['sent_id']}] {item['text']}" for item in visible
        )
        observation = (prefix + body)[: self.max_observation_chars]
        hit = {
            "doc_id": str(current.get("doc_id", "")),
            "title": self.current_title,
            "rank": current.get("rank", 1),
            "sentences": visible,
        }
        self.last_result = {
            "action": "search",
            "query": query,
            "status": "loaded",
            "retriever": result.get("mode"),
            "duplicate_query": False,
            "candidate_k": result.get("candidate_k"),
            "top_k": result.get("top_k"),
            "hits": [hit],
            "retrieved_hits": raw_logged_hits,
            "title": self.current_title,
            "sentences": visible,
            "latency_ms": result.get("latency_ms", {}),
        }
        self._seen_search_queries[normalized_query] = {
            "retriever": result.get("mode"),
            "candidate_k": result.get("candidate_k"),
            "retrieved_hits": raw_logged_hits,
            "retrieved_titles": [hit.get("title") for hit in hits],
        }
        return observation

    def search(self, query):
        query = str(query).strip().strip("'\"")
        normalized_query = self._normalize_query(query)
        if not normalized_query:
            return "Observation: Search query cannot be empty."

        if self.duplicate_search_guard and normalized_query in self._seen_search_queries:
            previous = self._seen_search_queries[normalized_query]
            self.last_result = {
                "action": "search",
                "query": query,
                "status": "duplicate_query",
                "retriever": previous.get("retriever"),
                "duplicate_query": True,
                "candidate_k": previous.get("candidate_k"),
                "top_k": self.search_top_k,
                "hits": [],
                "retrieved_hits": previous.get("retrieved_hits", []),
                "evidence_snippet_count": self.evidence_snippet_count,
                "max_evidence_snippets": self.max_evidence_snippets,
                "latency_ms": {"total": 0.0},
                "title": self.current_title,
                "sentences": [],
            }
            return f"Observation: The search query '{query}' was already performed earlier in this session."

        result = self.backend.search(query, top_k=self.search_top_k)
        hits = [dict(hit) for hit in result.get("hits", [])]
        raw_logged_hits = [self._raw_logged_hit(hit) for hit in hits]
        title_match_rank = next(
            (hit.get("rank") for hit in hits if self._normalize_title(hit.get("title", "")) == normalized_query),
            None,
        )
        exact_title_injected = False

        # Keep Point 4's future fast-path separate. We still allow exact-title
        # injection after first-stage retrieval, but it occupies one of the 15
        # local-reranker slots instead of creating a 16th pair.
        if self.use_local_reranking and title_match_rank is None and hasattr(self.backend, "get_doc_by_title"):
            direct_doc = self.backend.get_doc_by_title(query)
            if direct_doc:
                direct_id = str(direct_doc.get("doc_id", direct_doc.get("id", "")))
                existing_ids = {str(hit.get("doc_id", "")) for hit in hits}
                if direct_id not in existing_ids:
                    direct_hit = dict(direct_doc)
                    direct_hit["doc_id"] = direct_id
                    direct_hit["rank"] = (max([int(hit.get("rank") or 0) for hit in hits] or [0]) + 1)
                    direct_hit.setdefault("bm25_rank", None)
                    direct_hit.setdefault("bm25_score", None)
                    direct_hit.setdefault("dense_rank", None)
                    direct_hit.setdefault("dense_score", None)
                    direct_hit.setdefault("fused_score", None)
                    if len(hits) >= self.search_top_k:
                        hits = hits[: self.search_top_k - 1] + [direct_hit]
                    else:
                        hits.append(direct_hit)
                    title_match_rank = direct_hit["rank"]
                    exact_title_injected = True

        graph_expansion_logs = []
        if hits:
            hits, graph_expansion_logs = self._graph_expand_candidates(query, normalized_query, hits)
            hits = hits[: self.search_top_k]

        if not hits:
            self.last_result = {
                "action": "search",
                "query": query,
                "status": "not_found",
                "retriever": result.get("mode"),
                "duplicate_query": False,
                "candidate_k": result.get("candidate_k"),
                "top_k": result.get("top_k"),
                "hits": [],
                "retrieved_hits": raw_logged_hits,
                "evidence_snippet_count": self.evidence_snippet_count,
                "max_evidence_snippets": self.max_evidence_snippets,
                "latency_ms": result.get("latency_ms", {}),
                "title": None,
                "sentences": [],
            }
            self._seen_search_queries[normalized_query] = dict(
                self.last_result,
                retrieved_titles=[],
            )
            return f"Observation: FullWiki retrieval found no document for '{query}'."

        if self.use_local_reranking:
            return self._search_with_local_reranking(
                query,
                normalized_query,
                result,
                hits,
                raw_logged_hits,
                title_match_rank,
                exact_title_injected,
                graph_expansion_logs,
            )
        return self._search_without_local_reranking(
            query, normalized_query, result, hits, raw_logged_hits
        )

    def lookup(self, keyword):
        """Classic ReAct lookup over the current page, independent of memory.

        Repeating lookup with the same keyword advances through matching sentences
        on the same current page. A successful lookup promotes the exact matched
        sentence into persistent snippet memory, but memory never gates lookup.
        """
        keyword_raw = str(keyword).strip().strip("'\"")
        keyword_clean = keyword_raw.lower()
        if not keyword_clean:
            return "Observation: Lookup keyword cannot be empty."

        if not self.current_document or not self.current_title:
            self.last_result = {
                "action": "lookup",
                "query": keyword_raw,
                "status": "no_current_page",
                "title": None,
                "sentences": [],
                "hits": [],
            }
            return "Observation: No FullWiki document currently loaded. Perform a `search` first."

        if self._lookup_keyword != keyword_clean:
            self._lookup_keyword = keyword_clean
            self._lookup_matches = []
            self._lookup_index = 0
            for sent_id, text in enumerate(self.current_document.get("sentences", [])):
                if keyword_clean in str(text).lower():
                    self._lookup_matches.append({"sent_id": sent_id, "text": str(text)})

        if self._lookup_index >= len(self._lookup_matches):
            self.last_result = {
                "action": "lookup",
                "query": keyword_raw,
                "status": "no_more_results",
                "title": self.current_title,
                "sentences": [],
                "hits": [],
                "lookup_result_index": self._lookup_index,
                "lookup_result_count": len(self._lookup_matches),
            }
            if self._lookup_matches:
                return f"Observation: No more results for '{keyword_raw}' in [{self.current_title}]."
            return f"Observation: Could not find '{keyword_raw}' in [{self.current_title}]."

        match = dict(self._lookup_matches[self._lookup_index])
        self._lookup_index += 1
        total_matches = len(self._lookup_matches)
        current_doc_id = str(self.current_document.get("doc_id", self.current_document.get("id", "")))

        added_keys = []
        evicted_keys = []
        if self.use_local_reranking:
            key = self._upsert_lookup_snippet(match["sent_id"], match["text"], keyword_raw)
            added_keys, evicted_keys, _ = self._refresh_snippet_memory()
            if key not in self._active_snippet_keys:
                # This should be extremely rare (only if a single lookup sentence
                # cannot fit the character budget), but lookup itself still succeeds.
                pass

        hit = {
            "doc_id": current_doc_id,
            "title": self.current_title,
            "rank": 1,
            "sentences": [match],
        }
        self.last_result = {
            "action": "lookup",
            "query": keyword_raw,
            "status": "found",
            "title": self.current_title,
            "sentences": [match],
            "hits": [hit],
            "lookup_result_index": self._lookup_index,
            "lookup_result_count": total_matches,
            "current_page_rank": 1,
            "current_page_doc_id": current_doc_id,
            "new_evidence_snippets": [self._snippet_summary(key) for key in added_keys],
            "evicted_evidence_snippets": [self._snippet_summary(key) for key in evicted_keys],
            "evidence_snippet_count": self.evidence_snippet_count,
            "max_evidence_snippets": self.max_evidence_snippets,
        }

        prefix = f"Observation: (Result {self._lookup_index} / {total_matches}) "
        label = f"[{self.current_title} | sent {match['sent_id']}] "
        available = max(0, self.max_observation_chars - len(prefix) - len(label))
        text = match["text"]
        if len(text) > available:
            text = text[: max(0, available - 1)].rstrip() + "…"
            match["text"] = text
            hit["sentences"] = [match]
            self.last_result["sentences"] = [match]
            self.last_result["hits"] = [hit]
        return prefix + label + text
