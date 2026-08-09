import json
import os
import threading
import time
from collections import defaultdict

import numpy as np

from config import (
    DENSE_QUERY_DEVICE,
    FULLWIKI_BM25_INDEX_DIR,
    FULLWIKI_DENSE_INDEX_PATH,
    FULLWIKI_INDEX_MANIFEST,
    FULLWIKI_RRF_K,
    FULLWIKI_SEARCH_CANDIDATES,
)

from retrieval.reranker import CrossEncoderEvidenceReranker

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
        evidence_reranker_model=None,
        evidence_reranker_device="cpu",
        evidence_reranker_max_length=512,
        evidence_reranker_batch_size=16,
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

        # Thread-safe retrieval caches
        self._doc_id_cache = {}
        self._doc_cache_lock = threading.Lock()

        self._title_doc_cache = {}
        self._title_cache_lock = threading.Lock()

        self._query_embed_cache = {}
        self._embed_cache_lock = threading.Lock()

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

        self.evidence_reranker = None
        if evidence_reranker_model:
            self.evidence_reranker = CrossEncoderEvidenceReranker(
                evidence_reranker_model,
                device=evidence_reranker_device,
                max_length=evidence_reranker_max_length,
                batch_size=evidence_reranker_batch_size,
            )

        self.title_graph_path = os.path.join(os.path.dirname(self.manifest_path), "title_graph.json")
        self.title_graph = self._load_title_graph()

    def _load_title_graph(self):
        if os.path.isfile(self.title_graph_path):
            try:
                with open(self.title_graph_path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
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
        self.dense_index = faiss.read_index(self.dense_index_path)
        if hasattr(self.dense_index, "nprobe"):
            self.dense_index.nprobe = self.dense_nprobe
        self.encoder = SentenceTransformer(self.dense_model_name, device=self.dense_query_device)

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

        escaped_title = title.replace('"', '\\"')
        hits = self.lucene.search(f'title:"{escaped_title}"', k=5)
        found_doc = None
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
        query_key = (query, int(k))

        with self._embed_cache_lock:
            cached_encoded = self._query_embed_cache.get(query_key)

        if cached_encoded is None:
            with self._dense_lock:
                encoded = self.encoder.encode(
                    [BGE_QUERY_INSTRUCTION + query],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                ).astype("float32")
            with self._embed_cache_lock:
                if len(self._query_embed_cache) < 50000:
                    self._query_embed_cache[query_key] = encoded
        else:
            encoded = cached_encoded

        # FAISS C++ search is 100% thread-safe for read-only index: run OUTSIDE self._dense_lock!
        scores, ids = self.dense_index.search(encoded, k)

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

        if mode in {"bm25", "hybrid"}:
            sparse_hits, bm25_latency = self._bm25_search(query, candidate_k)
        if mode in {"dense", "hybrid"}:
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
                "total": round((bm25_latency + dense_latency + fusion_latency) * 1000, 3),
            },
        }

    @staticmethod
    def _reranker_passages_for_document(document):
        title = str(document.get("title", "")).strip()
        sentences = [str(x).strip() for x in document.get("sentences", []) if str(x).strip()]
        passages = [f"{title}: {sent}" for sent in sentences[:6]]
        full_lead = f"{title}\n{' '.join(sentences[:6])}".strip()
        if full_lead and full_lead not in passages:
            passages.append(full_lead)
        return passages or [title]

    def score_evidence_documents(self, question, documents):
        if self.evidence_reranker is None:
            raise RuntimeError("Evidence reranker is not configured on this FullWiki backend.")
        if not documents:
            return ([], []), 0.0

        all_passages = []
        doc_slice_bounds = []
        start = 0

        for doc in documents:
            passages = self._reranker_passages_for_document(doc)
            all_passages.extend(passages)
            end = start + len(passages)
            doc_slice_bounds.append((start, end))
            start = end

        scores, latency = self.evidence_reranker.score(question, all_passages)

        doc_max_scores = []
        doc_best_sents = []
        for doc, (s_idx, e_idx) in zip(documents, doc_slice_bounds):
            doc_scores = scores[s_idx:e_idx]
            max_rel_idx = max(range(len(doc_scores)), key=lambda i: doc_scores[i])
            doc_max_scores.append(doc_scores[max_rel_idx])

            sentences = [str(x).strip() for x in doc.get("sentences", []) if str(x).strip()]
            num_sents = len(sentences[:6])
            best_sent_id = max_rel_idx if max_rel_idx < num_sents else 0
            doc_best_sents.append(best_sent_id)

            sent_scores_dict = {
                idx: float(score) for idx, score in enumerate(doc_scores[:num_sents])
            }
            doc_sent_scores_list.append(sent_scores_dict)

        return (doc_max_scores, doc_best_sents, doc_sent_scores_list), latency

    def create_session(
        self,
        search_top_k=1,
        max_observation_chars=2200,
        max_evidence_documents=None,
        duplicate_search_guard=False,
        question=None,
    ):
        return FullWikiRetriever(
            self,
            search_top_k=search_top_k,
            max_observation_chars=max_observation_chars,
            max_evidence_documents=max_evidence_documents,
            duplicate_search_guard=duplicate_search_guard,
            question=question,
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
            "evidence_memory_reranker": (
                self.evidence_reranker.describe() if self.evidence_reranker is not None else None
            ),
            "index_manifest": self.manifest,
        }


class FullWikiRetriever:
    """Per-question ReAct-compatible session over a shared FullWiki backend.

    The backend always retrieves ``search_top_k`` results. When a shared evidence
    reranker is configured, every unique retrieved document is archived and scored
    against the original question and, when different, the current search sub-query.
    Rediscovered archive documents may improve their running relevance score under a
    later sub-query. Only the best ``max_evidence_documents`` remain in the recurrent
    Active Evidence Memory. Raw top-k results remain in ``last_result['retrieved_hits']``
    for later analysis. Without a reranker, the legacy first-seen bounded-memory
    behavior is preserved for compatibility.
    """

    def __init__(
        self,
        backend,
        search_top_k=1,
        max_observation_chars=2200,
        max_evidence_documents=None,
        duplicate_search_guard=False,
        question=None,
    ):
        self.backend = backend
        self.search_top_k = int(search_top_k)
        self.max_observation_chars = int(max_observation_chars)
        self.max_evidence_documents = (
            int(max_evidence_documents) if max_evidence_documents is not None else None
        )
        self.duplicate_search_guard = bool(duplicate_search_guard)
        self.question = str(question).strip() if question is not None else None
        self.use_reranked_memory = (
            getattr(self.backend, "evidence_reranker", None) is not None
            and self.max_evidence_documents is not None
            and bool(self.question)
        )
        self.current_title = None
        self.current_document = None
        self.last_result = None
        self.visited_pages = []
        self._evidence_doc_ids = []
        self._evidence_doc_id_set = set()
        self._evidence_archive = {}
        self._archive_order = 0
        self._active_memory_context = ""
        self._active_memory_hits = []
        self._seen_search_queries = {}
        self._lookup_keyword = None
        self._lookup_matches = []
        self._lookup_index = 0

    @property
    def current_page_title(self):
        return self.current_title

    @property
    def evidence_document_count(self):
        return len(self._evidence_doc_ids)

    def reset(self):
        self.current_title = None
        self.current_document = None
        self.last_result = None
        self.visited_pages = []
        self._evidence_doc_ids = []
        self._evidence_doc_id_set = set()
        self._evidence_archive = {}
        self._archive_order = 0
        self._active_memory_context = ""
        self._active_memory_hits = []
        self._seen_search_queries = {}
        self._lookup_keyword = None
        self._lookup_matches = []
        self._lookup_index = 0

    @staticmethod
    def _render_sentences(title, sentences):
        return "\n".join(
            f"[{title} | sent {item['sent_id']}] {item['text']}" for item in sentences
        )

    @staticmethod
    def _normalize_query(query):
        return " ".join(str(query).strip().strip("'\"").lower().split())

    @staticmethod
    def _normalize_title(title):
        return " ".join(str(title).strip().lower().split())

    @staticmethod
    def _raw_logged_hit(hit):
        return {
            "doc_id": str(hit["doc_id"]),
            "title": hit["title"],
            "rank": hit["rank"],
            "bm25_rank": hit.get("bm25_rank"),
            "bm25_score": hit.get("bm25_score"),
            "dense_rank": hit.get("dense_rank"),
            "dense_score": hit.get("dense_score"),
            "fused_score": hit.get("fused_score"),
            "sentences": [
                {"sent_id": sent_id, "text": text}
                for sent_id, text in enumerate(hit.get("sentences", []))
            ],
        }

    @staticmethod
    def _unpack_scores_and_bests(res):
        if isinstance(res, tuple) and len(res) == 3:
            return res[0], res[1], res[2]
        elif isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], list) and isinstance(res[1], list):
            return res[0], res[1], [{} for _ in res[0]]
        elif isinstance(res, list):
            return res, [0] * len(res), [{} for _ in res]
        return [], [], []

    def _visible_sentences(self, document, char_budget):
        doc_id = str(document.get("doc_id", ""))
        entry = self._evidence_archive.get(doc_id, {})
        best_sent_id = entry.get("best_sent_id", 0)
        sent_scores = entry.get("sentence_scores", {})

        all_sentences = document.get("sentences", [])
        if not all_sentences:
            return []

        # Always include sent 0 for title and lead sentence context
        selected_ids = {0}
        if 0 <= best_sent_id < len(all_sentences):
            selected_ids.add(best_sent_id)

        # Rank all non-lead sentences by Cross-Encoder score
        other_sent_indices = [idx for idx in range(1, len(all_sentences))]
        other_sent_indices.sort(key=lambda idx: sent_scores.get(idx, -999.0), reverse=True)

        for idx in other_sent_indices[:3]:
            selected_ids.add(idx)

        ordered_sent_ids = sorted(selected_ids)

        visible = []
        used = 0
        for sent_id in ordered_sent_ids:
            text = all_sentences[sent_id]
            label = f"[{document['title']} | sent {sent_id}] "
            available = char_budget - used - len(label)
            if available <= 0:
                break
            clean_text = str(text)
            if len(clean_text) > available:
                if not visible:
                    clean_text = clean_text[: max(0, available - 1)].rstrip() + "…"
                else:
                    break
            visible.append({"sent_id": sent_id, "text": clean_text})
            used += len(label) + len(clean_text) + 1
            if used >= char_budget:
                break
        return visible

    def _render_new_hits(self, hits, prefix):
        if not hits:
            return "", [], max(0, self.max_observation_chars - len(prefix))

        remaining = max(0, self.max_observation_chars - len(prefix))
        rendered_blocks = []
        logged_hits = []

        for index, hit in enumerate(hits):
            docs_left = len(hits) - index
            block_header = f"Loaded [{hit['title']}] (rank {hit['rank']}).\n"
            if len(block_header) >= remaining:
                break
            per_doc_budget = max(0, remaining // docs_left)
            sentence_budget = max(0, per_doc_budget - len(block_header) - 2)
            visible = self._visible_sentences(hit, sentence_budget)
            if not visible:
                continue
            block = block_header + self._render_sentences(hit["title"], visible)
            if len(block) > remaining:
                break
            rendered_blocks.append(block)
            remaining = max(0, remaining - len(block) - 2)

            logged_hits.append({
                "doc_id": str(hit["doc_id"]),
                "title": hit["title"],
                "rank": hit["rank"],
                "bm25_rank": hit.get("bm25_rank"),
                "bm25_score": hit.get("bm25_score"),
                "dense_rank": hit.get("dense_rank"),
                "dense_score": hit.get("dense_score"),
                "fused_score": hit.get("fused_score"),
                "sentences": visible,
            })

        return "\n\n".join(rendered_blocks), logged_hits, remaining

    @property
    def evidence_archive_count(self):
        return len(self._evidence_archive)

    def render_active_evidence(self):
        return self._active_memory_context

    def active_memory_snapshot(self):
        rows = []
        for memory_rank, doc_id in enumerate(self._evidence_doc_ids, 1):
            entry = self._evidence_archive.get(doc_id, {})
            document = entry.get("document", {})
            rows.append({
                "memory_rank": memory_rank,
                "doc_id": doc_id,
                "title": document.get("title"),
                "reranker_score": entry.get("reranker_score"),
                "first_seen_order": entry.get("first_seen_order"),
                "first_seen_query": entry.get("first_seen_query"),
            })
        return rows

    def _score_new_archive_documents(self, query, hits):
        # Partition hits into new vs previously archived documents before updating self._evidence_archive
        new_hits = [hit for hit in hits if str(hit["doc_id"]) not in self._evidence_archive]
        new_doc_ids = {str(hit["doc_id"]) for hit in new_hits}
        existing_hits = [
            hit for hit in hits
            if str(hit["doc_id"]) in self._evidence_archive and str(hit["doc_id"]) not in new_doc_ids
        ]

        added_titles = []
        latency = 0.0

        if new_hits:
            res_q, latency1 = self.backend.score_evidence_documents(self.question, new_hits)
            scores_q, b_sents_q, s_scores_q = self._unpack_scores_and_bests(res_q)

            # Sub-Query Bridge Protection (Max-Score Rule):
            # Score against both the original question and the specific search sub-query
            # for new documents to ensure intermediate bridge documents are not prematurely evicted.
            if query and self._normalize_query(query) != self._normalize_query(self.question):
                res_sub, latency2 = self.backend.score_evidence_documents(query, new_hits)
                scores_sub, b_sents_sub, s_scores_sub = self._unpack_scores_and_bests(res_sub)

                scores = []
                b_sents = []
                sent_scores_list = []
                for sq, ss, bq, bs, sq_dict, ss_dict in zip(scores_q, scores_sub, b_sents_q, b_sents_sub, s_scores_q, s_scores_sub):
                    if ss > sq:
                        scores.append(ss)
                        b_sents.append(bs)
                        sent_scores_list.append(ss_dict)
                    else:
                        scores.append(sq)
                        b_sents.append(bq)
                        sent_scores_list.append(sq_dict)
                latency += latency1 + latency2
            else:
                scores = scores_q
                b_sents = b_sents_q
                sent_scores_list = s_scores_q
                latency += latency1

            if len(scores) != len(new_hits):
                raise RuntimeError(
                    f"Evidence reranker returned {len(scores)} scores for {len(new_hits)} documents."
                )

            for hit, score, b_sent, s_dict in zip(new_hits, scores, b_sents, sent_scores_list):
                doc_id = str(hit["doc_id"])
                self._archive_order += 1
                self._evidence_archive[doc_id] = {
                    "document": dict(hit),
                    "reranker_score": float(score),
                    "best_sent_id": int(b_sent),
                    "sentence_scores": s_dict or {},
                    "first_seen_order": self._archive_order,
                    "first_seen_query": query,
                }
                added_titles.append(hit["title"])

        # Rediscovery Score Upgrade Rule:
        # If an already-archived document is explicitly rediscovered by a later sub-query,
        # score it against that sub-query and upgrade its stored score if higher.
        if query and self._normalize_query(query) != self._normalize_query(self.question) and existing_hits:
            res_ext, latency_ext = self.backend.score_evidence_documents(query, existing_hits)
            scores_existing, b_sents_existing, s_scores_existing = self._unpack_scores_and_bests(res_ext)
            latency += latency_ext
            for hit, score, b_sent, s_dict in zip(existing_hits, scores_existing, b_sents_existing, s_scores_existing):
                doc_id = str(hit["doc_id"])
                entry = self._evidence_archive[doc_id]
                if float(score) > float(entry["reranker_score"]):
                    entry["reranker_score"] = float(score)
                    entry["best_sent_id"] = int(b_sent)
                    entry["sentence_scores"] = s_dict or {}

        return added_titles, latency

    def _render_reranked_memory(self, current_doc_id=None):
        if not self._evidence_doc_ids:
            self._active_memory_context = ""
            self._active_memory_hits = []
            return [], []

        prefix = (
            "Active Evidence Memory (cross-encoder reranked; use ONLY sentence labels shown here "
            "or in lookup observations as evidence):\n"
        )
        remaining = max(0, self.max_observation_chars - len(prefix))
        rendered_blocks = []
        logged_hits = []
        omitted_due_to_char_cap = []

        # When current_doc_id is present in active memory, render it first so it
        # cannot be cut off by the global character budget on the turn it was searched.
        render_doc_ids = list(self._evidence_doc_ids)
        if current_doc_id in self._evidence_doc_id_set and render_doc_ids[0] != current_doc_id:
            render_doc_ids.remove(current_doc_id)
            render_doc_ids.insert(0, current_doc_id)
        memory_rank_by_id = {
            doc_id: memory_rank for memory_rank, doc_id in enumerate(self._evidence_doc_ids, 1)
        }

        for index, doc_id in enumerate(render_doc_ids):
            entry = self._evidence_archive[doc_id]
            document = entry["document"]
            memory_rank = memory_rank_by_id[doc_id]
            header = f"Memory [{memory_rank}] [{document['title']}]\n"
            if len(header) >= remaining:
                omitted_due_to_char_cap.extend(
                    self._evidence_archive[x]["document"]["title"]
                    for x in render_doc_ids[index:]
                )
                break

            sentence_budget = max(0, min(remaining - len(header) - 2, 3000))
            visible = self._visible_sentences(document, sentence_budget)
            if not visible:
                omitted_due_to_char_cap.append(document["title"])
                continue

            block = header + self._render_sentences(document["title"], visible)
            if len(block) > remaining:
                omitted_due_to_char_cap.extend(
                    self._evidence_archive[x]["document"]["title"]
                    for x in render_doc_ids[index:]
                )
                break

            rendered_blocks.append(block)
            remaining = max(0, remaining - len(block) - 2)
            logged_hits.append({
                "doc_id": doc_id,
                "title": document["title"],
                "rank": document.get("rank"),
                "memory_rank": memory_rank,
                "bm25_rank": document.get("bm25_rank"),
                "bm25_score": document.get("bm25_score"),
                "dense_rank": document.get("dense_rank"),
                "dense_score": document.get("dense_score"),
                "fused_score": document.get("fused_score"),
                "reranker_score": entry["reranker_score"],
                "sentences": visible,
            })

        body = "\n\n".join(rendered_blocks)
        self._active_memory_context = prefix + body if body else ""
        self._active_memory_hits = logged_hits
        return logged_hits, omitted_due_to_char_cap

    def _refresh_reranked_memory(self, current_doc_id=None, is_exact_title_search=False):
        previous_ids = list(self._evidence_doc_ids)
        previous_set = set(previous_ids)
        ranked = sorted(
            self._evidence_archive.items(),
            key=lambda item: (
                -float(item[1]["reranker_score"]),
                int(item[1]["first_seen_order"]),
                item[0],
            ),
        )
        limit = self.max_evidence_documents or len(ranked)
        selected_ids = [doc_id for doc_id, _ in ranked[:limit]]

        # Exact-Title Memory Reservation:
        # If an explicit exact title match exists (e.g. search[Inception] matched title "Inception"),
        # reserve one slot in active memory for current_doc_id so subsequent lookup[keyword]
        # calls on this exact entity page are guaranteed to succeed.
        if (
            is_exact_title_search
            and current_doc_id
            and current_doc_id in self._evidence_archive
            and current_doc_id not in selected_ids
        ):
            if selected_ids:
                selected_ids.pop()
            selected_ids.append(current_doc_id)

        self._evidence_doc_ids = selected_ids
        self._evidence_doc_id_set = set(self._evidence_doc_ids)

        added_ids = [doc_id for doc_id in self._evidence_doc_ids if doc_id not in previous_set]
        evicted_ids = [doc_id for doc_id in previous_ids if doc_id not in self._evidence_doc_id_set]
        active_hits, observation_omitted = self._render_reranked_memory(
            current_doc_id=current_doc_id
        )

        for hit in active_hits:
            if hit["title"] not in self.visited_pages:
                self.visited_pages.append(hit["title"])

        return added_ids, evicted_ids, active_hits, observation_omitted

    def _search_with_reranked_memory(
        self, query, normalized_query, result, hits, raw_logged_hits, title_match_rank
    ):
        new_archive_titles, reranker_latency = self._score_new_archive_documents(query, hits)
        current_doc_id = str(self.current_document.get("doc_id", "")) if self.current_document else None
        is_exact_title_search = title_match_rank is not None
        added_ids, evicted_ids, active_hits, observation_omitted = self._refresh_reranked_memory(
            current_doc_id=current_doc_id,
            is_exact_title_search=is_exact_title_search,
        )

        # Update current_document and current_title to Cross-Encoder #1 page when no exact title match
        if active_hits and not is_exact_title_search:
            top_doc_id = active_hits[0]["doc_id"]
            top_doc_entry = self._evidence_archive[top_doc_id]
            self.current_document = top_doc_entry["document"]
            self.current_title = top_doc_entry["document"]["title"]
            current_doc_id = str(self.current_document.get("doc_id", ""))

        active_titles = [hit["title"] for hit in active_hits]
        active_id_set = set(self._evidence_doc_ids)
        raw_omitted_titles = [
            hit["title"] for hit in hits if str(hit["doc_id"]) not in active_id_set
        ]
        added_titles = [self._evidence_archive[x]["document"]["title"] for x in added_ids]
        evicted_titles = [self._evidence_archive[x]["document"]["title"] for x in evicted_ids]
        rank1_in_memory = current_doc_id in active_id_set if current_doc_id else False

        observation = (
            f"Observation: FullWiki retrieved {len(hits)} candidates for '{query}'. "
            f"Cross-encoder evidence memory now retains {self.evidence_document_count} of "
            f"{self.evidence_archive_count} unique documents. Current rank-1 page: "
            f"[{self.current_title}]. Consult the Active Evidence Memory block above."
        )
        if not rank1_in_memory:
            observation += " The rank-1 page was not retained by the evidence-memory reranker."
        observation = observation[: self.max_observation_chars]

        self.last_result = {
            "action": "search",
            "query": query,
            "status": "loaded",
            "retriever": result.get("mode"),
            "duplicate_query": False,
            "query_matches_retrieved_title": title_match_rank is not None,
            "query_title_match_rank": title_match_rank,
            "candidate_k": result.get("candidate_k"),
            "top_k": result.get("top_k"),
            # hits = active evidence actually rendered into the recurrent model context.
            "hits": active_hits,
            # retrieved_hits = complete raw top-k search result before memory reranking.
            "retrieved_hits": raw_logged_hits,
            "new_archive_documents": new_archive_titles,
            "new_evidence_documents": added_titles,
            "evicted_evidence_documents": evicted_titles,
            "already_retained_documents": [
                title for title in active_titles if title not in added_titles
            ],
            "omitted_due_to_evidence_cap": raw_omitted_titles,
            "omitted_due_to_observation_cap": observation_omitted,
            "evidence_document_count": self.evidence_document_count,
            "evidence_archive_count": self.evidence_archive_count,
            "max_evidence_documents": self.max_evidence_documents,
            "memory_policy": "cross_encoder_top_k",
            "memory_reranker": self.backend.evidence_reranker.describe(),
            "active_memory_documents": self.active_memory_snapshot(),
            "rank1_in_active_memory": rank1_in_memory,
            "latency_ms": {
                **result.get("latency_ms", {}),
                "memory_reranker": round(reranker_latency * 1000, 3),
            },
            "title": self.current_title,
            "sentences": (
                next(
                    (hit["sentences"] for hit in active_hits if hit["doc_id"] == current_doc_id),
                    [],
                )
            ),
        }
        self._seen_search_queries[normalized_query] = {
            "retriever": result.get("mode"),
            "candidate_k": result.get("candidate_k"),
            "retrieved_hits": raw_logged_hits,
            "retrieved_titles": [hit["title"] for hit in hits],
            "retained_titles": active_titles,
            "query_matches_retrieved_title": title_match_rank is not None,
            "query_title_match_rank": title_match_rank,
        }
        return observation

    def search(self, query):
        query = str(query).strip().strip("'\"")
        normalized_query = self._normalize_query(query)

        if self.duplicate_search_guard and normalized_query in self._seen_search_queries:
            previous = self._seen_search_queries[normalized_query]
            self.last_result = {
                "action": "search",
                "query": query,
                "status": "duplicate_query",
                "retriever": previous.get("retriever"),
                "duplicate_query": True,
                "query_matches_retrieved_title": previous.get("query_matches_retrieved_title", False),
                "query_title_match_rank": previous.get("query_title_match_rank"),
                "candidate_k": previous.get("candidate_k"),
                "top_k": self.search_top_k,
                "hits": [],
                "retrieved_hits": previous.get("retrieved_hits", []),
                "new_evidence_documents": [],
                "already_retained_documents": previous.get("retained_titles", []),
                "omitted_due_to_evidence_cap": [],
                "omitted_due_to_observation_cap": [],
                "evidence_document_count": self.evidence_document_count,
                "max_evidence_documents": self.max_evidence_documents,
                "latency_ms": {"total": 0.0},
                "title": self.current_title,
                "sentences": [],
            }
            return f"Observation: The search query '{query}' was already performed earlier in this session."

        result = self.backend.search(query, top_k=self.search_top_k)
        hits = result.get("hits", [])
        raw_logged_hits = [self._raw_logged_hit(hit) for hit in hits]
        retrieved_titles = [hit["title"] for hit in hits]
        title_match_rank = next(
            (hit["rank"] for hit in hits if self._normalize_title(hit["title"]) == normalized_query),
            None,
        )

        # Direct Title Index Injection:
        # If an exact title match was missed by RRF (title_match_rank is None),
        # query the backend directly by title. If found, inject it at position 0 of hits.
        if self.use_reranked_memory and title_match_rank is None and hasattr(self.backend, "get_doc_by_title"):
            direct_doc = self.backend.get_doc_by_title(query)
            if direct_doc and str(direct_doc["doc_id"]) not in {str(h["doc_id"]) for h in hits}:
                direct_hit = dict(direct_doc, rank=1, fused_score=1.0)
                reordered_hits = [direct_hit] + hits
                hits = [dict(hit, rank=rank) for rank, hit in enumerate(reordered_hits, 1)]
                title_match_rank = 1

        # Smart Hyperlink Graph Candidate Expansion:
        # Extract 1-hop outgoing links from all Active Memory documents and visited pages.
        # Filter out generic titles and rank by overlap with question/query before fetching top 20.
        if self.use_reranked_memory and hasattr(self.backend, "get_outgoing_links"):
            GENERIC_TITLES = {
                "united states", "english language", "actor", "film director", "film",
                "television series", "california", "new york city", "united kingdom",
                "japan", "france", "germany", "music", "album", "single (music)",
                "rock music", "pop music", "hip hop music", "country", "city", "state", "year"
            }
            graph_targets = set()
            active_titles = [self._evidence_archive[doc_id]["document"]["title"] for doc_id in self._evidence_doc_ids]
            for page_title in active_titles + self.visited_pages:
                links = self.backend.get_outgoing_links(page_title)
                for target in links:
                    norm_target = str(target).strip().lower()
                    if norm_target not in GENERIC_TITLES and norm_target not in {t.lower() for t in self.visited_pages}:
                        graph_targets.add(target)

            # Rank candidate graph targets by token overlap with question and active query
            q_tokens = set(self._normalize_query(self.question).split() + normalized_query.split())

            def _graph_target_score(target):
                t_tokens = set(self._normalize_title(target).split())
                overlap = len(q_tokens & t_tokens)
                return overlap

            sorted_graph_targets = sorted(graph_targets, key=_graph_target_score, reverse=True)

            existing_hit_ids = {str(h["doc_id"]) for h in hits}
            injected_graph_count = 0
            for target_title in sorted_graph_targets[:30]:
                if injected_graph_count >= 20:
                    break
                graph_doc = self.backend.get_doc_by_title(target_title)
                if graph_doc and str(graph_doc["doc_id"]) not in existing_hit_ids:
                    graph_hit = dict(graph_doc, rank=len(hits) + 1, fused_score=0.5)
                    hits.append(graph_hit)
                    existing_hit_ids.add(str(graph_doc["doc_id"]))
                    injected_graph_count += 1

        # ReAct entity searches often name the exact Wikipedia page discovered on a
        # previous hop. When that exact title is already inside the retrieved top-k,
        # promote it to the session's rank-1/current page so lookup operates on the
        # intended entity. Keep raw_logged_hits above untouched for retrieval audits.
        if self.use_reranked_memory and title_match_rank not in {None, 1}:
            title_match_index = next(
                index
                for index, hit in enumerate(hits)
                if self._normalize_title(hit["title"]) == normalized_query
            )
            reordered_hits = (
                [hits[title_match_index]]
                + hits[:title_match_index]
                + hits[title_match_index + 1:]
            )
            hits = [dict(hit, rank=rank) for rank, hit in enumerate(reordered_hits, 1)]

        if not hits:
            self.last_result = {
                "action": "search",
                "query": query,
                "status": "not_found",
                "retriever": result.get("mode"),
                "duplicate_query": False,
                "query_matches_retrieved_title": False,
                "query_title_match_rank": None,
                "candidate_k": result.get("candidate_k"),
                "top_k": result.get("top_k"),
                "hits": [],
                "retrieved_hits": [],
                "new_evidence_documents": [],
                "already_retained_documents": [],
                "omitted_due_to_evidence_cap": [],
                "omitted_due_to_observation_cap": [],
                "evidence_document_count": self.evidence_document_count,
                "max_evidence_documents": self.max_evidence_documents,
                "latency_ms": result.get("latency_ms", {}),
                "title": None,
                "sentences": [],
            }
            self._seen_search_queries[normalized_query] = dict(self.last_result, retrieved_titles=[])
            return f"Observation: FullWiki retrieval found no document for '{query}'."

        # Rank 1 is the single classic ReAct current page used by lookup.
        # A successful search resets lookup iteration state, just like the
        # original WikiEnv.
        self.current_document = hits[0]
        self.current_title = hits[0]["title"]
        self._lookup_keyword = None
        self._lookup_matches = []
        self._lookup_index = 0

        if self.use_reranked_memory:
            return self._search_with_reranked_memory(
                query, normalized_query, result, hits, raw_logged_hits, title_match_rank
            )

        candidate_new_hits = []
        already_retained = []
        omitted = []
        remaining_slots = (
            None
            if self.max_evidence_documents is None
            else max(0, self.max_evidence_documents - self.evidence_document_count)
        )
        for hit in hits:
            doc_id = str(hit["doc_id"])
            if doc_id in self._evidence_doc_id_set:
                already_retained.append(hit["title"])
                continue
            if remaining_slots is not None and len(candidate_new_hits) >= remaining_slots:
                omitted.append(hit["title"])
                continue
            candidate_new_hits.append(hit)

        prefix = (
            "Observation: FullWiki retrieval returned ranked HotpotQA-aligned Wikipedia "
            "introductory paragraphs. Sentence IDs are exact 0-based HotpotQA sentence IDs.\n"
        )
        rendered, logged_hits, remaining_chars = self._render_new_hits(candidate_new_hits, prefix)
        exposed_ids = {str(hit["doc_id"]) for hit in logged_hits}
        exposed_titles = []
        for hit in candidate_new_hits:
            doc_id = str(hit["doc_id"])
            if doc_id not in exposed_ids:
                continue
            self._evidence_doc_ids.append(doc_id)
            self._evidence_doc_id_set.add(doc_id)
            exposed_titles.append(hit["title"])
            if hit["title"] not in self.visited_pages:
                self.visited_pages.append(hit["title"])

        observation_omitted = [
            hit["title"] for hit in candidate_new_hits if str(hit["doc_id"]) not in exposed_ids
        ]

        if rendered:
            observation = f"{prefix.rstrip()}\n{rendered}"
        else:
            observation = f"Observation: FullWiki retrieval found no new evidence for '{query}'."

        self.last_result = {
            "action": "search",
            "query": query,
            "status": "loaded",
            "retriever": result.get("mode"),
            "duplicate_query": False,
            "query_matches_retrieved_title": title_match_rank is not None,
            "query_title_match_rank": title_match_rank,
            "candidate_k": result.get("candidate_k"),
            "top_k": result.get("top_k"),
            # hits = evidence actually shown to the LLM on this turn.
            "hits": logged_hits,
            # retrieved_hits = complete raw top-k result, preserved for auditing.
            "retrieved_hits": raw_logged_hits,
            "new_evidence_documents": exposed_titles,
            "already_retained_documents": already_retained,
            "omitted_due_to_evidence_cap": omitted,
            "omitted_due_to_observation_cap": observation_omitted,
            "evidence_document_count": self.evidence_document_count,
            "max_evidence_documents": self.max_evidence_documents,
            "latency_ms": result.get("latency_ms", {}),
            "title": self.current_title,
            "sentences": (
                logged_hits[0]["sentences"]
                if logged_hits and logged_hits[0]["title"] == self.current_title
                else []
            ),
        }
        self._seen_search_queries[normalized_query] = {
            "retriever": result.get("mode"),
            "candidate_k": result.get("candidate_k"),
            "retrieved_hits": raw_logged_hits,
            "retrieved_titles": retrieved_titles,
            "retained_titles": list(dict.fromkeys(already_retained + exposed_titles)),
            "query_matches_retrieved_title": title_match_rank is not None,
            "query_title_match_rank": title_match_rank,
        }
        return observation

    def lookup(self, keyword):
        """Classic ReAct lookup over the single current rank-1 page.

        Search may expose multiple ranked passages, but only rank 1 becomes the
        current page. Repeating lookup with the same keyword advances through
        matching sentences on that page, mirroring Yao et al.'s WikiEnv. A
        lookup can never introduce a document that was not already admitted to
        the bounded working-evidence set.
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

        current_doc_id = str(self.current_document.get("doc_id", ""))
        if current_doc_id not in self._evidence_doc_id_set:
            self.last_result = {
                "action": "lookup",
                "query": keyword_raw,
                "status": "current_page_not_exposed",
                "title": self.current_title,
                "sentences": [],
                "hits": [],
                "evidence_document_count": self.evidence_document_count,
                "max_evidence_documents": self.max_evidence_documents,
            }
            return (
                f"Observation: The current rank-1 page [{self.current_title}] was not admitted "
                "to the bounded working evidence, so lookup cannot expose it."
            )

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
