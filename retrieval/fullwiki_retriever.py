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
        self.dense_nprobe = int(dense_manifest.get("nprobe", 16))
        self.faiss = faiss
        self.dense_index = faiss.read_index(self.dense_index_path)
        if hasattr(self.dense_index, "nprobe"):
            self.dense_index.nprobe = self.dense_nprobe
        self.encoder = SentenceTransformer(self.dense_model_name, device=self.dense_query_device)

    def _parse_lucene_doc(self, doc_id):
        doc = self.lucene.doc(str(doc_id))
        if doc is None:
            return None
        raw = doc.raw()
        if raw is None:
            return None
        data = json.loads(raw)
        data["id"] = str(data.get("id", doc_id))
        data["doc_id"] = data["id"]
        data["sentences"] = [str(sentence) for sentence in data.get("sentences", [])]
        return data

    def _bm25_search(self, query, k):
        started = time.perf_counter()
        hits = self.lucene.search(query, k=k)
        rows = [{"doc_id": str(hit.docid), "score": float(hit.score)} for hit in hits]
        return rows, time.perf_counter() - started

    def _dense_search(self, query, k):
        started = time.perf_counter()
        with self._dense_lock:
            encoded = self.encoder.encode(
                [BGE_QUERY_INSTRUCTION + query],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype("float32")
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

    def create_session(
        self,
        search_top_k=1,
        max_observation_chars=2200,
        max_evidence_documents=None,
        duplicate_search_guard=False,
    ):
        return FullWikiRetriever(
            self,
            search_top_k=search_top_k,
            max_observation_chars=max_observation_chars,
            max_evidence_documents=max_evidence_documents,
            duplicate_search_guard=duplicate_search_guard,
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
            "index_manifest": self.manifest,
        }


class FullWikiRetriever:
    """Per-question ReAct-compatible session over a shared FullWiki backend.

    The backend always retrieves ``search_top_k`` results. For iterative ReAct
    runs, ``max_evidence_documents`` bounds how many unique Wikipedia documents
    are ever exposed in the recurrent working context. Raw top-k results remain
    in ``last_result['retrieved_hits']`` for later analysis.
    """

    def __init__(
        self,
        backend,
        search_top_k=1,
        max_observation_chars=2200,
        max_evidence_documents=None,
        duplicate_search_guard=False,
    ):
        self.backend = backend
        self.search_top_k = int(search_top_k)
        self.max_observation_chars = int(max_observation_chars)
        self.max_evidence_documents = (
            int(max_evidence_documents) if max_evidence_documents is not None else None
        )
        self.duplicate_search_guard = bool(duplicate_search_guard)
        self.current_title = None
        self.current_document = None
        self.last_result = None
        self.visited_pages = []
        self._evidence_doc_ids = []
        self._evidence_doc_id_set = set()
        self._seen_search_queries = {}

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
        self._seen_search_queries = {}

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

    def _visible_sentences(self, document, char_budget):
        visible = []
        used = 0
        for sent_id, text in enumerate(document.get("sentences", [])):
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

        # Rank 1 remains the classic ReAct "current page" used by lookup.
        self.current_document = hits[0]
        self.current_title = hits[0]["title"]

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
        keyword_clean = str(keyword).strip().strip("'\"").lower()
        if not keyword_clean:
            return "Observation: Lookup keyword cannot be empty."

        active_hits = []
        if self.last_result and isinstance(self.last_result.get("hits"), list) and self.last_result["hits"]:
            active_hits = self.last_result["hits"]
        elif self.current_document:
            active_hits = [
                {
                    "doc_id": self.current_document["doc_id"],
                    "title": self.current_document["title"],
                    "sentences": [
                        {"sent_id": s_id, "text": text}
                        for s_id, text in enumerate(self.current_document.get("sentences", []))
                    ],
                }
            ]

        if not active_hits:
            self.last_result = {
                "action": "lookup",
                "query": keyword,
                "status": "no_current_page",
                "title": None,
                "sentences": [],
                "hits": [],
            }
            return "Observation: No FullWiki document currently loaded. Perform a `search` first."

        matched_hits = []
        rendered_blocks = []
        all_matches = []

        for hit in active_hits:
            doc_matches = []
            for item in hit.get("sentences", []):
                text = item.get("text", "")
                if keyword_clean in str(text).lower():
                    doc_matches.append({"sent_id": item["sent_id"], "text": text})
                    all_matches.append({"title": hit["title"], "sent_id": item["sent_id"], "text": text})
                    if len(all_matches) >= 5:
                        break
            if doc_matches:
                matched_hits.append({
                    "doc_id": str(hit.get("doc_id", "")),
                    "title": hit["title"],
                    "sentences": doc_matches,
                })
                rendered_blocks.append(
                    f"Found matches in [{hit['title']}].\n"
                    + self._render_sentences(hit["title"], doc_matches)
                )
            if len(all_matches) >= 5:
                break

        self.last_result = {
            "action": "lookup",
            "query": keyword,
            "status": "found" if matched_hits else "not_found",
            "title": self.current_title,
            "sentences": all_matches,
            "hits": matched_hits,
        }

        if not matched_hits:
            doc_names = ", ".join(f"[{h['title']}]" for h in active_hits[:3])
            return f"Observation: Could not find '{keyword}' in active documents ({doc_names})."

        return "Observation:\n" + "\n\n".join(rendered_blocks)
