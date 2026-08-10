import bz2
import io
import json
import tarfile
from concurrent.futures import ThreadPoolExecutor

import pytest

from retrieval.corpus import iter_hotpot_intro_records, normalize_intro_record
from retrieval.fullwiki_retriever import (
    FullWikiRetriever,
    FullWikiSearchBackend,
    reciprocal_rank_fusion,
)


def _hit(doc_id, title, sentences, rank=1):
    return {
        "doc_id": str(doc_id),
        "title": title,
        "rank": int(rank),
        "bm25_rank": int(rank),
        "bm25_score": float(20 - rank),
        "dense_rank": int(rank),
        "dense_score": 1.0 / rank,
        "fused_score": 1.0 / (60 + rank),
        "sentences": list(sentences),
    }


def test_normalize_intro_record_preserves_sentence_ids_and_title():
    record = {
        "id": 12,
        "title": "Anarchism",
        "url": "https://en.wikipedia.org/wiki?curid=12",
        "text": [
            "Anarchism is a political philosophy.",
            " It advocates self-governed societies.",
        ],
    }
    normalized = normalize_intro_record(record)
    assert normalized["id"] == "12"
    assert normalized["title"] == "Anarchism"
    assert normalized["sentences"][1] == " It advocates self-governed societies."
    assert normalized["contents"].startswith("Anarchism\n")

    with_blank = normalize_intro_record({
        "id": 13,
        "title": "Sentence IDs",
        "text": ["Sentence zero.", "", "Sentence two."],
    })
    assert with_blank["sentences"] == ["Sentence zero.", "", "Sentence two."]


def test_archive_reader_handles_bzip2_shards_inside_tar_bz2(tmp_path):
    source_rows = [
        {"id": 1, "title": "A", "text": ["A sentence."]},
        {"id": 2, "title": "B", "text": ["B sentence."]},
    ]
    payload = "".join(json.dumps(row) + "\n" for row in source_rows).encode("utf-8")
    compressed_payload = bz2.compress(payload)
    archive_path = tmp_path / "wiki.tar.bz2"

    with tarfile.open(archive_path, "w:bz2") as tar:
        info = tarfile.TarInfo("AA/wiki_00.bz2")
        info.size = len(compressed_payload)
        tar.addfile(info, io.BytesIO(compressed_payload))

    rows = list(iter_hotpot_intro_records(archive_path))
    assert [row["title"] for row in rows] == ["A", "B"]


def test_reciprocal_rank_fusion_combines_sparse_and_dense_rankings():
    sparse = [
        {"doc_id": "1", "score": 10.0},
        {"doc_id": "2", "score": 9.0},
    ]
    dense = [
        {"doc_id": "2", "score": 0.9},
        {"doc_id": "3", "score": 0.8},
    ]
    fused = reciprocal_rank_fusion(sparse, dense, rrf_k=60)
    assert fused[0]["doc_id"] == "2"
    assert fused[0]["bm25_rank"] == 2
    assert fused[0]["dense_rank"] == 1
    assert fused[1]["doc_id"] == "1"


class RecordingReranker:
    def __init__(self):
        self.max_length = 512
        self.calls = []

    def score_pairs(self, pairs):
        pairs = list(pairs)
        self.calls.append(pairs)
        return [float(index + 1) for index in range(len(pairs))], 0.001

    def describe(self):
        return {"model": "recording-reranker", "device": "cpu"}


def test_backend_page_stage_uses_one_full_intro_pair_and_sentence_stage_scores_all_nonempty_ids():
    backend = FullWikiSearchBackend.__new__(FullWikiSearchBackend)
    backend.local_reranker = RecordingReranker()

    documents = [
        _hit("1", "First", ["Opening.", "", "Middle.", "Final evidence."], rank=1),
        _hit("2", "Second", ["Only sentence."], rank=2),
    ]
    page_scores, _, _ = backend.score_page_documents("Question context", documents)

    assert len(page_scores) == 2
    page_pairs = backend.local_reranker.calls[0]
    assert len(page_pairs) == 2
    assert "Opening. Middle. Final evidence." in page_pairs[0][1]
    assert page_pairs[0][1].startswith("First\n")

    scores_by_doc, _, pair_count = backend.score_document_sentences(
        "Question context", [documents[0]]
    )
    sentence_pairs = backend.local_reranker.calls[1]
    assert pair_count == 3
    assert len(sentence_pairs) == 3
    assert set(scores_by_doc["1"]) == {0, 2, 3}
    assert sentence_pairs[-1][1] == "First: Final evidence."


def test_shared_backend_hybrid_search_hydrates_only_requested_top_k():
    backend = FullWikiSearchBackend.__new__(FullWikiSearchBackend)
    backend.mode = "hybrid"
    backend.candidate_k = 3
    backend.rrf_k = 60
    backend._hybrid_executor = ThreadPoolExecutor(max_workers=1)
    backend._bm25_search = lambda query, k: (
        [{"doc_id": "1", "score": 10.0}, {"doc_id": "2", "score": 9.0}],
        0.001,
    )
    backend._dense_search = lambda query, k: (
        [{"doc_id": "2", "score": 0.9}, {"doc_id": "3", "score": 0.8}],
        0.002,
    )
    docs = {
        "1": {"id": "1", "title": "A", "sentences": ["a"]},
        "2": {"id": "2", "title": "B", "sentences": ["b"]},
        "3": {"id": "3", "title": "C", "sentences": ["c"]},
    }
    backend._parse_lucene_doc = lambda doc_id: dict(docs[doc_id])

    try:
        result = backend.search("query", top_k=2)
    finally:
        backend._hybrid_executor.shutdown(wait=True)

    assert [hit["doc_id"] for hit in result["hits"]] == ["2", "1"]
    assert result["hits"][0]["bm25_rank"] == 2
    assert result["hits"][0]["dense_rank"] == 1
    assert result["candidate_k"] == 3
    assert result["top_k"] == 2


class LocalRerankBackend:
    def __init__(self, searches, page_scores, sentence_scores=None, exact_docs=None):
        self.searches = searches
        self.page_scores = page_scores
        self.sentence_scores = sentence_scores or {}
        self.exact_docs = {str(k).lower(): dict(v) for k, v in (exact_docs or {}).items()}
        self.calls = []
        self.page_stage_calls = []
        self.sentence_stage_calls = []
        self.local_reranker = RecordingReranker()

    def search(self, query, top_k=1):
        self.calls.append(query)
        hits = [dict(hit) for hit in self.searches[query]][:top_k]
        return {
            "query": query,
            "mode": "hybrid",
            "candidate_k": 50,
            "top_k": top_k,
            "hits": hits,
            "latency_ms": {"bm25": 1.0, "dense": 2.0, "fusion": 0.1, "total": 2.1},
        }

    def get_doc_by_title(self, title):
        doc = self.exact_docs.get(str(title).strip().lower())
        return dict(doc) if doc else None

    def score_page_documents(self, query_context, documents):
        documents = list(documents)
        self.page_stage_calls.append({
            "query_context": query_context,
            "doc_ids": [str(doc["doc_id"]) for doc in documents],
        })
        scores = [float(self.page_scores[(query_context, str(doc["doc_id"]))]) for doc in documents]
        return scores, 0.001, 0

    def score_document_sentences(self, query_context, documents):
        documents = list(documents)
        call = {
            "query_context": query_context,
            "doc_ids": [str(doc["doc_id"]) for doc in documents],
            "sent_ids": {},
        }
        output = {}
        pair_count = 0
        for document in documents:
            doc_id = str(document["doc_id"])
            output[doc_id] = {}
            call["sent_ids"][doc_id] = []
            for sent_id, text in enumerate(document.get("sentences", [])):
                if not str(text).strip():
                    continue
                score = self.sentence_scores.get(
                    (query_context, doc_id, sent_id),
                    1000.0 - (100.0 * int(document.get("rank", 1))) - sent_id,
                )
                output[doc_id][sent_id] = float(score)
                call["sent_ids"][doc_id].append(sent_id)
                pair_count += 1
        self.sentence_stage_calls.append(call)
        return output, 0.002, pair_count


def _query_context(question, search):
    if " ".join(question.lower().split()) == " ".join(search.lower().split()):
        return question
    return f"Question: {question}\nCurrent search: {search}"


def test_local_page_reranking_is_query_local_and_old_memory_cannot_hijack_current_page():
    question = "Who connects the two facts?"
    first_hits = [
        _hit("1", "Old Winner", ["Old evidence."], 1),
        _hit("2", "Other", ["Other evidence."], 2),
    ]
    second_hits = [
        _hit("1", "Old Winner", ["Old evidence."], 1),
        _hit("3", "New Winner", ["New evidence."], 2),
    ]
    q1 = _query_context(question, "first bridge")
    q2 = _query_context(question, "second bridge")
    page_scores = {
        (q1, "1"): 100.0,
        (q1, "2"): 1.0,
        (q2, "1"): -5.0,
        (q2, "3"): 8.0,
    }
    backend = LocalRerankBackend(
        {"first bridge": first_hits, "second bridge": second_hits}, page_scores
    )
    retriever = FullWikiRetriever(
        backend,
        search_top_k=2,
        local_rerank_page_count=2,
        question=question,
        max_evidence_snippets=12,
        max_evidence_chars=6000,
    )

    retriever.search("first bridge")
    assert retriever.current_page_title == "Old Winner"

    retriever.search("second bridge")
    assert retriever.current_page_title == "New Winner"
    assert retriever.last_result["current_page_selection_reason"] == "local_rerank"
    assert retriever.last_result["current_page_local_rank"] == 1
    assert retriever.last_result["local_reranked_hits"][0]["title"] == "New Winner"
    assert len(retriever._evidence_archive["1"]["search_history"]) == 2
    assert "reranker_score" not in retriever._evidence_archive["1"]


def test_fifteen_hydrated_pages_produce_fifteen_page_pairs_and_only_four_sentence_pages():
    question = "Question"
    search = "broad search"
    context = _query_context(question, search)
    hits = [
        _hit(str(i), f"Doc {i}", [f"Doc {i} sentence 0.", f"Doc {i} sentence 1."], i)
        for i in range(1, 16)
    ]
    # Reverse retrieval order: doc 15 should become local rank 1.
    page_scores = {(context, str(i)): float(i) for i in range(1, 16)}
    backend = LocalRerankBackend({search: hits}, page_scores)
    retriever = FullWikiRetriever(
        backend,
        search_top_k=15,
        local_rerank_page_count=4,
        question=question,
    )

    retriever.search(search)

    assert backend.page_stage_calls[0]["doc_ids"] == [str(i) for i in range(1, 16)]
    assert retriever.last_result["local_page_pair_count"] == 15
    assert backend.sentence_stage_calls[0]["doc_ids"] == ["15", "14", "13", "12"]
    assert len(retriever.last_result["sentence_rerank_pages"]) == 4
    assert retriever.current_page_title == "Doc 15"


def test_sentence_stage_scores_last_sentence_and_preserves_blank_sentence_ids():
    question = "Question"
    search = "find evidence"
    context = _query_context(question, search)
    hits = [
        _hit("1", "Page One", ["First.", "", "Third.", "", "Evidence is last."], 1),
        _hit("2", "Page Two", ["A."], 2),
        _hit("3", "Page Three", ["B."], 3),
        _hit("4", "Page Four", ["C."], 4),
        _hit("5", "Page Five", ["Should not be sentence-scored."], 5),
    ]
    page_scores = {(context, str(i)): float(10 - i) for i in range(1, 6)}
    backend = LocalRerankBackend({search: hits}, page_scores)
    retriever = FullWikiRetriever(
        backend,
        search_top_k=5,
        local_rerank_page_count=4,
        question=question,
    )

    retriever.search(search)
    call = backend.sentence_stage_calls[0]

    assert call["doc_ids"] == ["1", "2", "3", "4"]
    assert call["sent_ids"]["1"] == [0, 2, 4]
    assert 4 in retriever._evidence_archive["1"]["latest_sentence_scores"]
    assert "5" not in call["sent_ids"]


def test_sentence_scores_never_reorder_page_ranking():
    question = "Question"
    search = "search"
    context = _query_context(question, search)
    hits = [
        _hit("1", "Page One", ["Moderate sentence."], 1),
        _hit("2", "Page Two", ["Spectacular sentence."], 2),
        _hit("3", "Page Three", ["Other."], 3),
        _hit("4", "Page Four", ["Other."], 4),
    ]
    page_scores = {
        (context, "1"): 9.0,
        (context, "2"): 8.0,
        (context, "3"): 7.0,
        (context, "4"): 6.0,
    }
    sentence_scores = {
        (context, "1", 0): -100.0,
        (context, "2", 0): 10000.0,
    }
    backend = LocalRerankBackend({search: hits}, page_scores, sentence_scores=sentence_scores)
    retriever = FullWikiRetriever(
        backend,
        search_top_k=4,
        local_rerank_page_count=4,
        question=question,
    )

    retriever.search(search)

    assert retriever.current_page_title == "Page One"
    assert [row["title"] for row in retriever.last_result["local_reranked_hits"]] == [
        "Page One", "Page Two", "Page Three", "Page Four"
    ]


def test_exact_title_page_is_current_and_is_included_in_four_sentence_pages_without_reordering_pages():
    question = "Question"
    search = "Exact Page"
    context = _query_context(question, search)
    hits = [
        _hit("1", "High One", ["A."], 1),
        _hit("2", "High Two", ["B."], 2),
        _hit("3", "High Three", ["C."], 3),
        _hit("4", "High Four", ["D."], 4),
        _hit("5", "Exact Page", ["Exact evidence."], 5),
    ]
    page_scores = {
        (context, "1"): 10.0,
        (context, "2"): 9.0,
        (context, "3"): 8.0,
        (context, "4"): 7.0,
        (context, "5"): 1.0,
    }
    backend = LocalRerankBackend({search: hits}, page_scores)
    retriever = FullWikiRetriever(
        backend,
        search_top_k=5,
        local_rerank_page_count=4,
        question=question,
    )

    retriever.search(search)

    assert retriever.current_page_title == "Exact Page"
    assert retriever.last_result["current_page_selection_reason"] == "exact_title"
    assert retriever.last_result["current_page_local_rank"] == 5
    assert [row["title"] for row in retriever.last_result["local_reranked_hits"]][:4] == [
        "High One", "High Two", "High Three", "High Four"
    ]
    assert backend.sentence_stage_calls[0]["doc_ids"] == ["5", "1", "2", "3"]


def test_exact_title_injection_replaces_tail_and_keeps_page_pair_budget_fixed():
    question = "Question"
    search = "Exact Page"
    context = _query_context(question, search)
    hits = [_hit(str(i), f"Doc {i}", [f"Sentence {i}."], i) for i in range(1, 16)]
    exact = _hit("99", "Exact Page", ["Exact sentence."], 99)
    page_scores = {(context, str(i)): float(20 - i) for i in range(1, 16)}
    page_scores[(context, "99")] = -100.0
    backend = LocalRerankBackend(
        {search: hits}, page_scores, exact_docs={"Exact Page": exact}
    )
    retriever = FullWikiRetriever(
        backend,
        search_top_k=15,
        local_rerank_page_count=4,
        question=question,
    )

    retriever.search(search)

    assert retriever.last_result["exact_title_injected"] is True
    assert retriever.last_result["local_page_pair_count"] == 15
    assert "15" not in backend.page_stage_calls[0]["doc_ids"]
    assert "99" in backend.page_stage_calls[0]["doc_ids"]
    assert retriever.current_page_title == "Exact Page"


def test_snippet_memory_is_bounded_without_per_document_quota_and_without_cross_query_raw_score_ordering():
    question = "Question"
    search = "search"
    context = _query_context(question, search)
    hits = [
        _hit("1", "Dense Evidence", [f"Evidence {i}." for i in range(10)], 1),
        _hit("2", "Second", ["Second evidence."], 2),
        _hit("3", "Third", ["Third evidence."], 3),
        _hit("4", "Fourth", ["Fourth evidence."], 4),
    ]
    page_scores = {(context, str(i)): float(5 - i) for i in range(1, 5)}
    sentence_scores = {}
    for i in range(10):
        sentence_scores[(context, "1", i)] = 1000.0 - i
    sentence_scores[(context, "2", 0)] = 5.0
    sentence_scores[(context, "3", 0)] = 4.0
    sentence_scores[(context, "4", 0)] = 3.0

    backend = LocalRerankBackend(
        {search: hits}, page_scores, sentence_scores=sentence_scores
    )
    retriever = FullWikiRetriever(
        backend,
        search_top_k=4,
        local_rerank_page_count=4,
        question=question,
        max_evidence_snippets=6,
        max_evidence_chars=300,
    )

    retriever.search(search)
    memory = retriever.active_memory_snapshot()

    assert len(memory) <= 6
    assert len(retriever.render_active_evidence()) <= 300
    assert sum(row["doc_id"] == "1" for row in memory) > 2
    assert all("score" not in row for row in memory)
    assert all("last_local_score" not in row for row in memory)


def test_lookup_uses_current_page_even_if_not_in_active_memory_and_promotes_exact_sentence():
    question = "Question"
    search = "Exact Page"
    context = _query_context(question, search)
    hits = [
        _hit("1", "High One", ["High evidence."], 1),
        _hit("2", "High Two", ["Other evidence."], 2),
        _hit("3", "High Three", ["Other evidence."], 3),
        _hit("4", "High Four", ["Other evidence."], 4),
        _hit("5", "Exact Page", ["Target is here.", "Target appears again."], 5),
    ]
    page_scores = {
        (context, "1"): 10.0,
        (context, "2"): 9.0,
        (context, "3"): 8.0,
        (context, "4"): 7.0,
        (context, "5"): 1.0,
    }
    sentence_scores = {
        (context, "1", 0): 1000.0,
        (context, "2", 0): 900.0,
        (context, "3", 0): 800.0,
        (context, "5", 0): -100.0,
        (context, "5", 1): -101.0,
    }
    backend = LocalRerankBackend({search: hits}, page_scores, sentence_scores=sentence_scores)
    retriever = FullWikiRetriever(
        backend,
        search_top_k=5,
        local_rerank_page_count=4,
        question=question,
        max_evidence_snippets=1,
        max_evidence_chars=6000,
    )

    retriever.search(search)
    assert retriever.current_page_title == "Exact Page"
    assert retriever.active_memory_snapshot()[0]["doc_id"] == "1"
    assert "5" not in retriever._evidence_doc_id_set

    first = retriever.lookup("Target")
    assert "[Exact Page | sent 0]" in first
    assert retriever.current_page_title == "Exact Page"
    assert retriever.active_memory_snapshot()[0]["doc_id"] == "5"
    assert retriever.active_memory_snapshot()[0]["source"] == "lookup"

    second = retriever.lookup("Target")
    assert "[Exact Page | sent 1]" in second
    assert retriever.last_result["status"] == "found"


def test_duplicate_search_is_guarded_without_second_backend_or_reranker_call():
    question = "Question"
    search = "Radiohead lead singer"
    context = _query_context(question, search)
    hits = [_hit("1", "Radiohead", ["Thom Yorke is the lead singer."], 1)]
    backend = LocalRerankBackend({search: hits}, {(context, "1"): 1.0})
    retriever = FullWikiRetriever(
        backend,
        search_top_k=1,
        local_rerank_page_count=1,
        question=question,
        duplicate_search_guard=True,
    )

    retriever.search(search)
    duplicate_observation = retriever.search("  radiohead   lead singer  ")

    assert len(backend.calls) == 1
    assert len(backend.page_stage_calls) == 1
    assert retriever.last_result["duplicate_query"] is True
    assert retriever.last_result["status"] == "duplicate_query"
    assert "already performed earlier" in duplicate_observation


def test_fallback_without_local_reranker_keeps_classic_current_page_lookup_behavior():
    class FakeBackend:
        def search(self, query, top_k=1):
            hits = [
                _hit(
                    "10",
                    "Scott Derrickson",
                    [
                        "Scott Derrickson is an American filmmaker.",
                        "He was born in Denver, Colorado.",
                    ],
                    1,
                ),
                _hit("11", "Ed Wood", ["Edward D. Wood Jr. was an American filmmaker."], 2),
            ]
            return {
                "query": query,
                "mode": "hybrid",
                "candidate_k": 50,
                "top_k": top_k,
                "hits": hits[:top_k],
                "latency_ms": {"total": 1.0},
            }

    retriever = FullWikiRetriever(FakeBackend(), search_top_k=2, max_observation_chars=4000)
    observation = retriever.search("Scott Derrickson Ed Wood")

    assert retriever.current_page_title == "Scott Derrickson"
    assert "[Scott Derrickson | sent 0]" in observation
    lookup = retriever.lookup("Denver")
    assert "[Scott Derrickson | sent 1]" in lookup
