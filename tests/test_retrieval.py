import bz2
import io
import json
import tarfile

try:
    import pytest
except ImportError:
    class FakePytest:
        @staticmethod
        def approx(val, abs=1e-6):
            return val
    pytest = FakePytest()

from retrieval.corpus import iter_hotpot_intro_records, normalize_intro_record
from retrieval.fullwiki_retriever import FullWikiRetriever, reciprocal_rank_fusion


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


class FakeBackend:
    def search(self, query, top_k=1):
        hits = [
            {
                "doc_id": "10",
                "title": "Scott Derrickson",
                "rank": 1,
                "bm25_rank": 1,
                "bm25_score": 12.0,
                "dense_rank": 2,
                "dense_score": 0.8,
                "fused_score": 0.03,
                "sentences": [
                    "Scott Derrickson is an American filmmaker.",
                    "He was born in Denver, Colorado.",
                ],
            },
            {
                "doc_id": "11",
                "title": "Ed Wood",
                "rank": 2,
                "bm25_rank": 3,
                "bm25_score": 8.0,
                "dense_rank": 1,
                "dense_score": 0.9,
                "fused_score": 0.029,
                "sentences": ["Edward D. Wood Jr. was an American filmmaker."],
            },
        ]
        return {
            "query": query,
            "mode": "hybrid",
            "candidate_k": 20,
            "top_k": top_k,
            "hits": hits[:top_k],
            "latency_ms": {"bm25": 2.0, "dense": 4.0, "fusion": 0.1, "total": 6.1},
        }


def test_fullwiki_session_returns_multiple_ranked_documents_with_exact_sentence_ids():
    retriever = FullWikiRetriever(FakeBackend(), search_top_k=2, max_observation_chars=4000)
    observation = retriever.search("Scott Derrickson Ed Wood")

    assert "[Scott Derrickson | sent 0]" in observation
    assert "[Scott Derrickson | sent 1]" in observation
    assert "[Ed Wood | sent 0]" in observation
    assert retriever.visited_pages == ["Scott Derrickson", "Ed Wood"]
    assert len(retriever.last_result["hits"]) == 2
    assert retriever.last_result["hits"][1]["dense_rank"] == 1
    assert retriever.current_page_title == "Scott Derrickson"

    lookup = retriever.lookup("Denver")
    assert "[Scott Derrickson | sent 1]" in lookup
    assert retriever.last_result["sentences"] == [
        {"sent_id": 1, "text": "He was born in Denver, Colorado."}
    ]


def test_shared_backend_hybrid_search_hydrates_only_final_top_k():
    from retrieval.fullwiki_retriever import FullWikiSearchBackend

    backend = FullWikiSearchBackend.__new__(FullWikiSearchBackend)
    backend.mode = "hybrid"
    backend.candidate_k = 3
    backend.rrf_k = 60
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

    result = backend.search("query", top_k=2)
    assert [hit["doc_id"] for hit in result["hits"]] == ["2", "1"]
    assert result["hits"][0]["bm25_rank"] == 2
    assert result["hits"][0]["dense_rank"] == 1
    assert result["candidate_k"] == 3
    assert result["top_k"] == 2


class RollingFakeBackend:
    def __init__(self):
        self.calls = []

    def search(self, query, top_k=1):
        self.calls.append(query)
        base = len(self.calls) * 100
        hits = []
        for rank in range(1, top_k + 1):
            doc_id = str(base + rank)
            hits.append({
                "doc_id": doc_id,
                "title": f"Doc {doc_id}",
                "rank": rank,
                "bm25_rank": rank,
                "bm25_score": float(20 - rank),
                "dense_rank": rank,
                "dense_score": 1.0 / rank,
                "fused_score": 1.0 / (60 + rank),
                "sentences": [f"Sentence for {doc_id}."],
            })
        return {
            "query": query,
            "mode": "hybrid",
            "candidate_k": 20,
            "top_k": top_k,
            "hits": hits,
            "latency_ms": {"bm25": 1.0, "dense": 2.0, "fusion": 0.1, "total": 3.1},
        }


def test_react_session_caps_unique_working_evidence_but_logs_full_top_k():
    backend = RollingFakeBackend()
    retriever = FullWikiRetriever(
        backend,
        search_top_k=6,
        max_observation_chars=7200,
        max_evidence_documents=15,
        duplicate_search_guard=True,
    )

    retriever.search("first")
    retriever.search("second")
    third_observation = retriever.search("third")

    assert retriever.evidence_document_count == 15
    assert len(retriever.visited_pages) == 15
    assert len(retriever.last_result["retrieved_hits"]) == 6
    assert len(retriever.last_result["hits"]) == 3
    assert len(retriever.last_result["omitted_due_to_evidence_cap"]) == 3
    assert "found no new evidence" in third_observation


def test_react_session_duplicate_search_is_guarded_without_second_backend_call():
    backend = RollingFakeBackend()
    retriever = FullWikiRetriever(
        backend,
        search_top_k=6,
        max_observation_chars=7200,
        max_evidence_documents=15,
        duplicate_search_guard=True,
    )

    retriever.search("Radiohead lead singer")
    duplicate_observation = retriever.search("  radiohead   lead singer  ")

    assert len(backend.calls) == 1
    assert retriever.last_result["duplicate_query"] is True
    assert retriever.last_result["status"] == "duplicate_query"
    assert "already performed earlier" in duplicate_observation


def test_fullwiki_observation_respects_global_character_cap():
    backend = RollingFakeBackend()
    retriever = FullWikiRetriever(
        backend,
        search_top_k=6,
        max_observation_chars=500,
        max_evidence_documents=15,
    )

    observation = retriever.search("query")
    assert len(observation) <= 500
    assert len(retriever.last_result["retrieved_hits"]) == 6
