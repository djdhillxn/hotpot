import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    DENSE_EMBEDDING_MODEL,
    FULLWIKI_ARCHIVE_PATH,
    FULLWIKI_BM25_INDEX_DIR,
    FULLWIKI_CORPUS_DIR,
    FULLWIKI_DATA_DIR,
    FULLWIKI_DENSE_INDEX_PATH,
    FULLWIKI_DENSE_NPROBE,
    FULLWIKI_DENSE_TRAIN_SIZE,
    FULLWIKI_FAISS_FACTORY,
    FULLWIKI_INDEX_DIR,
    FULLWIKI_INDEX_MANIFEST,
    FULLWIKI_RRF_K,
)
from retrieval.corpus import ensure_official_corpus


def java_major_version():
    try:
        proc = subprocess.run(["java", "-version"], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    output = (proc.stderr or proc.stdout).splitlines()
    if not output:
        return None
    first = output[0]
    marker = 'version "'
    if marker not in first:
        return None
    version = first.split(marker, 1)[1].split('"', 1)[0]
    try:
        return int(version.split(".")[0])
    except ValueError:
        return None


def verify_environment(require_dense=True):
    if sys.version_info[:2] != (3, 11):
        print(
            f"NOTE: repository environment is pinned to Python 3.11; running {sys.version.split()[0]}.",
            file=sys.stderr,
        )

    major = java_major_version()
    if major is None or major < 21:
        raise RuntimeError(
            "Pyserini 1.6.0 requires Java 21. Install/open the conda environment from environment.yml "
            "and confirm `java -version` reports 21 before building the Lucene index."
        )

    try:
        import pyserini  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Pyserini is not installed. Recreate/update the project environment.") from exc

    if require_dense:
        try:
            import faiss  # noqa: F401
            import sentence_transformers  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Dense indexing requires faiss-cpu and sentence-transformers."
            ) from exc


def build_bm25_index(corpus_dir, index_dir, threads=8, force=False):
    index_dir = Path(index_dir)
    metadata_path = index_dir.parent / "bm25.json"
    if index_dir.exists() and metadata_path.exists() and not force:
        print(f"BM25 index already exists at {index_dir}; skipping.")
        with open(metadata_path, encoding="utf-8") as f:
            return json.load(f)
    if index_dir.exists():
        shutil.rmtree(index_dir)
    metadata_path.unlink(missing_ok=True)
    index_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(corpus_dir),
        "--index",
        str(index_dir),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        str(threads),
        "--storeRaw",
    ]
    print("Building Lucene BM25 index:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    metadata = {
        "engine": "Lucene/Anserini via Pyserini",
        "index_dir": os.path.abspath(str(index_dir)),
        "stored_raw_documents": True,
        "stored_positions": False,
        "stored_docvectors": False,
        "threads": int(threads),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata


def iter_corpus_rows(corpus_dir):
    for path in sorted(Path(corpus_dir).glob("part-*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def _make_faiss_index(faiss, dimension, factory):
    index = faiss.index_factory(dimension, factory, faiss.METRIC_INNER_PRODUCT)
    return index


def build_dense_index(
    corpus_dir,
    index_path,
    model_name=DENSE_EMBEDDING_MODEL,
    factory=FULLWIKI_FAISS_FACTORY,
    train_size=FULLWIKI_DENSE_TRAIN_SIZE,
    nprobe=FULLWIKI_DENSE_NPROBE,
    batch_size=128,
    device="cuda",
    sampling_seed=13,
    force=False,
):
    index_path = Path(index_path)
    metadata_path = Path(str(index_path) + ".json")
    if index_path.exists() and metadata_path.exists() and not force:
        print(f"Dense FAISS index already exists at {index_path}; skipping.")
        with open(metadata_path, encoding="utf-8") as f:
            return json.load(f)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    import faiss
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    if str(device).startswith("cuda"):
        model.half()
    dimension = model.get_sentence_embedding_dimension()
    index = _make_faiss_index(faiss, dimension, factory)

    # IVF/PQ training should see a representative sample of the whole corpus rather
    # than only the first Wikipedia page IDs. Reservoir sampling gives us a uniform
    # deterministic sample in one inexpensive CPU pass over the JSONL shards.
    rng = random.Random(sampling_seed)
    train_texts = []
    seen = 0
    for row in tqdm(iter_corpus_rows(corpus_dir), desc="Sampling dense index training texts", unit="doc"):
        seen += 1
        text = row["contents"]
        if len(train_texts) < train_size:
            train_texts.append(text)
            continue
        replacement = rng.randrange(seen)
        if replacement < train_size:
            train_texts[replacement] = text
    if not train_texts:
        raise RuntimeError("No corpus documents found for dense indexing.")

    def encode_texts(texts):
        return model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

    training_embeddings = []
    for start in tqdm(range(0, len(train_texts), batch_size), desc="Encoding dense training sample", unit="batch"):
        training_embeddings.append(encode_texts(train_texts[start : start + batch_size]))
    training_embeddings = np.concatenate(training_embeddings, axis=0)

    if not index.is_trained:
        print(f"Training FAISS index factory '{factory}' on {len(training_embeddings)} vectors...")
        index.train(training_embeddings)
    del training_embeddings
    del train_texts

    # Re-scan the corpus and encode every document exactly once for the persistent
    # index. Wikipedia numeric page IDs are used as FAISS IDs so dense hits can be
    # hydrated directly from the Lucene raw-document store.
    total_added = 0
    batch_rows = []
    pbar = tqdm(desc="Encoding + adding FullWiki dense vectors", unit="doc")
    for row in iter_corpus_rows(corpus_dir):
        batch_rows.append(row)
        if len(batch_rows) < batch_size:
            continue
        embeddings = encode_texts([item["contents"] for item in batch_rows])
        ids = np.array([int(item["id"]) for item in batch_rows], dtype="int64")
        index.add_with_ids(embeddings, ids)
        total_added += len(batch_rows)
        pbar.update(len(batch_rows))
        batch_rows = []

    if batch_rows:
        embeddings = encode_texts([item["contents"] for item in batch_rows])
        ids = np.array([int(item["id"]) for item in batch_rows], dtype="int64")
        index.add_with_ids(embeddings, ids)
        total_added += len(batch_rows)
        pbar.update(len(batch_rows))
    pbar.close()

    if hasattr(index, "nprobe"):
        index.nprobe = int(nprobe)
    faiss.write_index(index, str(index_path))
    metadata = {
        "model": model_name,
        "dimension": dimension,
        "factory": factory,
        "train_size": min(int(train_size), int(seen)),
        "training_sample_strategy": "uniform_reservoir",
        "training_sample_seed": int(sampling_seed),
        "nprobe": int(nprobe),
        "document_count": int(index.ntotal),
        "normalized_embeddings": True,
        "metric": "inner_product_on_L2_normalized_vectors",
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata


def verify_indexes(bm25_index_dir, dense_index_path, expected_docs=None):
    from pyserini.index.lucene import LuceneIndexReader
    import faiss

    reader = LuceneIndexReader(str(bm25_index_dir))
    bm25_docs = int(reader.stats()["documents"])
    dense = faiss.read_index(str(dense_index_path))
    dense_docs = int(dense.ntotal)

    if expected_docs is not None and bm25_docs != int(expected_docs):
        raise RuntimeError(f"BM25 document count {bm25_docs} != corpus count {expected_docs}.")
    if expected_docs is not None and dense_docs != int(expected_docs):
        raise RuntimeError(f"Dense document count {dense_docs} != corpus count {expected_docs}.")
    return {"bm25_document_count": bm25_docs, "dense_document_count": dense_docs}


def main():
    parser = argparse.ArgumentParser(description="Build global HotpotQA FullWiki BM25 + dense indexes")
    parser.add_argument("--archive", default=FULLWIKI_ARCHIVE_PATH)
    parser.add_argument("--corpus-dir", default=FULLWIKI_CORPUS_DIR)
    parser.add_argument("--index-dir", default=FULLWIKI_INDEX_DIR)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--dense-model", default=DENSE_EMBEDDING_MODEL)
    parser.add_argument("--dense-device", default="cuda")
    parser.add_argument("--dense-batch-size", type=int, default=128)
    parser.add_argument("--dense-factory", default=FULLWIKI_FAISS_FACTORY)
    parser.add_argument("--dense-train-size", type=int, default=FULLWIKI_DENSE_TRAIN_SIZE)
    parser.add_argument("--nprobe", type=int, default=FULLWIKI_DENSE_NPROBE)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-corpus", action="store_true")
    parser.add_argument("--force-bm25", action="store_true")
    parser.add_argument("--force-dense", action="store_true")
    args = parser.parse_args()

    verify_environment(require_dense=True)
    os.makedirs(FULLWIKI_DATA_DIR, exist_ok=True)
    os.makedirs(args.index_dir, exist_ok=True)

    corpus_manifest = ensure_official_corpus(
        archive_path=args.archive,
        corpus_dir=args.corpus_dir,
        force_download=args.force_download,
        force_prepare=args.force_corpus,
    )
    print(f"Prepared {corpus_manifest['document_count']:,} Wikipedia intro paragraphs.")

    bm25_dir = os.path.join(args.index_dir, "bm25")
    dense_path = os.path.join(args.index_dir, "dense.faiss")
    bm25_manifest = build_bm25_index(
        args.corpus_dir, bm25_dir, threads=args.threads, force=args.force_bm25
    )
    dense_manifest = build_dense_index(
        args.corpus_dir,
        dense_path,
        model_name=args.dense_model,
        factory=args.dense_factory,
        train_size=args.dense_train_size,
        nprobe=args.nprobe,
        batch_size=args.dense_batch_size,
        device=args.dense_device,
        force=args.force_dense,
    )
    counts = verify_indexes(
        bm25_dir,
        dense_path,
        expected_docs=corpus_manifest["document_count"],
    )

    manifest = {
        "corpus": corpus_manifest,
        "bm25": {
            **bm25_manifest,
            "document_count": counts["bm25_document_count"],
            "parameters": "Pyserini LuceneSearcher default BM25 (no tuned dataset-specific parameters)",
        },
        "dense": {
            **dense_manifest,
            "index_path": os.path.abspath(dense_path),
            "document_count": counts["dense_document_count"],
        },
        "hybrid": {
            "fusion": "reciprocal_rank_fusion",
            "rrf_k": FULLWIKI_RRF_K,
        },
    }
    manifest_path = Path(args.index_dir) / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\nFullWiki indexes are ready.")
    print(f"BM25:   {bm25_dir}")
    print(f"Dense:  {dense_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
