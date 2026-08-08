import bz2
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

from config import FULLWIKI_ARCHIVE_MD5, FULLWIKI_ARCHIVE_URL


def file_md5(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url, output_path, expected_md5=None, force=False):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        if expected_md5 is None or file_md5(output_path) == expected_md5:
            return str(output_path)
        raise RuntimeError(
            f"Existing archive checksum mismatch: {output_path}. "
            "Delete it or pass --force-download."
        )

    partial_path = output_path.with_suffix(output_path.suffix + ".part")
    with requests.get(url, stream=True, timeout=(15, 180)) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(partial_path, "wb") as f, tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {output_path.name}",
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))

    os.replace(partial_path, output_path)
    if expected_md5:
        actual = file_md5(output_path)
        if actual != expected_md5:
            output_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {output_path}: expected {expected_md5}, got {actual}."
            )
    return str(output_path)


def _iter_json_lines_from_binary(binary_stream):
    buffered = io.BufferedReader(binary_stream)
    signature = buffered.peek(3)[:3]
    if signature == b"BZh":
        stream = bz2.BZ2File(buffered, mode="rb")
    else:
        stream = buffered

    text_stream = io.TextIOWrapper(stream, encoding="utf-8", errors="replace")
    for line in text_stream:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def iter_hotpot_intro_records(archive_path):
    """Yield JSON objects from the official HotpotQA intro-paragraph archive.

    The released .tar.bz2 has historically contained compressed shards. This reader
    accepts both compressed and plain members and also falls back to a direct bz2
    JSON-lines stream for mirrors that repack the same data.
    """
    archive_path = str(archive_path)
    try:
        with tarfile.open(archive_path, mode="r:bz2") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                member_file = tar.extractfile(member)
                if member_file is None:
                    continue
                yield from _iter_json_lines_from_binary(member_file)
        return
    except tarfile.ReadError:
        pass

    with bz2.open(archive_path, mode="rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _flatten_sentences(raw_text):
    if not isinstance(raw_text, list):
        return [str(raw_text)] if raw_text is not None else []

    # Sentence positions are benchmark identifiers in HotpotQA. Never drop an
    # empty/blank sentence here: doing so would shift every later sentence_id.
    if all(isinstance(item, str) for item in raw_text):
        return list(raw_text)

    for paragraph in raw_text:
        if isinstance(paragraph, list):
            return [str(sentence) if sentence is not None else "" for sentence in paragraph]
        if isinstance(paragraph, str):
            return [paragraph]
    return []


def normalize_intro_record(record):
    title = str(record.get("title", "")).strip()
    raw_id = record.get("id")
    sentences = _flatten_sentences(record.get("text", []))

    if not title or raw_id is None or not sentences:
        return None

    try:
        numeric_id = int(raw_id)
    except (TypeError, ValueError):
        return None

    paragraph = "".join(sentences).strip()
    if not paragraph:
        return None

    return {
        "id": str(numeric_id),
        "title": title,
        "url": record.get("url", ""),
        "sentences": sentences,
        "contents": f"{title}\n{paragraph}",
    }


def prepare_jsonl_corpus(
    archive_path,
    output_dir,
    docs_per_shard=100000,
    force=False,
):
    output_dir = Path(output_dir)
    manifest_path = output_dir / "corpus_manifest.json"
    if manifest_path.exists() and not force:
        with open(manifest_path) as f:
            manifest = json.load(f)
        shard_paths = list(output_dir.glob("part-*.jsonl"))
        if len(shard_paths) == int(manifest.get("shard_count", -1)) and shard_paths:
            return manifest
        print("Corpus manifest exists but shards are incomplete; rebuilding corpus.")

    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("part-*.jsonl"):
        path.unlink()
    manifest_path.unlink(missing_ok=True)

    document_count = 0
    shard_count = 0
    shard_docs = 0
    output_file = None
    corpus_sha256 = hashlib.sha256()

    def open_next_shard():
        nonlocal shard_count, shard_docs
        shard_path = output_dir / f"part-{shard_count:05d}.jsonl"
        shard_count += 1
        shard_docs = 0
        return open(shard_path, "w", encoding="utf-8")

    try:
        output_file = open_next_shard()
        for raw_record in tqdm(iter_hotpot_intro_records(archive_path), desc="Preparing FullWiki corpus", unit="doc"):
            record = normalize_intro_record(raw_record)
            if record is None:
                continue
            if shard_docs >= docs_per_shard:
                output_file.close()
                output_file = open_next_shard()

            line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            output_file.write(line)
            corpus_sha256.update(line.encode("utf-8"))
            document_count += 1
            shard_docs += 1
    finally:
        if output_file is not None:
            output_file.close()

    manifest = {
        "source_archive": os.path.abspath(str(archive_path)),
        "source_archive_md5": file_md5(archive_path),
        "official_archive_url": FULLWIKI_ARCHIVE_URL,
        "official_archive_md5": FULLWIKI_ARCHIVE_MD5,
        "document_count": document_count,
        "shard_count": shard_count,
        "docs_per_shard": docs_per_shard,
        "corpus_sha256": corpus_sha256.hexdigest(),
        "sentence_ids": "HotpotQA/CoreNLP 0-based sentence IDs preserved from source text field",
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def ensure_official_corpus(archive_path, corpus_dir, force_download=False, force_prepare=False):
    archive_path = Path(archive_path)
    if not archive_path.exists() or force_download:
        download_file(
            FULLWIKI_ARCHIVE_URL,
            archive_path,
            expected_md5=FULLWIKI_ARCHIVE_MD5,
            force=force_download,
        )
    else:
        actual_md5 = file_md5(archive_path)
        if actual_md5 != FULLWIKI_ARCHIVE_MD5:
            raise RuntimeError(
                f"Official FullWiki archive MD5 mismatch: expected {FULLWIKI_ARCHIVE_MD5}, got {actual_md5}."
            )

    return prepare_jsonl_corpus(archive_path, corpus_dir, force=force_prepare)
