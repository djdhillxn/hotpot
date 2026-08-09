"""
===================================================================================
HOTPOTQA PRECISION HYPERLINK GRAPH PIPELINE & ARCHITECTURE OVERVIEW
===================================================================================

1. Purpose & Extraction (tools/build_hyperlink_graph.py):
   - Extracts sentence-level 1-hop Wikipedia HTML hyperlinks (<a href="Target">Anchor</a>)
     and outdegree statistics directly from HotpotQA's ~5M intro-paragraph corpus.
   - Builds an indexed, high-performance SQLite database: indexes/fullwiki/hyperlink_graph.db
   - SQLite Tables:
       * edges: (source_title, source_doc_id, source_sent_id, anchor_text, target_title, target_doc_id)
         Indexed with case-insensitive COLLATE NOCASE indexes on source_title and target_title.
       * outdegree: (target_title PRIMARY KEY, count INTEGER)
         Stores total incoming link frequency across Wikipedia for IDF outdegree penalization.
       * title_map: (norm_title PRIMARY KEY, doc_id TEXT)
         Provides O(1) title-to-doc_id lookups for direct title searches.

2. Shared Backend Loading (FullWikiSearchBackend in retrieval/fullwiki_retriever.py):
   - Loads SQLite DB via thread-safe read-only connection pooling (check_same_thread=False, mode=ro).
   - Prevents multi-gigabyte RAM bloat by querying disk indexes in <0.1ms per lookup.
   - Exposes O(1) backend methods:
       * backend.get_outgoing_edges(title)
       * backend.get_target_outdegree(target_title)

3. Precision Candidate Expansion (FullWikiRetriever in retrieval/fullwiki_retriever.py):
   - When use_graph_expansion is enabled (controlled via --enable-graph-expansion or YAML):
       a. Focuses expansion strictly on self.current_title and top N active-memory pages
          (graph_focus_doc_count, default: 2), avoiding noisy multi-page fan-out.
       b. Scores candidate target edges using multi-signal scoring:
            Score = (w_src * SourceSentScore) + (w_anchor * AnchorOverlap)
                  + (w_title * TitleOverlap) - (w_out * log(1 + TargetOutDegree))
       c. The -log(1 + TargetOutDegree) IDF prior automatically downweights generic hubs
          (e.g., "United States", "Film", "1985") without brittle hard-coded stop lists.
       d. Selects top candidate targets up to graph_candidate_quota (default: 10) and merges
          them into hits before Cross-Encoder reranking.
       e. Logs detailed edge expansion metrics to last_result["graph_expansion_info"].

4. Downstream Evaluation Integration (eval/run_eval.py & config/fullwiki.yaml):
   - Parameterized via YAML config and CLI flags:
       * --enable-graph-expansion / --disable-graph-expansion
   - Thread-safe across 64 concurrent ThreadPoolExecutor evaluation workers.
===================================================================================
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from config import FULLWIKI_ARCHIVE_PATH, FULLWIKI_INDEX_DIR
except Exception:
    FULLWIKI_INDEX_DIR = "indexes/fullwiki"
    FULLWIKI_ARCHIVE_PATH = os.path.join("data", "fullwiki", "enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2")

from retrieval.corpus import iter_hotpot_intro_records

import urllib.parse

# Matches <a href="TARGET_URL">ANCHOR_TEXT</a> or [[MediaWiki]]
HTML_LINK_REGEX = re.compile(r'<a\s+href="([^"]+)">([^<]+)</a>', re.IGNORECASE)
MEDIAWIKI_LINK_REGEX = re.compile(r"\[\[([^\]\|]+)(?:\|[^\]]+)?\]\]")


def clean_link_href(raw_href):
    decoded = urllib.parse.unquote(raw_href)
    if "#" in decoded:
        decoded = decoded.split("#", 1)[0]
    clean = " ".join(decoded.replace("_", " ").split()).strip()
    return clean


def extract_hyperlinks(record):
    title = str(record.get("title", "")).strip()
    if not title:
        return None, []

    text_with_links = record.get("text_with_links", [])
    if not text_with_links:
        return title, []

    linked_titles = []
    seen = set()

    for sentence in text_with_links:
        if isinstance(sentence, list):
            sentence_str = " ".join(str(x) for x in sentence)
        else:
            sentence_str = str(sentence)

        # Parse HTML links <a href="Target">Anchor</a>
        html_matches = HTML_LINK_REGEX.findall(sentence_str)
        for raw_href, anchor in html_matches:
            clean_target = clean_link_href(raw_href)
            if clean_target and clean_target.lower() != title.lower() and clean_target.lower() not in seen:
                seen.add(clean_target.lower())
                linked_titles.append(clean_target)

        # Fallback to MediaWiki [[Target]] links if HTML links not present
        if not html_matches:
            mw_matches = MEDIAWIKI_LINK_REGEX.findall(sentence_str)
            for raw_target in mw_matches:
                clean_target = clean_link_href(raw_target)
                if clean_target and clean_target.lower() != title.lower() and clean_target.lower() not in seen:
                    seen.add(clean_target.lower())
                    linked_titles.append(clean_target)

    return title, linked_titles


import sqlite3
from collections import Counter


def extract_detailed_hyperlinks(record):
    doc_id = str(record.get("_id", record.get("id", "")))
    title = str(record.get("title", "")).strip()
    if not title:
        return None, [], []

    text_with_links = record.get("text_with_links", [])
    if not text_with_links:
        return title, [], []

    edges = []
    linked_titles = []
    seen = set()

    for sent_id, sentence in enumerate(text_with_links):
        if isinstance(sentence, list):
            sentence_str = " ".join(str(x) for x in sentence)
        else:
            sentence_str = str(sentence)

        html_matches = HTML_LINK_REGEX.findall(sentence_str)
        for raw_href, anchor in html_matches:
            clean_target = clean_link_href(raw_href)
            clean_anchor = str(anchor).strip()
            if clean_target and clean_target.lower() != title.lower():
                edges.append((title, doc_id, sent_id, clean_anchor, clean_target, ""))
                if clean_target.lower() not in seen:
                    seen.add(clean_target.lower())
                    linked_titles.append(clean_target)

        if not html_matches:
            mw_matches = MEDIAWIKI_LINK_REGEX.findall(sentence_str)
            for raw_target in mw_matches:
                clean_target = clean_link_href(raw_target)
                if clean_target and clean_target.lower() != title.lower():
                    edges.append((title, doc_id, sent_id, clean_target, clean_target, ""))
                    if clean_target.lower() not in seen:
                        seen.add(clean_target.lower())
                        linked_titles.append(clean_target)

    return title, linked_titles, edges


def build_hyperlink_graph(archive_path=FULLWIKI_ARCHIVE_PATH, output_dir="indexes/fullwiki", force=False):
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    graph_path = output_dir / "title_graph.json"
    title_to_doc_id_path = output_dir / "title_to_doc_id.json"
    db_path = output_dir / "hyperlink_graph.db"

    if graph_path.exists() and title_to_doc_id_path.exists() and db_path.exists() and not force:
        print(f"Hyperlink graph database and indexes already exist at {output_dir}; skipping.")
        return str(graph_path)

    if not archive_path.exists():
        raise FileNotFoundError(f"Source archive not found at {archive_path}. Ensure corpus is downloaded first.")

    output_dir.mkdir(parents=True, exist_ok=True)
    graph = {}
    title_to_doc_id = {}
    outdegree_counts = Counter()
    total_links = 0

    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE edges (
            source_title TEXT,
            source_doc_id TEXT,
            source_sent_id INTEGER,
            anchor_text TEXT,
            target_title TEXT,
            target_doc_id TEXT
        )
    """)

    batch_edges = []

    for record in tqdm(iter_hotpot_intro_records(archive_path), desc="Extracting Wikipedia hyperlink graph & SQLite DB", unit="doc"):
        doc_id = str(record.get("_id", record.get("id", "")))
        title, links, detailed_edges = extract_detailed_hyperlinks(record)
        if title:
            if doc_id:
                title_to_doc_id[title.lower()] = doc_id
            if links:
                graph[title] = links
                total_links += len(links)
                for edge in detailed_edges:
                    outdegree_counts[edge[4].lower()] += 1
                    batch_edges.append(edge)

            if len(batch_edges) >= 50000:
                cursor.executemany("INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?)", batch_edges)
                batch_edges.clear()

    if batch_edges:
        cursor.executemany("INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?)", batch_edges)
        batch_edges.clear()

    print("Creating SQLite case-insensitive indexes on hyperlink_graph.db...")
    cursor.execute("CREATE INDEX idx_edges_source_nocase ON edges(source_title COLLATE NOCASE)")
    cursor.execute("CREATE INDEX idx_edges_target_nocase ON edges(target_title COLLATE NOCASE)")

    cursor.execute("CREATE TABLE outdegree (target_title TEXT PRIMARY KEY, count INTEGER)")
    outdegree_data = [(t, count) for t, count in outdegree_counts.items()]
    cursor.executemany("INSERT INTO outdegree VALUES (?, ?)", outdegree_data)
    cursor.execute("CREATE INDEX idx_outdegree_target ON outdegree(target_title)")

    cursor.execute("CREATE TABLE title_map (norm_title TEXT PRIMARY KEY, doc_id TEXT)")
    title_map_data = [(norm_t, did) for norm_t, did in title_to_doc_id.items()]
    cursor.executemany("INSERT INTO title_map VALUES (?, ?)", title_map_data)

    conn.commit()
    conn.close()

    print(f"Extracted hyperlinks for {len(graph)} documents ({total_links} total edges). Writing to {graph_path}...")
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=None, separators=(",", ":"))

    print(f"Writing title to doc_id index ({len(title_to_doc_id)} titles) to {title_to_doc_id_path}...")
    with open(title_to_doc_id_path, "w", encoding="utf-8") as f:
        json.dump(title_to_doc_id, f, ensure_ascii=False, indent=None, separators=(",", ":"))

    print(f"Hyperlink graph SQLite DB ({db_path}) and JSON indexes saved successfully to {output_dir}.")
    return str(graph_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Wikipedia Hyperlink Graph from HotpotQA archive.")
    parser.add_argument("--archive-path", default=FULLWIKI_ARCHIVE_PATH, help="Path to HotpotQA intro archive tar.bz2")
    parser.add_argument("--output-dir", default=FULLWIKI_INDEX_DIR, help="Output directory for hyperlink_graph.db and JSON indexes")
    parser.add_argument("--force", action="store_true", help="Force rebuild title_graph.json if it exists")
    args = parser.parse_args()

    build_hyperlink_graph(archive_path=args.archive_path, output_dir=args.output_dir, force=args.force)
