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
    from config import FULLWIKI_ARCHIVE_PATH, FULLWIKI_DATA_DIR
except Exception:
    FULLWIKI_DATA_DIR = "data/fullwiki"
    FULLWIKI_ARCHIVE_PATH = os.path.join(FULLWIKI_DATA_DIR, "enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2")

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


def build_hyperlink_graph(archive_path=FULLWIKI_ARCHIVE_PATH, output_dir="indexes/fullwiki", force=False):
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    graph_path = output_dir / "title_graph.json"
    title_to_doc_id_path = output_dir / "title_to_doc_id.json"

    if graph_path.exists() and title_to_doc_id_path.exists() and not force:
        print(f"Hyperlink graph and title index already exist at {graph_path}; skipping.")
        return str(graph_path)

    if not archive_path.exists():
        raise FileNotFoundError(f"Source archive not found at {archive_path}. Ensure corpus is downloaded first.")

    output_dir.mkdir(parents=True, exist_ok=True)
    graph = {}
    title_to_doc_id = {}
    total_links = 0

    for record in tqdm(iter_hotpot_intro_records(archive_path), desc="Extracting Wikipedia hyperlink graph & title map", unit="doc"):
        doc_id = str(record.get("_id", record.get("id", "")))
        title, links = extract_hyperlinks(record)
        if title:
            if doc_id:
                title_to_doc_id[title.lower()] = doc_id
            if links:
                graph[title] = links
                total_links += len(links)

    print(f"Extracted hyperlinks for {len(graph)} documents ({total_links} total edges). Writing to {graph_path}...")
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=None, separators=(",", ":"))

    print(f"Writing title to doc_id index ({len(title_to_doc_id)} titles) to {title_to_doc_id_path}...")
    with open(title_to_doc_id_path, "w", encoding="utf-8") as f:
        json.dump(title_to_doc_id, f, ensure_ascii=False, indent=None, separators=(",", ":"))

    print(f"Hyperlink graph and title index saved successfully to {output_dir}.")
    return str(graph_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Wikipedia Hyperlink Graph from HotpotQA archive.")
    parser.add_argument("--archive-path", default=FULLWIKI_ARCHIVE_PATH, help="Path to HotpotQA intro archive tar.bz2")
    parser.add_argument("--output-dir", default=FULLWIKI_DATA_DIR, help="Output directory for title_graph.json")
    parser.add_argument("--force", action="store_true", help="Force rebuild title_graph.json if it exists")
    args = parser.parse_args()

    build_hyperlink_graph(archive_path=args.archive_path, output_dir=args.output_dir, force=args.force)
