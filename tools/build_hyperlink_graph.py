import argparse
import json
import os
import re
import sys
from pathlib import Path
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

# Matches [[Target Title|Anchor]] or [[Target Title]]
LINK_REGEX = re.compile(r"\[\[([^\]\|]+)(?:\|[^\]]+)?\]\]")


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

        matches = LINK_REGEX.findall(sentence_str)
        for target in matches:
            clean_target = str(target).strip()
            if clean_target and clean_target.lower() != title.lower() and clean_target.lower() not in seen:
                seen.add(clean_target.lower())
                linked_titles.append(clean_target)

    return title, linked_titles


def build_hyperlink_graph(archive_path=FULLWIKI_ARCHIVE_PATH, output_dir=FULLWIKI_DATA_DIR, force=False):
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    graph_path = output_dir / "title_graph.json"

    if graph_path.exists() and not force:
        print(f"Hyperlink graph already exists at {graph_path}; skipping.")
        return str(graph_path)

    if not archive_path.exists():
        raise FileNotFoundError(f"Source archive not found at {archive_path}. Ensure corpus is downloaded first.")

    output_dir.mkdir(parents=True, exist_ok=True)
    graph = {}
    total_links = 0

    for record in tqdm(iter_hotpot_intro_records(archive_path), desc="Extracting Wikipedia hyperlink graph", unit="doc"):
        title, links = extract_hyperlinks(record)
        if title and links:
            graph[title] = links
            total_links += len(links)

    print(f"Extracted hyperlinks for {len(graph)} documents ({total_links} total edges). Writing to {graph_path}...")
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=None, separators=(",", ":"))

    print(f"Hyperlink graph saved successfully to {graph_path}.")
    return str(graph_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Wikipedia Hyperlink Graph from HotpotQA archive.")
    parser.add_argument("--archive-path", default=FULLWIKI_ARCHIVE_PATH, help="Path to HotpotQA intro archive tar.bz2")
    parser.add_argument("--output-dir", default=FULLWIKI_DATA_DIR, help="Output directory for title_graph.json")
    parser.add_argument("--force", action="store_true", help="Force rebuild title_graph.json if it exists")
    args = parser.parse_args()

    build_hyperlink_graph(archive_path=args.archive_path, output_dir=args.output_dir, force=args.force)
