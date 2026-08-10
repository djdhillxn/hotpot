import argparse
import json
import logging
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.metrics import compute_metrics_by_type, print_segmented_report


BASELINE_LABEL = "Reranked Single-Pass RAG"
REACT_LABEL = "ReAct Agent"
BASELINE_COLOR = "#6B7280"
REACT_COLOR = "#2563EB"
TEXT_COLOR = "#111827"
MUTED_COLOR = "#6B7280"
GRID_COLOR = "#E5E7EB"
RESCUE_COLOR = "#2563EB"
REGRESSION_COLOR = "#9CA3AF"

RESULT_KEY_BY_SUMMARY = {
    "em": "exact_match",
    "f1": "f1",
    "sp_em": "sp_em",
    "sp_f1": "sp_f1",
    "joint_em": "joint_em",
    "joint_f1": "joint_f1",
}

OFFICIAL_METRICS = [
    ("Answer EM", "em"),
    ("Answer F1", "f1"),
    ("Supporting Fact EM", "sp_em"),
    ("Supporting Fact F1", "sp_f1"),
    ("Joint EM", "joint_em"),
    ("Joint F1", "joint_f1"),
]

COMPARABILITY_FIELDS = [
    ("model", ("model",)),
    ("dataset source", ("dataset_source",)),
    ("dataset size", ("dataset_size",)),
    ("retrieval mode", ("retrieval_mode",)),
    ("retriever", ("retriever",)),
    ("concurrency", ("concurrency",)),
    ("candidate_k", ("retrieval_backend", "candidate_k")),
    ("rrf_k", ("retrieval_backend", "rrf_k")),
    ("dense model", ("retrieval_backend", "dense_model")),
    ("dense query device", ("retrieval_backend", "dense_query_device")),
    ("dense nprobe", ("retrieval_backend", "dense_nprobe")),
    (
        "corpus SHA-256",
        ("retrieval_backend", "index_manifest", "corpus", "corpus_sha256"),
    ),
    (
        "corpus source MD5",
        ("retrieval_backend", "index_manifest", "corpus", "source_archive_md5"),
    ),
    (
        "BM25 engine",
        ("retrieval_backend", "index_manifest", "bm25", "engine"),
    ),
    (
        "BM25 parameters",
        ("retrieval_backend", "index_manifest", "bm25", "parameters"),
    ),
    (
        "BM25 corpus document count",
        ("retrieval_backend", "index_manifest", "bm25", "document_count"),
    ),
    (
        "dense index factory",
        ("retrieval_backend", "index_manifest", "dense", "factory"),
    ),
    (
        "dense dimension",
        ("retrieval_backend", "index_manifest", "dense", "dimension"),
    ),
    (
        "dense corpus document count",
        ("retrieval_backend", "index_manifest", "dense", "document_count"),
    ),
    (
        "page reranker model",
        ("retrieval_backend", "local_reranker", "model"),
    ),
    (
        "page reranker device",
        ("retrieval_backend", "local_reranker", "device"),
    ),
    (
        "page reranker max length",
        ("retrieval_backend", "local_reranker", "max_length"),
    ),
    (
        "page reranker batch size",
        ("retrieval_backend", "local_reranker", "batch_size"),
    ),
]


class ComparisonValidationError(ValueError):
    pass


def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "comparison.log")

    logger = logging.getLogger("compare_runner")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode="w")
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, log_file


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def manifest_path_for_results(results_path):
    return str(Path(results_path).resolve().with_name("run_manifest.json"))


def nested_get(obj, path):
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def normalize_supporting_facts(facts):
    normalized = []
    for fact in facts or []:
        if not isinstance(fact, (list, tuple)) or len(fact) != 2:
            continue
        title, sent_id = fact
        try:
            sent_id = int(sent_id)
        except (TypeError, ValueError):
            continue
        normalized.append((str(title), sent_id))
    return sorted(set(normalized))


def index_records_by_id(records, label):
    indexed = {}
    duplicates = []
    missing_ids = []
    for position, record in enumerate(records):
        qid = str(record.get("id") or "").strip()
        if not qid:
            missing_ids.append(position)
            continue
        if qid in indexed:
            duplicates.append(qid)
        indexed[qid] = record

    if missing_ids or duplicates:
        problems = []
        if missing_ids:
            problems.append(f"{len(missing_ids)} records without IDs")
        if duplicates:
            problems.append(f"duplicate IDs: {duplicates[:5]}")
        raise ComparisonValidationError(f"{label} results are not uniquely pairable: " + "; ".join(problems))
    return indexed


def validate_record_alignment(baseline_records, react_records):
    baseline_by_id = index_records_by_id(baseline_records, BASELINE_LABEL)
    react_by_id = index_records_by_id(react_records, REACT_LABEL)

    baseline_ids = set(baseline_by_id)
    react_ids = set(react_by_id)
    if baseline_ids != react_ids:
        missing_in_react = sorted(baseline_ids - react_ids)
        missing_in_baseline = sorted(react_ids - baseline_ids)
        raise ComparisonValidationError(
            "Question-ID sets differ. "
            f"Missing in ReAct: {missing_in_react[:5]} ({len(missing_in_react)} total); "
            f"missing in baseline: {missing_in_baseline[:5]} ({len(missing_in_baseline)} total)."
        )

    mismatches = []
    for qid in baseline_by_id:
        baseline = baseline_by_id[qid]
        react = react_by_id[qid]
        checks = [
            ("question", str(baseline.get("question") or ""), str(react.get("question") or "")),
            ("gold_answer", str(baseline.get("gold_answer") or ""), str(react.get("gold_answer") or "")),
            ("question_type", str(baseline.get("question_type") or ""), str(react.get("question_type") or "")),
            (
                "gold_supporting_facts",
                normalize_supporting_facts(baseline.get("gold_supporting_facts")),
                normalize_supporting_facts(react.get("gold_supporting_facts")),
            ),
        ]
        for field, baseline_value, react_value in checks:
            if baseline_value != react_value:
                mismatches.append((qid, field, baseline_value, react_value))
                if len(mismatches) >= 10:
                    break
        if len(mismatches) >= 10:
            break

    if mismatches:
        lines = ["Paired result records disagree on evaluation identity:"]
        for qid, field, baseline_value, react_value in mismatches:
            lines.append(
                f"  {qid}: {field}: baseline={baseline_value!r}, react={react_value!r}"
            )
        raise ComparisonValidationError("\n".join(lines))

    return baseline_by_id, react_by_id


def validate_manifests(baseline_manifest, react_manifest):
    mismatches = []
    matched = {}
    for label, path in COMPARABILITY_FIELDS:
        baseline_value = nested_get(baseline_manifest, path)
        react_value = nested_get(react_manifest, path)
        if baseline_value is None or react_value is None:
            mismatches.append((label, baseline_value, react_value, "missing"))
        elif baseline_value != react_value:
            mismatches.append((label, baseline_value, react_value, "different"))
        else:
            matched[label] = baseline_value

    baseline_completed = int(baseline_manifest.get("completed_records") or 0)
    react_completed = int(react_manifest.get("completed_records") or 0)
    baseline_failed = int(baseline_manifest.get("failed_records") or 0)
    react_failed = int(react_manifest.get("failed_records") or 0)
    if baseline_completed != react_completed:
        mismatches.append(
            ("completed record count", baseline_completed, react_completed, "different")
        )
    if baseline_failed != 0 or react_failed != 0:
        mismatches.append(
            ("failed record count must be zero", baseline_failed, react_failed, "different")
        )

    if mismatches:
        lines = ["Run manifests are not apples-to-apples:"]
        for label, baseline_value, react_value, reason in mismatches:
            lines.append(
                f"  {label}: baseline={baseline_value!r}, react={react_value!r} ({reason})"
            )
        raise ComparisonValidationError("\n".join(lines))

    return matched



def validate_protocol_contracts(baseline_manifest, react_manifest):
    problems = []

    def require(condition, message):
        if not condition:
            problems.append(message)

    require(
        baseline_manifest.get("runner") == "single_pass_rag",
        f"baseline runner must be 'single_pass_rag', got {baseline_manifest.get('runner')!r}",
    )
    require(
        react_manifest.get("runner") == "react",
        f"ReAct runner must be 'react', got {react_manifest.get('runner')!r}",
    )
    require(
        int(baseline_manifest.get("retrieval_calls_per_question") or 0) == 1,
        "baseline must use exactly one retrieval call per question",
    )
    require(
        int(baseline_manifest.get("generation_calls_per_question") or 0) == 1,
        "baseline must use exactly one generation call per question",
    )
    require(
        baseline_manifest.get("page_reranker_enabled") is True,
        "baseline page reranker must be enabled",
    )
    require(
        baseline_manifest.get("sentence_reranker_enabled") is False,
        "baseline sentence reranker must be disabled",
    )

    baseline_hydrated = int(
        baseline_manifest.get("documents_hydrated_for_page_reranking") or 0
    )
    baseline_reader = int(baseline_manifest.get("documents_in_reader_context") or 0)
    react_search_docs = int(react_manifest.get("documents_per_search") or 0)
    react_sentence_pages = int(react_manifest.get("local_rerank_page_count") or 0)
    react_snippets = int(react_manifest.get("max_working_evidence_snippets") or 0)
    react_memory_chars = int(
        react_manifest.get("max_working_evidence_characters") or 0
    )

    require(baseline_hydrated > 0, "baseline must hydrate documents before page reranking")
    require(
        0 < baseline_reader <= baseline_hydrated,
        "baseline reader-context document count must be positive and no larger than hydrated count",
    )
    require(
        react_search_docs == baseline_hydrated,
        "baseline and ReAct must page-rerank the same number of hydrated candidates",
    )
    require(
        0 < react_sentence_pages <= react_search_docs,
        "ReAct sentence-rerank page count must be positive and within the page candidate set",
    )
    require(react_snippets > 0, "ReAct persistent snippet-memory size must be positive")
    require(react_memory_chars > 0, "ReAct persistent snippet-memory character budget must be positive")

    if problems:
        raise ComparisonValidationError(
            "Run protocol contracts are not the intended final baseline/ReAct experiment:\n  "
            + "\n  ".join(problems)
        )

    return {
        "baseline": {
            "retrieval_calls_per_question": 1,
            "generation_calls_per_question": 1,
            "documents_hydrated_for_page_reranking": baseline_hydrated,
            "documents_in_reader_context": baseline_reader,
            "page_reranker_enabled": True,
            "sentence_reranker_enabled": False,
        },
        "react": {
            "max_hops": int(react_manifest.get("max_hops") or 0),
            "documents_per_search": react_search_docs,
            "local_rerank_page_count": react_sentence_pages,
            "max_working_evidence_snippets": react_snippets,
            "max_working_evidence_characters": react_memory_chars,
            "working_evidence_policy": react_manifest.get("working_evidence_policy"),
        },
    }


def mean_pct(records, key):
    if not records:
        return 0.0
    return sum(float(record.get(key, 0.0) or 0.0) for record in records) / len(records) * 100.0


def percentile(values, percentile_value):
    values = sorted(float(value) for value in values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * (percentile_value / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction



def official_deltas_by_type(baseline_records, react_records):
    deltas = {}
    for group in ["overall", "bridge", "comparison"]:
        if group == "overall":
            baseline_group = baseline_records
            react_group = react_records
        else:
            baseline_group = [
                record for record in baseline_records
                if str(record.get("question_type") or "").lower() == group
            ]
            react_group = [
                record for record in react_records
                if str(record.get("question_type") or "").lower() == group
            ]
        deltas[group] = {}
        for _, summary_key in OFFICIAL_METRICS:
            result_key = RESULT_KEY_BY_SUMMARY[summary_key]
            deltas[group][summary_key] = (
                mean_pct(react_group, result_key) - mean_pct(baseline_group, result_key)
            )
    return deltas


def latency_summary(records):
    values = [float(record.get("latency", 0.0) or 0.0) for record in records]
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0}
    return {
        "mean": sum(values) / len(values),
        "median": median(values),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
    }


def evidence_summary(records):
    return {
        "supporting_document_f1": mean_pct(records, "doc_f1"),
        "observed_gold_document_recall": mean_pct(records, "observed_gold_document_recall"),
        "observed_gold_supporting_fact_recall": mean_pct(
            records, "observed_gold_supporting_fact_recall"
        ),
        "all_gold_supporting_facts_observed": mean_pct(
            records, "all_gold_supporting_facts_observed"
        ),
    }


def paired_outcomes(baseline_by_id, react_by_id, metric_key):
    counts = {
        "both_correct": 0,
        "baseline_only": 0,
        "react_only": 0,
        "both_wrong": 0,
    }
    for qid, baseline in baseline_by_id.items():
        react = react_by_id[qid]
        baseline_correct = bool(baseline.get(metric_key, False))
        react_correct = bool(react.get(metric_key, False))
        if baseline_correct and react_correct:
            counts["both_correct"] += 1
        elif baseline_correct:
            counts["baseline_only"] += 1
        elif react_correct:
            counts["react_only"] += 1
        else:
            counts["both_wrong"] += 1
    counts["net_gain"] = counts["react_only"] - counts["baseline_only"]
    return counts


def react_hop_summary(baseline_by_id, react_by_id):
    groups = defaultdict(list)
    for qid, record in react_by_id.items():
        groups[int(record.get("step_count", 0) or 0)].append(qid)

    summary = []
    for hops in sorted(groups):
        qids = groups[hops]
        react_records = [react_by_id[qid] for qid in qids]
        baseline_records = [baseline_by_id[qid] for qid in qids]
        summary.append(
            {
                "hops": hops,
                "count": len(qids),
                "react_joint_f1": mean_pct(react_records, "joint_f1"),
                "baseline_joint_f1_same_questions": mean_pct(
                    baseline_records, "joint_f1"
                ),
                "react_answer_f1": mean_pct(react_records, "f1"),
                "baseline_answer_f1_same_questions": mean_pct(
                    baseline_records, "f1"
                ),
            }
        )
    return summary


def reranker_workload(manifest, question_count):
    pairs = nested_get(manifest, ("retrieval_backend", "local_reranker", "total_pairs_scored"))
    pairs = int(pairs or 0)
    return {
        "total_pairs_scored": pairs,
        "pairs_per_question": pairs / question_count if question_count else 0.0,
    }


def configure_plot_style():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 10,
            "axes.edgecolor": "#D1D5DB",
            "axes.linewidth": 0.8,
            "xtick.color": MUTED_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig, output_dir, stem):
    png_path = os.path.join(output_dir, f"{stem}.png")
    svg_path = os.path.join(output_dir, f"{stem}.svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def plot_dumbbell(labels, baseline_values, react_values, title, subtitle, output_dir, stem, x_label="Score (%)", deltas=None):
    configure_plot_style()
    y_positions = list(range(len(labels)))[::-1]
    fig_height = max(4.4, len(labels) * 0.72 + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))

    maximum = max(baseline_values + react_values) if labels else 100.0
    minimum = min(baseline_values + react_values) if labels else 0.0
    x_min = max(0.0, minimum - 8.0)
    x_max = min(100.0, maximum + 14.0)
    if x_max - x_min < 30.0:
        x_min = max(0.0, x_min - 8.0)
        x_max = min(100.0, x_max + 8.0)

    for row_index, (y, baseline_value, react_value) in enumerate(zip(y_positions, baseline_values, react_values)):
        ax.plot(
            [baseline_value, react_value],
            [y, y],
            color="#CBD5E1",
            linewidth=3.0,
            solid_capstyle="round",
            zorder=1,
        )
        ax.scatter(baseline_value, y, s=90, color=BASELINE_COLOR, zorder=3)
        ax.scatter(react_value, y, s=105, color=REACT_COLOR, zorder=4)
        ax.text(
            baseline_value,
            y + 0.20,
            f"{baseline_value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=BASELINE_COLOR,
        )
        ax.text(
            react_value,
            y - 0.20,
            f"{react_value:.1f}",
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            color=REACT_COLOR,
        )
        ax.text(
            x_max - 0.2,
            y,
            f"{(deltas[row_index] if deltas is not None else react_value - baseline_value):+.2f} pp",
            ha="right",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=REACT_COLOR if (deltas[row_index] if deltas is not None else react_value - baseline_value) >= 0 else "#B91C1C",
        )

    ax.set_yticks(y_positions)
    ax.set_ylim(-0.45, len(labels) - 1 + 0.55)
    ax.set_yticklabels(labels)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(x_label)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, loc="left", fontweight="bold", pad=20)
    ax.text(
        0.0,
        1.02,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        color=MUTED_COLOR,
    )
    ax.scatter([], [], s=90, color=BASELINE_COLOR, label=BASELINE_LABEL)
    ax.scatter([], [], s=105, color=REACT_COLOR, label=REACT_LABEL)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.23),
        ncol=2,
        frameon=False,
    )

    return save_figure(fig, output_dir, stem)


def plot_official_metrics(b_summary, r_summary, deltas_by_type, output_dir):
    labels = [label for label, _ in OFFICIAL_METRICS]
    baseline_values = [b_summary["overall"][key] for _, key in OFFICIAL_METRICS]
    react_values = [r_summary["overall"][key] for _, key in OFFICIAL_METRICS]
    return plot_dumbbell(
        labels,
        baseline_values,
        react_values,
        "HotpotQA FullWiki: ReAct improves every official metric",
        "Same 7,405 questions, frozen Qwen2.5-7B reader, hybrid retriever, and BGE page reranker",
        output_dir,
        "official_metrics_comparison",
        deltas=[deltas_by_type["overall"][key] for _, key in OFFICIAL_METRICS],
    )


def plot_question_types(b_summary, r_summary, deltas_by_type, output_dir):
    labels = ["Overall", "Bridge", "Comparison"]
    baseline_values = [b_summary[group]["joint_f1"] for group in ["overall", "bridge", "comparison"]]
    react_values = [r_summary[group]["joint_f1"] for group in ["overall", "bridge", "comparison"]]
    return plot_dumbbell(
        labels,
        baseline_values,
        react_values,
        "Joint F1 gains persist across HotpotQA question types",
        "Percentage-point change in Joint F1; question type comes from the official dev annotations",
        output_dir,
        "joint_f1_by_question_type",
        deltas=[deltas_by_type[group]["joint_f1"] for group in ["overall", "bridge", "comparison"]],
    )


def plot_evidence_coverage(b_evidence, r_evidence, output_dir):
    rows = [
        ("Gold document recall observed", "observed_gold_document_recall"),
        ("Gold supporting-fact recall observed", "observed_gold_supporting_fact_recall"),
        ("Questions with all gold SP observed", "all_gold_supporting_facts_observed"),
        ("Supporting document F1", "supporting_document_f1"),
    ]
    labels = [label for label, _ in rows]
    baseline_values = [b_evidence[key] for _, key in rows]
    react_values = [r_evidence[key] for _, key in rows]
    return plot_dumbbell(
        labels,
        baseline_values,
        react_values,
        "Iterative retrieval exposes more of the gold evidence",
        "Diagnostic evidence-coverage metrics; these are not official HotpotQA leaderboard metrics",
        output_dir,
        "evidence_coverage_comparison",
    )


def plot_paired_transitions(answer_outcomes, joint_outcomes, output_dir):
    configure_plot_style()
    labels = ["Answer EM", "Joint EM"]
    rescued = [answer_outcomes["react_only"], joint_outcomes["react_only"]]
    regressed = [-answer_outcomes["baseline_only"], -joint_outcomes["baseline_only"]]
    nets = [answer_outcomes["net_gain"], joint_outcomes["net_gain"]]
    y_positions = [1, 0]

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.barh(y_positions, regressed, height=0.42, color=REGRESSION_COLOR, label="Baseline correct → ReAct wrong")
    ax.barh(y_positions, rescued, height=0.42, color=RESCUE_COLOR, label="Baseline wrong → ReAct correct")
    ax.axvline(0, color="#94A3B8", linewidth=1.0)

    extent = max(max(rescued), max(abs(v) for v in regressed))
    for y, left_value, right_value, net in zip(y_positions, regressed, rescued, nets):
        ax.text(left_value - extent * 0.025, y, f"{abs(left_value):,}", ha="right", va="center", color=TEXT_COLOR, fontweight="bold")
        ax.text(right_value + extent * 0.025, y, f"{right_value:,}", ha="left", va="center", color=TEXT_COLOR, fontweight="bold")
        ax.text(
            0,
            y + 0.31,
            f"Net +{net:,} correct questions" if net >= 0 else f"Net {net:,} correct questions",
            ha="center",
            va="bottom",
            color=REACT_COLOR if net >= 0 else "#B91C1C",
            fontweight="bold",
            fontsize=9,
        )

    ax.set_yticks(y_positions)
    ax.set_ylim(-0.45, 1.55)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Number of paired questions")
    ax.set_title("ReAct rescues more exact-match failures than it introduces", loc="left", fontweight="bold", pad=20)
    ax.text(
        0.0,
        1.02,
        "Paired by HotpotQA question ID; neutral cases are omitted from the bars",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=MUTED_COLOR,
        fontsize=9.5,
    )
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)
    return save_figure(fig, output_dir, "paired_outcome_transitions")


def plot_quality_cost(b_summary, r_summary, b_latency, r_latency, output_dir):
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(8.6, 5.3))
    points = [
        (b_latency["mean"], b_summary["overall"]["joint_f1"], BASELINE_LABEL, BASELINE_COLOR, -5),
        (r_latency["mean"], r_summary["overall"]["joint_f1"], REACT_LABEL, REACT_COLOR, 5),
    ]
    for latency, joint_f1, label, color, offset in points:
        ax.scatter(latency, joint_f1, s=170, color=color, zorder=3)
        ax.annotate(
            f"{label}\n{joint_f1:.2f} Joint F1  •  {latency:.2f}s mean",
            xy=(latency, joint_f1),
            xytext=(10, 12 + offset),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold" if label == REACT_LABEL else "normal",
            color=color,
        )

    ax.plot(
        [b_latency["mean"], r_latency["mean"]],
        [b_summary["overall"]["joint_f1"], r_summary["overall"]["joint_f1"]],
        color="#CBD5E1",
        linewidth=2,
        zorder=1,
    )
    ax.set_xlabel("Mean latency per question (seconds)")
    ax.set_ylabel("Joint F1 (%)")
    ax.set_title("Quality–latency tradeoff", loc="left", fontweight="bold", pad=20)
    ax.text(
        0.0,
        1.02,
        "ReAct trades additional tool calls for higher end-to-end answer + support quality",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=MUTED_COLOR,
        fontsize=9.5,
    )
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    return save_figure(fig, output_dir, "quality_cost_tradeoff")


def plot_react_by_hops(hop_summary, output_dir):
    configure_plot_style()
    hops = [row["hops"] for row in hop_summary]
    counts = [row["count"] for row in hop_summary]
    react_joint = [row["react_joint_f1"] for row in hop_summary]
    baseline_joint = [row["baseline_joint_f1_same_questions"] for row in hop_summary]

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax2 = ax.twinx()
    ax2.bar(hops, counts, width=0.66, color="#E5E7EB", zorder=0, label="Question count")
    ax.plot(hops, react_joint, marker="o", linewidth=2.5, color=REACT_COLOR, label="ReAct Joint F1")
    ax.plot(hops, baseline_joint, marker="o", linewidth=2.0, linestyle="--", color=BASELINE_COLOR, label="Baseline Joint F1 on same questions")

    ax.set_xlabel("ReAct trajectory length (hops)")
    ax.set_ylabel("Joint F1 (%)")
    ax2.set_ylabel("Questions", color=MUTED_COLOR)
    ax.set_xticks(hops)
    ax.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    ax.set_title("ReAct quality by trajectory length", loc="left", fontweight="bold", pad=20)
    ax.text(
        0.0,
        1.02,
        "Baseline line is recomputed on the exact question subset reaching each ReAct hop count",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=MUTED_COLOR,
        fontsize=9.5,
    )

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.27),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    return save_figure(fig, output_dir, "react_quality_by_hops")


def write_summary_json(
    path,
    validation,
    b_summary,
    r_summary,
    b_evidence,
    r_evidence,
    answer_outcomes,
    joint_outcomes,
    b_latency,
    r_latency,
    b_workload,
    r_workload,
    hop_summary,
    deltas_by_type,
):
    overall_deltas = {
        key: round(deltas_by_type["overall"][key], 2)
        for _, key in OFFICIAL_METRICS
    }
    by_type = {}
    for group in ["overall", "bridge", "comparison"]:
        by_type[group] = {
            "count": r_summary[group]["count"],
            "baseline": b_summary[group],
            "react": r_summary[group],
            "delta_pp": {
                key: round(deltas_by_type[group][key], 2)
                for _, key in OFFICIAL_METRICS
            },
        }

    payload = {
        "schema_version": "1.0",
        "comparison_validation": validation,
        "official_metrics": {
            "baseline": b_summary["overall"],
            "react": r_summary["overall"],
            "delta_pp": overall_deltas,
        },
        "by_question_type": by_type,
        "evidence_diagnostics": {
            "baseline": {key: round(value, 2) for key, value in b_evidence.items()},
            "react": {key: round(value, 2) for key, value in r_evidence.items()},
            "delta_pp": {
                key: round(r_evidence[key] - b_evidence[key], 2) for key in b_evidence
            },
        },
        "paired_outcomes": {
            "answer_exact_match": answer_outcomes,
            "joint_exact_match": joint_outcomes,
        },
        "efficiency": {
            "baseline": {
                "latency_seconds": {key: round(value, 3) for key, value in b_latency.items()},
                "avg_hops": b_summary["overall"]["steps"],
                "reranker": b_workload,
            },
            "react": {
                "latency_seconds": {key: round(value, 3) for key, value in r_latency.items()},
                "avg_hops": r_summary["overall"]["steps"],
                "reranker": r_workload,
            },
            "react_to_baseline_mean_latency_ratio": round(
                r_latency["mean"] / b_latency["mean"], 3
            )
            if b_latency["mean"]
            else None,
        },
        "react_quality_by_hops": hop_summary,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def write_report(
    path,
    figure_paths,
    validation,
    b_summary,
    r_summary,
    b_evidence,
    r_evidence,
    answer_outcomes,
    joint_outcomes,
    b_latency,
    r_latency,
    b_workload,
    r_workload,
    hop_summary,
    deltas_by_type,
    baseline_wall_seconds,
    react_wall_seconds,
):
    def figure(name, alt):
        return f"![{alt}]({os.path.basename(figure_paths[name][1])})"

    overall_delta = deltas_by_type["overall"]["joint_f1"]
    latency_ratio = r_latency["mean"] / b_latency["mean"] if b_latency["mean"] else 0.0

    with open(path, "w", encoding="utf-8") as f:
        f.write("# HotpotQA FullWiki: Final ReAct vs Reranked RAG Comparison\n\n")
        f.write(
            f"Across the same **{validation['question_count']:,} HotpotQA FullWiki dev questions**, "
            f"the ReAct agent improves Joint F1 from **{b_summary['overall']['joint_f1']:.2f}** "
            f"to **{r_summary['overall']['joint_f1']:.2f}** (**{overall_delta:+.2f} percentage points**) "
            f"over a reranked single-pass RAG baseline using the same frozen reader, hybrid retriever, "
            f"and BGE page reranker. The gain comes at **{latency_ratio:.2f}×** mean per-question latency.\n\n"
        )

        f.write("## 1. Experiment Validity\n\n")
        f.write("The comparison script pairs records by HotpotQA question ID and refuses to continue if the runs disagree on question/gold identity or the shared retrieval stack.\n\n")
        f.write("| Shared setting | Value |\n| :--- | :--- |\n")
        for label in [
            "model",
            "dataset source",
            "dataset size",
            "retrieval mode",
            "retriever",
            "concurrency",
            "candidate_k",
            "rrf_k",
            "dense model",
            "dense nprobe",
            "page reranker model",
        ]:
            f.write(f"| {label} | `{validation['shared_settings'][label]}` |\n")
        f.write("\n**Intended experimental difference:** the baseline performs one retrieval and one generation over the top seven reranked pages; ReAct performs adaptive `search` / `lookup` actions with query-local page reranking, sentence-level evidence selection, and bounded persistent snippet memory.\n\n")

        f.write("## 2. Headline Official Metrics\n\n")
        f.write("| Metric | Reranked RAG | ReAct | Gain |\n| :--- | ---: | ---: | ---: |\n")
        for label, key in OFFICIAL_METRICS:
            baseline_value = b_summary["overall"][key]
            react_value = r_summary["overall"][key]
            delta = deltas_by_type["overall"][key]
            f.write(
                f"| {label} | {baseline_value:.2f} | **{react_value:.2f}** | **{delta:+.2f} pp** |\n"
            )
        f.write("\n")
        f.write(figure("official", "Official metric comparison"))
        f.write("\n\n")

        f.write("## 3. Bridge vs Comparison Questions\n\n")
        f.write("| Question type | n | RAG Joint F1 | ReAct Joint F1 | Gain |\n| :--- | ---: | ---: | ---: | ---: |\n")
        for group in ["overall", "bridge", "comparison"]:
            baseline_value = b_summary[group]["joint_f1"]
            react_value = r_summary[group]["joint_f1"]
            delta = deltas_by_type[group]["joint_f1"]
            f.write(
                f"| {group.capitalize()} | {r_summary[group]['count']:,} | {baseline_value:.2f} | **{react_value:.2f}** | **{delta:+.2f} pp** |\n"
            )
        f.write("\n")
        f.write(figure("question_type", "Joint F1 by question type"))
        f.write("\n\n")

        f.write("## 4. Evidence Acquisition Diagnostics\n\n")
        f.write("These are diagnostic retrieval/exposure metrics, not official HotpotQA leaderboard metrics.\n\n")
        evidence_rows = [
            ("Observed gold document recall", "observed_gold_document_recall"),
            ("Observed gold supporting-fact recall", "observed_gold_supporting_fact_recall"),
            ("Questions with all gold SP observed", "all_gold_supporting_facts_observed"),
            ("Supporting document F1", "supporting_document_f1"),
        ]
        f.write("| Diagnostic | Reranked RAG | ReAct | Gain |\n| :--- | ---: | ---: | ---: |\n")
        for label, key in evidence_rows:
            baseline_value = b_evidence[key]
            react_value = r_evidence[key]
            f.write(
                f"| {label} | {baseline_value:.2f} | **{react_value:.2f}** | **{react_value - baseline_value:+.2f} pp** |\n"
            )
        f.write("\n")
        f.write(figure("evidence", "Evidence coverage comparison"))
        f.write("\n\n")

        f.write("## 5. Paired Exact-Match Outcomes\n\n")
        f.write("Because both systems answer the same questions, exact-match transitions show how often ReAct rescues or breaks a baseline outcome.\n\n")
        f.write("| Outcome | Answer EM | Joint EM |\n| :--- | ---: | ---: |\n")
        f.write(f"| Baseline wrong → ReAct correct | **{answer_outcomes['react_only']:,}** | **{joint_outcomes['react_only']:,}** |\n")
        f.write(f"| Baseline correct → ReAct wrong | {answer_outcomes['baseline_only']:,} | {joint_outcomes['baseline_only']:,} |\n")
        f.write(f"| Net additional correct | **+{answer_outcomes['net_gain']:,}** | **+{joint_outcomes['net_gain']:,}** |\n")
        f.write("\n")
        f.write(figure("transitions", "Paired outcome transitions"))
        f.write("\n\n")

        f.write("## 6. Efficiency / Quality Tradeoff\n\n")
        f.write("| Measure | Reranked RAG | ReAct |\n| :--- | ---: | ---: |\n")
        f.write(f"| Mean latency | {b_latency['mean']:.2f}s | {r_latency['mean']:.2f}s |\n")
        f.write(f"| Median latency | {b_latency['median']:.2f}s | {r_latency['median']:.2f}s |\n")
        f.write(f"| P90 latency | {b_latency['p90']:.2f}s | {r_latency['p90']:.2f}s |\n")
        f.write(f"| P95 latency | {b_latency['p95']:.2f}s | {r_latency['p95']:.2f}s |\n")
        f.write(f"| Average hops | {b_summary['overall']['steps']:.2f} | {r_summary['overall']['steps']:.2f} |\n")
        f.write(f"| Cross-encoder pairs / question | {b_workload['pairs_per_question']:.1f} | {r_workload['pairs_per_question']:.1f} |\n")
        f.write(f"| Total cross-encoder pairs | {b_workload['total_pairs_scored']:,} | {r_workload['total_pairs_scored']:,} |\n")
        if baseline_wall_seconds and react_wall_seconds:
            f.write(f"| Total evaluation wall time | {baseline_wall_seconds / 60.0:.2f} min | {react_wall_seconds / 60.0:.2f} min |\n")
            f.write(f"| Wall throughput | {validation['question_count'] / baseline_wall_seconds:.2f} q/s | {validation['question_count'] / react_wall_seconds:.2f} q/s |\n")
        f.write("\n")
        f.write(
            f"ReAct gains **{overall_delta:+.2f} Joint-F1 points** at **{latency_ratio:.2f}×** the mean per-question latency of the single-pass baseline.\n\n"
        )
        f.write(figure("quality_cost", "Quality latency tradeoff"))
        f.write("\n\n")

        f.write("## 7. ReAct Quality by Trajectory Length\n\n")
        f.write("Trajectory length is endogenous: questions reaching many hops are the unresolved/difficult tail, so this is a diagnostic rather than a causal comparison. The baseline column is recomputed on the exact questions in each ReAct hop-count bucket.\n\n")
        f.write("| ReAct hops | Questions | ReAct Joint F1 | Baseline Joint F1 on same questions |\n| ---: | ---: | ---: | ---: |\n")
        for row in hop_summary:
            f.write(
                f"| {row['hops']} | {row['count']:,} | {row['react_joint_f1']:.2f} | {row['baseline_joint_f1_same_questions']:.2f} |\n"
            )
        f.write("\n")
        f.write(figure("hops", "ReAct quality by hops"))
        f.write("\n\n")

        f.write("## 8. Interpretation\n\n")
        f.write(
            f"- ReAct improves **Joint F1 by {overall_delta:.2f} percentage points** overall while improving every official answer/support metric.\n"
        )
        f.write(
            f"- Joint-F1 gains are similar for bridge (**{deltas_by_type['bridge']['joint_f1']:+.2f} pp**) and comparison (**{deltas_by_type['comparison']['joint_f1']:+.2f} pp**) questions, so the improvement is not confined to one HotpotQA subtype.\n"
        )
        f.write(
            f"- Complete gold-support exposure increases by **{r_evidence['all_gold_supporting_facts_observed'] - b_evidence['all_gold_supporting_facts_observed']:+.2f} percentage points**, consistent with adaptive retrieval finding useful evidence beyond one-shot retrieval.\n"
        )
        f.write(
            f"- ReAct rescues **{answer_outcomes['react_only']:,}** baseline Answer-EM failures while regressing on **{answer_outcomes['baseline_only']:,}**, a net gain of **{answer_outcomes['net_gain']:,}** exactly-correct answers.\n"
        )
        f.write(
            f"- The quality gain has a clear systems cost: mean latency rises from **{b_latency['mean']:.2f}s** to **{r_latency['mean']:.2f}s** per question.\n"
        )


def publish_artifacts(output_dir, publish_dir, figure_paths, report_path, summary_path):
    if not publish_dir:
        return []
    os.makedirs(publish_dir, exist_ok=True)
    published = []
    for _, svg_path in figure_paths.values():
        destination = os.path.join(publish_dir, os.path.basename(svg_path))
        shutil.copy2(svg_path, destination)
        published.append(destination)
    for source in [report_path, summary_path]:
        destination = os.path.join(publish_dir, os.path.basename(source))
        shutil.copy2(source, destination)
        published.append(destination)
    return published


def compare_results(
    baseline_json_path,
    react_json_path,
    output_dir="eval_results/comparison",
    publish_dir="docs/results",
):
    logger, log_file = setup_logger(output_dir)

    for label, path in [("Baseline", baseline_json_path), ("ReAct", react_json_path)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} results file not found: {path}")

    baseline_manifest_path = manifest_path_for_results(baseline_json_path)
    react_manifest_path = manifest_path_for_results(react_json_path)
    for label, path in [("Baseline", baseline_manifest_path), ("ReAct", react_manifest_path)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} run manifest not found beside results.json: {path}")

    baseline_data = load_json(baseline_json_path)
    react_data = load_json(react_json_path)
    baseline_manifest = load_json(baseline_manifest_path)
    react_manifest = load_json(react_manifest_path)

    baseline_by_id, react_by_id = validate_record_alignment(baseline_data, react_data)
    shared_manifest_settings = validate_manifests(baseline_manifest, react_manifest)
    protocol_contracts = validate_protocol_contracts(baseline_manifest, react_manifest)

    question_count = len(baseline_by_id)
    if question_count != int(baseline_manifest.get("dataset_size") or 0):
        raise ComparisonValidationError(
            f"Paired result count ({question_count}) does not equal manifest dataset_size "
            f"({baseline_manifest.get('dataset_size')})."
        )

    validation = {
        "status": "PASSED",
        "question_count": question_count,
        "question_id_sets_identical": True,
        "question_and_gold_identity_verified": True,
        "failed_records": 0,
        "shared_settings": shared_manifest_settings,
        "baseline_runner": baseline_manifest.get("runner"),
        "react_runner": react_manifest.get("runner"),
        "protocol_contracts": protocol_contracts,
    }

    b_summary = compute_metrics_by_type(baseline_data)
    r_summary = compute_metrics_by_type(react_data)
    b_evidence = evidence_summary(baseline_data)
    r_evidence = evidence_summary(react_data)
    answer_outcomes = paired_outcomes(baseline_by_id, react_by_id, "exact_match")
    joint_outcomes = paired_outcomes(baseline_by_id, react_by_id, "joint_em")
    b_latency = latency_summary(baseline_data)
    r_latency = latency_summary(react_data)
    b_workload = reranker_workload(baseline_manifest, question_count)
    r_workload = reranker_workload(react_manifest, question_count)
    hop_summary = react_hop_summary(baseline_by_id, react_by_id)
    deltas_by_type = official_deltas_by_type(baseline_data, react_data)

    print_segmented_report(b_summary, model_name=BASELINE_LABEL)
    print_segmented_report(r_summary, model_name=REACT_LABEL)

    comparison_info = (
        "=== COMPARABILITY CHECKS: PASSED ===\n"
        f"Matched questions: {question_count:,} / {question_count:,}\n"
        f"Dataset: {shared_manifest_settings['dataset source']}\n"
        f"Model: {shared_manifest_settings['model']}\n"
        f"Hybrid retrieval: candidate_k={shared_manifest_settings['candidate_k']}, "
        f"rrf_k={shared_manifest_settings['rrf_k']}\n"
        f"Page reranker: {shared_manifest_settings['page reranker model']}\n"
        f"Concurrency: {shared_manifest_settings['concurrency']}\n"
    )
    print(comparison_info)
    logger.info(comparison_info)

    figure_paths = {
        "official": plot_official_metrics(b_summary, r_summary, deltas_by_type, output_dir),
        "question_type": plot_question_types(b_summary, r_summary, deltas_by_type, output_dir),
        "evidence": plot_evidence_coverage(b_evidence, r_evidence, output_dir),
        "transitions": plot_paired_transitions(answer_outcomes, joint_outcomes, output_dir),
        "quality_cost": plot_quality_cost(b_summary, r_summary, b_latency, r_latency, output_dir),
        "hops": plot_react_by_hops(hop_summary, output_dir),
    }

    summary_path = os.path.join(output_dir, "comparison_summary.json")
    summary_payload = write_summary_json(
        summary_path,
        validation,
        b_summary,
        r_summary,
        b_evidence,
        r_evidence,
        answer_outcomes,
        joint_outcomes,
        b_latency,
        r_latency,
        b_workload,
        r_workload,
        hop_summary,
        deltas_by_type,
    )
    baseline_wall = float(baseline_manifest.get("total_evaluation_seconds") or 0.0)
    react_wall = float(react_manifest.get("total_evaluation_seconds") or 0.0)
    summary_payload["efficiency"]["baseline"]["total_evaluation_seconds"] = round(baseline_wall, 3)
    summary_payload["efficiency"]["react"]["total_evaluation_seconds"] = round(react_wall, 3)
    summary_payload["efficiency"]["baseline"]["throughput_questions_per_second"] = round(question_count / baseline_wall, 3) if baseline_wall else None
    summary_payload["efficiency"]["react"]["throughput_questions_per_second"] = round(question_count / react_wall, 3) if react_wall else None
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    report_path = os.path.join(output_dir, "comparison_report.md")
    write_report(
        report_path,
        figure_paths,
        validation,
        b_summary,
        r_summary,
        b_evidence,
        r_evidence,
        answer_outcomes,
        joint_outcomes,
        b_latency,
        r_latency,
        b_workload,
        r_workload,
        hop_summary,
        deltas_by_type,
        baseline_wall,
        react_wall,
    )

    published = publish_artifacts(
        output_dir,
        publish_dir,
        figure_paths,
        report_path,
        summary_path,
    )

    headline = (
        f"Final Joint F1: {b_summary['overall']['joint_f1']:.2f} -> "
        f"{r_summary['overall']['joint_f1']:.2f} "
        f"({deltas_by_type['overall']['joint_f1']:+.2f} percentage points)\n"
        f"Mean latency: {b_latency['mean']:.2f}s -> {r_latency['mean']:.2f}s "
        f"({summary_payload['efficiency']['react_to_baseline_mean_latency_ratio']:.2f}x)\n"
        f"Answer-EM rescues/regressions: {answer_outcomes['react_only']:,} / "
        f"{answer_outcomes['baseline_only']:,}\n"
    )
    print(headline)
    logger.info(headline)

    print(f"Generated final comparison artifacts in: {output_dir}/")
    if published:
        print(f"Published SVG/report artifacts to: {publish_dir}/")
    print(f"Saved comparison log file to: {log_file}")
    return summary_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate and compare the final reranked RAG and ReAct HotpotQA runs"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="eval_results/baseline/results.json",
        help="Path to reranked single-pass baseline results.json",
    )
    parser.add_argument(
        "--react",
        type=str,
        default="eval_results/react/results.json",
        help="Path to ReAct results.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results/comparison",
        help="Full post-processing output directory",
    )
    parser.add_argument(
        "--publish-dir",
        type=str,
        default="docs/results",
        help="Trackable directory receiving SVGs, report, and summary JSON; use an empty string to disable",
    )

    args = parser.parse_args()
    compare_results(
        args.baseline,
        args.react,
        output_dir=args.output_dir,
        publish_dir=args.publish_dir or None,
    )
