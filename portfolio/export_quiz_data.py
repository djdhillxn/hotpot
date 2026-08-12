#!/usr/bin/env python3
"""Build the single compact JSON artifact used by the HotpotQA portfolio quiz.

The evaluator intentionally keeps exhaustive trajectories for research. This
exporter removes retrieval internals, duplicated raw model output, diagnostics,
and infrastructure telemetry while retaining every question, both answers, all
six official scores, supporting facts, visited pages, and the complete readable
Thought/Action/Observation trace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SCHEMA_VERSION = 1


def _as_number(value):
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def _first(record, *keys, default=None):
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def _clip_observation(value, max_chars):
    text = str(value or "").strip()
    if not text:
        return text, False
    if max_chars is None:
        return text, False
    if max_chars <= 0:
        return "", True
    if len(text) <= max_chars:
        return text, False

    marker = "\n… observation shortened for the portfolio …\n"
    available = max_chars - len(marker)
    if available <= 40:
        if max_chars == 1:
            return "…", True
        return text[: max_chars - 1].rstrip() + "…", True

    head = max(1, int(available * 0.68))
    tail = max(1, available - head)
    return f"{text[:head].rstrip()}{marker}{text[-tail:].lstrip()}", True


def _compact_step(step, max_observation_chars):
    observation, shortened = _clip_observation(
        step.get("observation", ""), max_observation_chars
    )
    compact = {
        "n": int(_first(step, "step", "n", default=0) or 0),
        "thought": str(step.get("thought") or "").strip(),
        "action": str(step.get("action") or "").strip(),
        "type": str(_first(step, "action_type", "type", default="") or "").strip(),
        "observation": observation,
    }
    if shortened:
        compact["observation_shortened"] = True
    return compact


def _compact_record(record, max_observation_chars):
    exact_match = _as_bool(_first(record, "exact_match"))
    return {
        "id": str(record.get("id") or record.get("idx") or ""),
        "question": str(record.get("question") or "").strip(),
        "type": str(_first(record, "question_type", "type", default="unknown")),
        "level": str(_first(record, "difficulty_level", "level", default="unknown")),
        "gold_answer": str(
            _first(record, "ground_truth", "gold_answer", default="") or ""
        ).strip(),
        "agent_answer": str(
            _first(record, "predicted_answer", "pred_answer", default="No Answer")
            or "No Answer"
        ).strip(),
        "scores": {
            "answer_em": exact_match,
            "answer_f1": _as_number(_first(record, "answer_f1", "f1")),
            "supporting_fact_em": _as_bool(
                _first(record, "supporting_fact_em", "sp_em")
            ),
            "supporting_fact_f1": _as_number(
                _first(record, "supporting_fact_f1", "sp_f1")
            ),
            "joint_em": _as_bool(record.get("joint_em")),
            "joint_f1": _as_number(record.get("joint_f1")),
        },
        "tool_steps": int(record.get("step_count") or 0),
        "gold_supporting_facts": record.get("gold_supporting_facts") or [],
        "agent_supporting_facts": (
            record.get("predicted_supporting_facts") or []
        ),
        "visited_pages": record.get("visited_pages") or [],
        "steps": [
            _compact_step(step, max_observation_chars)
            for step in (record.get("steps") or [])
        ],
    }


def _summary_from_records(records):
    metric_keys = {
        "answer_em": lambda item: (
            float(item["scores"]["answer_em"])
            if item["scores"]["answer_em"] is not None
            else None
        ),
        "answer_f1": lambda item: item["scores"]["answer_f1"],
        "supporting_fact_em": lambda item: (
            float(item["scores"]["supporting_fact_em"])
            if item["scores"]["supporting_fact_em"] is not None
            else None
        ),
        "supporting_fact_f1": lambda item: item["scores"]["supporting_fact_f1"],
        "joint_em": lambda item: (
            float(item["scores"]["joint_em"])
            if item["scores"]["joint_em"] is not None
            else None
        ),
        "joint_f1": lambda item: item["scores"]["joint_f1"],
    }
    summary = {"count": len(records)}
    for output_key, getter in metric_keys.items():
        values = [getter(record) for record in records]
        values = [value for value in values if value is not None]
        summary[output_key] = (
            round(sum(values) / len(values) * 100.0, 2) if values else None
        )
    return summary


def _summary_from_payload(metadata, records):
    source = metadata.get("metrics_summary") or metadata.get("metrics") or {}
    source = source.get("overall", source) if isinstance(source, dict) else {}
    if not source:
        return _summary_from_records(records)

    fallback = _summary_from_records(records)
    aliases = {
        "answer_em": ("answer_em", "em"),
        "answer_f1": ("answer_f1", "f1"),
        "supporting_fact_em": ("supporting_fact_em", "sp_em"),
        "supporting_fact_f1": ("supporting_fact_f1", "sp_f1"),
        "joint_em": ("joint_em",),
        "joint_f1": ("joint_f1",),
    }
    summary = {"count": int(source.get("count") or len(records))}
    for output_key, keys in aliases.items():
        value = _as_number(_first(source, *keys))
        summary[output_key] = fallback[output_key] if value is None else value
    return summary


def build_quiz_payload(payload, max_observation_chars=None, limit=None):
    if isinstance(payload, list):
        metadata = {}
        raw_records = payload
    elif isinstance(payload, dict):
        metadata = payload.get("metadata") or {}
        raw_records = (
            payload.get("trajectories")
            or payload.get("examples")
            or payload.get("records")
            or []
        )
    else:
        raise ValueError("Input must be a trajectory array or an object containing trajectories.")

    if limit is not None:
        raw_records = raw_records[:limit]

    records = [
        _compact_record(record, max_observation_chars)
        for record in raw_records
        if record.get("question")
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "demo": False,
        "source": {
            "dataset": "HotpotQA",
            "setting": "fullwiki",
            "split": "validation",
            "description": metadata.get("description")
            or "Compact portfolio export from a HotpotQA evaluation run.",
        },
        "metrics": _summary_from_payload(metadata, records),
        "examples": records,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a compact HotpotQA quiz artifact from evaluator trajectories."
    )
    parser.add_argument("input", type=Path, help="trajectories.json or portfolio_trajectories.json")
    parser.add_argument("output", type=Path, help="Destination quiz JSON path")
    parser.add_argument(
        "--observation-chars",
        type=int,
        default=None,
        help="Optional maximum characters per observation; the default retains full observations",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of examples to export; the default exports all examples",
    )
    parser.add_argument(
        "--pretty", action="store_true", help="Indent JSON for inspection (larger file)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.observation_chars is not None and args.observation_chars < 0:
        raise SystemExit("--observation-chars must be non-negative")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    with args.input.open(encoding="utf-8") as handle:
        source_payload = json.load(handle)

    quiz_payload = build_quiz_payload(
        source_payload,
        max_observation_chars=args.observation_chars,
        limit=args.limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(
            quiz_payload,
            handle,
            ensure_ascii=False,
            indent=2 if args.pretty else None,
            separators=None if args.pretty else (",", ":"),
        )
        handle.write("\n")

    print(
        f"Exported {len(quiz_payload['examples'])} examples to {args.output} "
        f"(schema v{SCHEMA_VERSION})."
    )


if __name__ == "__main__":
    main()
