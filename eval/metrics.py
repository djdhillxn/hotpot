import re
import string
from collections import Counter


_CATEGORICAL_ANSWERS = {"yes", "no", "noanswer"}


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction, ground_truth):
    """HotpotQA answer precision/recall/F1, including yes/no/noanswer handling."""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    if (
        normalized_prediction in _CATEGORICAL_ANSWERS
        or normalized_ground_truth in _CATEGORICAL_ANSWERS
    ) and normalized_prediction != normalized_ground_truth:
        return 0.0, 0.0, 0.0

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def supporting_fact_score(prediction, ground_truth):
    """Official HotpotQA supporting-fact score over exact (title, sentence_id) pairs."""
    pred_set = set(_valid_supporting_fact_tuples(prediction))
    gold_set = set(_valid_supporting_fact_tuples(ground_truth))

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    return precision, recall, f1, em


def supporting_document_score(visited_pages, gold_titles):
    """Diagnostic page-level retrieval score. This is NOT an official HotpotQA metric."""
    pred_set = set(normalize_answer(p) for p in visited_pages if p)
    gold_set = set(normalize_answer(t) for t in gold_titles if t)

    if not gold_set:
        return 1.0, 1.0, 1.0
    if not pred_set:
        return 0.0, 0.0, 0.0

    intersection = pred_set & gold_set
    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(gold_set)
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def supporting_facts_overlap(visited_pages, gold_titles):
    """Backward-compatible alias for the old page-level diagnostic."""
    return supporting_document_score(visited_pages, gold_titles)


def _valid_supporting_fact_tuples(facts):
    normalized = []
    for fact in facts or []:
        if not isinstance(fact, (list, tuple)) or len(fact) != 2:
            continue
        title, sent_id = fact
        if not isinstance(title, str):
            continue
        try:
            sent_id = int(sent_id)
        except (TypeError, ValueError):
            continue
        normalized.append((title, sent_id))
    return normalized


def evaluate_prediction(
    prediction,
    ground_truth,
    predicted_supporting_facts=None,
    gold_supporting_facts=None,
    visited_pages=None,
    gold_titles=None,
    step_count=0,
):
    """Evaluate one prediction with the official HotpotQA answer/SP/joint formulas.

    Page-title overlap is retained separately as a retrieval diagnostic and never mixed
    into the official supporting-fact or joint metrics.
    """
    em = exact_match_score(prediction, ground_truth)
    p, r, f1 = f1_score(prediction, ground_truth)

    sp_p, sp_r, sp_f1, sp_em = supporting_fact_score(
        predicted_supporting_facts or [], gold_supporting_facts or []
    )

    joint_precision = p * sp_p
    joint_recall = r * sp_r
    joint_f1 = (
        2 * joint_precision * joint_recall / (joint_precision + joint_recall)
        if joint_precision + joint_recall > 0
        else 0.0
    )
    joint_em = bool(em) and bool(sp_em)

    doc_p, doc_r, doc_f1 = supporting_document_score(
        visited_pages or [], gold_titles or []
    )

    return {
        "exact_match": bool(em),
        "f1": f1,
        "precision": p,
        "recall": r,
        "sp_em": bool(sp_em),
        "sp_precision": sp_p,
        "sp_recall": sp_r,
        "sp_f1": sp_f1,
        "joint_em": joint_em,
        "joint_precision": joint_precision,
        "joint_recall": joint_recall,
        "joint_f1": joint_f1,
        "doc_precision": doc_p,
        "doc_recall": doc_r,
        "doc_f1": doc_f1,
        "step_count": step_count,
    }


def compute_metrics_by_type(records):
    """Computes full HotpotQA metrics overall and segmented by question type ('bridge' vs 'comparison')."""
    groups = {
        "overall": [],
        "bridge": [],
        "comparison": [],
    }

    for item in records or []:
        q_type = str(item.get("question_type") or item.get("type") or "unknown").lower()

        em = float(item.get("exact_match", 0))
        f1 = float(item.get("f1", item.get("answer_f1", 0)))
        sp_em = float(item.get("sp_em", item.get("supporting_fact_em", 0)))
        sp_f1 = float(item.get("sp_f1", item.get("supporting_fact_f1", 0)))
        joint_em = float(item.get("joint_em", 0))
        joint_f1 = float(item.get("joint_f1", 0))
        steps = float(item.get("step_count", 1))
        lat = float(item.get("latency", item.get("latency_seconds", 0)))

        entry = {
            "em": em,
            "f1": f1,
            "sp_em": sp_em,
            "sp_f1": sp_f1,
            "joint_em": joint_em,
            "joint_f1": joint_f1,
            "steps": steps,
            "latency": lat,
        }

        groups["overall"].append(entry)
        if q_type in groups:
            groups[q_type].append(entry)

    summary = {}
    for group_name, entries in groups.items():
        count = len(entries)
        if count == 0:
            summary[group_name] = {
                "count": 0,
                "em": 0.0,
                "f1": 0.0,
                "sp_em": 0.0,
                "sp_f1": 0.0,
                "joint_em": 0.0,
                "joint_f1": 0.0,
                "steps": 0.0,
                "latency": 0.0,
            }
            continue

        summary[group_name] = {
            "count": count,
            "em": round(sum(e["em"] for e in entries) / count * 100.0, 2),
            "f1": round(sum(e["f1"] for e in entries) / count * 100.0, 2),
            "sp_em": round(sum(e["sp_em"] for e in entries) / count * 100.0, 2),
            "sp_f1": round(sum(e["sp_f1"] for e in entries) / count * 100.0, 2),
            "joint_em": round(sum(e["joint_em"] for e in entries) / count * 100.0, 2),
            "joint_f1": round(sum(e["joint_f1"] for e in entries) / count * 100.0, 2),
            "steps": round(sum(e["steps"] for e in entries) / count, 2),
            "latency": round(sum(e["latency"] for e in entries) / count, 2),
        }

    return summary


def print_segmented_report(summary, model_name="Evaluation Run"):
    """Print clean ASCII summary table broken down by HotpotQA question type."""
    print(f"\n{'='*75}")
    print(f"  HotpotQA Segmented Performance Report: {model_name}")
    print(f"{'='*75}")
    print(f"{'Type':<12} | {'Count':<6} | {'Ans EM':<8} | {'Ans F1':<8} | {'SP F1':<8} | {'Joint EM':<9} | {'Joint F1':<8}")
    print(f"{'-'*12}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}-+-{'-'*8}")

    for group_name in ["overall", "bridge", "comparison"]:
        data = summary.get(group_name, {"count": 0, "em": 0.0, "f1": 0.0, "sp_f1": 0.0, "joint_em": 0.0, "joint_f1": 0.0})
        label = group_name.capitalize()
        print(
            f"{label:<12} | {data['count']:<6} | {data['em']:<7.2f}% | {data['f1']:<7.2f}% | "
            f"{data['sp_f1']:<7.2f}% | {data['joint_em']:<8.2f}% | {data['joint_f1']:<7.2f}%"
        )

    print(f"{'='*75}\n")

