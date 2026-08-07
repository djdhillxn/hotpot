import re
import string
from collections import Counter

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
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def supporting_facts_overlap(visited_pages, gold_titles):
    pred_set = set(normalize_answer(p) for p in visited_pages if p)
    gold_set = set(normalize_answer(t) for t in gold_titles if t)

    if not gold_set:
        return 1.0, 1.0, 1.0
    if not pred_set:
        return 0.0, 0.0, 0.0

    intersection = pred_set & gold_set
    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(gold_set)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1


def evaluate_prediction(prediction, ground_truth, visited_pages, gold_titles, step_count=0):
    em = exact_match_score(prediction, ground_truth)
    p, r, f1 = f1_score(prediction, ground_truth)
    sp_p, sp_r, sp_f1 = supporting_facts_overlap(visited_pages, gold_titles)
    sp_em = (sp_r == 1.0)

    joint_em = em and sp_em
    joint_f1 = f1 * sp_f1

    return {
        "exact_match": em,
        "f1": f1,
        "precision": p,
        "recall": r,
        "sp_em": sp_em,
        "sp_precision": sp_p,
        "sp_recall": sp_r,
        "sp_f1": sp_f1,
        "joint_em": joint_em,
        "joint_f1": joint_f1,
        "step_count": step_count,
    }
