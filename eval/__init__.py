from .dataset import load_hotpot_dataset, SAMPLE_HOTPOT_QUESTIONS
from .metrics import evaluate_prediction, exact_match_score, f1_score, supporting_facts_overlap

# Alias for backwards compatibility
load_hotpot_samples = load_hotpot_dataset

__all__ = [
    "load_hotpot_dataset",
    "load_hotpot_samples",
    "SAMPLE_HOTPOT_QUESTIONS",
    "evaluate_prediction",
    "exact_match_score",
    "f1_score",
    "supporting_facts_overlap",
]
