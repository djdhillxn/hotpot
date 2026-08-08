import json
import os


def context_diagnostics(sample):
    context = sample.get("context", []) or []
    context_map = {}

    for paragraph in context:
        if isinstance(paragraph, dict):
            title = paragraph.get("title", "")
            sentences = paragraph.get("sentences", [])
        elif isinstance(paragraph, (list, tuple)) and len(paragraph) >= 2:
            title = paragraph[0]
            sentences = paragraph[1]
        else:
            continue

        if title:
            context_map[str(title)] = list(sentences) if isinstance(sentences, list) else [str(sentences)]

    gold_facts = []
    for fact in sample.get("supporting_facts", []) or []:
        if not isinstance(fact, (list, tuple)) or len(fact) != 2:
            continue
        try:
            gold_facts.append((str(fact[0]), int(fact[1])))
        except (TypeError, ValueError):
            continue

    gold_titles = {title for title, _ in gold_facts}
    context_titles = list(context_map.keys())
    available_titles = gold_titles & set(context_titles)
    available_facts = {
        (title, sent_id)
        for title, sent_id in gold_facts
        if title in context_map and 0 <= sent_id < len(context_map[title])
    }

    doc_recall = len(available_titles) / len(gold_titles) if gold_titles else 1.0
    fact_recall = len(available_facts) / len(set(gold_facts)) if gold_facts else 1.0

    return {
        "context_titles": context_titles,
        "gold_titles_in_context": sorted(available_titles),
        "gold_document_recall_in_context": doc_recall,
        "gold_supporting_fact_recall_in_context": fact_recall,
        "all_gold_supporting_facts_available": fact_recall == 1.0,
    }


def write_official_files(samples, trajectories, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    prediction = {"answer": {}, "sp": {}}
    for trajectory in trajectories:
        qid = str(trajectory["id"])
        prediction["answer"][qid] = trajectory.get("predicted_answer", "No Answer")
        prediction["sp"][qid] = trajectory.get("predicted_supporting_facts", [])

    prediction_path = os.path.join(output_dir, "official_predictions.json")
    with open(prediction_path, "w") as f:
        json.dump(prediction, f, indent=2, ensure_ascii=False)

    gold = []
    for sample in samples:
        gold.append(
            {
                "_id": str(sample["id"]),
                "answer": sample.get("answer", ""),
                "supporting_facts": sample.get("supporting_facts", []),
            }
        )

    gold_path = os.path.join(output_dir, "official_gold.json")
    with open(gold_path, "w") as f:
        json.dump(gold, f, indent=2, ensure_ascii=False)

    return prediction_path, gold_path


def write_run_manifest(output_dir, manifest):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "run_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return path
