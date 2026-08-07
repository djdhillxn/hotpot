import json
import requests
from config import HOTPOT_DEV_URL

SAMPLE_HOTPOT_QUESTIONS = [
    {
        "id": "sample_1",
        "question": "Were Scott Derrickson and Ed Wood born in the same state?",
        "answer": "no",
        "type": "comparison",
        "level": "medium",
        "supporting_facts": [["Scott Derrickson", 0], ["Ed Wood", 0]],
        "context": [
            {
                "title": "Scott Derrickson",
                "sentences": [
                    "Scott Derrickson (born July 16, 1966) is an American filmmaker.",
                    "He was born in Denver, Colorado."
                ]
            },
            {
                "title": "Ed Wood",
                "sentences": [
                    "Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker.",
                    "He was born in Poughkeepsie, New York."
                ]
            }
        ]
    },
    {
        "id": "sample_2",
        "question": "What is the birth date of the lead singer of Radiohead?",
        "answer": "7 October 1968",
        "type": "bridge",
        "level": "medium",
        "supporting_facts": [["Radiohead", 0], ["Thom Yorke", 0]],
        "context": [
            {
                "title": "Radiohead",
                "sentences": [
                    "Radiohead are an English rock band formed in Abingdon, Oxfordshire, in 1985.",
                    "The band consists of Thom Yorke (lead vocals, guitar), Jonny Greenwood, Colin Greenwood, Ed O'Brien, and Philip Selway."
                ]
            },
            {
                "title": "Thom Yorke",
                "sentences": [
                    "Thomas Edward Yorke (born 7 October 1968) is an English musician and main vocalist of the alternative rock band Radiohead."
                ]
            }
        ]
    },
    {
        "id": "sample_3",
        "question": "Which film directed by Christopher Nolan features the song 'Non, je ne regrette rien'?",
        "answer": "Inception",
        "type": "bridge",
        "level": "medium",
        "supporting_facts": [["Inception", 0], ["Non, je ne regrette rien", 0]],
        "context": [
            {
                "title": "Inception",
                "sentences": [
                    "Inception is a 2010 science fiction action film written and directed by Christopher Nolan.",
                    "The film stars Leonardo DiCaprio as a professional thief who steals information by infiltrating the subconscious of his targets.",
                    "The song 'Non, je ne regrette rien' by Édith Piaf plays a crucial role as a countdown signal for kicks throughout the film."
                ]
            },
            {
                "title": "Non, je ne regrette rien",
                "sentences": [
                    "Non, je ne regrette rien (meaning 'No, I regret nothing') is a French song composed by Charles Dumont.",
                    "It was recorded by Édith Piaf in 1960 and prominently featured in the 2010 movie Inception directed by Christopher Nolan."
                ]
            }
        ]
    },
    {
        "id": "sample_4",
        "question": "Are the directors of 'Avatar' and 'Titanic' the same person?",
        "answer": "yes",
        "type": "comparison",
        "level": "easy",
        "supporting_facts": [["Avatar (2009 film)", 0], ["Titanic (1997 film)", 0]],
        "context": [
            {
                "title": "Avatar (2009 film)",
                "sentences": [
                    "Avatar is a 2009 American epic science fiction film directed, written, produced, and co-edited by James Cameron."
                ]
            },
            {
                "title": "Titanic (1997 film)",
                "sentences": [
                    "Titanic is a 1997 American epic romance and disaster film directed, written, produced, and co-edited by James Cameron."
                ]
            }
        ]
    }
]


def load_hotpot_dataset(num_samples=None, source="sample"):
    if source == "sample":
        if num_samples:
            return SAMPLE_HOTPOT_QUESTIONS[:num_samples]
        return SAMPLE_HOTPOT_QUESTIONS

    if source == "huggingface":
        try:
            from datasets import load_dataset
            dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")
            samples = []
            for i, item in enumerate(dataset):
                if num_samples and i >= num_samples:
                    break
                gold_titles = list(set([fact[0] for fact in item.get("supporting_facts", [])]))
                samples.append({
                    "id": item.get("id", f"hf_{i}"),
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "type": item.get("type", "unknown"),
                    "level": item.get("level", "unknown"),
                    "supporting_facts": item.get("supporting_facts", []),
                    "gold_titles": gold_titles,
                    "context": item.get("context", []),
                })
            return samples
        except Exception as e:
            print(f"Warning: Could not load HuggingFace dataset ({str(e)}). Falling back to sample questions.")
            return SAMPLE_HOTPOT_QUESTIONS[:num_samples] if num_samples else SAMPLE_HOTPOT_QUESTIONS

    if source == "official_json":
        try:
            resp = requests.get(HOTPOT_DEV_URL, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if num_samples:
                data = data[:num_samples]
            return data
        except Exception as e:
            print(f"Warning: Could not download official JSON ({str(e)}). Falling back to sample questions.")
            return SAMPLE_HOTPOT_QUESTIONS[:num_samples] if num_samples else SAMPLE_HOTPOT_QUESTIONS

    return SAMPLE_HOTPOT_QUESTIONS[:num_samples] if num_samples else SAMPLE_HOTPOT_QUESTIONS
