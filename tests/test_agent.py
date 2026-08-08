import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agent.baseline_rag import run_single_pass_rag
from agent.engine import run_react_agent
from agent.parser import parse_baseline_output, parse_react_output, parse_supporting_facts
from eval.artifacts import context_diagnostics, write_official_files
from eval.metrics import (
    evaluate_prediction,
    exact_match_score,
    f1_score,
    supporting_document_score,
    supporting_fact_score,
)
from tools.local_retriever import LocalHotpotRetriever


def test_parse_react_output_standard():
    llm_output = """Thought: I need to search for Scott Derrickson to find his birthplace.
Action: search[Scott Derrickson]"""

    thought, raw_action, action_type, action_arg = parse_react_output(llm_output)

    assert thought == "I need to search for Scott Derrickson to find his birthplace."
    assert raw_action == "search[Scott Derrickson]"
    assert action_type == "search"
    assert action_arg == "Scott Derrickson"


def test_parse_react_output_preserves_parentheses_inside_brackets():
    llm_output = """Thought: I need the film page.
Action: search[Kiss and Tell (1945 film)]"""

    thought, raw_action, action_type, action_arg = parse_react_output(llm_output)

    assert thought == "I need the film page."
    assert raw_action == "search[Kiss and Tell (1945 film)]"
    assert action_type == "search"
    assert action_arg == "Kiss and Tell (1945 film)"


def test_parse_react_finish_and_supporting_facts():
    llm_output = """Thought: The evidence supports the answer.
Action: finish[no]
Support: [["Scott Derrickson", 1], ["Ed Wood", 1]]"""

    thought, raw_action, action_type, action_arg = parse_react_output(llm_output)
    supporting_facts = parse_supporting_facts(llm_output)

    assert thought == "The evidence supports the answer."
    assert raw_action == "finish[no]"
    assert action_type == "finish"
    assert action_arg == "no"
    assert supporting_facts == [["Scott Derrickson", 1], ["Ed Wood", 1]]


def test_parse_baseline_json_output():
    answer, supporting_facts = parse_baseline_output(
        '{"answer": "no", "supporting_facts": [["Scott Derrickson", 1], ["Ed Wood", 1]]}'
    )
    assert answer == "no"
    assert supporting_facts == [["Scott Derrickson", 1], ["Ed Wood", 1]]


def test_local_retriever_preserves_hotpot_sentence_ids():
    context = [
        {
            "title": "Scott Derrickson",
            "sentences": [
                "Scott Derrickson is a director.",
                "He was born in Denver, Colorado.",
            ],
        }
    ]
    retriever = LocalHotpotRetriever(context_paragraphs=context)

    obs_search = retriever.search("Scott Derrickson")
    assert "Loaded [Scott Derrickson]" in obs_search
    assert "[Scott Derrickson | sent 0]" in obs_search
    assert "[Scott Derrickson | sent 1] He was born in Denver, Colorado." in obs_search
    assert retriever.last_result["sentences"][1]["sent_id"] == 1

    obs_lookup = retriever.lookup("Denver")
    assert "[Scott Derrickson | sent 1] He was born in Denver, Colorado." in obs_lookup
    assert retriever.last_result["sentences"] == [
        {"sent_id": 1, "text": "He was born in Denver, Colorado."}
    ]


def test_hotpot_answer_f1_categorical_behavior():
    assert exact_match_score("Denver, Colorado", "denver colorado") is True
    precision, recall, f1 = f1_score("No, they were different.", "no")
    assert (precision, recall, f1) == (0.0, 0.0, 0.0)


def test_official_supporting_fact_score_uses_title_sentence_pairs():
    precision, recall, f1, em = supporting_fact_score(
        [["A", 0], ["B", 2]],
        [["A", 0], ["B", 1]],
    )
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)
    assert f1 == pytest.approx(0.5)
    assert em == 0.0


def test_joint_f1_uses_official_joint_precision_and_recall_formula():
    result = evaluate_prediction(
        prediction="alpha beta",
        ground_truth="alpha",
        predicted_supporting_facts=[["A", 0]],
        gold_supporting_facts=[["A", 0], ["B", 0]],
        visited_pages=["A"],
        gold_titles=["A", "B"],
    )

    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(1.0)
    assert result["sp_precision"] == pytest.approx(1.0)
    assert result["sp_recall"] == pytest.approx(0.5)
    assert result["joint_precision"] == pytest.approx(0.5)
    assert result["joint_recall"] == pytest.approx(0.5)
    assert result["joint_f1"] == pytest.approx(0.5)


def test_document_score_is_separate_diagnostic():
    precision, recall, f1 = supporting_document_score(
        ["Scott Derrickson", "Ed Wood"],
        ["Scott Derrickson", "Ed Wood"],
    )
    assert (precision, recall, f1) == (1.0, 1.0, 1.0)


def test_context_diagnostics_records_candidate_pool_oracle_coverage():
    sample = {
        "supporting_facts": [["A", 1], ["B", 0]],
        "context": [
            {"title": "A", "sentences": ["a0", "a1"]},
            {"title": "C", "sentences": ["c0"]},
        ],
    }
    diagnostics = context_diagnostics(sample)
    assert diagnostics["gold_document_recall_in_context"] == pytest.approx(0.5)
    assert diagnostics["gold_supporting_fact_recall_in_context"] == pytest.approx(0.5)
    assert diagnostics["all_gold_supporting_facts_available"] is False


def test_single_pass_rag_baseline_returns_answer_and_support():
    fake_llm = FakeListChatModel(
        responses=['{"answer": "no", "supporting_facts": [["Scott Derrickson", 0]]}']
    )
    context = [
        {"title": "Scott Derrickson", "sentences": ["Scott Derrickson was born in Denver, Colorado."]}
    ]
    retriever = LocalHotpotRetriever(context_paragraphs=context)

    question = "Were Scott Derrickson and Ed Wood born in the same state?"
    state = run_single_pass_rag(question=question, llm=fake_llm, toolset=retriever)

    assert state["final_answer"] == "no"
    assert state["predicted_supporting_facts"] == [["Scott Derrickson", 0]]
    assert state["step_count"] == 1
    assert len(state["steps"]) == 1
    assert "raw_model_output" in state["steps"][0]
    assert state["steps"][0]["retrieval"]["title"] == "Scott Derrickson"


def test_react_agent_end_to_end_with_supporting_facts():
    fake_responses = [
        "Thought: I need to search for Scott Derrickson.\nAction: search[Scott Derrickson]",
        "Thought: Scott Derrickson was born in Denver, Colorado. Now I search for Ed Wood.\nAction: search[Ed Wood]",
        'Thought: They were born in different states.\nAction: finish[no]\nSupport: [["Scott Derrickson", 0], ["Ed Wood", 0]]',
    ]
    fake_llm = FakeListChatModel(responses=fake_responses)

    context = [
        {"title": "Scott Derrickson", "sentences": ["Scott Derrickson was born in Denver, Colorado."]},
        {"title": "Ed Wood", "sentences": ["Ed Wood was born in Poughkeepsie, New York."]},
    ]
    retriever = LocalHotpotRetriever(context_paragraphs=context)

    question = "Were Scott Derrickson and Ed Wood born in the same state?"
    final_state = run_react_agent(question=question, llm=fake_llm, toolset=retriever)

    assert final_state["final_answer"] == "no"
    assert final_state["predicted_supporting_facts"] == [["Scott Derrickson", 0], ["Ed Wood", 0]]
    assert len(final_state["steps"]) == 3
    assert final_state["visited_pages"] == ["Scott Derrickson", "Ed Wood"]
    assert final_state["steps"][0]["retrieval"]["title"] == "Scott Derrickson"


def test_react_agent_forces_synthesis_at_hop_budget_with_support():
    fake_responses = [
        "Thought: I need to search for Scott Derrickson.\nAction: search[Scott Derrickson]",
        'Action: finish[American]\nSupport: [["Scott Derrickson", 0]]',
    ]
    fake_llm = FakeListChatModel(responses=fake_responses)

    context = [
        {"title": "Scott Derrickson", "sentences": ["Scott Derrickson is an American filmmaker."]},
    ]
    retriever = LocalHotpotRetriever(context_paragraphs=context)

    final_state = run_react_agent(
        question="What nationality is Scott Derrickson?",
        llm=fake_llm,
        toolset=retriever,
        max_hops=1,
    )

    assert final_state["step_count"] == 1
    assert final_state["final_answer"] == "American"
    assert final_state["predicted_supporting_facts"] == [["Scott Derrickson", 0]]
    assert final_state["steps"][-1]["action_type"] == "finish"
    assert final_state["steps"][-1]["action"] == "finish[American]"
    assert len(final_state["steps"]) == 2


def test_official_prediction_artifacts_have_hotpot_schema(tmp_path):
    samples = [
        {
            "id": "q1",
            "answer": "no",
            "supporting_facts": [["A", 0], ["B", 1]],
        }
    ]
    trajectories = [
        {
            "id": "q1",
            "predicted_answer": "no",
            "predicted_supporting_facts": [["A", 0], ["B", 1]],
        }
    ]

    prediction_path, gold_path = write_official_files(samples, trajectories, str(tmp_path))
    prediction = json.loads(open(prediction_path).read())
    gold = json.loads(open(gold_path).read())

    assert prediction == {
        "answer": {"q1": "no"},
        "sp": {"q1": [["A", 0], ["B", 1]]},
    }
    assert gold[0]["_id"] == "q1"
    assert gold[0]["supporting_facts"] == [["A", 0], ["B", 1]]
