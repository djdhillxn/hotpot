import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agent.parser import parse_react_output
from agent.engine import run_react_agent
from tools.wikipedia import WikipediaToolSet
from tools.local_retriever import LocalHotpotRetriever
from eval.metrics import exact_match_score, f1_score, supporting_facts_overlap, evaluate_prediction


def test_parse_react_output_standard():
    llm_output = """Thought: I need to search for Scott Derrickson to find his birthplace.
Action: search[Scott Derrickson]"""

    thought, raw_action, action_type, action_arg = parse_react_output(llm_output)

    assert thought == "I need to search for Scott Derrickson to find his birthplace."
    assert raw_action == "search[Scott Derrickson]"
    assert action_type == "search"
    assert action_arg == "Scott Derrickson"


def test_parse_react_output_finish():
    llm_output = """Thought: Scott Derrickson was born in Colorado, and Ed Wood was born in New York.
Action: finish[no]"""

    thought, raw_action, action_type, action_arg = parse_react_output(llm_output)

    assert thought == "Scott Derrickson was born in Colorado, and Ed Wood was born in New York."
    assert raw_action == "finish[no]"
    assert action_type == "finish"
    assert action_arg == "no"


def test_parse_react_output_fallback_parentheses():
    llm_output = """Thought: Let me lookup the lead singer.
Action: search(Radiohead)"""

    thought, raw_action, action_type, action_arg = parse_react_output(llm_output)

    assert action_type == "search"
    assert action_arg == "Radiohead"


def test_local_retriever():
    context = [
        {
            "title": "Scott Derrickson",
            "sentences": ["Scott Derrickson is a director.", "He was born in Denver, Colorado."]
        }
    ]
    retriever = LocalHotpotRetriever(context_paragraphs=context)

    obs_search = retriever.search("Scott Derrickson")
    assert "Loaded [Scott Derrickson]" in obs_search
    assert "Denver, Colorado" in obs_search

    obs_lookup = retriever.lookup("director")
    assert "Found matches" in obs_lookup
    assert "director" in obs_lookup


def test_eval_metrics():
    assert exact_match_score("Denver, Colorado", "denver colorado") is True
    assert exact_match_score("Denver, Colorado", "New York") is False

    p, r, f1 = f1_score("7 October 1968", "7 October 1968")
    assert f1 == 1.0

    p, r, f1 = f1_score("October 1968", "7 October 1968")
    assert 0.0 < f1 < 1.0

    sp_p, sp_r, sp_f1 = supporting_facts_overlap(["Scott Derrickson", "Ed Wood"], ["Scott Derrickson", "Ed Wood"])
    assert sp_f1 == 1.0

    res = evaluate_prediction("no", "no", ["Scott Derrickson", "Ed Wood"], ["Scott Derrickson", "Ed Wood"])
    assert res["joint_em"] is True
    assert res["joint_f1"] == 1.0


def test_react_agent_end_to_end():
    fake_responses = [
        "Thought: I need to search for Scott Derrickson.\nAction: search[Scott Derrickson]",
        "Thought: Scott Derrickson was born in Denver, Colorado. Now I search for Ed Wood.\nAction: search[Ed Wood]",
        "Thought: Ed Wood was born in New York. So they were born in different states.\nAction: finish[no]"
    ]
    fake_llm = FakeListChatModel(responses=fake_responses)

    context = [
        {"title": "Scott Derrickson", "sentences": ["Scott Derrickson was born in Denver, Colorado."]},
        {"title": "Ed Wood", "sentences": ["Ed Wood was born in Poughkeepsie, New York."]}
    ]
    retriever = LocalHotpotRetriever(context_paragraphs=context)

    question = "Were Scott Derrickson and Ed Wood born in the same state?"
    final_state = run_react_agent(question=question, llm=fake_llm, toolset=retriever)

    assert final_state["final_answer"] == "no"
    assert len(final_state["steps"]) == 3
    assert final_state["visited_pages"] == ["Scott Derrickson", "Ed Wood"]
    assert len(final_state["evidence_graph"]) >= 1
