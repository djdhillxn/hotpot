import time

from agent.parser import parse_baseline_output
from agent.state import create_initial_state
from tools.wikipedia import WikipediaToolSet

SINGLE_PASS_RAG_PROMPT = """Answer the following question directly and concisely using ONLY the provided context text.
The context may contain sentence labels such as [Wikipedia Title | sent 2]. Those labels identify exact HotpotQA sentence IDs.

Return EXACTLY one JSON object and no other text:
{{"answer": "<concise answer>", "supporting_facts": [["Exact Wikipedia title", <sentence_id>], ...]}}

For supporting_facts, cite ONLY sentence labels that appear in the provided context and are needed to justify the answer. Copy titles and sentence IDs exactly. If the context provides no supporting evidence, return an empty list rather than inventing evidence.

Context:
{context}

Question: {question}
"""


def run_single_pass_rag(question, llm, toolset=None):
    """Execute the one-retrieval, one-generation RAG baseline."""
    if toolset is None:
        toolset = WikipediaToolSet()
    elif hasattr(toolset, "reset"):
        toolset.reset()

    retrieval_started = time.perf_counter()
    obs = toolset.search(question)
    retrieval_latency = time.perf_counter() - retrieval_started
    visited_pages = list(toolset.visited_pages) if hasattr(toolset, "visited_pages") and toolset.visited_pages else []
    if not visited_pages and getattr(toolset, "current_page_title", None):
        visited_pages = [toolset.current_page_title]

    prompt = SINGLE_PASS_RAG_PROMPT.format(context=obs, question=question)
    generation_started = time.perf_counter()
    response = llm.invoke(prompt)
    generation_latency = time.perf_counter() - generation_started
    response_text = response.content if hasattr(response, "content") else str(response)
    answer, parsed_supporting_facts = parse_baseline_output(response_text)

    retrieval = getattr(toolset, "last_result", None)
    observed_supporting_facts = []
    if isinstance(retrieval, dict) and retrieval.get("title"):
        for sentence in retrieval.get("sentences", []):
            if isinstance(sentence, dict) and "sent_id" in sentence:
                observed_supporting_facts.append([retrieval["title"], int(sentence["sent_id"])])

    observed_set = {tuple(fact) for fact in observed_supporting_facts}
    supporting_facts = [fact for fact in parsed_supporting_facts if tuple(fact) in observed_set]
    invalid_supporting_facts = [fact for fact in parsed_supporting_facts if tuple(fact) not in observed_set]

    state = create_initial_state(question)
    state["final_answer"] = answer
    state["predicted_supporting_facts"] = supporting_facts
    state["observed_supporting_facts"] = observed_supporting_facts
    state["invalid_supporting_facts"] = invalid_supporting_facts
    state["step_count"] = 1
    state["visited_pages"] = visited_pages
    state["steps"] = [
        {
            "step": 1,
            "thought": "Single-pass direct prompt generation using initial retrieved context.",
            "action": f"search[{question}]",
            "action_type": "search",
            "action_arg": question,
            "supporting_facts": supporting_facts,
            "invalid_supporting_facts": invalid_supporting_facts,
            "raw_model_output": response_text,
            "tool_latency_seconds": round(retrieval_latency, 6),
            "llm_latency_seconds": round(generation_latency, 6),
            "observation": obs,
            "retrieval": retrieval,
        }
    ]
    state["evidence_graph"] = [
        {"source": "Question", "target": visited_pages[0] if visited_pages else "Context", "label": "Single-pass search"}
    ]

    return state
