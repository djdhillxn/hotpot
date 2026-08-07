from agent.state import create_initial_state
from tools.wikipedia import WikipediaToolSet

SINGLE_PASS_RAG_PROMPT = """Answer the following question directly and concisely using ONLY the provided context text.
Do not invent information. Be brief and output only the direct answer.

Context:
{context}

Question: {question}
Answer:"""


def run_single_pass_rag(question, llm, toolset=None):
    """
    Executes a Single-Pass RAG (Direct Prompting Baseline):
    1. Performs a single retrieval query for the question.
    2. Passes the retrieved context to the LLM in one single turn without ReAct hop loops.
    """
    if toolset is None:
        toolset = WikipediaToolSet()
    elif hasattr(toolset, "reset"):
        toolset.reset()

    # Step 1: Single retrieval step
    obs = toolset.search(question)
    visited_pages = list(toolset.visited_pages) if hasattr(toolset, "visited_pages") and toolset.visited_pages else []
    if not visited_pages and getattr(toolset, "current_page_title", None):
        visited_pages = [toolset.current_page_title]

    # Step 2: Direct Single-Pass LLM Generation
    prompt = SINGLE_PASS_RAG_PROMPT.format(context=obs, question=question)
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    answer = answer.strip().strip('"\'')

    state = create_initial_state(question)
    state["final_answer"] = answer
    state["step_count"] = 1
    state["visited_pages"] = visited_pages
    state["steps"] = [
        {
            "step": 1,
            "thought": "Single-pass direct prompt generation using initial retrieved context.",
            "action": f"search[{question}]",
            "action_type": "search",
            "action_arg": question,
            "observation": obs,
        }
    ]
    state["evidence_graph"] = [
        {"source": "Question", "target": visited_pages[0] if visited_pages else "Context", "label": "Single-pass search"}
    ]

    return state
