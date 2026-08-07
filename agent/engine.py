from langgraph.graph import StateGraph, END
from agent.prompt import REACT_PROMPT_SYSTEM
from agent.parser import parse_react_output
from agent.state import create_initial_state
from tools.wikipedia import WikipediaToolSet
from config import MAX_AGENT_HOPS

def create_react_agent_graph(llm, toolset, max_hops=MAX_AGENT_HOPS):
    def agent_node(state):
        prompt = REACT_PROMPT_SYSTEM.format(
            question=state["question"], scratchpad=state["scratchpad"]
        )

        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)

        thought, raw_action, action_type, action_arg = parse_react_output(response_text)

        step_record = {
            "step": state["step_count"] + 1,
            "thought": thought,
            "action": raw_action,
            "action_type": action_type,
            "action_arg": action_arg,
            "observation": "",
        }

        updates = {
            "current_action_type": action_type,
            "current_action_arg": action_arg,
            "steps": state["steps"] + [step_record],
        }

        if action_type == "finish":
            updates["final_answer"] = action_arg

        return updates

    def tool_node(state):
        action_type = state["current_action_type"]
        action_arg = state["current_action_arg"]
        last_step = state["steps"][-1]

        observation = ""
        evidence_graph = list(state.get("evidence_graph", []))
        visited_pages = list(state.get("visited_pages", []))

        prev_page = toolset.current_page_title

        if action_type == "search":
            observation = toolset.search(action_arg)
            new_page = toolset.current_page_title
            if new_page and new_page not in visited_pages:
                visited_pages.append(new_page)
                if prev_page and prev_page != new_page:
                    evidence_graph.append(
                        {"source": prev_page, "target": new_page, "label": f"Searched '{action_arg}'"}
                    )
                elif not prev_page and new_page:
                    evidence_graph.append(
                        {"source": "Question", "target": new_page, "label": f"Initial search '{action_arg}'"}
                    )
        elif action_type == "lookup":
            observation = toolset.lookup(action_arg)
        elif action_type == "invalid":
            observation = f"Observation: {action_arg}"
        else:
            observation = "Observation: Unknown action type."

        updated_steps = list(state["steps"])
        updated_steps[-1]["observation"] = observation

        new_scratchpad = (
            state["scratchpad"]
            + f"Thought: {last_step['thought']}\nAction: {last_step['action']}\n{observation}\n"
        )

        return {
            "steps": updated_steps,
            "scratchpad": new_scratchpad,
            "step_count": state["step_count"] + 1,
            "visited_pages": visited_pages,
            "evidence_graph": evidence_graph,
        }

    def should_continue(state):
        if state["current_action_type"] == "finish":
            return "end"
        if state["step_count"] >= max_hops:
            return "end"
        return "tool"

    workflow = StateGraph(dict)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tool", tool_node)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tool": "tool",
            "end": END,
        },
    )
    workflow.add_edge("tool", "agent")

    return workflow.compile()


def run_react_agent(question, llm, toolset=None, max_hops=MAX_AGENT_HOPS):
    if toolset is None:
        toolset = WikipediaToolSet()
    elif hasattr(toolset, "reset"):
        toolset.reset()

    graph = create_react_agent_graph(llm, toolset, max_hops)
    initial_state = create_initial_state(question)
    final_state = graph.invoke(initial_state)
    return final_state
