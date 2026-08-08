from langgraph.graph import StateGraph, END
from agent.prompt import REACT_PROMPT_SYSTEM, FORCED_SYNTHESIS_PROMPT_SYSTEM
from agent.parser import parse_react_output
from agent.state import create_initial_state
from tools.wikipedia import WikipediaToolSet
from config import MAX_AGENT_HOPS


def _get_response_text(response):
    return response.content if hasattr(response, "content") else str(response)


def create_react_agent_graph(llm, toolset, max_hops=MAX_AGENT_HOPS):
    def agent_node(state):
        prompt = REACT_PROMPT_SYSTEM.format(
            question=state["question"], scratchpad=state.get("scratchpad", "")
        )

        response = llm.invoke(prompt)
        response_text = _get_response_text(response)

        thought, raw_action, action_type, action_arg = parse_react_output(response_text)

        step_record = {
            "step": state.get("step_count", 0) + 1,
            "thought": thought,
            "action": raw_action,
            "action_type": action_type,
            "action_arg": action_arg,
            "observation": "",
        }

        updates = dict(state)
        updates["current_action_type"] = action_type
        updates["current_action_arg"] = action_arg
        updates["steps"] = list(state.get("steps", [])) + [step_record]

        if action_type == "finish":
            updates["final_answer"] = action_arg

        return updates

    def tool_node(state):
        action_type = state.get("current_action_type")
        action_arg = state.get("current_action_arg")
        steps = list(state.get("steps", []))
        last_step = steps[-1] if steps else {"thought": "", "action": ""}

        observation = ""
        evidence_graph = list(state.get("evidence_graph", []))
        visited_pages = list(state.get("visited_pages", []))

        prev_page = getattr(toolset, "current_page_title", None)

        if action_type == "search":
            observation = toolset.search(action_arg)
            new_page = getattr(toolset, "current_page_title", None)
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

        updated_steps = list(steps)
        if updated_steps:
            updated_steps[-1]["observation"] = observation

        new_scratchpad = (
            state.get("scratchpad", "")
            + f"Thought: {last_step['thought']}\nAction: {last_step['action']}\n{observation}\n"
        )

        updates = dict(state)
        updates["steps"] = updated_steps
        updates["scratchpad"] = new_scratchpad
        updates["step_count"] = state.get("step_count", 0) + 1
        updates["visited_pages"] = visited_pages
        updates["evidence_graph"] = evidence_graph

        return updates

    def synthesis_node(state):
        prompt = FORCED_SYNTHESIS_PROMPT_SYSTEM.format(
            question=state["question"], scratchpad=state.get("scratchpad", "")
        )

        response = llm.invoke(prompt)
        response_text = _get_response_text(response)
        thought, raw_action, action_type, action_arg = parse_react_output(response_text)

        if action_type == "finish" and action_arg.strip():
            final_answer = action_arg.strip()
            final_action = f"finish[{final_answer}]"
        else:
            final_answer = response_text.strip() or "unknown"
            final_action = f"finish[{final_answer}]"

        step_record = {
            "step": state.get("step_count", 0) + 1,
            "thought": "Search budget exhausted; synthesizing the best answer from gathered evidence.",
            "action": final_action,
            "action_type": "finish",
            "action_arg": final_answer,
            "observation": "",
        }

        updates = dict(state)
        updates["current_action_type"] = "finish"
        updates["current_action_arg"] = final_answer
        updates["final_answer"] = final_answer
        updates["steps"] = list(state.get("steps", [])) + [step_record]
        return updates

    def route_after_agent(state):
        action_type = state.get("current_action_type")
        step_count = state.get("step_count", 0)
        if action_type == "finish":
            return "end"
        if step_count >= max_hops:
            return "synthesize"
        return "tool"

    def route_after_tool(state):
        if state.get("step_count", 0) >= max_hops:
            return "synthesize"
        return "agent"

    workflow = StateGraph(dict)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tool", tool_node)
    workflow.add_node("synthesize", synthesis_node)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tool": "tool",
            "synthesize": "synthesize",
            "end": END,
        },
    )
    workflow.add_conditional_edges(
        "tool",
        route_after_tool,
        {
            "agent": "agent",
            "synthesize": "synthesize",
        },
    )
    workflow.add_edge("synthesize", END)

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
