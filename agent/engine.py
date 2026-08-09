import time
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from agent.prompt import REACT_SYSTEM_PROMPT, FORCED_SYNTHESIS_PROMPT_SYSTEM
from agent.parser import parse_react_output, parse_supporting_facts
from agent.state import create_initial_state
from tools.wikipedia import WikipediaToolSet
from config import MAX_AGENT_HOPS


def _get_response_text(response):
    return response.content if hasattr(response, "content") else str(response)


def _validate_supporting_facts(facts, observed_facts):
    observed = {tuple(fact) for fact in observed_facts or []}
    valid = [fact for fact in facts if tuple(fact) in observed]
    invalid = [fact for fact in facts if tuple(fact) not in observed]
    return valid, invalid


def create_react_agent_graph(llm, toolset, max_hops=MAX_AGENT_HOPS):
    # Bind stop sequence and max token generation constraints for Qwen / LLM inference
    bound_llm = llm.bind(stop=["\nObservation:", "Observation:"], max_tokens=150)

    def agent_node(state):
        messages = [
            SystemMessage(content=REACT_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Question: {state['question']}\n{state.get('scratchpad', '')}Thought: "
            ),
        ]

        llm_started = time.perf_counter()
        response = bound_llm.invoke(messages)
        llm_latency = time.perf_counter() - llm_started
        raw_text = _get_response_text(response)

        # Prepend 'Thought: ' if LLM continued straight from the prompt prefix
        if not raw_text.startswith("Thought:") and not raw_text.startswith("Action:"):
            response_text = f"Thought: {raw_text}"
        else:
            response_text = raw_text

        thought, raw_action, action_type, action_arg = parse_react_output(response_text)
        parsed_supporting_facts = parse_supporting_facts(response_text) if action_type == "finish" else []
        supporting_facts, invalid_supporting_facts = _validate_supporting_facts(
            parsed_supporting_facts, state.get("observed_supporting_facts", [])
        )

        step_record = {
            "step": state.get("step_count", 0) + 1,
            "thought": thought,
            "action": raw_action,
            "action_type": action_type,
            "action_arg": action_arg,
            "supporting_facts": supporting_facts,
            "invalid_supporting_facts": invalid_supporting_facts,
            "raw_model_output": response_text,
            "llm_latency_seconds": round(llm_latency, 6),
            "observation": "",
        }

        updates = dict(state)
        updates["current_action_type"] = action_type
        updates["current_action_arg"] = action_arg
        updates["steps"] = list(state.get("steps", [])) + [step_record]

        if action_type == "finish":
            updates["final_answer"] = action_arg
            updates["predicted_supporting_facts"] = supporting_facts
            updates["invalid_supporting_facts"] = invalid_supporting_facts

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
        tool_started = time.perf_counter()

        # Consecutive duplicate searches are useless for deterministic retrieval,
        # but repeated lookup[keyword] is valid classic ReAct behavior: each call
        # advances to the next matching sentence on the current page.
        is_repetition = (
            len(steps) >= 2
            and steps[-1].get("action") == steps[-2].get("action")
            and action_type == "search"
        )

        if is_repetition:
            observation = (
                f"Observation: You previously executed this exact action '{last_step['action']}'. "
                "Do not repeat identical actions. Try a different query term or use Lookup."
            )
        elif action_type == "search":
            observation = toolset.search(action_arg)
        elif action_type == "lookup":
            observation = toolset.lookup(action_arg)
        elif action_type == "invalid":
            observation = (
                "Observation: Invalid action format. Use Action: search[query], "
                "Action: lookup[keyword], or Action: finish[answer]."
            )
        else:
            observation = "Observation: Unknown action type."

        tool_latency = time.perf_counter() - tool_started
        updated_steps = list(steps)
        observed_supporting_facts = list(state.get("observed_supporting_facts", []))
        if updated_steps:
            updated_steps[-1]["observation"] = observation
            updated_steps[-1]["tool_latency_seconds"] = round(tool_latency, 6)
            retrieval = getattr(toolset, "last_result", None)
            if retrieval is not None:
                updated_steps[-1]["retrieval"] = retrieval
                if isinstance(retrieval, dict):
                    retrieval_hits = retrieval.get("hits") or []
                    if retrieval_hits:
                        for hit in retrieval_hits:
                            title = hit.get("title")
                            if title and title not in visited_pages:
                                visited_pages.append(title)
                                evidence_graph.append({
                                    "source": prev_page if prev_page in visited_pages else "Question",
                                    "target": title,
                                    "label": f"Retrieved for '{action_arg}'",
                                })
                            for sentence in hit.get("sentences", []):
                                if title and isinstance(sentence, dict) and "sent_id" in sentence:
                                    fact = [title, int(sentence["sent_id"])]
                                    if fact not in observed_supporting_facts:
                                        observed_supporting_facts.append(fact)
                    else:
                        title = retrieval.get("title")
                        if title and title not in visited_pages:
                            visited_pages.append(title)
                            evidence_graph.append({
                                "source": prev_page if prev_page in visited_pages else "Question",
                                "target": title,
                                "label": f"Retrieved for '{action_arg}'",
                            })
                        for sentence in retrieval.get("sentences", []):
                            if title and isinstance(sentence, dict) and "sent_id" in sentence:
                                fact = [title, int(sentence["sent_id"])]
                                if fact not in observed_supporting_facts:
                                    observed_supporting_facts.append(fact)

        new_scratchpad = (
            state.get("scratchpad", "")
            + f"Thought: {last_step['thought']}\nAction: {last_step['action']}\n{observation}\n"
        )

        updates = dict(state)
        updates["steps"] = updated_steps
        updates["scratchpad"] = new_scratchpad
        updates["step_count"] = state.get("step_count", 0) + 1
        updates["visited_pages"] = visited_pages
        updates["observed_supporting_facts"] = observed_supporting_facts
        updates["evidence_graph"] = evidence_graph

        return updates

    def synthesis_node(state):
        synthesis_llm = llm.bind(max_tokens=150)
        messages = [
            SystemMessage(content=FORCED_SYNTHESIS_PROMPT_SYSTEM),
            HumanMessage(
                content=f"Question: {state['question']}\n{state.get('scratchpad', '')}"
            ),
        ]

        llm_started = time.perf_counter()
        response = synthesis_llm.invoke(messages)
        llm_latency = time.perf_counter() - llm_started
        response_text = _get_response_text(response)

        thought, raw_action, action_type, action_arg = parse_react_output(response_text)
        parsed_supporting_facts = parse_supporting_facts(response_text)
        supporting_facts, invalid_supporting_facts = _validate_supporting_facts(
            parsed_supporting_facts, state.get("observed_supporting_facts", [])
        )

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
            "supporting_facts": supporting_facts,
            "invalid_supporting_facts": invalid_supporting_facts,
            "raw_model_output": response_text,
            "llm_latency_seconds": round(llm_latency, 6),
            "observation": "",
        }

        updates = dict(state)
        updates["current_action_type"] = "finish"
        updates["current_action_arg"] = final_answer
        updates["final_answer"] = final_answer
        updates["predicted_supporting_facts"] = supporting_facts
        updates["invalid_supporting_facts"] = invalid_supporting_facts
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
