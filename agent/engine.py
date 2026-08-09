import threading
import time
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from agent.prompt import REACT_SYSTEM_PROMPT, FORCED_SYNTHESIS_PROMPT_SYSTEM
from agent.parser import parse_react_output, parse_supporting_facts, _canonical_answer_text
from agent.state import create_initial_state
from tools.wikipedia import WikipediaToolSet
from config import MAX_AGENT_HOPS


_COMPILED_GRAPH_CACHE = {}
_GRAPH_CACHE_LOCK = threading.Lock()


def _get_response_text(response):
    return response.content if hasattr(response, "content") else str(response)


def _normalize_fact_title(title):
    return " ".join(str(title or "").strip().lower().replace(",", "").replace(".", "").split())


def _validate_supporting_facts(facts, observed_facts):
    observed_map = {}
    for fact in observed_facts or []:
        if isinstance(fact, (list, tuple)) and len(fact) == 2:
            norm_t = _normalize_fact_title(fact[0])
            try:
                sent_id = int(fact[1])
                observed_map[(norm_t, sent_id)] = [str(fact[0]).strip(), sent_id]
            except (ValueError, TypeError):
                pass

    valid = []
    invalid = []
    for fact in facts or []:
        if isinstance(fact, (list, tuple)) and len(fact) == 2:
            norm_t = _normalize_fact_title(fact[0])
            try:
                sent_id = int(fact[1])
                key = (norm_t, sent_id)
                if key in observed_map:
                    valid.append(observed_map[key])
                else:
                    invalid.append([str(fact[0]).strip(), sent_id])
            except (ValueError, TypeError):
                invalid.append(fact)
    return valid, invalid


def create_react_agent_graph(llm, toolset=None, max_hops=MAX_AGENT_HOPS):
    # Bind stop sequence and max token generation constraints for Qwen / LLM inference
    bound_llm = llm.bind(stop=["\nObservation:", "Observation:"], max_tokens=150)

    def agent_node(state):
        evidence_context = state.get("active_evidence_context", "").strip()
        evidence_block = f"{evidence_context}\n\n" if evidence_context else ""
        messages = [
            SystemMessage(content=REACT_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Question: {state['question']}\n"
                    f"{evidence_block}"
                    f"Reasoning / Action History:\n{state.get('scratchpad', '')}"
                    "Thought: "
                )
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
        active_toolset = state.get("toolset") or toolset
        action_type = state.get("current_action_type")
        action_arg = state.get("current_action_arg")
        steps = list(state.get("steps", []))
        last_step = steps[-1] if steps else {"thought": "", "action": ""}

        observation = ""
        evidence_graph = list(state.get("evidence_graph", []))
        visited_pages = list(state.get("visited_pages", []))
        observed_supporting_facts = list(state.get("observed_supporting_facts", []))

        prev_page = getattr(active_toolset, "current_page_title", None) or "Question"
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
            observation = active_toolset.search(action_arg)
        elif action_type == "lookup":
            observation = active_toolset.lookup(action_arg)
        elif action_type == "invalid":
            observation = (
                "Observation: Invalid action format. Use Action: search[query], "
                "Action: lookup[keyword], or Action: finish[answer]."
            )
        else:
            observation = "Observation: Unknown action type."

        tool_latency = time.perf_counter() - tool_started
        updated_steps = list(steps)

        last_step_updated = dict(updated_steps[-1])
        last_step_updated["observation"] = observation
        last_step_updated["tool_latency_seconds"] = round(tool_latency, 6)

        retrieval = getattr(active_toolset, "last_result", None)
        if retrieval is not None:
            last_step_updated["retrieval"] = retrieval
            if isinstance(retrieval, dict):
                hits = retrieval.get("hits") or []
                if hits:
                    for hit in hits:
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

        updated_steps[-1] = last_step_updated

        new_scratchpad = state.get("scratchpad", "")
        if action_type in {"search", "lookup", "invalid"}:
            t_text = str(last_step.get("thought", "")).strip()
            a_text = str(last_step.get("action", "")).strip()
            if t_text and not t_text.startswith("Thought:"):
                t_text = f"Thought: {t_text}"
            if a_text and not a_text.startswith("Action:"):
                a_text = f"Action: {a_text}"

            new_scratchpad += f"{t_text}\n{a_text}\n{observation}\n"

        active_evidence_context = (
            active_toolset.render_active_evidence()
            if hasattr(active_toolset, "render_active_evidence")
            else (
                active_toolset.active_evidence_context()
                if hasattr(active_toolset, "active_evidence_context")
                else ""
            )
        )

        updates = dict(state)
        updates["steps"] = updated_steps
        updates["scratchpad"] = new_scratchpad
        updates["active_evidence_context"] = active_evidence_context
        updates["observed_supporting_facts"] = observed_supporting_facts
        updates["visited_pages"] = visited_pages
        updates["step_count"] = state.get("step_count", 0) + 1
        updates["evidence_graph"] = evidence_graph

        return updates

    def synthesis_node(state):
        synthesis_llm = llm.bind(max_tokens=150)
        evidence_context = state.get("active_evidence_context", "").strip()
        evidence_block = f"{evidence_context}\n\n" if evidence_context else ""
        messages = [
            SystemMessage(content=FORCED_SYNTHESIS_PROMPT_SYSTEM),
            HumanMessage(
                content=(
                    f"Question: {state['question']}\n"
                    f"{evidence_block}"
                    f"Reasoning / Action History:\n{state.get('scratchpad', '')}"
                )
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

        candidate_answer = ""
        if action_type == "finish" and action_arg.strip():
            candidate_answer = _canonical_answer_text(action_arg)

        if not candidate_answer:
            import re
            cleaned_text = re.sub(r"(?i)\n?\s*Support\s*:\s*\[.*\]\s*$", "", response_text).strip()
            lines = [l.strip() for l in cleaned_text.split("\n") if l.strip()]
            for line in reversed(lines):
                cleaned_line = _canonical_answer_text(line)
                if cleaned_line and not cleaned_line.lower().startswith("support"):
                    candidate_answer = cleaned_line
                    break

        if not candidate_answer:
            candidate_answer = "unknown"

        final_answer = candidate_answer
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


def get_compiled_react_agent_graph(llm, max_hops=MAX_AGENT_HOPS):
    key = (id(llm), int(max_hops))
    with _GRAPH_CACHE_LOCK:
        if key in _COMPILED_GRAPH_CACHE:
            return _COMPILED_GRAPH_CACHE[key]

    graph = create_react_agent_graph(llm, toolset=None, max_hops=max_hops)
    with _GRAPH_CACHE_LOCK:
        _COMPILED_GRAPH_CACHE[key] = graph
    return graph


def run_react_agent(question, llm, toolset=None, max_hops=MAX_AGENT_HOPS):
    if toolset is None:
        toolset = WikipediaToolSet()
    elif hasattr(toolset, "reset"):
        toolset.reset()

    graph = get_compiled_react_agent_graph(llm, max_hops=max_hops)
    initial_state = create_initial_state(question)
    initial_state["toolset"] = toolset
    final_state = graph.invoke(initial_state)
    return final_state
