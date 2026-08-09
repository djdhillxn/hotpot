import json
import re


_SENTENCE_LABEL_RE = re.compile(r"^\s*(.*?)\s*\|\s*sent\s+(\d+)\s*$", re.IGNORECASE)
_ANSWER_PREFIX_RE = re.compile(r"^\s*(?:final\s+answer|answer)\s*:\s*", re.IGNORECASE)
_THE_ANSWER_IS_RE = re.compile(r"^\s*the\s+answer\s+is\s+", re.IGNORECASE)


def _extract_delimited_action(cleaned_output):
    """Extract Action: tool[argument] or Action: tool(argument) with paired delimiters."""
    prefix_match = re.search(r"Action:\s*(\w+)\s*([\[(])", cleaned_output, re.IGNORECASE)
    if not prefix_match:
        return None

    action_type = prefix_match.group(1).lower()
    opening = prefix_match.group(2)
    closing = "]" if opening == "[" else ")"
    arg_start = prefix_match.end()
    depth = 1

    for index in range(arg_start, len(cleaned_output)):
        char = cleaned_output[index]
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return prefix_match, action_type, cleaned_output[arg_start:index]

    return None


def _canonical_answer_text(value):
    """Conservatively strip common answer wrappers without rewriting content."""
    answer = str(value or "").strip().strip("'\"")
    answer = _ANSWER_PREFIX_RE.sub("", answer, count=1)
    answer = _THE_ANSWER_IS_RE.sub("", answer, count=1)
    return answer.strip().strip("'\"")


def _canonical_support_fact(fact):
    """Normalize a HotpotQA [title, sent_id] pair, including copied visual labels."""
    if not isinstance(fact, (list, tuple)) or len(fact) != 2:
        return None

    title, sent_id = fact
    if not isinstance(title, str):
        return None

    label_match = _SENTENCE_LABEL_RE.match(title)
    if label_match:
        title = label_match.group(1).strip()
        sent_id = label_match.group(2)

    try:
        sent_id = int(sent_id)
    except (TypeError, ValueError):
        return None

    title = title.strip()
    if not title:
        return None
    return [title, sent_id]


def parse_supporting_facts(llm_output):
    """Parse a JSON list of [title, sentence_id] pairs from a Support line."""
    match = re.search(
        r"(?:Supporting\s+Facts|Support)\s*:\s*(\[.*\])\s*$",
        llm_output.strip(),
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return []

    try:
        raw_facts = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []

    facts = []
    if not isinstance(raw_facts, list):
        return facts

    for fact in raw_facts:
        canonical = _canonical_support_fact(fact)
        if canonical is not None:
            facts.append(canonical)

    return facts


def parse_baseline_output(llm_output):
    """Parse baseline JSON and repair copied ``Title | sent N`` labels safely."""
    text = llm_output.strip()
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
            answer = _canonical_answer_text(payload.get("answer", ""))
            supporting_facts = payload.get("supporting_facts", [])
            facts = []
            if isinstance(supporting_facts, list):
                for fact in supporting_facts:
                    canonical = _canonical_support_fact(fact)
                    if canonical is not None and canonical not in facts:
                        facts.append(canonical)
            if answer:
                return answer, facts
        except json.JSONDecodeError:
            pass

    return _canonical_answer_text(text), []


def parse_react_output(llm_output):
    cleaned_output = llm_output.strip()

    if "Observation:" in cleaned_output:
        cleaned_output = cleaned_output.split("Observation:")[0].strip()

    action_match = _extract_delimited_action(cleaned_output)
    if action_match:
        prefix_match, action_type, extracted_arg = action_match
        action_arg = extracted_arg.strip().strip("'\"")
        if action_type == "finish":
            action_arg = _canonical_answer_text(action_arg)
        raw_action = f"{action_type}[{action_arg}]"

        thought_part = cleaned_output[: prefix_match.start()].strip()
        thought = re.sub(r"^Thought:\s*", "", thought_part, flags=re.IGNORECASE).strip()
        if not thought:
            thought = "Analyzing available information."
        return thought, raw_action, action_type, action_arg

    fallback_action = re.search(
        r"Action:\s*(search|lookup|finish)\s+(.+)", cleaned_output, re.IGNORECASE
    )
    if fallback_action:
        action_type = fallback_action.group(1).lower()
        action_arg = fallback_action.group(2).strip().strip("'\"")
        if action_type == "finish":
            action_arg = _canonical_answer_text(action_arg)
        raw_action = f"{action_type}[{action_arg}]"

        thought_part = cleaned_output[: fallback_action.start()].strip()
        thought = re.sub(r"^Thought:\s*", "", thought_part, flags=re.IGNORECASE).strip()
        if not thought:
            thought = "Analyzing available information."
        return thought, raw_action, action_type, action_arg

    if "finish" in cleaned_output.lower() or "final answer" in cleaned_output.lower():
        lines = cleaned_output.split("\n")
        last_line = lines[-1].strip()
        arg = re.sub(r"^(finish|final answer):\s*", "", last_line, flags=re.IGNORECASE).strip("[] ")
        arg = _canonical_answer_text(arg)
        return cleaned_output, f"finish[{arg}]", "finish", arg

    thought = re.sub(r"^Thought:\s*", "", cleaned_output, flags=re.IGNORECASE).strip()
    return (
        thought,
        "invalid",
        "invalid",
        "Invalid action format. Follow Action: search[entity], lookup[keyword], or finish[answer].",
    )
