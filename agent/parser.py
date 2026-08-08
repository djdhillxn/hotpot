import re


def _extract_delimited_action(cleaned_output):
    """Extract Action: tool[argument] or Action: tool(argument) with paired delimiters."""
    prefix_match = re.search(
        r"Action:\s*(\w+)\s*([\[(])", cleaned_output, re.IGNORECASE
    )
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


def parse_react_output(llm_output):
    cleaned_output = llm_output.strip()

    if "Observation:" in cleaned_output:
        cleaned_output = cleaned_output.split("Observation:")[0].strip()

    action_match = _extract_delimited_action(cleaned_output)

    if action_match:
        prefix_match, action_type, extracted_arg = action_match
        action_arg = extracted_arg.strip().strip("'\"")
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
        return cleaned_output, f"finish[{arg}]", "finish", arg

    thought = re.sub(r"^Thought:\s*", "", cleaned_output, flags=re.IGNORECASE).strip()
    return (
        thought,
        "invalid",
        "invalid",
        "Invalid action format. Follow Action: search[entity], lookup[keyword], or finish[answer].",
    )
