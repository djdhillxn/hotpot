import re

def parse_react_output(llm_output):
    cleaned_output = llm_output.strip()

    if "Observation:" in cleaned_output:
        cleaned_output = cleaned_output.split("Observation:")[0].strip()

    action_match = re.search(
        r"Action:\s*(\w+)\s*(?:\[|\()(.*?)(?:\]|\))", cleaned_output, re.DOTALL | re.IGNORECASE
    )

    if action_match:
        action_type = action_match.group(1).lower()
        action_arg = action_match.group(2).strip().strip("'\"")
        raw_action = f"{action_type}[{action_arg}]"

        thought_part = cleaned_output[: action_match.start()].strip()
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
