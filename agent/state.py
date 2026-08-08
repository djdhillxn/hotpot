def create_initial_state(question):
    return {
        "question": question,
        "scratchpad": "",
        "steps": [],
        "current_action_type": None,
        "current_action_arg": None,
        "step_count": 0,
        "final_answer": None,
        "predicted_supporting_facts": [],
        "observed_supporting_facts": [],
        "invalid_supporting_facts": [],
        "visited_pages": [],
        "evidence_graph": [],
        "error": None,
    }
