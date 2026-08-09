import unittest
from unittest.mock import MagicMock
from agent.engine import run_react_agent


class MockToolset:
    def __init__(self):
        self.current_page_title = "Mondelez International"
        self.current_document = {
            "title": "Mondelez International",
            "sentences": [
                "Mondelez International is an American multinational snack food company.",
                "It was founded in Deerfield, Illinois."
            ],
        }
        self.last_result = {
            "mode": "fullwiki",
            "candidate_k": 100,
            "hits": [
                {
                    "doc_id": "123",
                    "title": "Mondelez International",
                    "rank": 1,
                    "sentences": [
                        {"sent_id": 0, "text": "Mondelez International is an American multinational snack food company."}
                    ],
                }
            ],
        }

    def search(self, query):
        return (
            "Observation: Loaded [Mondelez International] (rank 1).\n"
            "[Mondelez International | sent 0] Mondelez International is an American multinational snack food company."
        )

    def lookup(self, term):
        return "Observation: No matching line found."

    def render_active_evidence(self):
        return (
            "Active Evidence Memory:\n"
            "[Mondelez International | sent 0] Mondelez International is an American multinational snack food company."
        )

    def reset(self):
        pass


class MockLLM:
    def __init__(self):
        self.calls = []

    def bind(self, **kwargs):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        call_count = len(self.calls)
        if call_count == 1:
            res = MagicMock()
            res.content = "I need to search for Mondelez International. Action: search[Mondelez International]"
            return res
        else:
            res = MagicMock()
            res.content = "I found the answer. Action: finish[Mondelez International] Support: [[\"Mondelez International\", 0]]"
            return res


class TestReactMemoryContract(unittest.TestCase):
    def test_react_memory_contract(self):
        mock_llm = MockLLM()
        mock_toolset = MockToolset()

        final_state = run_react_agent(
            question="What company makes Mondelez snacks?",
            llm=mock_llm,
            toolset=mock_toolset,
            max_hops=3,
        )

        # 1. Assert 2nd LLM prompt contains active evidence memory sentence label
        self.assertGreaterEqual(len(mock_llm.calls), 2, "Expected at least 2 LLM calls")
        second_call_human_msg = mock_llm.calls[1][1].content
        self.assertIn("Active Evidence Memory:", second_call_human_msg, "Active Evidence Memory missing from 2nd prompt")
        self.assertIn("[Mondelez International | sent 0]", second_call_human_msg, "Active evidence sentence label missing from 2nd prompt")

        # 2. Assert last_result attached to step 1
        steps = final_state.get("steps", [])
        self.assertGreaterEqual(len(steps), 1)
        self.assertIn("retrieval", steps[0])
        self.assertEqual(steps[0]["retrieval"]["hits"][0]["title"], "Mondelez International")

        # 3. Assert observed_supporting_facts contains ONLY rendered sentences
        observed_facts = final_state.get("observed_supporting_facts", [])
        self.assertIn(["Mondelez International", 0], observed_facts)
        self.assertNotIn(["Mondelez International", 1], observed_facts)

        # 4. Assert evidence_graph edges created
        evidence_graph = final_state.get("evidence_graph", [])
        self.assertGreaterEqual(len(evidence_graph), 1)
        self.assertEqual(evidence_graph[0]["target"], "Mondelez International")

        # 5. Assert scratchpad contains explicit Thought: and Action: prefixes
        scratchpad = final_state.get("scratchpad", "")
        self.assertIn("Thought: I need to search for Mondelez International.", scratchpad)
        self.assertIn("Action: search[Mondelez International]", scratchpad)


if __name__ == "__main__":
    unittest.main()
