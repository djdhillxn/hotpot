REACT_PROMPT_SYSTEM = """Solve a multi-step, multi-hop question-answering task by alternating between Thought, Action, and Observation.

You have access to the following tools:
(1) search[entity]: Searches Wikipedia for the exact entity page and returns its lead paragraph summary.
(2) lookup[keyword]: Searches the currently loaded Wikipedia page for paragraphs containing the exact keyword.
(3) finish[answer]: Ends the task and provides the final concise answer based ONLY on evidence observed in Wikipedia.

STRICT FORMATTING RULES:
Each turn MUST follow this exact format:
Thought: <your step-by-step reasoning on what to search or extract next>
Action: <one of: search[entity], lookup[keyword], or finish[answer]>

DO NOT write "Observation:" yourself. The system will provide the Observation after your Action.
DO NOT hallucinate facts. Every claim in your Thought must be backed by previous Observations.

---
EXAMPLE 1:
Question: Were Scott Derrickson and Ed Wood born in the same state?
Thought: I need to search for Scott Derrickson to find his birthplace.
Action: search[Scott Derrickson]
Observation: Loaded [Scott Derrickson]: Scott Derrickson (born July 16, 1966) is an American filmmaker. He was born in Denver, Colorado.
Thought: Scott Derrickson was born in Denver, Colorado. Now I need to search for Ed Wood to find his birthplace.
Action: search[Ed Wood]
Observation: Loaded [Ed Wood]: Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker. He was born in Poughkeepsie, New York.
Thought: Ed Wood was born in Poughkeepsie, New York. Scott Derrickson was born in Colorado, while Ed Wood was born in New York. They were born in different states.
Action: finish[no]

---
EXAMPLE 2:
Question: What is the birth date of the lead singer of the band that performed 'Creep'?
Thought: I need to search for the song 'Creep' to identify the band that performed it.
Action: search[Creep (Radiohead song)]
Observation: Loaded [Creep (Radiohead song)]: "Creep" is the debut single by the English alternative rock band Radiohead.
Thought: The lead singer of Radiohead needs to be identified. Let me search for Radiohead.
Action: search[Radiohead]
Observation: Loaded [Radiohead]: Radiohead are an English rock band formed in Abingdon, Oxfordshire, in 1985. The band consists of Thom Yorke (vocals, guitar)...
Thought: Thom Yorke is the lead singer of Radiohead. Now I need to search for Thom Yorke to find his birth date.
Action: search[Thom Yorke]
Observation: Loaded [Thom Yorke]: Thomas Edward Yorke (born 7 October 1968) is an English musician and main vocalist of Radiohead.
Thought: Thom Yorke was born on 7 October 1968.
Action: finish[7 October 1968]

---
Task Question: {question}
{scratchpad}Thought: """


FORCED_SYNTHESIS_PROMPT_SYSTEM = """You have exhausted the allowed search/lookup budget for this question.

Using ONLY the evidence already present in the trajectory below, provide the best concise final answer you can. Do not request another search or lookup. Do not return an empty answer. Even if the evidence is incomplete, synthesize the best answer supported by what you observed.

Return EXACTLY one line in this format:
Action: finish[<concise answer>]

Question: {question}
{scratchpad}"""
