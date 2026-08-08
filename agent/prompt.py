REACT_PROMPT_SYSTEM = """Solve a multi-step, multi-hop question-answering task by alternating between Thought, Action, and Observation.

You have access to the following tools:
(1) search[entity]: Searches the available HotpotQA context for an entity page and returns sentence-labeled evidence.
(2) lookup[keyword]: Searches the currently loaded page for sentences containing the keyword.
(3) finish[answer]: Ends the task and provides the final concise answer based ONLY on observed evidence.

STRICT FORMATTING RULES:
For search and lookup turns, output exactly:
Thought: <your reasoning about the next evidence needed>
Action: <search[entity] or lookup[keyword]>

For the final turn, output exactly:
Thought: <brief evidence-based synthesis>
Action: finish[<concise answer>]
Support: [["Exact Wikipedia title", <sentence_id>], ...]

DO NOT write "Observation:" yourself. The system provides the Observation after search/lookup.
DO NOT hallucinate facts or supporting evidence.
When an Observation contains labels such as [Radiohead | sent 1], the integer is the exact HotpotQA sentence ID for that page.
For Support, cite ONLY sentence labels that were actually observed, copy the Wikipedia title exactly, and include every sentence needed to justify the answer.
If no observed sentence supports the answer, return Support: [] rather than inventing evidence.

---
EXAMPLE 1:
Question: Were Scott Derrickson and Ed Wood born in the same state?
Thought: I need Scott Derrickson's birthplace.
Action: search[Scott Derrickson]
Observation: Loaded [Scott Derrickson]. Sentence IDs are HotpotQA sentence IDs; use them exactly when citing supporting facts.
[Scott Derrickson | sent 0] Scott Derrickson is an American filmmaker.
[Scott Derrickson | sent 1] He was born in Denver, Colorado.
Thought: Scott Derrickson was born in Colorado. I now need Ed Wood's birthplace.
Action: search[Ed Wood]
Observation: Loaded [Ed Wood]. Sentence IDs are HotpotQA sentence IDs; use them exactly when citing supporting facts.
[Ed Wood | sent 0] Edward Davis Wood Jr. was an American filmmaker.
[Ed Wood | sent 1] He was born in Poughkeepsie, New York.
Thought: The two birthplaces are in different states.
Action: finish[no]
Support: [["Scott Derrickson", 1], ["Ed Wood", 1]]

---
EXAMPLE 2:
Question: What is the birth date of the lead singer of Radiohead?
Thought: I need to identify Radiohead's lead singer.
Action: search[Radiohead]
Observation: Loaded [Radiohead]. Sentence IDs are HotpotQA sentence IDs; use them exactly when citing supporting facts.
[Radiohead | sent 0] Radiohead are an English rock band formed in Abingdon, Oxfordshire, in 1985.
[Radiohead | sent 1] The band consists of Thom Yorke (lead vocals, guitar), Jonny Greenwood, Colin Greenwood, Ed O'Brien, and Philip Selway.
Thought: Thom Yorke is the lead singer. I need his birth date.
Action: search[Thom Yorke]
Observation: Loaded [Thom Yorke]. Sentence IDs are HotpotQA sentence IDs; use them exactly when citing supporting facts.
[Thom Yorke | sent 0] Thomas Edward Yorke (born 7 October 1968) is an English musician and main vocalist of Radiohead.
Thought: The observed evidence identifies Thom Yorke as Radiohead's lead vocalist and gives his birth date.
Action: finish[7 October 1968]
Support: [["Radiohead", 1], ["Thom Yorke", 0]]

---
Task Question: {question}
{scratchpad}Thought: """


FORCED_SYNTHESIS_PROMPT_SYSTEM = """You have exhausted the allowed search/lookup budget for this question.

Using ONLY the evidence already present in the trajectory below, provide the best concise final answer you can. Do not request another search or lookup. Do not return an empty answer. Even if the evidence is incomplete, synthesize the best answer supported by what you observed.

Return EXACTLY these two lines:
Action: finish[<concise answer>]
Support: [["Exact Wikipedia title", <sentence_id>], ...]

For Support, cite ONLY sentence labels that were actually observed. Copy titles and sentence IDs exactly. If no observed sentence supports the answer, return Support: [].

Question: {question}
{scratchpad}"""
