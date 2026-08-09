REACT_SYSTEM_PROMPT = """Solve a multi-step, multi-hop question-answering task by alternating between Thought, Action, and Observation.

You have access to the following tools:
(1) search[query]: Retrieves the top FullWiki candidates and updates a bounded, cross-encoder-reranked Active Evidence Memory. The system shows that current memory separately on every turn.
(2) lookup[keyword]: Searches ONLY the currently loaded rank-1 page, when that page is retained in Active Evidence Memory, and returns the next sentence containing the keyword. Repeating the same lookup advances to the next match on that same page.
(3) finish[answer]: Ends the task and provides the final concise answer based ONLY on observed evidence.

STRICT FORMATTING RULES:
For search and lookup turns, output exactly:
Thought: <your reasoning about the next evidence needed>
Action: <search[query] or lookup[keyword]>

For the final turn, output exactly:
Thought: <brief evidence-based synthesis>
Action: finish[<short canonical answer only>]
Support: [["Exact Wikipedia title", <sentence_id>], ...]

FINAL ANSWER RULES:
- Derive the answer only from observed evidence. For entity, date, number, and phrase answers, copy the exact concise wording from the evidence whenever possible. For yes/no and comparison questions, infer the concise answer ("yes" or "no") from the cited facts.
- Prefer the exact concise answer wording from the evidence whenever possible; do not paraphrase, generalize, or shorten a multiword entity or noun phrase.
- Use exactly "yes" or "no" for yes/no questions.
- Do NOT write "The answer is ...", "Answer: ...", explanations, evidence, or full sentences inside finish[...].
- Put evidence citations only on the Support line.

DO NOT write "Observation:" yourself. The system provides the Observation after search/lookup.
DO NOT hallucinate facts or supporting evidence.
Active Evidence Memory is the current persistent evidence set; later searches may replace weaker documents with stronger ones. Base reasoning on the current memory plus any explicit lookup observations in the history.
When Active Evidence Memory or an Observation contains labels such as [Albert Einstein | sent 1], the integer is the exact HotpotQA sentence ID for that page.
For Support, cite ONLY sentence labels that were actually observed, copy the Wikipedia title exactly, and include every sentence needed to justify the answer.
If no observed sentence supports the answer, return Support: [] rather than inventing evidence.
If an observation states that a search query was already performed, reformulate the query or use lookup; do not repeat identical searches.
When you identify a bridge entity or exact Wikipedia title, search the entity or title directly rather than repeating the full original question.
Finish immediately as soon as you observe sufficient evidence to answer and cite the required supporting facts; do not perform redundant confirmation searches.

---
HERE ARE FEW-SHOT EXAMPLES OF THE REQUIRED FORMAT:

EXAMPLE 1:
Question: Were Albert Einstein and Niels Bohr born in the same country?
Thought: I need to find Albert Einstein's birth country first.
Action: search[Albert Einstein]
Observation: Loaded [Albert Einstein] (rank 1).
[Albert Einstein | sent 0] Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist.
[Albert Einstein | sent 1] He was born in Ulm, in the Kingdom of Württemberg in the German Empire.
Thought: Albert Einstein was born in Germany. Now I need Niels Bohr's birth country.
Action: search[Niels Bohr]
Observation: Loaded [Niels Bohr] (rank 1).
[Niels Bohr | sent 0] Niels Henrik David Bohr (7 October 1885 – 18 November 1962) was a Danish physicist.
[Niels Bohr | sent 1] He was born in Copenhagen, Denmark.
Thought: Einstein was born in Germany and Bohr was born in Denmark. They were born in different countries.
Action: finish[no]
Support: [["Albert Einstein", 1], ["Niels Bohr", 1]]

---
EXAMPLE 2:
Question: What is the birthplace of the director of the 2010 film Inception?
Thought: I need to find who directed the film Inception.
Action: search[Inception]
Observation: Loaded [Inception] (rank 1).
[Inception | sent 0] Inception is a 2010 science fiction action film written and directed by Christopher Nolan.
Thought: Christopher Nolan directed Inception. Now I need to search for Christopher Nolan to find his birthplace.
Action: search[Christopher Nolan]
Observation: Loaded [Christopher Nolan] (rank 1).
[Christopher Nolan | sent 0] Christopher Edward Nolan (born 30 July 1970) is a British-American film director.
[Christopher Nolan | sent 1] He was born in London, England.
Thought: Observed evidence states Christopher Nolan was born in London, England.
Action: finish[London, England]
Support: [["Inception", 0], ["Christopher Nolan", 1]]

---
EXAMPLE 3:
Question: In what city is the headquarters of the company founded by Bill Gates located?
Thought: I need to search for Bill Gates to find the company he founded.
Action: search[Bill Gates]
Observation: Loaded [Bill Gates] (rank 1).
[Bill Gates | sent 0] William Henry Gates III (born October 28, 1955) is an American business magnate.
[Bill Gates | sent 1] He is best known for co-founding software giant Microsoft along with Paul Allen.
Thought: Bill Gates co-founded Microsoft. Now I need to search for Microsoft to find its headquarters.
Action: search[Microsoft]
Observation: Loaded [Microsoft] (rank 1).
[Microsoft | sent 0] Microsoft Corporation is an American multinational technology corporation.
Thought: I will lookup where Microsoft's headquarters is located on the Microsoft page.
Action: lookup[headquarters]
Observation: (Result 1/1) [Microsoft | sent 4] Its corporate headquarters is located in Redmond, Washington.
Thought: Microsoft's headquarters is in Redmond, Washington. I have sufficient evidence to answer immediately.
Action: finish[Redmond, Washington]
Support: [["Bill Gates", 1], ["Microsoft", 4]]
"""

FORCED_SYNTHESIS_PROMPT_SYSTEM = """You have exhausted the allowed search/lookup budget for this question.

Using ONLY the evidence already present in the trajectory below, provide the best supported final answer. Do not request another search or lookup. Do not return an empty answer.

Return EXACTLY these two lines:
Action: finish[<short canonical answer only>]
Support: [["Exact Wikipedia title", <sentence_id>], ...]

FINAL ANSWER RULES:
- Derive the answer only from observed evidence. For entity, date, number, and phrase answers, copy the exact concise wording from the evidence whenever possible. For yes/no and comparison questions, infer the concise answer ("yes" or "no") from the cited facts.
- Prefer the exact concise answer wording from the evidence whenever possible; do not paraphrase, generalize, or shorten a multiword entity or noun phrase.
- Use exactly "yes" or "no" for yes/no questions.
- Do NOT include "The answer is ...", "Answer: ...", explanations, evidence, or full sentences inside finish[...].
- Put evidence citations only on the Support line.

For Support, cite ONLY sentence labels that were actually observed. Copy titles and sentence IDs exactly. If no observed sentence supports the answer, return Support: [].

Examples of valid final answers:
Action: finish[yes]
Action: finish[no]
Action: finish[London, England]
Action: finish[Redmond, Washington]
"""

REACT_PROMPT_SYSTEM = REACT_SYSTEM_PROMPT
