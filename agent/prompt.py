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
- The content inside finish[...] must be ONLY the exact concise answer appearing in the observed evidence.
- Prefer the exact concise answer wording from the evidence whenever possible; do not paraphrase, generalize, or shorten a multiword entity or noun phrase.
- Use exactly "yes" or "no" for yes/no questions.
- Do NOT write "The answer is ...", "Answer: ...", explanations, evidence, or full sentences inside finish[...].
- Put evidence citations only on the Support line.

DO NOT write "Observation:" yourself. The system provides the Observation after search/lookup.
DO NOT hallucinate facts or supporting evidence.
Active Evidence Memory is the current persistent evidence set; later searches may replace weaker documents with stronger ones. Base reasoning on the current memory plus any explicit lookup observations in the history.
When Active Evidence Memory or an Observation contains labels such as [Radiohead | sent 1], the integer is the exact HotpotQA sentence ID for that page.
For Support, cite ONLY sentence labels that were actually observed, copy the Wikipedia title exactly, and include every sentence needed to justify the answer.
If no observed sentence supports the answer, return Support: [] rather than inventing evidence.
If an observation states that a search query was already performed, reformulate the query or use lookup; do not repeat identical searches.
When you identify a bridge entity or exact Wikipedia title, search the entity or title directly rather than repeating the full original question.
Finish immediately as soon as you observe sufficient evidence to answer and cite the required supporting facts; do not perform redundant confirmation searches.

---
HERE ARE FEW-SHOT EXAMPLES OF THE REQUIRED FORMAT:

EXAMPLE 1:
Question: Were Scott Derrickson and Ed Wood born in the same state?
Thought: I need to find Scott Derrickson's birthplace first.
Action: search[Scott Derrickson]
Observation: Loaded [Scott Derrickson] (rank 1).
[Scott Derrickson | sent 0] Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.
[Scott Derrickson | sent 1] He was born in Denver, Colorado.
Thought: Scott Derrickson was born in Colorado. Now I need Ed Wood's birthplace.
Action: search[Ed Wood]
Observation: Loaded [Ed Wood] (rank 1).
[Ed Wood | sent 0] Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker, actor, writer, producer, and director.
[Ed Wood | sent 1] He was born in Poughkeepsie, New York.
Thought: Scott Derrickson was born in Colorado and Ed Wood was born in New York. They were born in different states.
Action: finish[no]
Support: [["Scott Derrickson", 1], ["Ed Wood", 1]]

---
EXAMPLE 2:
Question: What is the birth date of the lead singer of Radiohead?
Thought: I need to identify the lead singer of Radiohead.
Action: search[Radiohead]
Observation: Loaded [Radiohead] (rank 1).
[Radiohead | sent 0] Radiohead are an English rock band formed in Abingdon, Oxfordshire, in 1985.
[Radiohead | sent 1] The band consists of Thom Yorke (lead vocals, guitar), Jonny Greenwood, Colin Greenwood, Ed O'Brien, and Philip Selway.
Thought: Thom Yorke is the lead singer. Now I need Thom Yorke's birth date.
Action: search[Thom Yorke]
Observation: Loaded [Thom Yorke] (rank 1).
[Thom Yorke | sent 0] Thomas Edward Yorke (born 7 October 1968) is an English musician and main vocalist of Radiohead.
Thought: Observed evidence gives Thom Yorke's birth date as 7 October 1968.
Action: finish[7 October 1968]
Support: [["Radiohead", 1], ["Thom Yorke", 0]]

---
EXAMPLE 3:
Question: What profession do Nicholas Ray and Elia Kazan have in common?
Thought: I need to find Nicholas Ray's professions.
Action: search[Nicholas Ray]
Observation: Loaded [Nicholas Ray] (rank 1).
[Nicholas Ray | sent 0] Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for Rebel Without a Cause.
Thought: Nicholas Ray was a film director, screenwriter, and actor. Now I need Elia Kazan's professions.
Action: search[Elia Kazan]
Observation: Loaded [Elia Kazan] (rank 1).
[Elia Kazan | sent 0] Elia Kazan was an American film and theatre director, producer, screenwriter, and actor.
Thought: Both Nicholas Ray and Elia Kazan were film directors, screenwriters, and actors. The primary shared profession is director.
Action: finish[director]
Support: [["Nicholas Ray", 0], ["Elia Kazan", 0]]

---
EXAMPLE 4:
Question: Which film directed by Christopher Nolan stars Leonardo DiCaprio?
Thought: I need to search for Christopher Nolan's filmography.
Action: search[Christopher Nolan]
Observation: Loaded [Christopher Nolan] (rank 1).
[Christopher Nolan | sent 0] Christopher Edward Nolan (born 30 July 1970) is a British-American film director, screenwriter, and producer.
[Christopher Nolan | sent 1] His films include Memento, The Dark Knight Trilogy, Inception, Interstellar, and Oppenheimer.
Thought: I should search for Inception to see if it stars Leonardo DiCaprio.
Action: search[Inception]
Observation: Loaded [Inception] (rank 1).
[Inception | sent 0] Inception is a 2010 science fiction action film written and directed by Christopher Nolan.
[Inception | sent 1] The film stars Leonardo DiCaprio as a professional thief who steals information by infiltrating the subconscious.
Thought: Inception was directed by Christopher Nolan and stars Leonardo DiCaprio.
Action: finish[Inception]
Support: [["Christopher Nolan", 1], ["Inception", 0], ["Inception", 1]]

---
EXAMPLE 5:
Question: In what city is the headquarters of the company founded by Bill Gates located?
Thought: I need to search for Bill Gates to find the company he founded.
Action: search[Bill Gates]
Observation: Loaded [Bill Gates] (rank 1).
[Bill Gates | sent 0] William Henry Gates III (born October 28, 1955) is an American business magnate, investor, and philanthropist.
[Bill Gates | sent 1] He is best known for co-founding software giant Microsoft along with his late childhood friend Paul Allen.
Thought: Bill Gates co-founded Microsoft. Now I need to search for Microsoft and find its headquarters location.
Action: search[Microsoft]
Observation: Loaded [Microsoft] (rank 1).
[Microsoft | sent 0] Microsoft Corporation is an American multinational technology corporation.
[Microsoft | sent 1] It produces computer software, consumer electronics, personal computers, and related services.
Thought: I need to lookup where Microsoft's headquarters is located on the Microsoft page.
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
- The content inside finish[...] must be ONLY the exact concise answer appearing in the observed evidence.
- Prefer the exact concise answer wording from the evidence whenever possible; do not paraphrase, generalize, or shorten a multiword entity or noun phrase.
- Use exactly "yes" or "no" for yes/no questions.
- Do NOT include "The answer is ...", "Answer: ...", explanations, evidence, or full sentences inside finish[...].
- Put evidence citations only on the Support line.

For Support, cite ONLY sentence labels that were actually observed. Copy titles and sentence IDs exactly. If no observed sentence supports the answer, return Support: [].

Examples of valid final answers:
Action: finish[yes]
Action: finish[no]
Action: finish[Thom Yorke]
Action: finish[7 October 1968]
"""

REACT_PROMPT_SYSTEM = REACT_SYSTEM_PROMPT
