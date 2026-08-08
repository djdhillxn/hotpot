class LocalHotpotRetriever:
    def __init__(self, context_paragraphs=None, max_observation_chars=800):
        self.corpus = {}
        self.max_observation_chars = max_observation_chars
        if context_paragraphs:
            self.load_context(context_paragraphs)
        self.current_title = None
        self.last_result = None

    @property
    def current_page_title(self):
        return self.current_title

    def reset(self):
        self.current_title = None
        self.last_result = None

    def load_context(self, paragraphs):
        self.corpus.clear()
        if not paragraphs:
            return

        # HuggingFace Parquet dict-of-lists format.
        if isinstance(paragraphs, dict) and "title" in paragraphs and "sentences" in paragraphs:
            titles = paragraphs["title"]
            sentences_groups = paragraphs["sentences"]
            for title, sentence_group in zip(titles, sentences_groups):
                if title:
                    self._store_paragraph(title, sentence_group)
            return

        # List of dicts or [title, sentences] pairs.
        if isinstance(paragraphs, list):
            for paragraph in paragraphs:
                title = ""
                sentences = []
                if isinstance(paragraph, dict):
                    title = paragraph.get("title", "")
                    sentences = paragraph.get("sentences", [])
                elif isinstance(paragraph, (list, tuple)) and len(paragraph) >= 2:
                    title = str(paragraph[0])
                    sentences = paragraph[1]

                if title:
                    self._store_paragraph(title, sentences)

    def _store_paragraph(self, title, sentences):
        if isinstance(sentences, list):
            sentence_list = [str(sentence) for sentence in sentences]
        else:
            sentence_list = [str(sentences)]

        self.corpus[title.lower()] = {
            "title": title,
            "sentences": sentence_list,
        }

    def _visible_sentences(self, page, sentence_ids=None):
        if sentence_ids is None:
            sentence_ids = range(len(page["sentences"]))

        visible = []
        char_count = 0
        for sent_id in sentence_ids:
            if sent_id < 0 or sent_id >= len(page["sentences"]):
                continue
            text = page["sentences"][sent_id]
            rendered = f"[{page['title']} | sent {sent_id}] {text}"
            if visible and char_count + len(rendered) > self.max_observation_chars:
                break
            visible.append({"sent_id": sent_id, "text": text})
            char_count += len(rendered)

        return visible

    @staticmethod
    def _render_sentences(title, sentences):
        if not sentences:
            return "(no sentence text available)"
        return "\n".join(
            f"[{title} | sent {item['sent_id']}] {item['text']}" for item in sentences
        )

    def _load_page(self, page, query, match_type):
        self.current_title = page["title"]
        visible = self._visible_sentences(page)
        self.last_result = {
            "action": "search",
            "query": query,
            "status": "loaded",
            "match_type": match_type,
            "title": page["title"],
            "sentences": visible,
        }
        return (
            f"Observation: Loaded [{page['title']}]. Sentence IDs are HotpotQA sentence IDs; "
            "use them exactly when citing supporting facts.\n"
            + self._render_sentences(page["title"], visible)
        )

    def search(self, query):
        query_clean = query.strip().strip("'\"").lower()

        if query_clean in self.corpus:
            return self._load_page(self.corpus[query_clean], query, "exact_title")

        for key, page in self.corpus.items():
            if query_clean in key or key in query_clean:
                return self._load_page(page, query, "title_substring")

        for page in self.corpus.values():
            joined = " ".join(page["sentences"])
            if query_clean in joined.lower():
                return self._load_page(page, query, "text_substring")

        available_titles = [page["title"] for page in self.corpus.values()][:5]
        self.last_result = {
            "action": "search",
            "query": query,
            "status": "not_found",
            "match_type": None,
            "title": None,
            "sentences": [],
            "available_titles": available_titles,
        }
        return f"Observation: Could not find page for '{query}'. Available documents: {available_titles}"

    def lookup(self, keyword):
        if not self.current_title:
            self.last_result = {
                "action": "lookup",
                "query": keyword,
                "status": "no_current_page",
                "title": None,
                "sentences": [],
            }
            return "Observation: No document currently loaded. Perform a `search` first."

        keyword_clean = keyword.strip().strip("'\"").lower()
        page = self.corpus.get(self.current_title.lower())
        if page is None:
            return f"Observation: Current document [{self.current_title}] is unavailable."

        matching_ids = [
            sent_id
            for sent_id, sentence in enumerate(page["sentences"])
            if keyword_clean in sentence.lower()
        ]
        visible = self._visible_sentences(page, matching_ids[:3])

        if visible:
            self.last_result = {
                "action": "lookup",
                "query": keyword,
                "status": "found",
                "title": page["title"],
                "sentences": visible,
            }
            return (
                f"Observation: Found matches in [{page['title']}]. Sentence IDs are HotpotQA sentence IDs.\n"
                + self._render_sentences(page["title"], visible)
            )

        self.last_result = {
            "action": "lookup",
            "query": keyword,
            "status": "not_found",
            "title": page["title"],
            "sentences": [],
        }
        return f"Observation: Could not find '{keyword}' in [{page['title']}]."
