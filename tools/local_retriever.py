class LocalHotpotRetriever:
    def __init__(self, context_paragraphs=None):
        self.corpus = {}
        if context_paragraphs:
            self.load_context(context_paragraphs)
        self.current_title = None

    @property
    def current_page_title(self):
        return self.current_title

    def reset(self):
        self.current_title = None

    def load_context(self, paragraphs):
        self.corpus.clear()
        if not paragraphs:
            return

        # HuggingFace Parquet dict of lists format: {"title": [...], "sentences": [...]}
        if isinstance(paragraphs, dict) and "title" in paragraphs and "sentences" in paragraphs:
            titles = paragraphs["title"]
            sentences_groups = paragraphs["sentences"]
            for title, s_group in zip(titles, sentences_groups):
                text = " ".join(s_group) if isinstance(s_group, list) else str(s_group)
                if title:
                    self.corpus[title.lower()] = (title, text)
            return

        # List of items (dicts or lists/tuples)
        if isinstance(paragraphs, list):
            for p in paragraphs:
                title = ""
                text = ""
                if isinstance(p, dict):
                    title = p.get("title", "")
                    sentences = p.get("sentences", [])
                    text = " ".join(sentences) if isinstance(sentences, list) else str(sentences)
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    title = str(p[0])
                    sentences = p[1]
                    text = " ".join(sentences) if isinstance(sentences, list) else str(sentences)

                if title:
                    self.corpus[title.lower()] = (title, text)

    def search(self, query):
        query_clean = query.strip().strip("'\"").lower()

        if query_clean in self.corpus:
            orig_title, text = self.corpus[query_clean]
            self.current_title = orig_title
            return f"Observation: Loaded [{orig_title}]: {text[:800]}"

        for key, (orig_title, text) in self.corpus.items():
            if query_clean in key or key in query_clean:
                self.current_title = orig_title
                return f"Observation: Loaded [{orig_title}]: {text[:800]}"

        for orig_title, text in self.corpus.values():
            if query_clean in text.lower():
                self.current_title = orig_title
                return f"Observation: Loaded [{orig_title}]: {text[:800]}"

        available_titles = list(set([t for t, _ in self.corpus.values()]))[:5]
        return f"Observation: Could not find page for '{query}'. Available documents: {available_titles}"

    def lookup(self, keyword):
        if not self.current_title:
            return "Observation: No document currently loaded. Perform a `search` first."

        keyword_clean = keyword.strip().strip("'\"").lower()
        _, text = self.corpus.get(self.current_title.lower(), ("", ""))

        if keyword_clean in text.lower():
            sentences = text.split(". ")
            matches = [s for s in sentences if keyword_clean in s.lower()]
            return f"Observation: Found matches in [{self.current_title}]: " + ". ".join(matches[:3])

        return f"Observation: Could not find '{keyword}' in [{self.current_title}]."
