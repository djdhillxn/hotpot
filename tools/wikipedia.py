import requests
from config import WIKIPEDIA_USER_AGENT

class WikipediaToolSet:
    def __init__(self, user_agent=WIKIPEDIA_USER_AGENT):
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        
        self.current_page_title = None
        self.current_page_paragraphs = []
        self.visited_pages = set()

    def search(self, query):
        query = query.strip().strip("'\"")
        if not query:
            return "Observation: Search query cannot be empty."

        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 5,
        }

        try:
            resp = self.session.get(search_url, params=search_params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            results = data.get("query", {}).get("search", [])
            if not results:
                return f"Observation: Could not find any Wikipedia page for '{query}'."

            target_title = results[0]["title"]

            extract_params = {
                "action": "query",
                "prop": "extracts",
                "titles": target_title,
                "explaintext": True,
                "format": "json",
                "redirects": True,
            }
            page_resp = self.session.get(search_url, params=extract_params, timeout=10)
            page_resp.raise_for_status()
            pages_data = page_resp.json().get("query", {}).get("pages", {})

            page_info = next(iter(pages_data.values()), {})
            if "missing" in page_info:
                titles = [r["title"] for r in results]
                return f"Observation: Could not find exact page '{query}'. Similar pages: {titles}"

            full_text = page_info.get("extract", "")
            resolved_title = page_info.get("title", target_title)

            raw_paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
            self.current_page_title = resolved_title
            self.current_page_paragraphs = raw_paragraphs
            self.visited_pages.add(resolved_title)

            lead_summary = raw_paragraphs[0] if raw_paragraphs else "Page is empty."
            if len(lead_summary) > 800:
                lead_summary = lead_summary[:800] + "..."

            return f"Observation: Loaded [{resolved_title}]: {lead_summary}"

        except Exception as e:
            return f"Observation: Error searching Wikipedia for '{query}': {str(e)}"

    def lookup(self, keyword):
        keyword = keyword.strip().strip("'\"")
        if not self.current_page_title or not self.current_page_paragraphs:
            return "Observation: No Wikipedia page currently loaded. Perform a `search` first."

        if not keyword:
            return "Observation: Lookup keyword cannot be empty."

        matches = []
        keyword_lower = keyword.lower()

        for p in self.current_page_paragraphs:
            if keyword_lower in p.lower():
                matches.append(p)
                if len(matches) >= 3:
                    break

        if matches:
            combined = "\n\n".join(matches)
            if len(combined) > 1000:
                combined = combined[:1000] + "..."
            return f"Observation: Found matches for '{keyword}' in [{self.current_page_title}]:\n{combined}"
        else:
            return f"Observation: Could not find '{keyword}' in [{self.current_page_title}]."

    def reset(self):
        self.current_page_title = None
        self.current_page_paragraphs = []
        self.visited_pages.clear()
