import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from retrieval.fullwiki_retriever import FullWikiSearchBackend
from tools.build_hyperlink_graph import extract_detailed_hyperlinks


class TestPrecisionGraphExpansion(unittest.TestCase):
    def test_extract_detailed_hyperlinks_html(self):
        record = {
            "_id": "doc123",
            "title": "Main Subject",
            "text_with_links": [
                'Main Subject was born in <a href="Paris">Paris</a>, France.',
                'He studied at <a href="University_of_Oxford">University of Oxford</a>.',
            ],
        }
        title, links, edges = extract_detailed_hyperlinks(record)
        self.assertEqual(title, "Main Subject")
        self.assertIn("Paris", links)
        self.assertIn("University of Oxford", links)
        self.assertEqual(len(edges), 2)
        self.assertEqual(edges[0][0], "Main Subject")
        self.assertEqual(edges[0][1], "doc123")
        self.assertEqual(edges[0][2], 0)  # sentence id
        self.assertEqual(edges[0][3], "Paris")  # anchor text
        self.assertEqual(edges[0][4], "Paris")  # target title

    def test_sqlite_backend_edges_and_outdegree(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "hyperlink_graph.db")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE edges (
                    source_title TEXT, source_doc_id TEXT, source_sent_id INTEGER,
                    anchor_text TEXT, target_title TEXT, target_doc_id TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE outdegree (target_title TEXT PRIMARY KEY, count INTEGER)
            """)
            cursor.execute("""
                CREATE TABLE title_map (norm_title TEXT PRIMARY KEY, doc_id TEXT)
            """)

            cursor.execute(
                "INSERT INTO edges VALUES ('Alpha', 'doc1', 0, 'Beta Link', 'Beta', 'doc2')"
            )
            cursor.execute(
                "INSERT INTO edges VALUES ('Alpha', 'doc1', 1, 'Gamma Link', 'United States', 'doc3')"
            )
            cursor.execute("INSERT INTO outdegree VALUES ('beta', 5)")
            cursor.execute("INSERT INTO outdegree VALUES ('united states', 50000)")
            conn.commit()
            conn.close()

            # Create mock backend
            class MockBackend(FullWikiSearchBackend):
                def __init__(self, db_p):
                    self.db_path = db_p
                    self.title_graph = {}
                    self.title_to_doc_id = {"alpha": "doc1", "beta": "doc2", "united states": "doc3"}

            backend = MockBackend(db_path)
            edges = backend.get_outgoing_edges("Alpha")
            self.assertEqual(len(edges), 2)
            self.assertEqual(edges[0]["target_title"], "Beta")
            self.assertEqual(edges[0]["anchor_text"], "Beta Link")

            beta_outdegree = backend.get_target_outdegree("Beta")
            us_outdegree = backend.get_target_outdegree("United States")
            self.assertEqual(beta_outdegree, 5)
            self.assertEqual(us_outdegree, 50000)


if __name__ == "__main__":
    unittest.main()
