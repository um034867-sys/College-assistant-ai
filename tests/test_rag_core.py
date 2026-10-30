import unittest
from pathlib import Path

from langchain_core.documents import Document

from rag_core import (
    RetrievedPassage,
    build_grounded_prompt,
    citation_label,
    format_passages,
    is_relevant,
    split_documents,
)

ROOT = Path(__file__).resolve().parents[1]


class RagCoreTests(unittest.TestCase):
    def test_chunking_preserves_source_metadata(self):
        source = Document(page_content=("A policy sentence. " * 60), metadata={"source": "guide.txt"})
        chunks = split_documents([source], chunk_size=250, chunk_overlap=40)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.metadata["source"] == "guide.txt" for chunk in chunks))

    def test_chunking_rejects_overlap_larger_than_chunk(self):
        with self.assertRaises(ValueError):
            split_documents([Document(page_content="text")], chunk_size=200, chunk_overlap=200)

    def test_relevance_gate_accepts_score_at_threshold(self):
        self.assertTrue(is_relevant(0.30, 0.30))

    def test_relevance_gate_rejects_lower_score(self):
        self.assertFalse(is_relevant(0.29, 0.30))

    def test_pdf_zero_based_page_is_displayed_as_page_one(self):
        passage = format_passages([(Document(page_content="Policy.", metadata={"source": "rules.pdf", "page": 0}), 0.8)])[0]
        self.assertEqual(passage.page, 1)
        self.assertEqual(citation_label(passage), "rules.pdf, page 1")

    def test_text_source_has_no_fake_page_number(self):
        passage = format_passages([(Document(page_content="Policy.", metadata={"source": "rules.txt"}), 0.8)])[0]
        self.assertIsNone(passage.page)
        self.assertEqual(citation_label(passage), "rules.txt")

    def test_prompt_requires_abstention_and_no_general_knowledge(self):
        passage = RetrievedPassage("Deadline is July 15.", "admissions.txt", None, 0.8)
        prompt = build_grounded_prompt("When is the deadline?", [passage])
        self.assertIn("using ONLY the source passages", prompt)
        self.assertIn("Do not guess or use general knowledge", prompt)

    def test_prompt_requires_reporting_conflicts(self):
        passages = [
            RetrievedPassage("Starts Sep 3.", "calendar_notice_A.txt", None, 0.7),
            RetrievedPassage("Starts Sep 8.", "calendar_notice_B.txt", None, 0.7),
        ]
        prompt = build_grounded_prompt("When does term start?", passages)
        self.assertIn("do not pick a winner", prompt)
        self.assertIn("calendar_notice_A.txt", prompt)
        self.assertIn("calendar_notice_B.txt", prompt)

    def test_five_baseline_sample_documents_exist(self):
        expected = {"admissions.txt", "scholarships.txt", "library.txt", "attendance.txt", "campus_services.txt"}
        present = {path.name for path in (ROOT / "sample_docs").glob("*.txt")}
        self.assertTrue(expected.issubset(present))

    def test_conflict_fixture_contains_both_dates(self):
        folder = ROOT / "sample_docs" / "conflict_demo"
        notice_a = (folder / "calendar_notice_A.txt").read_text(encoding="utf-8")
        notice_b = (folder / "calendar_notice_B.txt").read_text(encoding="utf-8")
        self.assertIn("September 3, 2026", notice_a)
        self.assertIn("September 8, 2026", notice_b)


if __name__ == "__main__":
    unittest.main()
