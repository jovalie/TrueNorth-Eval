import unittest
from langchain_core.documents import Document
from truenorth.agent.state import ChatState, CitationSource, CitedSource
from truenorth.utils.citation_manager import CitationManager


class TestCitationManager(unittest.TestCase):

    def setUp(self):
        # Create dummy documents
        self.pdf_doc = Document(page_content="Stress causes cortisol spikes which increase oil production.", metadata={"source": "/data/dermatology_handbook.pdf", "page": 42, "author": "Dr. Smith", "title": "Dermatology Handbook", "year": "2023"})

        self.web_doc = Document(
            page_content="Top 10 tips for oily skin: 1. Drink water. 2. Use salicylic acid.",
            metadata={
                "url": "https://skincare.com/tips",
                "title": "Oily Skin Tips",
                # Web scraper often doesn't find author/year
            },
        )

    def test_normalize_pdf(self):
        source = CitationManager.normalize_document(self.pdf_doc)
        self.assertEqual(source.type, "pdf")
        self.assertEqual(source.author, "Dr. Smith")
        self.assertIn("api/pdf/dermatology_handbook.pdf", source.url)
        self.assertIn("page=42", source.url)
        self.assertEqual(source.filename, "dermatology_handbook.pdf")

    def test_normalize_web(self):
        source = CitationManager.normalize_document(self.web_doc)
        self.assertEqual(source.type, "web")
        self.assertEqual(source.url, "https://skincare.com/tips")
        self.assertEqual(source.filename, "web_source")
        # Test fallback author extraction from domain
        self.assertEqual(source.author, "skincare.com")

    def test_process_documents_mixed(self):
        state = ChatState(question="test")
        state.documents = [self.pdf_doc, self.web_doc]

        state = CitationManager.process_documents(state)

        registry = state.citation_registry
        self.assertEqual(len(registry), 2)

        # Check numbering
        self.assertEqual(registry[1].type, "pdf")
        self.assertEqual(registry[2].type, "web")

        # Check source_id injection into docs
        self.assertEqual(state.documents[0].metadata["source_id"], 1)
        self.assertEqual(state.documents[1].metadata["source_id"], 2)

    def test_deduplication(self):
        state = ChatState(question="test")
        # Add same doc twice
        state.documents = [self.pdf_doc, self.pdf_doc]

        state = CitationManager.process_documents(state)

        # Should only be 1 in registry
        self.assertEqual(len(state.citation_registry), 1)
        # Both docs should point to ID 1
        self.assertEqual(state.documents[0].metadata["source_id"], 1)
        self.assertEqual(state.documents[1].metadata["source_id"], 1)

    def test_get_context_string(self):
        state = ChatState(question="test")
        state.documents = [self.pdf_doc]
        state = CitationManager.process_documents(state)

        context = CitationManager.get_context_string(state)

        self.assertIn("Source [1]:", context)
        self.assertIn("Dr. Smith", context)
        self.assertIn("cortisol spikes", context)

    def test_resolve_citations_renumbering(self):
        """
        Verify that we renumber citations sequentially (1, 2, 3)
        and update the text accordingly.
        """
        state = ChatState(question="test")
        state.documents = [self.pdf_doc, self.web_doc]  # ID 1 and 2
        state = CitationManager.process_documents(state)

        # Scenario: LLM only uses ID 2 (Web).
        # We expect it to be renumbered to [1]
        state.generated_citations = [CitedSource(source_id=2, quote="Use salicylic acid")]
        state.generation = "You should use salicylic acid [2] for your skin."

        final_citations, renumbered_text = CitationManager.resolve_citations(state)

        self.assertEqual(len(final_citations), 1)

        # Check Citation Object
        c1 = final_citations[0]
        self.assertEqual(c1["id"], 1)  # Renumbered from 2 -> 1
        self.assertEqual(c1["url"], "https://skincare.com/tips")

        # Check Text Update
        self.assertEqual(renumbered_text, "You should use salicylic acid [1] for your skin.")

    def test_resolve_citations_fallback(self):
        """
        If LLM returns empty quote, fallback to snippet.
        """
        state = ChatState(question="test")
        state.documents = [self.pdf_doc]
        state = CitationManager.process_documents(state)

        # Simulate LLM returning ID but empty quote
        state.generated_citations = [CitedSource(source_id=1, quote="")]

        final_citations, _ = CitationManager.resolve_citations(state)

        self.assertEqual(len(final_citations), 1)
        # Should fall back to the content snippet (first 200 chars)
        self.assertTrue(final_citations[0]["snippet"].startswith("Stress causes"))


if __name__ == "__main__":
    unittest.main()
