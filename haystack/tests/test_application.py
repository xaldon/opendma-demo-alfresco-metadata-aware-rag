"""Offline tests for the Haystack demo application."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from haystack import Document
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore

from opendma_demo_alfresco_haystack_metadata_aware_rag.cli import (
    parse_args,
    unique_site_ids,
)
from opendma_demo_alfresco_haystack_metadata_aware_rag.content import TextDocumentFilter
from opendma_demo_alfresco_haystack_metadata_aware_rag.output import (
    DEMO_QUESTION,
    print_interactive_intro,
)
from opendma_demo_alfresco_haystack_metadata_aware_rag.rag import (
    SearchParameterParser,
    build_filters,
)
from opendma_demo_alfresco_haystack_metadata_aware_rag.sites import (
    SiteInfo,
    select_requested_sites,
)


class SiteSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sites = [
            SiteInfo("swsdp", "Sample site", "Sample project content"),
            SiteInfo("engineering", "Engineering", "Engineering content"),
        ]

    def test_requested_order_is_preserved(self) -> None:
        selected = select_requested_sites(self.sites, ["engineering", "swsdp"])
        self.assertEqual([site.id for site in selected], ["engineering", "swsdp"])

    def test_unknown_site_reports_available_sites(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"Alfresco site\(s\) not found: finance.*Available sites: engineering, swsdp",
        ):
            select_requested_sites(self.sites, ["finance"])


class QueryAnalysisTests(unittest.TestCase):
    def test_parser_builds_semantic_query_and_site_filter(self) -> None:
        parser = SearchParameterParser({"swsdp", "engineering"})
        result = parser.run(
            [ChatMessage.from_assistant('{"query":"website localisation status","site":"swsdp"}')]
        )

        self.assertEqual(result["query"], "website localisation status")
        self.assertEqual(result["site"], "swsdp")
        self.assertEqual(result["filters"], build_filters("swsdp"))

    def test_parser_allows_no_site_filter(self) -> None:
        parser = SearchParameterParser({"swsdp"})
        result = parser.run(
            [ChatMessage.from_assistant('{"query":"retention policy","site":null}')]
        )

        self.assertIsNone(result["site"])
        self.assertEqual(result["filters"], build_filters(None))

    def test_parser_rejects_unknown_site(self) -> None:
        parser = SearchParameterParser({"swsdp"})
        with self.assertRaisesRegex(ValueError, "unknown site ID: finance"):
            parser.run(
                [ChatMessage.from_assistant('{"query":"quarterly report","site":"finance"}')]
            )


class RetrieverTests(unittest.TestCase):
    def test_haystack_filter_restricts_site_and_checked_out_status(self) -> None:
        store = InMemoryDocumentStore(embedding_similarity_function="cosine")
        store.write_documents(
            [
                Document(
                    content="Relevant meeting notes",
                    embedding=[1.0, 0.0],
                    meta={"alfresco:Site": "swsdp", "opendma:CheckedOut": False},
                ),
                Document(
                    content="Distracting engineering content",
                    embedding=[1.0, 0.0],
                    meta={"alfresco:Site": "engineering", "opendma:CheckedOut": False},
                ),
                Document(
                    content="Checked-out draft",
                    embedding=[1.0, 0.0],
                    meta={"alfresco:Site": "swsdp", "opendma:CheckedOut": True},
                ),
            ]
        )
        retriever = InMemoryEmbeddingRetriever(document_store=store, top_k=10)

        documents = retriever.run(
            query_embedding=[1.0, 0.0],
            filters=build_filters("swsdp"),
        )["documents"]

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].content, "Relevant meeting notes")


class ConversionTests(unittest.TestCase):
    def test_empty_documents_are_removed(self) -> None:
        result = TextDocumentFilter().run(
            [Document(content="content"), Document(content="  "), Document(content=None)]
        )
        self.assertEqual([document.content for document in result["documents"]], ["content"])


class ConsoleTests(unittest.TestCase):
    def capture_intro(self, indexed_site_ids: list[str]) -> str:
        output = io.StringIO()
        with redirect_stdout(output):
            print_interactive_intro(compare=True, indexed_site_ids=indexed_site_ids)
        return output.getvalue()

    def test_demo_question_is_printed_when_swsdp_was_indexed(self) -> None:
        self.assertIn(DEMO_QUESTION, self.capture_intro(["swsdp", "engineering"]))

    def test_demo_question_is_not_printed_without_swsdp(self) -> None:
        self.assertNotIn(DEMO_QUESTION, self.capture_intro(["engineering"]))


class CommandLineTests(unittest.TestCase):
    def test_interactive_compare_arguments(self) -> None:
        args = parse_args(["swsdp", "engineering", "--compare"])
        self.assertEqual(args.sites, ["swsdp", "engineering"])
        self.assertTrue(args.compare)
        self.assertIsNone(args.question)

    def test_duplicate_sites_are_removed_in_order(self) -> None:
        self.assertEqual(
            unique_site_ids(["swsdp", "engineering", "swsdp"]),
            ["swsdp", "engineering"],
        )


if __name__ == "__main__":
    unittest.main()
