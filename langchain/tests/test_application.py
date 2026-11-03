"""Offline tests for the LangChain demo application."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from typing import Any

from langchain_core.documents import Document

from opendma_demo_alfresco_langchain_metadata_aware_rag.cli import (
    parse_args,
    unique_site_ids,
)
from opendma_demo_alfresco_langchain_metadata_aware_rag.output import (
    DEMO_QUESTION,
    print_interactive_intro,
)
from opendma_demo_alfresco_langchain_metadata_aware_rag.rag import (
    MetadataAwareRag,
    build_search_schema,
)
from opendma_demo_alfresco_langchain_metadata_aware_rag.sites import (
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


class QuerySchemaTests(unittest.TestCase):
    def test_schema_allows_only_indexed_sites_or_no_filter(self) -> None:
        sites = [SiteInfo("swsdp", "Sample site", "Sample project content")]
        schema = build_search_schema(sites)

        choices = schema["properties"]["site"]["oneOf"]
        self.assertEqual(choices[0]["const"], "swsdp")
        self.assertEqual(choices[1]["type"], "null")
        self.assertEqual(len(choices), 2)


class FakeStructuredModel:
    def invoke(self, messages: list[dict[str, str]]) -> dict[str, str]:
        return {"query": "website localisation status", "site": "swsdp"}


class FakeChatModel:
    def __init__(self) -> None:
        self.schema: dict[str, Any] | None = None

    def with_structured_output(self, schema: dict[str, Any]) -> FakeStructuredModel:
        self.schema = schema
        return FakeStructuredModel()

    def invoke(self, messages: list[dict[str, str]]) -> SimpleNamespace:
        return SimpleNamespace(content="The answer comes from the filtered context.")


class FakeVectorStore:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.queries: list[str] = []

    def similarity_search(
        self,
        query: str,
        *,
        k: int,
        filter: Any,
    ) -> list[Document]:
        self.queries.append(query)
        return [document for document in self.documents if filter(document)][:k]


class RagWorkflowTests(unittest.TestCase):
    def test_three_step_workflow_uses_analyzed_query_and_site_filter(self) -> None:
        vector_store = FakeVectorStore(
            [
                Document(
                    page_content="Relevant meeting notes",
                    metadata={"alfresco:Site": "swsdp", "opendma:CheckedOut": False},
                ),
                Document(
                    page_content="Distracting engineering content",
                    metadata={"alfresco:Site": "engineering", "opendma:CheckedOut": False},
                ),
            ]
        )
        model = FakeChatModel()
        application = MetadataAwareRag(
            vector_store=vector_store,  # type: ignore[arg-type]
            sites=[SiteInfo("swsdp", "Sample site", "Sample project content")],
            llm=model,  # type: ignore[arg-type]
            top_k=10,
        )

        result = application.run_metadata_aware("What is the localisation status?")

        self.assertEqual(vector_store.queries, ["website localisation status"])
        self.assertEqual(
            result.analysis,
            {"query": "website localisation status", "site": "swsdp"},
        )
        self.assertEqual(
            [doc.metadata["alfresco:Site"] for doc in result.context],
            ["swsdp"],
        )
        self.assertEqual(result.answer, "The answer comes from the filtered context.")


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
