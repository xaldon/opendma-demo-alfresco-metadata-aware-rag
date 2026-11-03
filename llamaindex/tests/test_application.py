"""Offline tests for the LlamaIndex demo application."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from typing import Any

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import NodeWithScore, TextNode

from opendma_demo_alfresco_llamaindex_metadata_aware_rag.cli import (
    parse_args,
    unique_site_ids,
)
from opendma_demo_alfresco_llamaindex_metadata_aware_rag.content import (
    build_sentence_splitter,
)
from opendma_demo_alfresco_llamaindex_metadata_aware_rag.output import (
    DEMO_QUESTION,
    print_interactive_intro,
)
from opendma_demo_alfresco_llamaindex_metadata_aware_rag.rag import (
    AnalyzedQuery,
    MetadataAwareSiteRetriever,
    build_metadata_filters,
)
from opendma_demo_alfresco_llamaindex_metadata_aware_rag.sites import (
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


class FakeLlm:
    def structured_predict(self, *args: Any, **kwargs: Any) -> AnalyzedQuery:
        return AnalyzedQuery(query="website localisation status", site="swsdp")


class FakeInnerRetriever:
    def __init__(self, owner: FakeIndex) -> None:
        self.owner = owner

    def retrieve(self, query: str) -> list[NodeWithScore]:
        self.owner.query = query
        return [
            NodeWithScore(
                node=TextNode(
                    text="Relevant meeting notes",
                    metadata={"alfresco:Site": "swsdp", "opendma:CheckedOut": False},
                ),
                score=0.9,
            )
        ]


class FakeIndex:
    def __init__(self) -> None:
        self.similarity_top_k: int | None = None
        self.filters: Any = None
        self.query: str | None = None

    def as_retriever(self, *, similarity_top_k: int, filters: Any) -> FakeInnerRetriever:
        self.similarity_top_k = similarity_top_k
        self.filters = filters
        return FakeInnerRetriever(self)


class RetrieverTests(unittest.TestCase):
    def test_query_analysis_drives_semantic_query_and_site_filter(self) -> None:
        index = FakeIndex()
        retriever = MetadataAwareSiteRetriever(
            index=index,  # type: ignore[arg-type]
            sites=[SiteInfo("swsdp", "Sample site", "Sample project content")],
            similarity_top_k=10,
            llm=FakeLlm(),  # type: ignore[arg-type]
        )

        nodes = retriever.retrieve("What is the localisation status?")

        self.assertEqual(index.query, "website localisation status")
        self.assertEqual(index.similarity_top_k, 10)
        self.assertEqual(
            retriever.last_analysis,
            AnalyzedQuery(query="website localisation status", site="swsdp"),
        )
        filter_values = {item.key: item.value for item in index.filters.filters}
        self.assertEqual(
            filter_values,
            {"opendma:CheckedOut": 0, "alfresco:Site": "swsdp"},
        )
        self.assertEqual(len(nodes), 1)

    def test_no_site_analysis_omits_site_filter(self) -> None:
        filters = build_metadata_filters(None)
        self.assertEqual(
            [(item.key, item.value) for item in filters.filters],
            [("opendma:CheckedOut", 0)],
        )

    def test_filters_work_with_boolean_opendma_metadata_in_memory(self) -> None:
        index = VectorStoreIndex.from_documents(
            [
                Document(
                    text="Relevant meeting notes",
                    metadata={"alfresco:Site": "swsdp", "opendma:CheckedOut": False},
                ),
                Document(
                    text="Distracting content",
                    metadata={"alfresco:Site": "engineering", "opendma:CheckedOut": False},
                ),
                Document(
                    text="Checked-out draft",
                    metadata={"alfresco:Site": "swsdp", "opendma:CheckedOut": True},
                ),
            ],
            embed_model=MockEmbedding(embed_dim=8),
        )

        nodes = index.as_retriever(
            similarity_top_k=10,
            filters=build_metadata_filters("swsdp"),
        ).retrieve("website localisation")

        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].metadata["alfresco:Site"], "swsdp")
        self.assertFalse(nodes[0].metadata["opendma:CheckedOut"])


class IndexingTests(unittest.TestCase):
    def test_sentence_splitter_does_not_load_nltk_data(self) -> None:
        splitter = build_sentence_splitter()
        nodes = splitter.get_nodes_from_documents([Document(text="website localisation " * 1500)])

        self.assertEqual(splitter.class_name(), "SentenceSplitter")
        self.assertGreater(len(nodes), 1)


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
