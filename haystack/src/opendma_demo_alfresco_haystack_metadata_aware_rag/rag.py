"""Two-step baseline and three-step metadata-aware Haystack RAG."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from haystack import Document, Pipeline, component
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret

from opendma_demo_alfresco_haystack_metadata_aware_rag.config import Settings
from opendma_demo_alfresco_haystack_metadata_aware_rag.sites import SiteInfo

ANSWER_PROMPT = [
    ChatMessage.from_system(
        "You answer questions using content retrieved from Alfresco Content Services. "
        "Use only the provided context. "
        "If the context does not contain the answer, say that you do not know."
    ),
    ChatMessage.from_user(
        """
Question: {{ question }}

Context:
{% for document in documents %}
Title: {{ document.meta.get("opendma:Title", "") }}
Site: {{ document.meta.get("alfresco:Site", "") }}
Path: {{ document.meta.get("alfresco:Path", "") }}
Content:
{{ document.content }}

{% endfor %}
"""
    ),
]

ANALYSIS_PROMPT = [
    ChatMessage.from_system(
        "Split the user's question into a focused semantic search query and an "
        "optional Alfresco site filter. Select a site only when the question gives "
        "a useful project, team, department, business-area, or site signal. Otherwise "
        "return null for site. Return only compact JSON with keys query and site. "
        "The site must be null or one of the listed site IDs. Never invent a site ID."
    ),
    ChatMessage.from_user(
        """
Available sites:
{{ site_descriptions }}

Question:
{{ question }}
"""
    ),
]


@dataclass(frozen=True)
class AnalyzedQuery:
    """Structured output passed from query analysis to retrieval."""

    query: str
    site: str | None


@dataclass(frozen=True)
class RagRun:
    """Result returned to the CLI for display."""

    question: str
    context: list[Document]
    answer: str
    analysis: AnalyzedQuery | None = None


def build_filters(site: str | None) -> dict[str, Any]:
    """Filter out checked-out versions and optionally restrict retrieval by site."""
    conditions: list[dict[str, Any]] = [
        {
            "field": "meta.opendma:CheckedOut",
            "operator": "==",
            "value": False,
        }
    ]
    if site is not None:
        conditions.insert(
            0,
            {
                "field": "meta.alfresco:Site",
                "operator": "==",
                "value": site,
            },
        )
    return {"operator": "AND", "conditions": conditions}


@component
class SearchParameterParser:
    """Validate JSON query analysis and build a Haystack metadata filter."""

    def __init__(self, site_ids: set[str]) -> None:
        self.site_ids = site_ids

    @component.output_types(query=str, filters=dict[str, Any], site=str | None)
    def run(self, replies: list[ChatMessage]) -> dict[str, Any]:
        if not replies or not replies[0].text:
            raise ValueError("Query analysis returned no JSON response.")
        data = json.loads(replies[0].text)
        query = str(data.get("query") or "").strip()
        if not query:
            raise ValueError("Query analysis returned an empty semantic query.")

        raw_site = data.get("site")
        site = str(raw_site).strip() if raw_site is not None else None
        if site == "":
            site = None
        if site is not None and site not in self.site_ids:
            raise ValueError(f"Query analysis returned an unknown site ID: {site}")

        return {
            "query": query,
            "filters": build_filters(site),
            "site": site,
        }


def _build_answer_components(
    pipeline: Pipeline,
    *,
    settings: Settings,
    document_store: InMemoryDocumentStore,
    top_k: int,
    retriever_filters: dict[str, Any] | None,
) -> None:
    pipeline.add_component(
        "query_embedder",
        OpenAITextEmbedder(
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            model=settings.embedding_model,
        ),
    )
    pipeline.add_component(
        "retriever",
        InMemoryEmbeddingRetriever(
            document_store=document_store,
            filters=retriever_filters,
            top_k=top_k,
        ),
    )
    pipeline.add_component(
        "prompt_builder",
        ChatPromptBuilder(
            template=ANSWER_PROMPT,
            required_variables={"question", "documents"},
        ),
    )
    pipeline.add_component(
        "llm",
        OpenAIChatGenerator(
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            model=settings.chat_model,
            generation_kwargs={"temperature": 0},
        ),
    )
    pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "llm.messages")


def build_baseline_pipeline(
    settings: Settings,
    document_store: InMemoryDocumentStore,
    top_k: int,
) -> Pipeline:
    """Build the two-step comparison pipeline."""
    pipeline = Pipeline()
    _build_answer_components(
        pipeline,
        settings=settings,
        document_store=document_store,
        top_k=top_k,
        retriever_filters=build_filters(None),
    )
    return pipeline


def build_metadata_pipeline(
    settings: Settings,
    document_store: InMemoryDocumentStore,
    sites: list[SiteInfo],
    top_k: int,
) -> Pipeline:
    """Build query analysis, filtered retrieval, and response generation."""
    pipeline = Pipeline()
    pipeline.add_component(
        "query_analysis_prompt_builder",
        ChatPromptBuilder(
            template=ANALYSIS_PROMPT,
            required_variables={"question", "site_descriptions"},
        ),
    )
    pipeline.add_component(
        "query_analysis_llm",
        OpenAIChatGenerator(
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            model=settings.chat_model,
            generation_kwargs={
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
        ),
    )
    pipeline.add_component(
        "search_parameter_parser",
        SearchParameterParser({site.id for site in sites}),
    )
    _build_answer_components(
        pipeline,
        settings=settings,
        document_store=document_store,
        top_k=top_k,
        retriever_filters=None,
    )
    pipeline.connect(
        "query_analysis_prompt_builder.prompt",
        "query_analysis_llm.messages",
    )
    pipeline.connect(
        "query_analysis_llm.replies",
        "search_parameter_parser.replies",
    )
    pipeline.connect("search_parameter_parser.query", "query_embedder.text")
    pipeline.connect("search_parameter_parser.filters", "retriever.filters")
    return pipeline


class MetadataAwareRag:
    """Run baseline and metadata-aware Haystack pipelines."""

    def __init__(
        self,
        *,
        settings: Settings,
        document_store: InMemoryDocumentStore,
        sites: list[SiteInfo],
        top_k: int,
    ) -> None:
        self.site_descriptions = "\n".join(
            f"- id: {site.id}, title: {site.title or site.id}, "
            f"description: {site.description or site.title or site.id}"
            for site in sites
        )
        self.baseline_pipeline = build_baseline_pipeline(
            settings,
            document_store,
            top_k,
        )
        self.metadata_pipeline = build_metadata_pipeline(
            settings,
            document_store,
            sites,
            top_k,
        )

    @staticmethod
    def _answer(result: dict[str, Any]) -> str:
        return result["llm"]["replies"][0].text or ""

    def run_unfiltered(self, question: str) -> RagRun:
        """Run the original two-step RAG for comparison."""
        result = self.baseline_pipeline.run(
            {
                "query_embedder": {"text": question},
                "prompt_builder": {"question": question},
            },
            include_outputs_from={"retriever"},
        )
        return RagRun(
            question=question,
            context=result["retriever"]["documents"],
            answer=self._answer(result),
        )

    def run_metadata_aware(self, question: str) -> RagRun:
        """Run query analysis, filtered retrieval, and response generation."""
        result = self.metadata_pipeline.run(
            {
                "query_analysis_prompt_builder": {
                    "question": question,
                    "site_descriptions": self.site_descriptions,
                },
                "prompt_builder": {"question": question},
            },
            include_outputs_from={"search_parameter_parser", "retriever"},
        )
        parser_output = result["search_parameter_parser"]
        return RagRun(
            question=question,
            analysis=AnalyzedQuery(
                query=parser_output["query"],
                site=parser_output["site"],
            ),
            context=result["retriever"]["documents"],
            answer=self._answer(result),
        )
