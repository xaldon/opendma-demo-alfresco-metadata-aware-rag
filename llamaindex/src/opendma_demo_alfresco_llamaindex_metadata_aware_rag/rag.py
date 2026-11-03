"""Two-step baseline and three-step metadata-aware LlamaIndex RAG."""

from __future__ import annotations

from dataclasses import dataclass

from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import LLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

from opendma_demo_alfresco_llamaindex_metadata_aware_rag.sites import SiteInfo

ANALYSIS_PROMPT = PromptTemplate(
    """Split the user's question into a focused semantic search query and an optional
Alfresco site filter.

Select a site only when the question gives a useful project, team, department,
business-area, or site signal. Otherwise leave the site empty. Use only a site ID
from the list below and never invent a site ID.

Available sites:
{site_descriptions}

Question:
{question}
"""
)


class AnalyzedQuery(BaseModel):
    """Structured output passed from query analysis to retrieval."""

    query: str = Field(description="Focused semantic search query.")
    site: str | None = Field(
        default=None,
        description="Alfresco site ID to search, or null when no site filter is useful.",
    )


@dataclass(frozen=True)
class RagRun:
    """Result returned to the CLI for display."""

    question: str
    context: list[NodeWithScore]
    answer: str
    analysis: AnalyzedQuery | None = None


def build_metadata_filters(site: str | None) -> MetadataFilters:
    """Filter out checked-out versions and optionally restrict retrieval by site."""
    filters = [MetadataFilter(key="opendma:CheckedOut", value=0)]
    if site is not None:
        filters.append(MetadataFilter(key="alfresco:Site", value=site))
    return MetadataFilters(filters=filters)


class MetadataAwareSiteRetriever(BaseRetriever):
    """Analyse a question and retrieve nodes from the selected Alfresco site."""

    def __init__(
        self,
        *,
        index: VectorStoreIndex,
        sites: list[SiteInfo],
        similarity_top_k: int,
        llm: LLM,
    ) -> None:
        super().__init__()
        self.index = index
        self.sites = sites
        self.site_ids = {site.id for site in sites}
        self.similarity_top_k = similarity_top_k
        self.llm = llm
        self.last_analysis: AnalyzedQuery | None = None

    def _analyse_query(self, question: str) -> AnalyzedQuery:
        site_descriptions = "\n".join(
            f"- id: {site.id}, title: {site.title or site.id}, "
            f"description: {site.description or site.title or site.id}"
            for site in self.sites
        )
        result = self.llm.structured_predict(
            AnalyzedQuery,
            ANALYSIS_PROMPT,
            site_descriptions=site_descriptions,
            question=question,
        )
        query = result.query.strip() or question
        site = result.site.strip() if result.site else None
        if site is not None and site not in self.site_ids:
            raise ValueError(f"Query analysis returned an unknown site ID: {site}")
        return AnalyzedQuery(query=query, site=site)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        analysis = self._analyse_query(query_bundle.query_str)
        self.last_analysis = analysis
        retriever = self.index.as_retriever(
            similarity_top_k=self.similarity_top_k,
            filters=build_metadata_filters(analysis.site),
        )
        return retriever.retrieve(analysis.query)


class MetadataAwareRag:
    """Run baseline and metadata-aware RAG against one vector index."""

    def __init__(
        self,
        *,
        index: VectorStoreIndex,
        sites: list[SiteInfo],
        llm: LLM,
        top_k: int,
    ) -> None:
        baseline_retriever = index.as_retriever(
            similarity_top_k=top_k,
            filters=build_metadata_filters(None),
        )
        self.metadata_retriever = MetadataAwareSiteRetriever(
            index=index,
            sites=sites,
            similarity_top_k=top_k,
            llm=llm,
        )
        self.baseline_engine = RetrieverQueryEngine.from_args(
            baseline_retriever,
            llm=llm,
        )
        self.metadata_engine = RetrieverQueryEngine.from_args(
            self.metadata_retriever,
            llm=llm,
        )

    @staticmethod
    def _to_run(
        question: str,
        response: RESPONSE_TYPE,
        analysis: AnalyzedQuery | None = None,
    ) -> RagRun:
        return RagRun(
            question=question,
            context=list(response.source_nodes),
            answer=str(response),
            analysis=analysis,
        )

    def run_metadata_aware(self, question: str) -> RagRun:
        """Run query analysis, filtered retrieval, and response synthesis."""
        response = self.metadata_engine.query(question)
        analysis = self.metadata_retriever.last_analysis
        if analysis is None:
            raise RuntimeError("Metadata-aware retrieval did not analyse the query.")
        return self._to_run(question, response, analysis)

    def run_unfiltered(self, question: str) -> RagRun:
        """Run the original two-step RAG for comparison."""
        response = self.baseline_engine.query(question)
        return self._to_run(question, response)
