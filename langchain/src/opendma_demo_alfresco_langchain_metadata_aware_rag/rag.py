"""Two-step baseline and three-step metadata-aware LangChain RAG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import END, START, StateGraph

from opendma_demo_alfresco_langchain_metadata_aware_rag.output import message_text
from opendma_demo_alfresco_langchain_metadata_aware_rag.sites import SiteInfo

SYSTEM_PROMPT = (
    "You answer questions using documents retrieved from Alfresco Content Services. "
    "Use only the provided context. If the context does not contain the answer, "
    "say that you do not know."
)

ANALYSIS_PROMPT = """
Split the user's question into a focused semantic search query and an optional
Alfresco site filter.

Select a site only when the question gives a useful project, team, department,
business-area, or site signal. Otherwise return null for the site. Use only the
site IDs permitted by the response schema. Never invent a site ID.
""".strip()


class AnalyzedQuery(TypedDict):
    """Structured output passed from query analysis to retrieval."""

    query: str
    site: str | None


class RagState(TypedDict, total=False):
    """State carried through the metadata-aware graph."""

    question: str
    analysis: AnalyzedQuery
    context: list[Document]
    answer: str


@dataclass(frozen=True)
class RagRun:
    """Result returned to the CLI for display."""

    question: str
    context: list[Document]
    answer: str
    analysis: AnalyzedQuery | None = None


def build_search_schema(sites: list[SiteInfo]) -> dict[str, Any]:
    """Build a structured-output schema limited to sites in the vector index."""
    site_choices: list[dict[str, Any]] = [
        {
            "const": site.id,
            "title": site.title or site.id,
            "description": site.description or site.title or site.id,
        }
        for site in sites
    ]
    site_choices.append(
        {
            "type": "null",
            "title": "No site filter",
            "description": "Use when the question does not identify a useful site scope.",
        }
    )
    return {
        "title": "AnalyzedQuery",
        "description": "Semantic query and optional Alfresco site filter.",
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Focused semantic search query.",
            },
            "site": {
                "description": "Alfresco site to search, or null for all indexed sites.",
                "oneOf": site_choices,
            },
        },
        "required": ["query", "site"],
        "additionalProperties": False,
    }


def generate_answer(llm: BaseChatModel, question: str, context: list[Document]) -> str:
    """Generate an answer using only retrieved Alfresco content."""
    docs_content = "\n\n---\n\n".join(document.page_content for document in context)
    response = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {question}\n\nContext:\n{docs_content}",
            },
        ]
    )
    return message_text(response)


def _is_current_document(document: Document) -> bool:
    checked_out = document.metadata.get("opendma:CheckedOut")
    return checked_out is False or checked_out == 0


class MetadataAwareRag:
    """Run baseline and metadata-aware RAG against one vector store."""

    def __init__(
        self,
        *,
        vector_store: InMemoryVectorStore,
        sites: list[SiteInfo],
        llm: BaseChatModel,
        top_k: int,
    ) -> None:
        self.vector_store = vector_store
        self.sites = sites
        self.llm = llm
        self.top_k = top_k
        self.site_ids = {site.id for site in sites}
        self.structured_llm = llm.with_structured_output(build_search_schema(sites))
        self.graph = self._build_graph()

    def _analyse(self, state: RagState) -> dict[str, AnalyzedQuery]:
        raw = self.structured_llm.invoke(
            [
                {"role": "system", "content": ANALYSIS_PROMPT},
                {"role": "user", "content": state["question"]},
            ]
        )
        if not isinstance(raw, dict):
            raise TypeError("Structured query analysis did not return a dictionary.")

        query = str(raw.get("query") or state["question"]).strip()
        raw_site = raw.get("site")
        site = str(raw_site).strip() if raw_site is not None else None
        if site == "":
            site = None
        if site is not None and site not in self.site_ids:
            raise ValueError(f"Query analysis returned an unknown site ID: {site}")
        return {"analysis": {"query": query, "site": site}}

    def _retrieve(self, state: RagState) -> dict[str, list[Document]]:
        analysis = state["analysis"]
        selected_site = analysis["site"]

        def metadata_filter(document: Document) -> bool:
            return _is_current_document(document) and (
                selected_site is None or document.metadata.get("alfresco:Site") == selected_site
            )

        documents = self.vector_store.similarity_search(
            analysis["query"],
            k=self.top_k,
            filter=metadata_filter,
        )
        return {"context": documents}

    def _generate(self, state: RagState) -> dict[str, str]:
        return {
            "answer": generate_answer(
                self.llm,
                question=state["question"],
                context=state["context"],
            )
        }

    def _build_graph(self) -> Any:
        graph_builder = StateGraph(RagState)
        graph_builder.add_node("analyse_query", self._analyse)
        graph_builder.add_node("retrieve", self._retrieve)
        graph_builder.add_node("generate", self._generate)
        graph_builder.add_edge(START, "analyse_query")
        graph_builder.add_edge("analyse_query", "retrieve")
        graph_builder.add_edge("retrieve", "generate")
        graph_builder.add_edge("generate", END)
        return graph_builder.compile()

    def run_metadata_aware(self, question: str) -> RagRun:
        """Run query analysis, filtered retrieval, and generation."""
        state = self.graph.invoke({"question": question})
        return RagRun(
            question=question,
            analysis=state["analysis"],
            context=state["context"],
            answer=state["answer"],
        )

    def run_unfiltered(self, question: str) -> RagRun:
        """Run the original two-step RAG for comparison."""
        context = self.vector_store.similarity_search(
            question,
            k=self.top_k,
            filter=_is_current_document,
        )
        return RagRun(
            question=question,
            context=context,
            answer=generate_answer(self.llm, question, context),
        )
