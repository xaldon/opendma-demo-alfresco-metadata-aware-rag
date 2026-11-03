"""Console formatting for the LangChain demo."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document

if TYPE_CHECKING:
    from opendma_demo_alfresco_langchain_metadata_aware_rag.rag import AnalyzedQuery, RagRun


DEMO_QUESTION = "What is the state of localisation of our new website design?"


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace for compact terminal output."""
    text = text.replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def truncate(text: str, length: int = 240) -> str:
    """Return a compact content preview."""
    normalized = normalize_whitespace(text)
    if len(normalized) <= length:
        return normalized
    return normalized[: length - 3].rstrip() + "..."


def message_text(message: Any) -> str:
    """Extract printable text from common LangChain message shapes."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def print_query_analysis(analysis: AnalyzedQuery) -> None:
    """Print the structured query analysis."""
    print("Query analysis")
    print(f"Semantic query: {analysis['query']}")
    print(f"Site filter: {analysis['site'] or '(none)'}")


def print_documents(documents: list[Document]) -> None:
    """Print every retrieved chunk so dilution is visible."""
    print(f"Retrieved chunks: {len(documents)}")
    for index, document in enumerate(documents, start=1):
        print()
        print(f"Result {index}")
        print(f"Title: {document.metadata.get('opendma:Title')}")
        print(f"Site: {document.metadata.get('alfresco:Site')}")
        print(f"Path: {document.metadata.get('alfresco:Path')}")
        print(f"Preview: {truncate(document.page_content)}")


def print_run(run: RagRun, *, include_analysis: bool) -> None:
    """Print one complete RAG run."""
    if include_analysis and run.analysis is not None:
        print_query_analysis(run.analysis)
        print()
    print_documents(run.context)
    print()
    print("Answer")
    print(run.answer)


def print_interactive_intro(*, compare: bool, indexed_site_ids: list[str]) -> None:
    """Explain the loop and conditionally show the reproducible demo question."""
    print()
    print("The in-memory vector index is ready.")
    if compare:
        print(
            "For each question, the application first runs an unfiltered RAG and then "
            "the three-step metadata-aware RAG against the same index."
        )
    else:
        print(
            "For each question, the application analyses the query, applies an optional "
            "Alfresco site filter, and generates an answer from the retrieved context."
        )

    if "swsdp" in indexed_site_ids:
        print()
        print("Try this question by copying it into the prompt:")
        print(DEMO_QUESTION)

    print()
    print("Type 'exit' or 'quit' to end the application.")
    print()
