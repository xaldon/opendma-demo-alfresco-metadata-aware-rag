"""Fetch, convert, split, embed, and index Alfresco content with Haystack."""

from __future__ import annotations

from dataclasses import dataclass

from haystack import Document, Pipeline, component
from haystack.components.converters import (
    HTMLToDocument,
    PyPDFToDocument,
    TextFileToDocument,
)
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.routers import FileTypeRouter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.components.converters.docling import DoclingConverter
from opendma_haystack import AlfrescoFetcher

from opendma_demo_alfresco_haystack_metadata_aware_rag.config import Settings

SUPPORTED_MIME_TYPES = [
    "text/plain",
    "text/html",
    "application/pdf",
    "application/msword",
]


@component
class TextDocumentFilter:
    """Drop converted documents without text content."""

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        return {
            "documents": [
                document
                for document in documents
                if document.content is not None and document.content.strip()
            ]
        }


@dataclass(frozen=True)
class IndexingApplication:
    """Reusable indexing pipeline and its in-memory document store."""

    pipeline: Pipeline
    document_store: InMemoryDocumentStore


def build_indexing_application(settings: Settings) -> IndexingApplication:
    """Build the tutorial indexing pipeline for repeated site ingestion."""
    document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
    pipeline = Pipeline()
    pipeline.add_component(
        "fetcher",
        AlfrescoFetcher(
            endpoint=settings.endpoint,
            username=settings.username,
            password=settings.password,
            repository_id=settings.repository_id,
        ),
    )
    pipeline.add_component("router", FileTypeRouter(mime_types=SUPPORTED_MIME_TYPES))
    pipeline.add_component("text_converter", TextFileToDocument())
    pipeline.add_component("html_converter", HTMLToDocument())
    pipeline.add_component("pdf_converter", PyPDFToDocument())
    pipeline.add_component("word_converter", DoclingConverter())
    pipeline.add_component("text_filter", TextDocumentFilter())
    pipeline.add_component("joiner", DocumentJoiner())
    pipeline.add_component(
        "splitter",
        DocumentSplitter(split_by="word", split_length=250, split_overlap=40),
    )
    pipeline.add_component(
        "embedder",
        OpenAIDocumentEmbedder(
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            model=settings.embedding_model,
        ),
    )
    pipeline.add_component(
        "writer",
        DocumentWriter(
            document_store=document_store,
            policy=DuplicatePolicy.OVERWRITE,
        ),
    )

    pipeline.connect("fetcher.streams", "router.sources")
    pipeline.connect("router.text/plain", "text_converter.sources")
    pipeline.connect("router.text/html", "html_converter.sources")
    pipeline.connect("router.application/pdf", "pdf_converter.sources")
    pipeline.connect("router.application/msword", "word_converter.sources")
    pipeline.connect("text_converter.documents", "text_filter.documents")
    pipeline.connect("html_converter.documents", "text_filter.documents")
    pipeline.connect("pdf_converter.documents", "text_filter.documents")
    pipeline.connect("word_converter.documents", "text_filter.documents")
    pipeline.connect("text_filter.documents", "joiner.documents")
    pipeline.connect("joiner.documents", "splitter.documents")
    pipeline.connect("splitter.documents", "embedder.documents")
    pipeline.connect("embedder.documents", "writer.documents")
    return IndexingApplication(pipeline=pipeline, document_store=document_store)


def index_site(application: IndexingApplication, site_id: str) -> list[Document]:
    """Run the shared indexing pipeline for one Alfresco site."""
    result = application.pipeline.run(
        {"fetcher": {"sites": [site_id]}},
        include_outputs_from={"splitter"},
    )
    return result["splitter"]["documents"]
