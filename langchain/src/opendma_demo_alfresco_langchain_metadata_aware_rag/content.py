"""Load Alfresco content and build the LangChain vector store."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_opendma import AlfrescoLoader
from langchain_opendma.content_handlers import UnstructuredLoaderContentHandler

from opendma_demo_alfresco_langchain_metadata_aware_rag.config import Settings


@dataclass(frozen=True)
class LoadedSite:
    """Documents loaded for one Alfresco site."""

    site_id: str
    documents: list[Document]


def build_content_handler() -> UnstructuredLoaderContentHandler:
    """Use the same conversion and chunking settings as the tutorial."""
    return UnstructuredLoaderContentHandler(
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3000,
        combine_text_under_n_chars=1000,
    )


def load_site(settings: Settings, site_id: str) -> LoadedSite:
    """Load and chunk all content below one Alfresco site."""
    loader = AlfrescoLoader(
        endpoint=settings.endpoint,
        username=settings.username,
        password=settings.password,
        repository_id=settings.repository_id,
        sites=[site_id],
        recurse_folders=True,
        content_handlers=[build_content_handler()],
    )
    return LoadedSite(site_id=site_id, documents=loader.load())


def build_vector_store(
    settings: Settings,
    loaded_sites: list[LoadedSite],
) -> InMemoryVectorStore:
    """Build one in-memory vector store containing all requested sites."""
    embeddings = OpenAIEmbeddings(model=settings.embedding_model)
    vector_store = InMemoryVectorStore(embeddings)
    for loaded_site in loaded_sites:
        vector_store.add_documents(loaded_site.documents)
    return vector_store
