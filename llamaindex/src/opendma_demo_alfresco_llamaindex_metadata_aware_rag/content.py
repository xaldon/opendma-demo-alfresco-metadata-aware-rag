"""Load Alfresco content and build the LlamaIndex vector index."""

from __future__ import annotations

from dataclasses import dataclass

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser.text.utils import split_by_regex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.docling import DoclingReader
from llama_index.readers.opendma import AlfrescoReader

from opendma_demo_alfresco_llamaindex_metadata_aware_rag.config import Settings


@dataclass(frozen=True)
class LoadedSite:
    """Documents loaded for one Alfresco site."""

    site_id: str
    documents: list[Document]


def load_site(settings: Settings, site_id: str) -> LoadedSite:
    """Load all supported content below one Alfresco site."""
    reader = AlfrescoReader(
        endpoint=settings.endpoint,
        username=settings.username,
        password=settings.password,
        repository_id=settings.repository_id,
        sites=[site_id],
        recursive=True,
        file_extractor_per_mimetype={
            "application/msword": DoclingReader(),
        },
    )
    return LoadedSite(site_id=site_id, documents=reader.load_data())


def build_sentence_splitter() -> SentenceSplitter:
    """Create tutorial-style sentence chunking without loading NLTK data files."""
    sentence_pattern = r"[^.!?\n]+(?:[.!?]+|\n+|$)|[.!?]+"
    return SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=200,
        chunking_tokenizer_fn=split_by_regex(sentence_pattern),
    )


def build_index(settings: Settings, loaded_sites: list[LoadedSite]) -> VectorStoreIndex:
    """Build one in-memory vector index containing all requested sites."""
    documents = [document for loaded_site in loaded_sites for document in loaded_site.documents]
    embed_model = OpenAIEmbedding(model=settings.embedding_model)
    return VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        transformations=[build_sentence_splitter()],
    )
