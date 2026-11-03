# Haystack implementation

This project implements the metadata-aware Alfresco RAG demo with Haystack. It
uses `AlfrescoFetcher`, an in-memory document store, a query-analysis component,
and a metadata filter passed to `InMemoryEmbeddingRetriever`.

For shared Alfresco setup, sample data, configuration, and command-line usage,
see the [repository README](../README.md).

Install this implementation:

```bash
uv sync
```

Run its interactive comparison from this directory:

```bash
uv run alfresco-metadata-rag swsdp engineering --compare
```

The implementation follows the
[Haystack metadata-aware retrieval tutorial](https://github.com/OpenDMA/opendma-haystack/blob/main/docs/tutorials/02_metadata_aware_retrieval.md).

