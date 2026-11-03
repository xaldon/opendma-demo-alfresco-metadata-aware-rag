# LangChain implementation

This project implements the metadata-aware Alfresco RAG demo with LangChain and
LangGraph. It uses `AlfrescoLoader`, an in-memory LangChain vector store,
structured model output for site selection, and a three-node graph for query
analysis, filtered retrieval, and answer generation.

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
[LangChain metadata-aware retrieval tutorial](https://github.com/OpenDMA/langchain-opendma/blob/main/docs/tutorials/02_metadata_aware_retrieval.md).

