# LlamaIndex implementation

This project implements the metadata-aware Alfresco RAG demo with LlamaIndex. It
uses `AlfrescoReader`, `VectorStoreIndex`, structured prediction, and
`MetadataFilters` to apply the selected Alfresco site during retrieval.

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
[LlamaIndex metadata-aware retrieval tutorial](https://github.com/OpenDMA/llama-index-opendma/blob/main/docs/tutorials/02_metadata_aware_retrieval.md).

