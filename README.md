# Metadata-Aware RAG for Alfresco with OpenDMA

This repository contains implementations of the metadata-aware RAG example described in
the accompanying article
[Prevent Retrieval Dilution in Alfresco RAG with Metadata Filters](https://dev.to/stefankopfxaldon/prevent-retrieval-dilution-in-alfresco-rag-with-metadata-filters-4la4).

It implements the same basic principle with the three popular frameworks LangChain,
LlamaIndex, and Haystack.

All three applications implement the same fixed workflow:

```text
question -> query analysis -> retrieval with metadata filter -> answer generation
```

The implementations use their framework's native document model, vector retrieval,
metadata filters, and response synthesis. They expose the same command-line interface.

The examples are intentionally small. They demonstrate retrieval dilution and
metadata filtering, not a production RAG architecture.

## What the demo shows

The standard Alfresco `swsdp` site contains meeting notes for the Sample: Web
Site Design Project. These meeting notes answer the question:

> What is the state of localisation of our new website design?

The additional `engineering` site contains a deliberately distracting document
which discusses website localisation extensively without answering the question.
After both sites have been indexed, an unfiltered similarity search is likely to
return chunks from this document instead of the relevant meeting notes.

In comparison mode, each application runs the question twice:

1. with an unfiltered similarity search;
2. with an LLM-generated semantic query and Alfresco site filter.

Both searches use the same vector index.

## Prerequisites

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/)
- an Alfresco deployment
- an OpenDMA REST endpoint for Alfresco
- an OpenAI API key

The defaults use the Alfresco tutorial credentials `admin/admin`. Do not use
these credentials as production guidance.

## Running an OpenDMA endpoint for Alfresco

The applications need an OpenDMA REST endpoint for Alfresco. The quickest way to
map Alfresco to the OpenDMA data model and expose such an endpoint is to run the
[ECI Server](https://github.com/xaldon/eci-server). It is available free of
charge for non-production use.

Start it with Docker Compose:

```bash
git clone https://github.com/xaldon/eci-server.git
cd eci-server/docker_compose
docker compose up -d
```

After the service is running:

1. Open the web UI at `http://localhost:7070`.
2. Initialize the admin account.
3. Accept the license agreement.
4. Install a free-of-charge license key.
5. Navigate to "Admin" > "Connections".
6. Add a new connection to `Alfresco Content Services`.
7. Choose "Automatically detect parameters with Smart Setup" for your Alfresco Content Services as target server.
   (If you are running Alfresco in docker as well, use `host.docker.internal`)
8. Save the new connection as `Alfresco @ <hostname>`.
9. Navigate to "Admin" > "REST Endpoints".
10. Add a new REST Endpoint.
11. Set the slug to `opendma/alf`.
12. Select the `Alfresco @ <hostname>` connection.
13. Keep the proposed "Inbound Authentication" ("HTTP Basic" and "Propagate Inbound").
14. Save the new REST Endpoint.

The examples expect the endpoint at:

```text
http://localhost:7070/opendma/alf
```

Open `http://localhost:7070/opendma/alf/` with a trailing slash and authenticate
with your Alfresco credentials to verify it.

## Prepare the demo content

New Alfresco deployments contain the **Sample: Web Site Design Project** site
with short name `swsdp`.

Add the distracting content used by this demo:

1. Log in to Alfresco Share.
2. Create a public Collaboration Site named `Engineering`.
3. Set its description to `All product engineering related documents`.
4. Open its Document Library.
5. Upload `sample-data/product-webui-design.txt`.

The applications are read-only. They do not create the site or upload the file.

## Configuration

Copy `.env.example` to `.env` in the repository root and set your API key:

```text
OPENAI_API_KEY=...
```

Alternatively, you can set the `OPENAI_API_KEY` environment variable before running
the application.

Available settings:

```text
OPENDMA_ALFRESCO_ENDPOINT=http://localhost:7070/opendma/alf
ALFRESCO_USERNAME=admin
ALFRESCO_PASSWORD=admin
ALFRESCO_REPOSITORY_ID=Alfresco
OPENAI_API_KEY=...
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

## Choose a framework

Install one implementation from the repository root:

```bash
uv sync --project langchain
```

```bash
uv sync --project llamaindex
```

```bash
uv sync --project haystack
```

Each project has its own lockfile and virtual environment.

## Run the interactive comparison

The primary demo is the interactive loop in comparison mode. Choose a framework
by changing the value passed to `--project`:

```bash
uv run --project langchain alfresco-metadata-rag swsdp engineering --compare
```

```bash
uv run --project llamaindex alfresco-metadata-rag swsdp engineering --compare
```

```bash
uv run --project haystack alfresco-metadata-rag swsdp engineering --compare
```

Allow some time for the application to ingest the content from Alfresco and
build the vector index.

The application prompts for a question and then tries to answer twice, first
without metadata filtering and a second time filtering chunks by site.

Type `exit` or `quit` to end the application.

You can also run from inside a framework directory. The command itself remains
the same:

```bash
cd langchain
uv run alfresco-metadata-rag swsdp engineering --compare
```

## Ask one question and exit

Use `--question` when you do not want to enter the interactive loop:

```bash
uv run --project langchain alfresco-metadata-rag swsdp engineering \
  --compare \
  --question "What is the state of localisation of our new website design?"
```

## Command-line arguments

```text
alfresco-metadata-rag SITES... [OPTIONS]

SITES...                 Alfresco site short names to index
--question, -q TEXT      Ask one question and exit
--compare                Show unfiltered and metadata-aware retrieval
--top-k INTEGER          Number of chunks or nodes to retrieve; default: 10
--verbose                Show conversion logs and warnings
```

## Expected differences between frameworks

The three applications implement the same workflow, but their document
conversion and splitting implementations differ. Chunk counts, similarity
scores, ranking, and generated answer text are therefore not expected to match.

The relevant comparison is behavioural:

- both Alfresco sites are present in one vector index;
- unfiltered retrieval can be diluted by Engineering content;
- query analysis selects `swsdp` for the demo question;
- metadata-aware retrieval returns only chunks from the selected site.

## Security

A vector-store metadata filter is not an access-control mechanism. Production
applications must preserve and enforce the permissions of the effective Alfresco
user during ingestion and retrieval.

## Framework documentation

- [LangChain implementation](./langchain/README.md)
- [LlamaIndex implementation](./llamaindex/README.md)
- [Haystack implementation](./haystack/README.md)

## Further reading

- [OpenDMA](https://opendma.org/)
- ECI Server: https://xaldon.com/products/eci-server
- [LangChain metadata-aware retrieval tutorial](https://github.com/OpenDMA/langchain-opendma/blob/main/docs/tutorials/02_metadata_aware_retrieval.md)
- [LlamaIndex metadata-aware retrieval tutorial](https://github.com/OpenDMA/llama-index-opendma/blob/main/docs/tutorials/02_metadata_aware_retrieval.md)
- [Haystack metadata-aware retrieval tutorial](https://github.com/OpenDMA/opendma-haystack/blob/main/docs/tutorials/02_metadata_aware_retrieval.md)

