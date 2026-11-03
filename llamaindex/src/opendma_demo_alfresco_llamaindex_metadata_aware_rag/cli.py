"""Command-line application for metadata-aware LlamaIndex RAG."""

from __future__ import annotations

import argparse
import logging
import warnings

from llama_index.llms.openai import OpenAI

from opendma_demo_alfresco_llamaindex_metadata_aware_rag.config import (
    load_settings,
    require_openai_api_key,
)
from opendma_demo_alfresco_llamaindex_metadata_aware_rag.content import (
    build_index,
    load_site,
)
from opendma_demo_alfresco_llamaindex_metadata_aware_rag.output import (
    print_interactive_intro,
    print_run,
)
from opendma_demo_alfresco_llamaindex_metadata_aware_rag.rag import MetadataAwareRag
from opendma_demo_alfresco_llamaindex_metadata_aware_rag.sites import (
    discover_sites,
    select_requested_sites,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run metadata-aware RAG over Alfresco content with LlamaIndex.",
    )
    parser.add_argument("sites", nargs="+", help="Alfresco site short names to index.")
    parser.add_argument("--question", "-q", help="Ask one question and exit.")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show unfiltered and metadata-aware retrieval for each question.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of nodes to retrieve. Default: 10.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show conversion logs and warnings.",
    )
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero")
    return args


def configure_verbosity(verbose: bool) -> None:
    """Hide recoverable conversion messages unless explicitly requested."""
    if verbose:
        logging.disable(logging.NOTSET)
        return
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")


def unique_site_ids(site_ids: list[str]) -> list[str]:
    """Remove duplicate command-line site IDs while retaining their order."""
    return list(dict.fromkeys(site_ids))


def answer_question(application: MetadataAwareRag, question: str, compare: bool) -> None:
    """Run and print one metadata-aware answer, optionally with a baseline."""
    if compare:
        print()
        print("=" * 80)
        print("Unfiltered RAG")
        print("=" * 80)
        print_run(application.run_unfiltered(question), include_analysis=False)

    print()
    print("=" * 80)
    print("Metadata-aware RAG")
    print("=" * 80)
    print_run(application.run_metadata_aware(question), include_analysis=True)
    print()


def run_question_loop(application: MetadataAwareRag, compare: bool) -> None:
    """Prompt until the user exits."""
    while True:
        question = input("Question: ").strip()
        if question.lower() in {"exit", "quit"}:
            return
        if not question:
            print("Enter a question, or use 'exit' or 'quit' to end the application.")
            continue
        answer_question(application, question, compare)


def main() -> int:
    """Load the selected Alfresco sites and run the question application."""
    args = parse_args()
    configure_verbosity(args.verbose)
    site_ids = unique_site_ids(args.sites)

    print(
        "Starting the metadata-aware RAG demo. The application will load Alfresco "
        "content and build an in-memory vector index. This can take some time.",
        flush=True,
    )

    settings = load_settings()
    require_openai_api_key()

    requested_sites = select_requested_sites(discover_sites(settings), site_ids)
    print("Selected Alfresco sites:")
    for site in requested_sites:
        print(f"- {site.id}: {site.title or site.id} - {site.description or 'No description'}")

    loaded_sites = []
    for site in requested_sites:
        print()
        print(f"Loading and converting Alfresco site '{site.id}'...", flush=True)
        loaded_site = load_site(settings, site.id)
        loaded_sites.append(loaded_site)
        print(f"Loaded {len(loaded_site.documents)} documents from '{site.id}'.")

    print()
    print("Calculating embeddings and building the in-memory vector index...", flush=True)
    index = build_index(settings, loaded_sites)
    print(f"Indexed {len(index.ref_doc_info)} reference documents.")
    indexed_nodes = getattr(index.storage_context.docstore, "docs", None)
    if indexed_nodes is not None:
        print(f"Indexed {len(indexed_nodes)} nodes.")

    llm = OpenAI(model=settings.chat_model, temperature=0)
    application = MetadataAwareRag(
        index=index,
        sites=requested_sites,
        llm=llm,
        top_k=args.top_k,
    )

    if args.question:
        answer_question(application, args.question, args.compare)
        return 0

    print_interactive_intro(compare=args.compare, indexed_site_ids=site_ids)
    run_question_loop(application, args.compare)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
