"""Command-line application for metadata-aware Haystack RAG."""

from __future__ import annotations

import argparse
import logging
import warnings

from opendma_demo_alfresco_haystack_metadata_aware_rag.config import (
    load_settings,
    require_openai_api_key,
)
from opendma_demo_alfresco_haystack_metadata_aware_rag.content import (
    build_indexing_application,
    index_site,
)
from opendma_demo_alfresco_haystack_metadata_aware_rag.output import (
    print_interactive_intro,
    print_run,
)
from opendma_demo_alfresco_haystack_metadata_aware_rag.rag import MetadataAwareRag
from opendma_demo_alfresco_haystack_metadata_aware_rag.sites import (
    discover_sites,
    select_requested_sites,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run metadata-aware RAG over Alfresco content with Haystack.",
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
        help="Number of chunks to retrieve. Default: 10.",
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

    indexing_application = build_indexing_application(settings)
    for site in requested_sites:
        print()
        print(f"Loading, converting, and indexing Alfresco site '{site.id}'...", flush=True)
        chunks = index_site(indexing_application, site.id)
        print(f"Indexed {len(chunks)} document chunks from '{site.id}'.")

    document_count = indexing_application.document_store.count_documents()
    print()
    print(f"The in-memory document store contains {document_count} chunks.")

    application = MetadataAwareRag(
        settings=settings,
        document_store=indexing_application.document_store,
        sites=requested_sites,
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
