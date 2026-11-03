"""Configuration shared by the LangChain demo application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def load_root_dotenv() -> Path | None:
    """Load the first .env found in the current directory or one of its parents."""
    for directory in (Path.cwd(), *Path.cwd().parents):
        candidate = directory / ".env"
        if candidate.is_file():
            load_dotenv(candidate)
            return candidate
    return None


@dataclass(frozen=True)
class Settings:
    endpoint: str
    username: str
    password: str
    repository_id: str
    chat_model: str
    embedding_model: str


def load_settings() -> Settings:
    """Load settings with defaults matching the local Alfresco tutorial setup."""
    load_root_dotenv()
    return Settings(
        endpoint=os.environ.get(
            "OPENDMA_ALFRESCO_ENDPOINT",
            "http://localhost:7070/opendma/alf",
        ),
        username=os.environ.get("ALFRESCO_USERNAME", "admin"),
        password=os.environ.get("ALFRESCO_PASSWORD", "admin"),
        repository_id=os.environ.get("ALFRESCO_REPOSITORY_ID", "Alfresco"),
        chat_model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        embedding_model=os.environ.get(
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-large",
        ),
    )


def require_openai_api_key() -> None:
    """Fail before loading Alfresco content when OpenAI is not configured."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY must be set. Copy the root .env.example to .env and add your API key."
        )
