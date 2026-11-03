"""Discover Alfresco sites through OpenDMA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import opendma.remote
from opendma.api import OdmaId, OdmaQName

from opendma_demo_alfresco_llamaindex_metadata_aware_rag.config import Settings


@dataclass(frozen=True)
class SiteInfo:
    """Metadata used to constrain query analysis."""

    id: str
    title: str
    description: str


def _property_string(obj: Any, qname: str) -> str:
    prop = obj.get_property(OdmaQName.from_string(qname))
    if prop is None:
        return ""
    return prop.get_string() or ""


def discover_sites(settings: Settings) -> list[SiteInfo]:
    """Return all Alfresco sites visible through the configured endpoint."""
    session = opendma.remote.connect(
        endpoint=settings.endpoint,
        username=settings.username,
        password=settings.password,
    )
    try:
        result = session.search(
            OdmaId(settings.repository_id),
            OdmaQName.from_string("alfresco:afts"),
            'TYPE:"st:site"',
        )
        return [
            SiteInfo(
                id=_property_string(site, "alfresco:cm:name"),
                title=_property_string(site, "alfresco:cm:title"),
                description=_property_string(site, "alfresco:cm:description"),
            )
            for site in result.get_objects()
        ]
    finally:
        session.close()


def select_requested_sites(
    discovered_sites: list[SiteInfo],
    requested_ids: list[str],
) -> list[SiteInfo]:
    """Validate requested IDs and preserve their command-line order."""
    sites_by_id = {site.id: site for site in discovered_sites if site.id}
    missing = [site_id for site_id in requested_ids if site_id not in sites_by_id]
    if missing:
        available = ", ".join(sorted(sites_by_id)) or "none"
        missing_text = ", ".join(missing)
        raise ValueError(
            f"Alfresco site(s) not found: {missing_text}. Available sites: {available}."
        )
    return [sites_by_id[site_id] for site_id in requested_ids]
