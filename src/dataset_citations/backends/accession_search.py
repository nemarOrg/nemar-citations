"""Find papers that mention a dataset accession via OpenAlex full-text search.

Datasets are cited by accession number in running text ("ds002718",
"on005964", "nm000207"), not by DOI (OpenNeuro / NEMAR DOIs aren't indexed in
OpenAlex). OpenAlex's `fulltext.search` filter surfaces those works. This
backend reuses opencite's `OpenAlexClient` + `Config` (no new HTTP client) and
maps each hit to the same citation-detail dict the opencite pipeline emits,
tagged `discovery_method="accession_mention"`. Sync facade over the async
client, mirroring `OpenCiteBackend`. Issue #169.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, ClassVar

import httpx
from opencite.clients.openalex import OpenAlexClient
from opencite.config import Config
from opencite.exceptions import OpenCiteError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AccessionSearchResult:
    """Citations found for a dataset's accession terms, plus what failed.

    `failed_terms` exists because a per-term failure is NOT fatal here (a
    rate-limited or timed-out term is logged and skipped), so an empty
    `citations` list is ambiguous on its own: it means either "genuinely no
    papers mention this accession" or "we never got an answer". The freshness
    gate in `cli/find_mentions.py` must tell those apart, since recording a
    degraded search as a completed one would hide the dataset for a full
    `--max-age-days` window with no citation coverage and no error.
    """

    citations: list[dict[str, Any]]
    failed_terms: list[str]

    @property
    def degraded(self) -> bool:
        """True if any term failed, so this dataset was not fully searched."""
        return bool(self.failed_terms)


# Safety bound on cursor pagination per term (50 * 100 = 5000 works). Accession
# searches return far fewer, but this caps a pathological term and pairs with
# the empty-page break below to make the loop always terminate.
_MAX_PAGES = 50


def reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str | None:
    """Rebuild plain abstract text from OpenAlex's abstract_inverted_index."""
    if not inverted:
        return None
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted.items():
        positions.extend((i, word) for i in idxs)
    if not positions:
        return None
    positions.sort()
    return " ".join(word for _, word in positions)


def _bare_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    stripped = (
        doi.replace("https://doi.org/", "").replace("http://doi.org/", "").lower()
    )
    return stripped or None


def _short_id(url_id: str | None) -> str | None:
    if not url_id:
        return None
    return url_id.rsplit("/", 1)[-1] or None


def work_to_citation(work: dict[str, Any], matched_accession: str) -> dict[str, Any]:
    """Map a raw OpenAlex work JSON to a citation-detail dict.

    Mirrors `opencite_pipeline._citing_work_to_dict` so accession-mention
    citations are schema-compatible with anchor-based ones, plus the
    `discovery_method` / `matched_accession` tags that mark the dataset bucket.
    """
    authors = [
        a.get("author", {}).get("display_name")
        for a in work.get("authorships", [])
        if a.get("author", {}).get("display_name")
    ]
    source = (work.get("primary_location") or {}).get("source") or {}
    pmid = (work.get("ids") or {}).get("pmid")
    if pmid:
        pmid = pmid.rsplit("/", 1)[-1]
    doi = _bare_doi(work.get("doi"))
    return {
        "title": work.get("title"),
        "author": ", ".join(authors) if authors else "n/a",
        "venue": source.get("display_name") or "n/a",
        "year": work.get("publication_year") or 0,
        "url": f"https://doi.org/{doi}" if doi else (work.get("id") or ""),
        "cited_by": work.get("cited_by_count") or 0,
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "doi": doi,
        "pmid": pmid,
        "openalex_id": _short_id(work.get("id")),
        "source_doi": None,
        "source_relation": None,
        "discovery_backend": "openalex",
        "discovery_method": "accession_mention",
        "matched_accession": matched_accession,
    }


async def _search_term(client: OpenAlexClient, term: str) -> list[dict[str, Any]]:
    """Fully paginate OpenAlex `fulltext.search:term`, returning mapped hits.

    Pagination is decoupled from any result cap so the full hit set is collected
    deterministically; truncation happens after a stable sort in `search`. The
    loop terminates on a falsy cursor, on an empty results page (OpenAlex index
    lag can return `results: []` with `next_cursor` still set), or at the
    `_MAX_PAGES` safety bound. A page that is not a JSON object raises
    TypeError.
    """
    out: list[dict[str, Any]] = []
    cursor: str | None = "*"
    pages = 0
    while cursor and pages < _MAX_PAGES:
        resp = await client.get(
            "works",
            params={
                "filter": f"fulltext.search:{term}",
                "per-page": 100,
                "cursor": cursor,
            },
        )
        data = resp.json()
        if not isinstance(data, dict):
            # An upstream contract break, not a transient failure, so it
            # escapes instead of being recorded as a failed term.
            raise TypeError(
                f"OpenAlex search for {term} returned a {type(data).__name__}, "
                "not a results page"
            )
        results = data.get("results", [])
        if not results:
            break
        out.extend(work_to_citation(work, term) for work in results)
        pages += 1
        cursor = (data.get("meta") or {}).get("next_cursor")
    if pages >= _MAX_PAGES and cursor:
        logger.warning(
            "accession term %s hit the %d-page cap; results truncated", term, _MAX_PAGES
        )
    return out


def _stable_key(citation: dict[str, Any]) -> str:
    """Dedup / sort key: DOI, else OpenAlex id, else title (stripped+lowercased)."""
    for field in ("doi", "openalex_id", "title"):
        value = citation.get(field)
        if value:
            return str(value).strip().lower()
    return ""


def dedup_citations(hit_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Merge per-term hit lists into one deduped, stably-sorted list.

    A paper that matches more than one accession term (e.g. an `on-` dataset's
    `on*` and `ds*` ids) appears once. Pure function -> unit-testable without
    network. The stable sort is what makes a later truncation deterministic.
    """
    deduped: dict[str, dict[str, Any]] = {}
    for hits in hit_lists:
        for citation in hits:
            key = _stable_key(citation)
            if key and key not in deduped:
                deduped[key] = citation
    return sorted(deduped.values(), key=_stable_key)


class AccessionSearchBackend:
    """Sync facade: search OpenAlex full text for dataset accession mentions."""

    # The OpenAlex client class `search` opens; a test points a subclass at a
    # local HTTP server to exercise the real requests, paging, and error
    # handling.
    _openalex_client_cls: ClassVar[type[OpenAlexClient]] = OpenAlexClient

    def __init__(self, config: Config | None = None, *, max_results: int = 200) -> None:
        self._config = config or Config.from_env()
        self._max_results = max_results

    def search(self, terms: list[str]) -> AccessionSearchResult:
        """Return citations mentioning any of `terms`, plus the terms that failed.

        Citations are sorted by a stable key and truncated to `max_results`
        AFTER the sort, so the kept subset is identical across runs even when a
        term has more hits than the cap (content-idempotent writes depend on
        this). Per-term failures are logged and skipped rather than raised, so
        one dead term still yields the others' hits; the failures are reported
        in `failed_terms` so the caller can tell a complete search from a
        degraded one.
        """
        if not terms:
            return AccessionSearchResult(citations=[], failed_terms=[])
        deduped, failed = asyncio.run(self._search_all(terms))
        if self._max_results and len(deduped) > self._max_results:
            deduped = deduped[: self._max_results]
        return AccessionSearchResult(citations=deduped, failed_terms=failed)

    async def _search_all(
        self, terms: list[str]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        hit_lists: list[list[dict[str, Any]]] = []
        failed: list[str] = []
        async with self._openalex_client_cls(self._config) as client:
            assert isinstance(client, OpenAlexClient), (  # noqa: S101 - upstream contract guard
                "opencite changed OpenAlexClient.__aenter__ return type; "
                f"got {type(client).__name__}"
            )
            for term in terms:
                try:
                    hit_lists.append(await _search_term(client, term))
                except (TimeoutError, OpenCiteError, httpx.HTTPError, OSError) as e:
                    logger.warning("accession search failed for %s: %s", term, e)
                    failed.append(term)
        return dedup_citations(hit_lists), failed
