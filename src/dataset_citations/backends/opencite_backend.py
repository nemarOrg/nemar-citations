"""Sync facade over `opencite.citations.CitationExplorer`.

opencite is async-only; this module hides the asyncio boilerplate so the
rest of the pipeline (sync, pandas-first) can stay sync. Batches reuse a
single `CitationExplorer` session so we get connection pooling and one
catalog warm-up per call.

Source selection and rate limiting are delegated to opencite, not
hand-rolled here. `CitationExplorer` fans out the cited-by lookup across
its enabled sources (OpenAlex + Semantic Scholar today), honoring
`config.disabled_sources` so a deployment can drop a source via
`OPENCITE_DISABLED_SOURCES` without a code change. opencite's process-wide
per-source shared rate limiter (e.g. Semantic Scholar at 1 req/s under the
`"s2"` key) keeps a slow source from triggering 429 storms, which is what
the old `S2_SKIP_PREFIXES` router existed to avoid; that local routing has
been removed (epic #180, phase 2).

Errors are surfaced through `FetchResult`, never silently absorbed. The
Phase 1 review found that an opencite rate limit was being cached as
"this DOI has zero citing works"; here a rate limit returns
`FetchError("rate_limit", ...)` and the caller decides whether to retry.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from typing import Any

import httpx
from opencite.citations import CitationExplorer
from opencite.clients import openalex as _openalex
from opencite.clients.openalex import OpenAlexClient
from opencite.config import Config
from opencite.exceptions import APIError, APIKeyError, RateLimitError

from dataset_citations.sources.models import (
    Author,
    CitingWork,
    DoiReference,
    FetchError,
    FetchResult,
    FetchSuccess,
)

logger = logging.getLogger(__name__)

# The field list opencite's own DOI lookup requests; None asks for the full work.
_OPENALEX_WORK_FIELDS: str | None = getattr(_openalex, "_WORK_FIELDS", None)


class OpenCiteBackend:
    """Resolve `DoiReference` anchors into lists of `CitingWork` via opencite."""

    # The OpenAlex client class `get_paper` opens; a test points a subclass at a
    # local HTTP server to exercise the real error handling.
    _openalex_client_cls: type[OpenAlexClient] = OpenAlexClient

    def __init__(
        self,
        config: Config | None = None,
        *,
        max_results_per_doi: int = 100,
        concurrency: int = 4,
    ) -> None:
        self._config = config or Config.from_env()
        self._max_results_per_doi = max_results_per_doi
        self._concurrency = max(1, concurrency)

    def get_citing_works(self, ref: DoiReference) -> FetchResult[list[CitingWork]]:
        """Look up citing works for a single anchor."""
        return asyncio.run(self._batch_async([ref]))[ref.identifier]

    def get_citing_works_batch(
        self, refs: Iterable[DoiReference]
    ) -> dict[str, FetchResult[list[CitingWork]]]:
        """Look up citing works for many anchors, reusing one opencite session."""
        ref_list = list(refs)
        if not ref_list:
            return {}
        return asyncio.run(self._batch_async(ref_list))

    def get_paper(self, doi: str) -> FetchResult[CitingWork]:
        """Look up the anchor paper's own bib + abstract via OpenAlex.

        Returns a `CitingWork` populated with the paper's title, abstract,
        authors, venue, year, and ID fields. The `source_doi` is set to the
        looked-up DOI and `source_relation` defaults to "References" since
        the model only carries the four DataCite values; callers treating
        this as a self-lookup should ignore both fields.

        Used by epic #76's anchor adjudication path (probe + judge-anchors
        CLI) to assemble the candidate paper context for the LLM prompt.

        Only an OpenAlex 404 (or a record without a title) is `not_found`; the
        judge treats that as permanent and retries it monthly. Any other
        failure is transient: a 429 is `rate_limit`, and a 5xx or network
        error that outlasts opencite's retries is `network`.
        """
        return asyncio.run(self._get_paper_async(doi))

    async def _get_paper_async(self, doi: str) -> FetchResult[CitingWork]:
        # Narrow exception list: classify_error is built for network-shaped
        # failures, so we only route those through it. Programmer errors
        # (AttributeError on opencite API drift, AssertionError on the
        # isinstance invariant below, TypeError) must propagate to surface
        # the contract break instead of being silently demoted to
        # FetchError("other", ...). Mirror this list against the existing
        # _lookup branch's handling if it widens in the future.
        try:
            async with self._openalex_client_cls(self._config) as openalex_base:
                assert isinstance(openalex_base, OpenAlexClient), (  # noqa: S101 - upstream contract guard
                    "opencite changed OpenAlexClient.__aenter__ return type; "
                    "expected an OpenAlexClient bound, got "
                    f"{type(openalex_base).__name__}"
                )
                # Not `lookup_doi`: it swallows every error and returns None,
                # so an OpenAlex outage would read as an unindexed DOI and the
                # judge would lock the anchor out for a month (#241 review).
                params = (
                    {"select": _OPENALEX_WORK_FIELDS} if _OPENALEX_WORK_FIELDS else None
                )
                try:
                    resp = await openalex_base.get(f"/works/doi:{doi}", params=params)
                except APIError as exc:
                    status = _status_code_of(exc)
                    if status == 404:
                        return FetchError(
                            "not_found",
                            f"{doi}: not found in OpenAlex (anchor self-lookup)",
                        )
                    if status is None and not isinstance(
                        exc, (RateLimitError, APIKeyError)
                    ):
                        # opencite gave up after retrying a 5xx or a network error.
                        return FetchError("network", f"{doi}: {exc}")
                    return classify_error(exc, doi)
                paper = openalex_base._parse_work(resp.json())
        except (TimeoutError, httpx.HTTPError, OSError) as exc:
            return classify_error(exc, doi)

        if paper is None or not paper.title:
            return FetchError(
                "not_found", f"{doi}: not found in OpenAlex (anchor self-lookup)"
            )

        # Synthesize a self-referential anchor so we can reuse the existing
        # paper -> CitingWork mapper. Callers treat source_relation as opaque
        # for self-lookups; the dataclass invariant just requires a non-empty
        # source_doi and a valid RelationType literal.
        self_ref = DoiReference(
            identifier=doi,
            identifier_type="doi",
            relation_type="References",
            source="nemar_catalog",
            source_field="self_lookup",
        )
        return FetchSuccess(_paper_to_citing_work(paper, self_ref))

    async def _batch_async(
        self, refs: list[DoiReference]
    ) -> dict[str, FetchResult[list[CitingWork]]]:
        results: dict[str, FetchResult[list[CitingWork]]] = {}
        semaphore = asyncio.Semaphore(self._concurrency)
        # One CitationExplorer session for the whole batch. opencite fans the
        # cited-by lookup across its enabled sources and applies its per-source
        # shared rate limiter internally; source selection is config-driven
        # (`config.disabled_sources`), so there is no per-anchor routing here.
        async with CitationExplorer(self._config) as explorer:

            async def run_one(ref: DoiReference) -> FetchResult[list[CitingWork]]:
                async with semaphore:
                    return await self._lookup(explorer, ref)

            # return_exceptions=True isolates a single failing lookup from the
            # whole batch; we convert each leak-through to FetchError below.
            outcomes = await asyncio.gather(
                *(run_one(r) for r in refs), return_exceptions=True
            )
        for ref, outcome in zip(refs, outcomes, strict=True):
            # `gather(return_exceptions=True)` also captures BaseExceptions like
            # asyncio.CancelledError (not an Exception subclass since 3.8). A
            # cancellation must propagate, not be demoted to FetchError("other"),
            # or an outer timeout / Ctrl+C silently turns into phantom results.
            if isinstance(outcome, BaseException) and not isinstance(
                outcome, Exception
            ):
                raise outcome
            if isinstance(outcome, Exception):
                logger.error(
                    "uncaught exception in opencite lookup for %s",
                    ref.identifier,
                    exc_info=outcome,
                )
                results[ref.identifier] = classify_error(outcome, ref.identifier)
            else:
                results[ref.identifier] = outcome
        return results

    async def _lookup(
        self, explorer: CitationExplorer, ref: DoiReference
    ) -> FetchResult[list[CitingWork]]:
        try:
            result = await explorer.citing_papers(
                ref.identifier,
                max_results=self._max_results_per_doi,
                sort="citations",
            )
        except Exception as exc:
            # opencite raises bare Exception subclasses for HTTP / parse fails;
            # narrow when neuromechanist/opencite#33 adds typed errors. We let
            # KeyboardInterrupt / SystemExit / asyncio.CancelledError escape.
            return classify_error(exc, ref.identifier)

        # opencite >= v0.5.4 no longer raises when the anchor itself can't be
        # resolved; `citing_papers` returns an empty result whose seed carries
        # no OpenAlex/S2 id (a synthetic "Unknown" seed). Surface that as
        # not_found so the caller does not record a genuinely-missing anchor as
        # "0 citing works" (the Phase 1 review's exact concern).
        #
        # Residual gap (upstream, neuromechanist/opencite#33): if the seed
        # resolves but BOTH `citing_papers` tasks fail, opencite's
        # `_gather_papers` swallows them with its own WARNING and returns an
        # empty list, which we cannot distinguish from a genuinely uncited
        # anchor at this layer. Fixing that requires opencite to expose
        # per-source task status; logging a warning here would fire on every
        # legitimately-uncited anchor, so we do not.
        if _is_unresolved_seed(result):
            return FetchError(
                "not_found",
                f"{ref.identifier}: anchor not resolved on any opencite source",
            )

        works = [
            _paper_to_citing_work(paper, ref)
            for paper in (result.papers or [])
            if paper.title
        ]
        return FetchSuccess(works)


def _paper_to_citing_work(paper: Any, ref: DoiReference) -> CitingWork:
    ids = paper.ids
    venue = paper.source_venue.name if paper.source_venue else None
    authors = tuple(
        _to_author(a) for a in (paper.authors or []) if getattr(a, "name", None)
    )
    return CitingWork(
        title=paper.title,
        doi=ids.doi or None,
        pmid=ids.pmid or None,
        openalex_id=ids.openalex_id or None,
        year=paper.year,
        authors=authors,
        venue=venue,
        abstract=paper.abstract or None,
        citation_count=(
            paper.citation_count if paper.citation_count is not None else None
        ),
        source_doi=ref.identifier,
        source_relation=ref.relation_type,
    )


def _to_author(a: Any) -> Author:
    return Author(
        name=a.name,
        family=getattr(a, "family_name", None) or None,
        given=getattr(a, "given_name", None) or None,
        orcid=getattr(a, "orcid", None) or None,
    )


def _is_unresolved_seed(result: Any) -> bool:
    """True when opencite could not resolve the anchor on any source.

    opencite >= v0.5.4 returns a `CitationResult` whose `seed_paper` is a
    synthetic "Unknown" paper carrying only the input id (no `openalex_id`,
    no `s2_id`) when `_lookup_seed` fails. A resolved anchor always has at
    least one of those ids (the S2 `lookup` or the OpenAlex fallback in
    opencite's `_lookup_seed`), so both-empty is the robust unresolved
    signal. Checking the ids rather than `title == "Unknown"` pins the
    structural invariant instead of a string constant that could change.
    """
    seed_ids = result.seed_paper.ids
    return not seed_ids.openalex_id and not seed_ids.s2_id


def classify_error(exc: BaseException, identifier: str) -> FetchError:
    """Map an opencite / httpx exception to a FetchError reason.

    Precedence order (first match wins):
      rate_limit -> auth -> not_found -> network -> other.
    Rate-limit and auth signatures are checked first so that an upstream
    error message that happens to mention 'not found' or 'connect' (e.g.,
    'Cannot connect to host: not found in DNS' on a 429 reroute) is still
    routed to the correct bucket.

    opencite's typed exceptions are checked first: `RateLimitError` and
    `APIKeyError` carry no `status_code` and their messages contain none of
    the substrings below, so without the isinstance checks a missing API key
    would be misclassified as 'other' and a retry would burn rate budget.
    After that, uses the exception's `status_code` (httpx or opencite
    `APIError`) and finally substring matching. Tracked for cleanup in
    neuromechanist/opencite#33 (typed exceptions upstream).
    """
    msg = f"{type(exc).__name__}: {exc}"
    # Typed opencite exceptions first (most precise). RateLimitError and
    # APIKeyError both subclass APIError, so check them before any generic
    # APIError/status handling below.
    if isinstance(exc, RateLimitError):
        return FetchError("rate_limit", msg)
    if isinstance(exc, APIKeyError):
        return FetchError("auth", msg)
    status = _status_code_of(exc)
    if status == 429:
        return FetchError("rate_limit", msg)
    if status in (401, 403):
        return FetchError("auth", msg)
    if status == 404:
        return FetchError("not_found", msg)

    text = str(exc).lower()
    if "429" in text or "rate limit" in text or "too many requests" in text:
        return FetchError("rate_limit", msg)
    if "401" in text or "403" in text or "unauthorized" in text or "forbidden" in text:
        return FetchError("auth", msg)
    if "404" in text or "not found" in text:
        return FetchError("not_found", msg)
    if "timeout" in text or "timed out" in text or "connect" in text:
        return FetchError("network", msg)
    logger.error("unclassified opencite failure for %s", identifier, exc_info=exc)
    return FetchError("other", msg)


def _status_code_of(exc: BaseException) -> int | None:
    """Return the HTTP status code an exception carries, or None.

    Handles httpx.HTTPStatusError-style exceptions where the response attr
    exposes `.status_code`, plus the convention some libraries use of
    setting `exc.status_code` directly.
    """
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    if isinstance(code, int):
        return code
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    code = getattr(exc, "status", None)
    if isinstance(code, int):
        return code
    return None
