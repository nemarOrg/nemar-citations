"""Phase 3 orchestrator: fetch dataset citations via the opencite backend.

Picks a source extractor by dataset-ID prefix (`nm*` / `on*` -> nemar metadata,
`ds*` -> OpenNeuro BIDS description), resolves DOI/PMID/arXiv references,
optionally seeds the dataset's own catalog DOI as an additional anchor, asks
the opencite backend for citing works, deduplicates them across source
anchors, and returns a citation JSON dict in the schema-v2 shape that
`citation_utils.add_discovery_provenance` produces.

A per-dataset checkpoint store (see `core.checkpoint`) records each anchor's
opencite result the moment it lands. A subsequent run only re-fetches anchors
that hadn't completed successfully, which is what lets a 6h CI window resume
where the previous one timed out.

This module produces the canonical schema-v2.1 `citations/json_opencite/`
output that the dashboard aggregator and reporting tools read. (The pre-opencite
`citation_utils.save_citation_json` write path whose shape it originally
mirrored was removed in epic #180 phase 5.)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dataset_citations.backends import OpenCiteBackend
from dataset_citations.core.checkpoint import (
    DEFAULT_CHECKPOINT_DIR,
    CheckpointStore,
)
from dataset_citations.core.citation_identity import (
    base_doi,
    dedupe_citations,
    normalize_title,
)
from dataset_citations.core.citation_utils import (
    add_discovery_provenance,
    stamp_dataset_metadata,
)
from dataset_citations.quality.anchor_judgment_io import (
    DEFAULT_JUDGMENTS_DIR,
    JudgmentSidecar,
    canonical_anchor_key,
    load_judgment_sidecar,
)
from dataset_citations.sources import (
    EMPTY_NEMAR_DATASET_METADATA,
    BidsMetadataSource,
    CitingWork,
    DoiReference,
    FetchSuccess,
    NemarDatasetMetadata,
    NemarMetadataSource,
)
from dataset_citations.sources.doi import (
    is_openneuro_dataset_doi,
    normalize_doi,
    validate_identifier,
)

logger = logging.getLogger(__name__)

# The single classification that survives the partition into "fetch citations
# from this anchor". Everything else (umbrella, methodology, related_work,
# irrelevant) is recorded as context only.
_FETCH_CLASSIFICATION = "data_paper"


def fetch_dataset_citations_via_opencite(
    dataset_id: str,
    *,
    backend: OpenCiteBackend | None = None,
    nemar_source: NemarMetadataSource | None = None,
    bids_source: BidsMetadataSource | None = None,
    github_token: str | None = None,
    fetch_date: datetime | None = None,
    catalog_doi: str | None = None,
    checkpoint_store: CheckpointStore | None = None,
    use_checkpoint: bool = False,
    judgments_dir: Path | str | None = None,
    local_metadata_dir: str | None = None,
) -> dict[str, Any]:
    """Return a schema-v2 citation JSON dict for one dataset.

    Steps:
      1. Pick the source by dataset prefix (`nm` / `on` -> nemar, `ds` -> bids).
      2. Fetch DOI/PMID anchors from the source.
      3. If `catalog_doi` is set, add it as an extra anchor with
         `relation_type=References` and `source=nemar_catalog`.
      4. Dedupe anchors by normalized identifier.
      4a. *Anchor judgment* (epic #76 phase 3): load
          `citations/anchor_judgments/<id>.json`. Anchors classified as
          `data_paper` are kept for the backend call; everything else
          (umbrella / methodology / related_work / irrelevant) becomes
          context-only. Every anchor (kept or not) is recorded in
          `metadata.anchors[]` with a `kept` flag (schema v2.1). When the
          sidecar is missing (or an anchor has no entry) the pipeline falls
          back to its pre-phase-3 behavior and fetches that anchor — phase 4
          flips this to mandatory once the backfill is complete.
      5. If `use_checkpoint=True`, consult the checkpoint store for any
         anchors already fetched successfully on a previous run; skip them
         in the backend call. (Default is off so unit tests aren't surprised
         by ambient filesystem state; the CLI opts in.)
      6. Ask `OpenCiteBackend.get_citing_works_batch` for the remaining anchors.
      7. Persist each new anchor outcome to the checkpoint as it lands.
      8. Aggregate works across all anchors (checkpointed + freshly fetched),
         dedupe by (normalized DOI || title).
      9. Build the JSON dict with discovery provenance fields. On full
         success the checkpoint file is removed.

    Always returns a dict; never raises for expected fetch failures.
    `metadata.fetch_status` distinguishes outcomes:
      - "success": every anchor returned a successful result.
      - "partial": at least one anchor returned a FetchError but others
        produced citing works; see `metadata.anchor_errors` for details.
      - Any other value (e.g. "rate_limit", "not_found", "no_doi_references",
        "unsupported_prefix:...") indicates a stub payload with zero citations.
    """
    when = fetch_date or datetime.now(UTC)
    backend = backend or OpenCiteBackend()
    store = checkpoint_store or (
        CheckpointStore(base_dir=DEFAULT_CHECKPOINT_DIR) if use_checkpoint else None
    )
    sidecar_dir: Path | str = (
        judgments_dir if judgments_dir is not None else DEFAULT_JUDGMENTS_DIR
    )

    if dataset_id.startswith(("nm", "on")):
        # nm-* = NEMAR-native; on-* = OpenNeuro imported into NEMAR. Both carry
        # .nemar/metadata.json so the same source handles them.
        source: Any = nemar_source or NemarMetadataSource(github_token=github_token)
    elif dataset_id.startswith("ds"):
        source = bids_source or BidsMetadataSource(
            github_token=github_token, local_metadata_dir=local_metadata_dir
        )
    else:
        return _stub_payload(
            dataset_id, when, fetch_status=f"unsupported_prefix:{dataset_id[:2]}"
        )

    refs_result = source.get_doi_references(dataset_id)
    if not isinstance(refs_result, FetchSuccess):
        logger.warning(
            "source lookup for %s failed: %s (%s)",
            dataset_id,
            refs_result.reason,
            refs_result.detail,
        )
        return _stub_payload(dataset_id, when, fetch_status=refs_result.reason)

    # Source resolved: pull the descriptive dataset metadata (keywords /
    # methods_description / funding) once for the schema-v2.1 fields. Done here,
    # before the `no_doi_references` early-out, so even DOI-less datasets carry
    # their rich metadata. Capability-gated so the BIDS source / test stubs
    # without the method fall back to empty.
    dataset_metadata = _fetch_dataset_metadata(source, dataset_id)

    refs: list[DoiReference] = _merge_anchors(refs_result.value, catalog_doi)
    if not refs:
        return _stub_payload(
            dataset_id,
            when,
            fetch_status="no_doi_references",
            dataset_metadata=dataset_metadata,
        )

    sidecar = load_judgment_sidecar(dataset_id, judgments_dir=sidecar_dir)
    fetch_refs = _partition_by_judgment(refs, sidecar, dataset_id)
    fetch_ids = {ref.identifier for ref in fetch_refs}
    anchors = _build_anchor_records(refs, fetch_ids, sidecar)
    searched_dois = [
        ref.identifier for ref in fetch_refs if ref.identifier_type == "doi"
    ]
    judgment_model = sidecar.model if sidecar.present else None

    if not fetch_refs:
        return _stub_payload(
            dataset_id,
            when,
            fetch_status="no_data_paper_anchor",
            anchor_count=len(refs),
            anchor_judgment_model=judgment_model,
            anchors=anchors,
            searched_dois=searched_dois,
            dataset_metadata=dataset_metadata,
        )

    pending_refs, checkpointed_works = _split_against_checkpoint(
        fetch_refs, store, dataset_id
    )

    batch: dict[str, Any] = {}
    if pending_refs:
        batch = backend.get_citing_works_batch(pending_refs)
        if store is not None:
            for ref in pending_refs:
                outcome = batch.get(ref.identifier)
                if outcome is None:
                    continue
                store.record_anchor(dataset_id, ref.identifier, outcome, when=when)

    citing_works, per_anchor_errors = _flatten_batch(
        batch, prior_works=checkpointed_works
    )

    if not citing_works and per_anchor_errors:
        reason = _dominant_error_reason(per_anchor_errors)
        return _stub_payload(
            dataset_id,
            when,
            fetch_status=reason,
            anchor_count=len(fetch_refs),
            anchor_errors=per_anchor_errors,
            anchor_judgment_model=judgment_model,
            anchors=anchors,
            searched_dois=searched_dois,
            dataset_metadata=dataset_metadata,
        )

    citation_details, duplicates_dropped = dedupe_citations(
        [_citing_work_to_dict(w) for w in citing_works]
    )
    if duplicates_dropped:
        logger.info(
            "%s: merged %d duplicate citing work(s) (version/preprint variants)",
            dataset_id,
            duplicates_dropped,
        )
    # Counts must come from the deduped list, not `citing_works`, or the
    # cumulative total keeps counting a preprint and its published version.
    total_cumulative_citations = sum(
        int(c.get("cited_by") or 0) for c in citation_details
    )

    payload: dict[str, Any] = {
        "dataset_id": dataset_id,
        "num_citations": len(citation_details),
        "date_last_updated": when.isoformat(),
        "metadata": {
            "total_cumulative_citations": int(total_cumulative_citations),
            "fetch_date": when.isoformat(),
            "processing_version": "1.0",
            # schema_version + discovery_backend are stamped by
            # _finalize_payload (the single authoritative exit), so they are
            # intentionally not duplicated here.
            "fetch_status": "success" if not per_anchor_errors else "partial",
            "anchor_count": len(fetch_refs),
            "anchor_errors": per_anchor_errors,
        },
        "citation_details": citation_details,
    }

    if store is not None and not per_anchor_errors:
        store.clear(dataset_id)

    return _finalize_payload(
        payload,
        anchors=anchors,
        searched_dois=searched_dois,
        anchor_judgment_model=judgment_model,
        dataset_metadata=dataset_metadata,
    )


def _merge_anchors(
    source_refs: list[DoiReference], catalog_doi: str | None
) -> list[DoiReference]:
    """Combine source-derived refs with an optional catalog-row DOI.

    Catalog DOI is added with relation_type=References and source=nemar_catalog.
    The result is deduplicated by normalized identifier, preserving the
    source-derived entry when there's a collision (more specific relation type).
    """
    seen: dict[str, DoiReference] = {}
    for ref in source_refs:
        seen.setdefault(ref.identifier, ref)

    if catalog_doi:
        normalized = normalize_doi(catalog_doi)
        if not validate_identifier(normalized):
            logger.warning("ignoring malformed catalog DOI %r", catalog_doi)
        elif is_openneuro_dataset_doi(normalized):
            logger.info(
                "skipping OpenNeuro dataset DOI from catalog (not indexed): %s",
                normalized,
            )
        elif normalized not in seen:
            seen[normalized] = DoiReference(
                identifier=normalized,
                identifier_type="doi",
                relation_type="References",
                source="nemar_catalog",
                source_field="catalog_doi",
            )

    return list(seen.values())


def _split_against_checkpoint(
    refs: list[DoiReference],
    store: CheckpointStore | None,
    dataset_id: str,
) -> tuple[list[DoiReference], list[CitingWork]]:
    """Partition anchors into (to-fetch, already-fetched-works).

    A checkpoint hit with status="success" yields its works directly; any
    other status (including a miss) puts the anchor in the to-fetch bucket
    so the backend retries it on this run. Prior non-success outcomes are
    intentionally not surfaced as `errors` here: the current run will
    re-attempt those anchors and the fresh outcome takes precedence.
    """
    if store is None:
        return list(refs), []

    checkpoint = store.load(dataset_id)
    pending: list[DoiReference] = []
    prior_works: list[CitingWork] = []

    for ref in refs:
        if checkpoint.is_success(ref.identifier):
            prior_works.extend(checkpoint.successful_works(ref.identifier))
        else:
            pending.append(ref)

    if prior_works:
        logger.info(
            "resuming %s with %d works from checkpoint across %d anchors",
            dataset_id,
            len(prior_works),
            sum(1 for r in refs if checkpoint.is_success(r.identifier)),
        )

    return pending, prior_works


def _flatten_batch(
    batch: dict[str, Any],
    *,
    prior_works: list[CitingWork] | None = None,
) -> tuple[list[CitingWork], dict[str, str]]:
    """Dedupe citing works across all anchors; return (works, per-anchor errors).

    `prior_works` carries successful results from previous checkpoint hits
    and is merged into the dedupe set before the fresh batch is consumed.
    Errors come exclusively from the current run; checkpoint-resident errors
    are not surfaced (the corresponding anchors are re-fetched and produce
    a fresh outcome that lands here).
    """
    seen: set[str] = set()
    works: list[CitingWork] = []
    errors: dict[str, str] = {}

    def _take(work: CitingWork) -> None:
        # A DOI identifies the work once version suffixes are stripped; only
        # when there is no DOI does the title become the identity. Keying on
        # the PAIR (the pre-#216 behavior) meant a single punctuation
        # difference between OpenAlex and S2 produced two records for one
        # paper. Preprint-vs-published pairs carry genuinely different DOIs,
        # so they survive this pass and are resolved by the
        # `dedupe_citations` call that runs next (defined earlier in this
        # file), which can record the superseded DOI on the survivor.
        key = base_doi(work.doi) or normalize_title(work.title)
        if not key:
            # No usable identity: no DOI, and a title that normalizes to
            # nothing (a fully non-Latin-script title, since normalize_title
            # keeps only ASCII alphanumerics). Keep it as its own record
            # rather than drop it. An empty key must never mean "same as the
            # other empty-key record", which is the choice dedupe_citations
            # and merge_accession_mentions both already make.
            logger.warning(
                "citing work has no usable identity (doi=%r, title=%r); "
                "keeping it as a distinct record",
                work.doi,
                work.title,
            )
            works.append(work)
            return
        if key in seen:
            return
        seen.add(key)
        works.append(work)

    for work in prior_works or []:
        _take(work)

    for anchor_id, result in batch.items():
        if not isinstance(result, FetchSuccess):
            errors[anchor_id] = result.reason
            continue
        for work in result.value:
            _take(work)

    return works, errors


def _citing_work_to_dict(work: CitingWork) -> dict[str, Any]:
    return {
        "title": work.title,
        "author": ", ".join(a.name for a in work.authors) if work.authors else "n/a",
        "venue": work.venue or "n/a",
        "year": work.year or 0,
        "url": _doi_to_url(work.doi),
        "cited_by": work.citation_count or 0,
        "abstract": work.abstract,
        "doi": work.doi,
        "pmid": work.pmid,
        "openalex_id": work.openalex_id,
        "source_doi": work.source_doi,
        "source_relation": work.source_relation,
        "discovery_backend": "opencite",
    }


def _doi_to_url(doi: str | None) -> str:
    return f"https://doi.org/{doi}" if doi else "n/a"


def _dominant_error_reason(errors: dict[str, str]) -> str:
    counts: dict[str, int] = {}
    for reason in errors.values():
        counts[reason] = counts.get(reason, 0) + 1
    return max(counts, key=lambda r: counts[r]) if counts else "unknown"


def _stub_payload(
    dataset_id: str,
    when: datetime,
    *,
    fetch_status: str,
    anchor_count: int = 0,
    anchor_errors: dict[str, str] | None = None,
    anchor_judgment_model: str | None = None,
    anchors: list[dict[str, Any]] | None = None,
    searched_dois: list[str] | None = None,
    dataset_metadata: NemarDatasetMetadata = EMPTY_NEMAR_DATASET_METADATA,
) -> dict[str, Any]:
    payload = {
        "dataset_id": dataset_id,
        "num_citations": 0,
        "date_last_updated": when.isoformat(),
        "metadata": {
            "total_cumulative_citations": 0,
            "fetch_date": when.isoformat(),
            "processing_version": "1.0",
            # schema_version + discovery_backend are stamped by _finalize_payload.
            "fetch_status": fetch_status,
            "anchor_count": anchor_count,
            "anchor_errors": anchor_errors or {},
        },
        "citation_details": [],
    }
    return _finalize_payload(
        payload,
        anchors=anchors or [],
        searched_dois=searched_dois or [],
        anchor_judgment_model=anchor_judgment_model,
        dataset_metadata=dataset_metadata,
    )


def _finalize_payload(
    payload: dict[str, Any],
    *,
    anchors: list[dict[str, Any]],
    searched_dois: list[str],
    anchor_judgment_model: str | None,
    dataset_metadata: NemarDatasetMetadata,
) -> dict[str, Any]:
    """Stamp schema-v2.1 provenance + descriptive metadata onto a payload.

    The single exit point shared by the success/partial path and every
    `_stub_payload`, so all outcomes emit an identical `metadata` shape.
    """
    add_discovery_provenance(
        payload,
        discovery_backend="opencite",
        anchor_judgment_model=anchor_judgment_model,
        anchors=anchors,
        searched_dois=searched_dois,
    )
    stamp_dataset_metadata(
        payload,
        keywords=list(dataset_metadata.keywords),
        methods_description=dataset_metadata.methods_description,
        funding=list(dataset_metadata.funding),
    )
    return payload


def _fetch_dataset_metadata(source: Any, dataset_id: str) -> NemarDatasetMetadata:
    """Pull descriptive dataset metadata (keywords / methods / funding) once.

    Capability-gated: only `NemarMetadataSource` implements
    `get_dataset_metadata`. The BIDS source (legacy ds-*) and unit-test stubs
    without the method fall back to empty, so those datasets still get the
    schema-v2.1 keys with empty values.
    """
    getter = getattr(source, "get_dataset_metadata", None)
    if getter is None:
        return EMPTY_NEMAR_DATASET_METADATA
    try:
        return getter(dataset_id)
    except Exception:
        # Rich metadata is provenance, never a hard dependency: an unexpected
        # source error must degrade to empty, not abort the citation fetch.
        # Logged (not swallowed) so the gap is visible in the cron log.
        logger.exception(
            "unexpected error fetching rich metadata for %s; using empty",
            dataset_id,
        )
        return EMPTY_NEMAR_DATASET_METADATA


def _partition_by_judgment(
    refs: list[DoiReference],
    sidecar: JudgmentSidecar,
    dataset_id: str,
) -> list[DoiReference]:
    """Return the anchors to fetch citations from, per phase 2's sidecar.

    Rules:
      * Sidecar missing entirely: log a single INFO line and fetch all anchors
        (the legacy-compatible fallback while the backfill is still in flight).
      * Sidecar present, anchor classified `data_paper`: fetch.
      * Sidecar present, anchor classified anything else: skip. It still lands
        in `metadata.anchors[]` (with `kept=False`), built separately by
        `_build_anchor_records`.
      * Sidecar present but anchor has no entry: fetch (fall back to fetching);
        surfaced via a single WARN per dataset until phase 4 makes judgment
        mandatory.
    """
    if not sidecar.present:
        logger.info(
            "%s: anchor-judgment sidecar missing; fetching all %d anchors (fallback)",
            dataset_id,
            len(refs),
        )
        return list(refs)

    fetch_refs: list[DoiReference] = []
    unjudged_count = 0

    for ref in refs:
        key = canonical_anchor_key(ref.identifier, ref.identifier_type)
        classification = sidecar.lookup.get(key) if key is not None else None
        if classification is None:
            # Anchor not present in sidecar -> fall back to fetching it.
            # Phase 4 will make judgment mandatory once the backfill is
            # complete; until then we surface the drift via a single WARN
            # per dataset so operators can see at-a-glance how many
            # anchors slipped past the judgment step.
            fetch_refs.append(ref)
            unjudged_count += 1
            continue
        if classification == _FETCH_CLASSIFICATION:
            fetch_refs.append(ref)

    if unjudged_count:
        logger.warning(
            "%s: %d/%d anchors in this dataset have no judgment in the sidecar; "
            "they will be fetched as fallback. Re-run dataset-citations-judge-anchors "
            "to close the gap.",
            dataset_id,
            unjudged_count,
            len(refs),
        )

    return fetch_refs


def _build_anchor_records(
    refs: list[DoiReference],
    fetch_ids: set[str],
    sidecar: JudgmentSidecar,
) -> list[dict[str, Any]]:
    """Build `metadata.anchors[]`: every anchor, kept or context (schema v2.1).

    `kept` is membership in `fetch_ids` (the identifiers the pipeline actually
    sent to the backend), so unjudged-fallback anchors correctly show
    `kept=True` with a `None` classification. `classification` / `paper_*` /
    `reason` come from phase 2's sidecar (`context_details` is populated for
    `data_paper` anchors too, so kept anchors carry their own paper title).
    `judgment_model` mirrors `metadata.anchor_judgment_model` for per-anchor
    self-description. `identifier` stays the pipeline's canonical form so it
    matches `searched_dois` and each citation's `source_doi`.

    Replaces the v2.0 `context_anchors[]`; that key was just the `kept=False`
    subset, which a consumer now derives via `[a for a in anchors if not a["kept"]]`.
    """
    model = sidecar.model if sidecar.present else None
    records: list[dict[str, Any]] = []
    for ref in refs:
        key = canonical_anchor_key(ref.identifier, ref.identifier_type)
        classification = sidecar.lookup.get(key) if key is not None else None
        details = sidecar.context_details.get(key) if key is not None else None
        records.append(
            {
                "identifier": ref.identifier,
                "identifier_type": ref.identifier_type,
                "source_relation": ref.relation_type,
                "classification": classification,
                "kept": ref.identifier in fetch_ids,
                "paper_title": details.get("paper_title") if details else None,
                "paper_year": details.get("paper_year") if details else None,
                "paper_venue": details.get("paper_venue") if details else None,
                "reason": details.get("reason") if details else None,
                "judgment_model": model,
            }
        )
    return records
