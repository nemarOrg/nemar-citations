"""Merge accession-mention citations into a dataset's citation JSON.

Accession-mention citations (papers that name the dataset accession in text)
are discovered by `backends/accession_search.py` and folded into the same
`citation_details[]` list as anchor-based citations, each tagged
`discovery_method="accession_mention"`. The toggle (#169) buckets:
  - "cites dataset"   = accession mentions + anchor citations whose
                        source_relation is IsVersionOf / IsIdenticalTo
  - "cites data paper"= the remaining anchor citations (References /
                        IsDerivedFrom / IsDescribedBy)

`IsDescribedBy` (a data paper that describes the dataset, e.g. the on000117
data paper) is intentionally NOT in `_DATASET_RELATIONS`: its citations are
citations of the data paper, not of the dataset record itself.

This module is pure (no network / I/O) and deterministic so repeated runs are
content-idempotent (issue #165). Issue #169.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from dataset_citations.core.citation_identity import (
    base_doi,
    normalize_title,
)

# Intentional subset of RelationType: only the relations whose anchor IS the
# dataset record (vs a paper about it). IsDescribedBy/References/IsDerivedFrom
# denote papers, so they stay out.
_DATASET_RELATIONS = frozenset({"IsVersionOf", "IsIdenticalTo"})


def _ids(citation: dict[str, Any]) -> list[str]:
    """Strong identifiers (DOI + OpenAlex id) for dedup matching.

    The DOI goes through `base_doi` so a concept DOI and its versioned form
    (`10.82901/nemar.on004842` / `...on004842.v1.0.0`) match each other, and
    any superseded DOI already
    folded into this record still identifies it -- otherwise a mention
    carrying the preprint DOI would re-append a paper we merged earlier.
    """
    out: list[str] = []
    doi = base_doi(citation.get("doi"))
    if doi:
        out.append(doi)
    for superseded in citation.get("superseded_dois") or []:
        key = base_doi(superseded)
        if key:
            out.append(key)
    openalex_id = citation.get("openalex_id")
    if openalex_id:
        out.append(str(openalex_id).strip().lower())
    return out


def _title_key(citation: dict[str, Any]) -> str:
    """Normalized title, used as a dedup fallback only when no strong ids exist.

    Shares `normalize_title` with the fetch-side dedup so punctuation drift
    between sources ("... replication." vs "... replication") does not let a
    mention re-enter as a second copy.
    """
    return normalize_title(citation.get("title"))


def cites_dataset(citation: dict[str, Any]) -> bool:
    """True if `citation` belongs in the 'cites dataset' bucket."""
    if citation.get("discovery_method") == "accession_mention":
        return True
    if citation.get("mentions_accession"):
        return True
    return citation.get("source_relation") in _DATASET_RELATIONS


def merge_accession_mentions(
    citation_json: dict[str, Any],
    mentions: list[dict[str, Any]],
    searched_accessions: list[str],
    *,
    when: datetime | None = None,
) -> dict[str, Any]:
    """Fold accession `mentions` into `citation_json` (mutated and returned).

    - A mention whose paper is already a citation flags that existing entry
      (`mentions_accession=True` + `matched_accession`) so a paper that both
      cites the data paper and names the accession lands in BOTH buckets. An
      existing accession-mention entry is left untouched (idempotent).
    - A mention not already present is appended.
    - When new mentions are appended, the stale top-level `confidence_scoring`
      block is dropped so the downstream score step (its `--skip-existing`
      skips files that already carry the block) re-scores the dataset.
    - `num_citations` is recomputed; `metadata` records the searched accessions
      and the per-bucket breakdown.
    - `date_last_updated` advances to `when` (default now) whenever a mention is
      appended or an existing citation is newly flagged. `date_last_updated`
      means "last content change" (issue #165); this step changes content just
      as much as the opencite fetch does, and was previously the one write path
      that silently left the field stale (issue #229).

    All counts derive from the final state (not deltas), so a re-run with the
    same mentions produces byte-identical output.
    """
    details: list[dict[str, Any]] = citation_json.setdefault("citation_details", [])
    # Index existing citations by EVERY strong identifier (DOI and OpenAlex id),
    # plus title as a fallback, so a mention is recognised as a duplicate when it
    # shares ANY identifier with an existing citation -- not just the first one.
    # This is what prevents double-counting a paper found by both the anchor and
    # accession-mention paths when the two copies don't carry the same id set.
    by_id: dict[str, dict[str, Any]] = {}
    by_title: dict[str, dict[str, Any]] = {}
    for citation in details:
        for ident in _ids(citation):
            by_id.setdefault(ident, citation)
        title = _title_key(citation)
        if title:
            by_title.setdefault(title, citation)

    def _register(citation: dict[str, Any]) -> None:
        for ident in _ids(citation):
            by_id.setdefault(ident, citation)
        title = _title_key(citation)
        if title:
            by_title.setdefault(title, citation)

    appended = 0
    flagged = 0
    for mention in mentions:
        mention_ids = _ids(mention)
        existing = next((by_id[i] for i in mention_ids if i in by_id), None)
        if existing is None and not mention_ids:
            # No DOI / OpenAlex id to match on; fall back to exact title.
            existing = by_title.get(_title_key(mention))
        if existing is None:
            if not mention_ids and not _title_key(mention):
                continue  # nothing to dedup or display by; drop it
            details.append(mention)
            _register(mention)
            appended += 1
        elif existing.get("discovery_method") != "accession_mention":
            # Anchor citation that also names the accession -> flag for both
            # buckets (no-op if already flagged, keeping re-runs idempotent).
            if existing.get("mentions_accession") is not True:
                flagged += 1
            existing["mentions_accession"] = True
            existing.setdefault("matched_accession", mention.get("matched_accession"))

    if appended:
        citation_json.pop("confidence_scoring", None)
    if appended or flagged:
        citation_json["date_last_updated"] = (when or datetime.now(UTC)).isoformat()

    citation_json["num_citations"] = len(details)
    meta = citation_json.setdefault("metadata", {})
    meta["searched_accessions"] = list(searched_accessions)
    # Provenance: papers found purely via the accession search.
    meta["num_accession_mentions"] = sum(
        1 for c in details if c.get("discovery_method") == "accession_mention"
    )
    # The two toggle buckets, recorded explicitly so the dashboard / leaderboard
    # keys on them rather than re-deriving (and never compares a raw
    # num_citations from a find-mentions-processed dataset against a not-yet-
    # processed one). num_citations stays the deduped total of both buckets.
    dataset_bucket = sum(1 for c in details if cites_dataset(c))
    meta["num_dataset_citations"] = dataset_bucket
    meta["num_datapaper_citations"] = len(details) - dataset_bucket
    return citation_json
