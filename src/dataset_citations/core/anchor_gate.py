"""The anchor gate: which anchors may contribute citations to a dataset.

Issue #241. The gate FAILS CLOSED. An anchor contributes citations only when:

* it is the dataset's own NEMAR concept DOI (`10.82901/nemar.<id>`): citing it
  is citing the dataset, no judgment needed; or
* a successful anchor judgment classified it as the dataset's `data_paper`,
  and it is not a standards / software / platform paper (`never_anchor`); or
* it has no successful judgment but its DataCite relation says it is another
  record of the same data (`IsIdenticalTo` / `IsVersionOf`, e.g. the figshare
  or PhysioNet deposit), and it is not a `never_anchor` paper. Such a record is
  often not a paper opencite can resolve for the judge, and the relation
  itself asserts identity; a successful judgment still overrides it.

Everything else is context only: anchors the judge called umbrella /
methodology / related_work / irrelevant (including the `References` papers a
dataset merely cites; apart from the identity relations above, the DataCite
relation is a hint, never the decision), and other anchors with no successful
judgment by the trusted judge model (judge down, lookup failed, never run, or
judged by a retired model). The previous fallback fetched unjudged anchors,
which is how a dead judge turned every related-work paper's citers into
"citations" of the dataset.

The same decision function backs the fetch path (`core.opencite_pipeline`) and
the offline sweep that re-applies the gate to files already on disk
(`cli.gate_anchors`), so their per-anchor verdicts agree. The sweep can only
remove: an anchor that newly qualifies is recorded `awaiting_fetch` until the
next refetch (`cli.update` refetches such a dataset right away).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from dataset_citations.core.citation_identity import base_doi
from dataset_citations.quality.anchor_judgment_io import (
    JudgmentSidecar,
    canonical_anchor_key,
)
from dataset_citations.quality.never_anchor import is_never_anchor, is_spec_title
from dataset_citations.sources.doi import NEMAR_DOI_PREFIX, is_own_dataset_doi

logger = logging.getLogger(__name__)

DATA_PAPER = "data_paper"
# DataCite relations whose anchor IS a record of the dataset itself, not a
# paper about it. `cites_dataset` (core.accession_mentions) buckets their
# citers as citing the dataset.
DATASET_RECORD_RELATIONS = frozenset({"IsIdenticalTo", "IsVersionOf"})

KEPT_OWN_DOI = "own_doi"
KEPT_DATASET_RECORD = "dataset_record"
KEPT_DATA_PAPER = "judged_data_paper"
DROPPED_NEVER_ANCHOR = "never_anchor"
DROPPED_NOT_DATA_PAPER = "judged_not_data_paper"
DROPPED_UNJUDGED = "unjudged"
# Sweep only: the gate would keep this anchor, but its citers were never fetched.
AWAITING_FETCH = "awaiting_fetch"

# `metadata.fetch_status` of a file in which no anchor survives the gate.
NO_DATA_PAPER_ANCHOR = "no_data_paper_anchor"

# A citing work may carry the publication year of a version that predates the
# anchor's version of record by one year (online-first vs issue year, a
# preprint merged into the journal version). More than that is impossible.
PRE_ANCHOR_TOLERANCE_YEARS = 1


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Whether an anchor contributes citations, and why (`reason`, persisted as
    `metadata.anchors[].kept_reason`)."""

    kept: bool
    reason: str


def gate_anchor(
    *,
    dataset_id: str,
    identifier: str,
    identifier_type: str,
    classification: str | None,
    paper_title: str | None,
    source_relation: str | None = None,
) -> GateDecision:
    """Decide one anchor. `classification` is None when there is no successful
    judgment for it; `paper_title` is the resolved title when known;
    `source_relation` is the anchor's DataCite relation."""
    is_doi = identifier_type.lower() == "doi"
    if is_doi and is_own_dataset_doi(identifier, dataset_id):
        return GateDecision(kept=True, reason=KEPT_OWN_DOI)
    blocked = (
        is_never_anchor(identifier, paper_title)
        if is_doi
        else is_spec_title(paper_title)
    )
    if blocked:
        if classification == DATA_PAPER:
            # The judge said data paper and the curated list or the title rule
            # overrode it; say so, since a wrong override is otherwise silent.
            logger.info(
                "%s: never-anchor overrides the judge's data_paper verdict for %s (%s)",
                dataset_id,
                identifier,
                paper_title or "no title",
            )
        return GateDecision(kept=False, reason=DROPPED_NEVER_ANCHOR)
    if classification is None:
        if source_relation in DATASET_RECORD_RELATIONS:
            return GateDecision(kept=True, reason=KEPT_DATASET_RECORD)
        return GateDecision(kept=False, reason=DROPPED_UNJUDGED)
    if classification == DATA_PAPER:
        return GateDecision(kept=True, reason=KEPT_DATA_PAPER)
    return GateDecision(kept=False, reason=DROPPED_NOT_DATA_PAPER)


def gate_against_sidecar(
    *,
    dataset_id: str,
    identifier: str,
    identifier_type: str,
    source_relation: str | None,
    sidecar: JudgmentSidecar,
    known_title: str | None = None,
) -> tuple[GateDecision, dict[str, Any] | None]:
    """Gate one anchor with the verdict a loaded sidecar holds for it.

    Returns the decision and the sidecar's details for the anchor (None when
    it has no successful judgment there). `known_title` is a title recorded
    elsewhere (the citation file), used for the spec-title rule when the
    sidecar has none.
    """
    key = canonical_anchor_key(identifier, identifier_type)
    details = sidecar.context_details.get(key) if key is not None else None
    decision = gate_anchor(
        dataset_id=dataset_id,
        identifier=identifier,
        identifier_type=identifier_type,
        classification=sidecar.lookup.get(key) if key is not None else None,
        paper_title=(details or {}).get("paper_title") or known_title,
        source_relation=source_relation,
    )
    return decision, details


def newly_kept_anchors(
    payload: dict[str, Any], sidecar: JudgmentSidecar, dataset_id: str
) -> list[str]:
    """Recorded anchors the file never fetched through that the gate now keeps.

    A new `data_paper` verdict (or a relabel to an identity relation) makes
    such an anchor qualify, but its citers are not in the file until the
    dataset is refetched; `cli.update` refetches when this is non-empty
    instead of waiting out its freshness window.
    """
    metadata = payload.get("metadata")
    anchors = metadata.get("anchors") if isinstance(metadata, dict) else None
    out: list[str] = []
    for anchor in anchors if isinstance(anchors, list) else []:
        if not isinstance(anchor, dict) or anchor.get("kept") is True:
            continue
        identifier = anchor.get("identifier")
        if not isinstance(identifier, str):
            continue
        decision, _ = gate_against_sidecar(
            dataset_id=dataset_id,
            identifier=identifier,
            identifier_type=anchor.get("identifier_type") or "doi",
            source_relation=anchor.get("source_relation"),
            sidecar=sidecar,
            known_title=anchor.get("paper_title"),
        )
        if decision.kept:
            out.append(identifier)
    return out


def surfaced_by_mention(citation: dict[str, Any]) -> bool:
    """True for records that cite the dataset whatever anchor surfaced them:
    accession mentions, anchor citations that also name the accession, and
    records with no source anchor at all."""
    return (
        citation.get("discovery_method") == "accession_mention"
        or citation.get("mentions_accession") is True
        or not citation.get("source_doi")
    )


def drop_pre_anchor_citations(
    details: list[dict[str, Any]], anchor_years: dict[str, int]
) -> tuple[list[dict[str, Any]], int]:
    """Drop citing works published well before the anchor they supposedly cite.

    A paper cannot cite a paper that did not exist yet, so such a record is a
    bibliographic error (issue #241: PREP, 2015, listed as citing a dataset
    whose data paper is from 2019). One year of slack
    (`PRE_ANCHOR_TOLERANCE_YEARS`) keeps citers of an online-first or preprint
    version. `anchor_years` maps an anchor identifier to its publication year;
    records whose anchor year or own year is unknown are kept, since there is
    nothing to compare, and records surfaced by an accession mention are never
    touched. Returns (kept, dropped_count).
    """
    years = {base_doi(k): v for k, v in anchor_years.items() if v}
    kept: list[dict[str, Any]] = []
    for record in details:
        anchor_year = years.get(base_doi(record.get("source_doi")))
        too_early = predates_anchor(record.get("year"), anchor_year)
        if too_early and not surfaced_by_mention(record):
            continue
        kept.append(record)
    return kept, len(details) - len(kept)


def predates_anchor(year: Any, anchor_year: int | None) -> bool:
    """True when a work from `year` is too old to cite a paper from `anchor_year`.

    False when either year is unknown (0 / None), since there is nothing to
    compare. Allows `PRE_ANCHOR_TOLERANCE_YEARS` of slack.
    """
    return bool(
        anchor_year
        and isinstance(year, int)
        and 0 < year < anchor_year - PRE_ANCHOR_TOLERANCE_YEARS
    )


def drop_self_citations(
    details: list[dict[str, Any]], dataset_id: str
) -> tuple[list[dict[str, Any]], int]:
    """Drop records whose citing work is the dataset's own DOI record.

    OpenAlex indexes the dataset's DataCite record (e.g.
    `10.82901/nemar.on004554.v1.0.0`) as a work that "cites" the papers in its
    related identifiers, so without this the dataset counts itself as a citer
    of its own data paper. Returns (kept, dropped_count).
    """
    own = f"{NEMAR_DOI_PREFIX}nemar.{dataset_id.lower()}"
    kept = [r for r in details if base_doi(r.get("doi")) != own]
    return kept, len(details) - len(kept)
