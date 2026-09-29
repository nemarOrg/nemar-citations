"""The anchor gate: which anchors may contribute citations to a dataset.

Issue #241. The gate FAILS CLOSED. An anchor contributes citations only when:

* it is the dataset's own NEMAR concept DOI (`10.82901/nemar.<id>`): citing it
  is citing the dataset, no judgment needed; or
* a successful anchor judgment classified it as the dataset's `data_paper`,
  and it is not a standards / software / platform paper (`never_anchor`).

Everything else is context only: `References` papers the dataset merely cites,
anchors the judge called umbrella / methodology / related_work / irrelevant,
and anchors with no successful judgment (judge down, lookup failed, or never
run). The previous fallback fetched unjudged anchors, which is how a dead judge
turned every related-work paper's citers into "citations" of the dataset.

The same functions back the fetch path (`core.opencite_pipeline`) and the
offline sweep that re-applies the gate to files already on disk
(`cli.gate_anchors`), so the two can never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dataset_citations.core.citation_identity import base_doi
from dataset_citations.quality.never_anchor import is_never_anchor, is_spec_title
from dataset_citations.sources.doi import is_own_dataset_doi

DATA_PAPER = "data_paper"

KEPT_OWN_DOI = "own_doi"
KEPT_DATA_PAPER = "judged_data_paper"
DROPPED_NEVER_ANCHOR = "never_anchor"
DROPPED_NOT_DATA_PAPER = "judged_not_data_paper"
DROPPED_UNJUDGED = "unjudged"


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Whether an anchor contributes citations, and why (`kept_reason`)."""

    kept: bool
    reason: str


def gate_anchor(
    *,
    dataset_id: str,
    identifier: str,
    identifier_type: str,
    classification: str | None,
    paper_title: str | None,
) -> GateDecision:
    """Decide one anchor. `classification` is None when there is no successful
    judgment for it; `paper_title` is the resolved title when known."""
    is_doi = identifier_type.lower() == "doi"
    if is_doi and is_own_dataset_doi(identifier, dataset_id):
        return GateDecision(kept=True, reason=KEPT_OWN_DOI)
    blocked = (
        is_never_anchor(identifier, paper_title)
        if is_doi
        else is_spec_title(paper_title)
    )
    if blocked:
        return GateDecision(kept=False, reason=DROPPED_NEVER_ANCHOR)
    if classification is None:
        return GateDecision(kept=False, reason=DROPPED_UNJUDGED)
    if classification == DATA_PAPER:
        return GateDecision(kept=True, reason=KEPT_DATA_PAPER)
    return GateDecision(kept=False, reason=DROPPED_NOT_DATA_PAPER)


def drop_pre_anchor_citations(
    details: list[dict[str, Any]], anchor_years: dict[str, int]
) -> tuple[list[dict[str, Any]], int]:
    """Drop citing works published before the anchor they supposedly cite.

    A paper cannot cite a paper that did not exist yet, so such a record is a
    bibliographic error (issue #241: PREP, 2015, listed as citing a dataset
    whose data paper is from 2019). `anchor_years` maps an anchor identifier to
    its publication year; records whose anchor year or own year is unknown are
    kept, since there is nothing to compare. Records with no `source_doi`
    (accession mentions) are never touched. Returns (kept, dropped_count).
    """
    years = {base_doi(k): v for k, v in anchor_years.items() if v}
    kept: list[dict[str, Any]] = []
    for record in details:
        anchor_year = years.get(base_doi(record.get("source_doi")))
        year = record.get("year")
        if anchor_year and isinstance(year, int) and 0 < year < anchor_year:
            continue
        kept.append(record)
    return kept, len(details) - len(kept)
