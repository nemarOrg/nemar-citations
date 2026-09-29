"""Re-apply the anchor gate to citation JSON files already on disk.

Issue #241. The fetch path (`core.opencite_pipeline`) gates anchors when it
writes a file, but `update` refetches a dataset only once its 7-day freshness
window lapses, and a file written before the gate existed (or before its
anchors were re-judged) can hold citations pulled in through anchors that are
not the dataset's data paper. This sweep brings every file in line with the
current judgments without touching the network:

1. Rebuild `metadata.anchors[]` from the sidecars in `citations/anchor_judgments/`
   through `core.anchor_gate.gate_anchor` (the exact rules the fetch path uses).
2. Drop citations whose source anchor is not kept. Accession mentions and
   citations that also name the dataset accession are never dropped: they cite
   the dataset regardless of which anchor surfaced them.
3. Drop citing works published before their anchor.
4. Recompute every derived count.

A citing work that cited both a dropped anchor and a kept one may have been
recorded under the dropped anchor (the fetch dedupes first-seen), so the sweep
can undercount until that dataset's next refetch restores it. That is the
conservative direction and the intended trade.

Idempotent: a second run is a no-op, so steady-state nights add no git churn.
When a file loses citations its stale `confidence_scoring` block is dropped
(the score step then re-scores it) and `date_last_updated` advances, mirroring
`dedupe_citations`.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dataset_citations.core.accession_mentions import refresh_derived_counts
from dataset_citations.core.anchor_gate import drop_pre_anchor_citations, gate_anchor
from dataset_citations.core.citation_identity import base_doi
from dataset_citations.quality.anchor_judgment_io import (
    DEFAULT_JUDGMENTS_DIR,
    canonical_anchor_key,
    load_judgment_sidecar,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# fetch_status the pipeline writes when no anchor survives the gate.
_NO_DATA_PAPER = "no_data_paper_anchor"


class CitationFileUnreadable(Exception):
    """A citation JSON could not be read or parsed."""


@dataclass(frozen=True, slots=True)
class GateOutcome:
    """What the sweep did to one file."""

    changed: bool
    dropped: int


def _surfaced_by_mention(citation: dict[str, Any]) -> bool:
    """True for records that cite the dataset whatever anchor surfaced them."""
    return (
        citation.get("discovery_method") == "accession_mention"
        or citation.get("mentions_accession") is True
        or not citation.get("source_doi")
    )


def gate_citation_file(
    path: Path,
    *,
    judgments_dir: Path | str = DEFAULT_JUDGMENTS_DIR,
    dry_run: bool = False,
    when: datetime | None = None,
) -> GateOutcome:
    """Re-gate one citation file in place. Raises `CitationFileUnreadable`."""
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("%s: unreadable (%s); skipping", path.name, exc)
        raise CitationFileUnreadable(path.name) from exc

    metadata = payload.get("metadata")
    anchors = metadata.get("anchors") if isinstance(metadata, dict) else None
    if not isinstance(metadata, dict) or not isinstance(anchors, list) or not anchors:
        return GateOutcome(changed=False, dropped=0)

    dataset_id = str(payload.get("dataset_id") or path.name.split("_")[0])
    before = json.dumps(payload, sort_keys=True)
    sidecar = load_judgment_sidecar(dataset_id, judgments_dir=judgments_dir)

    kept_ids: set[str] = set()
    anchor_years: dict[str, int] = {}
    for anchor in anchors:
        identifier = anchor.get("identifier")
        identifier_type = anchor.get("identifier_type") or "doi"
        if not isinstance(identifier, str):
            continue
        key = canonical_anchor_key(identifier, identifier_type)
        details = sidecar.context_details.get(key) if key is not None else None
        classification = sidecar.lookup.get(key) if key is not None else None
        decision = gate_anchor(
            dataset_id=dataset_id,
            identifier=identifier,
            identifier_type=identifier_type,
            classification=classification,
            paper_title=(details or {}).get("paper_title") or anchor.get("paper_title"),
        )
        anchor["classification"] = classification
        anchor["kept"] = decision.kept
        anchor["kept_reason"] = decision.reason
        for field in ("paper_title", "paper_year", "paper_venue", "reason"):
            anchor[field] = details.get(field) if details else None
        anchor["judgment_model"] = sidecar.model if sidecar.present else None
        if decision.kept:
            kept_ids.add(base_doi(identifier))
            year = anchor["paper_year"]
            if isinstance(year, int) and year > 0:
                anchor_years[identifier] = year

    details_in = payload.get("citation_details") or []
    kept = [
        c
        for c in details_in
        if _surfaced_by_mention(c) or base_doi(c.get("source_doi")) in kept_ids
    ]
    kept, _ = drop_pre_anchor_citations(kept, anchor_years)
    dropped = len(details_in) - len(kept)

    kept_anchors = [a for a in anchors if a.get("kept")]
    metadata["anchor_count"] = len(kept_anchors)
    metadata["searched_dois"] = [
        a["identifier"]
        for a in kept_anchors
        if (a.get("identifier_type") or "doi").lower() == "doi"
    ]
    metadata["anchor_judgment_model"] = sidecar.model if sidecar.present else None
    if not kept_anchors and metadata.get("fetch_status") in ("success", "partial"):
        metadata["fetch_status"] = _NO_DATA_PAPER
    payload["citation_details"] = kept
    refresh_derived_counts(payload)
    if dropped:
        # Scores referred to the pre-gate list; the score step recomputes.
        payload.pop("confidence_scoring", None)
        payload["date_last_updated"] = (when or datetime.now(UTC)).isoformat()

    changed = json.dumps(payload, sort_keys=True) != before
    if changed and not dry_run:
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    if dropped:
        logger.info(
            "%s: dropped %d citation(s) through non-data-paper anchors",
            path.name,
            dropped,
        )
    return GateOutcome(changed=changed, dropped=dropped)


def run_gate(args: argparse.Namespace) -> None:
    citations_dir = Path(args.citations_dir)
    if not citations_dir.is_dir():
        logger.error("citations-dir %s is not a directory; aborting", citations_dir)
        raise SystemExit(1)

    files = sorted(citations_dir.glob("*_citations.json"))
    touched = total_dropped = unreadable = 0
    for path in files:
        try:
            outcome = gate_citation_file(
                path, judgments_dir=args.judgments_dir, dry_run=args.dry_run
            )
        except CitationFileUnreadable:
            unreadable += 1
            continue
        touched += int(outcome.changed)
        total_dropped += outcome.dropped

    verb = "would update" if args.dry_run else "updated"
    logger.info(
        "anchor gate complete: %s %d of %d file(s), %d citation(s) dropped, "
        "%d unreadable",
        verb,
        touched,
        len(files),
        total_dropped,
        unreadable,
    )
    if unreadable:
        # Non-zero so the cron's guard fires on a corrupt file instead of
        # publishing it ungated.
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-apply the fail-closed anchor gate to citation JSON files using "
            "the current anchor judgments (offline)."
        )
    )
    parser.add_argument(
        "--citations-dir",
        default="citations/json_opencite",
        help="Directory of *_citations.json files (default: citations/json_opencite)",
    )
    parser.add_argument(
        "--judgments-dir",
        default=str(DEFAULT_JUDGMENTS_DIR),
        help=f"Anchor-judgment sidecars (default: {DEFAULT_JUDGMENTS_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing any file.",
    )
    run_gate(parser.parse_args())


if __name__ == "__main__":
    main()
