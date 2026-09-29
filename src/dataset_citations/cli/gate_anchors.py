"""Re-apply the anchor gate to citation JSON files already on disk.

Issue #241. The fetch path (`core.opencite_pipeline`) gates anchors when it
writes a file, but `update` refetches a dataset only once its 7-day freshness
window lapses, and a file written before the gate existed (or before its
anchors were re-judged) can hold citations pulled in through anchors that are
not the dataset's data paper. This sweep brings every file in line with the
current judgments without touching the network:

1. Re-derive each entry of `metadata.anchors[]` (classification, kept,
   kept_reason, judge reason) from the trusted judge's sidecar in
   `citations/anchor_judgments/` through `core.anchor_gate` (the exact rules
   the fetch path uses). A file written before schema 2.1 has no `anchors[]`;
   its anchors are rebuilt from its citations' `source_doi` first, so it is
   gated like every other file instead of skipped.
2. Drop citations whose source anchor is not kept. Accession mentions and
   citations that also name the dataset accession are never dropped: they cite
   the dataset regardless of which anchor surfaced them.
3. Drop citing works published before their anchor (mentions exempt again),
   and citing works that are NEMAR or OpenNeuro dataset records.
4. Recompute every derived count.

The sweep only ever removes. An anchor the file did not fetch through that now
qualifies (a new `data_paper` verdict) stays `kept: false` with
`kept_reason: "awaiting_fetch"`, so `kept` and `searched_dois` keep meaning
"fetched"; `update` refetches such a dataset on its next run.

A citing work that cited both a dropped anchor and a kept one may have been
recorded under the dropped anchor (the fetch dedupes first-seen), so the sweep
can undercount until that dataset's next refetch restores it. That is the
conservative direction and the intended trade.

Nothing is written unless the whole corpus checks out: an unreadable citation
file or sidecar, a missing judgments directory, or most datasets lacking a
trusted sidecar (the judge did not run, or ran as another model) exits 1 before
any file is touched, because gating against missing verdicts would wipe every
dataset's citations.

Idempotent: a second run is a no-op, so steady-state nights add no git churn.
When a file loses citations its stale `confidence_scoring` block is dropped
(the score step then re-scores it) and `date_last_updated` advances, mirroring
`dedupe_citations`.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dataset_citations.core.accession_mentions import refresh_derived_counts
from dataset_citations.core.anchor_gate import (
    AWAITING_FETCH,
    NO_DATA_PAPER_ANCHOR,
    drop_dataset_record_citers,
    drop_pre_anchor_citations,
    gate_against_sidecar,
    surfaced_by_mention,
)
from dataset_citations.core.citation_identity import base_doi
from dataset_citations.core.citation_utils import write_json_atomic
from dataset_citations.quality.anchor_judgment_io import (
    DEFAULT_JUDGMENTS_DIR,
    SIDECAR_OK,
    SIDECAR_UNREADABLE,
    load_judgment_sidecar,
)
from dataset_citations.quality.llm_client import trusted_judge_model
from dataset_citations.sources.doi import is_own_dataset_doi, normalize_doi

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Abort when more than this share of the files that need a judgment have no
# trusted sidecar. On a normal night the judge has just written one for every
# discovered dataset, so only a handful of legacy files lack one; a majority
# means a wrong --judgments-dir or a judge that did not run.
_DEFAULT_MAX_MISSING_SHARE = 0.5


class CitationFileUnreadable(Exception):
    """A citation JSON could not be read or parsed."""


@dataclass(frozen=True, slots=True)
class GateOutcome:
    """What the sweep did (or, in a dry run, would do) to one file.

    `dropped` counts all removed citations; `dropped_unkept`, `dropped_early`,
    and `dropped_records` split it by cause. `sidecar_status` is the trusted
    sidecar's `JudgmentSidecar.status`; `needs_judgment` is True when the file
    has an anchor that only a judgment can keep (anything but the dataset's
    own DOI). A file whose sidecar is unreadable is left untouched.
    """

    changed: bool
    dropped: int
    dropped_unkept: int
    dropped_early: int
    dropped_records: int
    sidecar_status: str
    needs_judgment: bool


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("%s: unreadable (%s)", path.name, exc)
        raise CitationFileUnreadable(path.name) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        logger.error("%s: not a citation JSON object with metadata", path.name)
        raise CitationFileUnreadable(path.name)
    return payload


def _derived_anchors(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Anchors rebuilt from citation records, for a file with no `anchors[]`.

    Every non-mention record names the anchor that surfaced it in `source_doi`,
    so those anchors were fetched through (`kept: true` as recorded).
    """
    anchors: dict[str, dict[str, Any]] = {}
    for citation in payload.get("citation_details") or []:
        if not isinstance(citation, dict) or surfaced_by_mention(citation):
            continue
        identifier = normalize_doi(str(citation["source_doi"]))
        anchors.setdefault(
            identifier,
            {
                "identifier": identifier,
                "identifier_type": "doi",
                "source_relation": citation.get("source_relation"),
                "kept": True,
            },
        )
    return list(anchors.values())


def _needs_judgment(anchors: list[dict[str, Any]], dataset_id: str) -> bool:
    return any(
        not (
            (a.get("identifier_type") or "doi").lower() == "doi"
            and is_own_dataset_doi(str(a.get("identifier") or ""), dataset_id)
        )
        for a in anchors
    )


def gate_citation_file(
    path: Path,
    *,
    judgments_dir: Path | str = DEFAULT_JUDGMENTS_DIR,
    judge_model: str | None = None,
    dry_run: bool = False,
    when: datetime | None = None,
    quiet: bool = False,
) -> GateOutcome:
    """Re-gate one citation file in place. Raises `CitationFileUnreadable`.

    Only a sidecar judged by `judge_model` (default: `trusted_judge_model()`)
    counts; any other sidecar gates like a missing one. `quiet` silences the
    per-file drop log (the planning pass).
    """
    payload = _load_payload(path)
    metadata: dict[str, Any] = payload["metadata"]
    dataset_id = str(payload.get("dataset_id") or path.name.split("_")[0])

    recorded = metadata.get("anchors")
    if isinstance(recorded, list) and recorded:
        anchors = [a for a in recorded if isinstance(a, dict)]
    else:
        anchors = _derived_anchors(payload)
    needs_judgment = _needs_judgment(anchors, dataset_id)
    sidecar = load_judgment_sidecar(
        dataset_id,
        judgments_dir=judgments_dir,
        expected_model=judge_model or trusted_judge_model(),
    )
    if sidecar.status == SIDECAR_UNREADABLE:
        # Gating against a corrupt sidecar would read as "nothing judged" and
        # wipe the dataset; leave it for an operator (the caller exits 1).
        logger.error("%s: anchor-judgment sidecar is unreadable", path.name)
        return GateOutcome(False, 0, 0, 0, 0, sidecar.status, needs_judgment)
    if not anchors:
        return GateOutcome(False, 0, 0, 0, 0, sidecar.status, needs_judgment)

    before = copy.deepcopy(payload)
    model = sidecar.model if sidecar.present else None
    kept_ids: set[str] = set()
    anchor_years: dict[str, int] = {}
    for anchor in anchors:
        identifier = anchor.get("identifier")
        if not isinstance(identifier, str) or not identifier.strip():
            continue
        decision, details = gate_against_sidecar(
            dataset_id=dataset_id,
            identifier=identifier,
            identifier_type=anchor.get("identifier_type") or "doi",
            source_relation=anchor.get("source_relation"),
            sidecar=sidecar,
            known_title=anchor.get("paper_title"),
        )
        # Only remove: an anchor never fetched through cannot start counting
        # here, since its citers are not in the file.
        fetched = anchor.get("kept") is True
        anchor["classification"] = details.get("classification") if details else None
        anchor["kept"] = decision.kept and fetched
        anchor["kept_reason"] = (
            AWAITING_FETCH if decision.kept and not fetched else decision.reason
        )
        # Paper facts come from the resolved record and stay valid without a
        # trusted verdict; the judge's reasoning does not.
        for field in ("paper_title", "paper_year", "paper_venue"):
            if details:
                anchor[field] = details.get(field)
            else:
                anchor.setdefault(field, None)
        anchor["reason"] = details.get("reason") if details else None
        anchor["judgment_model"] = model
        if anchor["kept"]:
            kept_ids.add(base_doi(identifier))
            year = anchor["paper_year"]
            if isinstance(year, int) and year > 0:
                anchor_years[identifier] = year

    details_in = [
        c for c in payload.get("citation_details") or [] if isinstance(c, dict)
    ]
    kept = [
        c
        for c in details_in
        if surfaced_by_mention(c) or base_doi(c.get("source_doi")) in kept_ids
    ]
    dropped_unkept = len(details_in) - len(kept)
    kept, dropped_early = drop_pre_anchor_citations(kept, anchor_years)
    kept, dropped_records = drop_dataset_record_citers(kept)
    dropped = dropped_unkept + dropped_early + dropped_records

    kept_anchors = [a for a in anchors if a.get("kept")]
    metadata["anchors"] = anchors
    metadata["anchor_count"] = len(kept_anchors)
    metadata["searched_dois"] = [
        a["identifier"]
        for a in kept_anchors
        if (a.get("identifier_type") or "doi").lower() == "doi"
    ]
    metadata["anchor_judgment_model"] = model
    if not kept_anchors and metadata.get("fetch_status") in ("success", "partial"):
        metadata["fetch_status"] = NO_DATA_PAPER_ANCHOR
    payload["citation_details"] = kept
    refresh_derived_counts(payload)
    if dropped:
        # Scores referred to the pre-gate list; the score step recomputes.
        payload.pop("confidence_scoring", None)
        payload["date_last_updated"] = (when or datetime.now(UTC)).isoformat()

    changed = payload != before
    if changed and not dry_run:
        write_json_atomic(path, payload)
    if dropped and not quiet:
        logger.info(
            "%s: %s %d citation(s): %d through anchors that are not kept, %d "
            "older than their anchor, %d dataset record(s)",
            path.name,
            "would drop" if dry_run else "dropped",
            dropped,
            dropped_unkept,
            dropped_early,
            dropped_records,
        )
    return GateOutcome(
        changed=changed,
        dropped=dropped,
        dropped_unkept=dropped_unkept,
        dropped_early=dropped_early,
        dropped_records=dropped_records,
        sidecar_status=sidecar.status,
        needs_judgment=needs_judgment,
    )


def run_gate(args: argparse.Namespace) -> None:
    """Gate every citation file; exit 1 (writing nothing) if the corpus is not safe to gate."""
    citations_dir = Path(args.citations_dir)
    if not citations_dir.is_dir():
        logger.error("citations-dir %s is not a directory; aborting", citations_dir)
        raise SystemExit(1)
    judgments_dir = Path(args.judgments_dir)
    if not judgments_dir.is_dir() or not any(judgments_dir.glob("*.json")):
        # Every anchor would read as unjudged and every dataset would lose its
        # citations; a typo or a wrong working directory must not do that.
        logger.error(
            "judgments-dir %s is missing or holds no sidecars; aborting",
            judgments_dir,
        )
        raise SystemExit(1)
    judge_model = args.judge_model or trusted_judge_model()
    files = sorted(citations_dir.glob("*_citations.json"))

    # Pass 1 plans every file without writing, so nothing is touched unless
    # the whole corpus can be gated.
    unreadable = bad_sidecars = needing = missing = 0
    for path in files:
        try:
            outcome = gate_citation_file(
                path,
                judgments_dir=judgments_dir,
                judge_model=judge_model,
                dry_run=True,
                quiet=True,
            )
        except CitationFileUnreadable:
            unreadable += 1
            continue
        if outcome.sidecar_status == SIDECAR_UNREADABLE:
            bad_sidecars += 1
        elif outcome.needs_judgment:
            needing += 1
            if outcome.sidecar_status != SIDECAR_OK:
                missing += 1
                logger.warning(
                    "%s: no sidecar judged by %s (%s); its anchors gate as unjudged",
                    path.name,
                    judge_model,
                    outcome.sidecar_status,
                )
    if unreadable or bad_sidecars:
        logger.error(
            "anchor gate aborted: %d unreadable citation file(s), %d unreadable "
            "sidecar(s); no file was changed",
            unreadable,
            bad_sidecars,
        )
        raise SystemExit(1)
    if needing and missing > args.max_missing_share * needing:
        logger.error(
            "anchor gate aborted: %d of %d dataset(s) with anchors to judge have "
            "no sidecar from %s (limit %.0f%%); did the judge run, and is "
            "--judgments-dir right? No file was changed.",
            missing,
            needing,
            judge_model,
            100 * args.max_missing_share,
        )
        raise SystemExit(1)

    touched = total_dropped = 0
    for path in files:
        outcome = gate_citation_file(
            path,
            judgments_dir=judgments_dir,
            judge_model=judge_model,
            dry_run=args.dry_run,
        )
        touched += int(outcome.changed)
        total_dropped += outcome.dropped
    logger.info(
        "anchor gate complete: %s %d of %d file(s), %d citation(s) dropped, "
        "%d of %d dataset(s) with anchors to judge had no trusted sidecar",
        "would update" if args.dry_run else "updated",
        touched,
        len(files),
        total_dropped,
        missing,
        needing,
    )


def build_parser() -> argparse.ArgumentParser:
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
        "--judge-model",
        default=None,
        help=(
            "Only sidecars judged by this model count (default: "
            "ANCHOR_JUDGE_MODEL, else claude-sonnet-5-5)."
        ),
    )
    parser.add_argument(
        "--max-missing-share",
        type=float,
        default=_DEFAULT_MAX_MISSING_SHARE,
        help=(
            "Abort without writing when more than this share of the datasets "
            "with anchors to judge have no trusted sidecar (default: "
            f"{_DEFAULT_MAX_MISSING_SHARE})."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing any file.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    run_gate(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
