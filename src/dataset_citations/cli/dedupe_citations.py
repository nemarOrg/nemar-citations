"""Collapse duplicate citing works inside existing citation JSON files.

The fetch path (`core.opencite_pipeline`) dedupes what it writes, but a file
produced before issue #216 can still hold several records for one paper:

- the same DOI twice, because the old key paired DOI with the raw title and the
  sources punctuate titles differently;
- a concept DOI beside its versioned form (`10.82901/nemar.on004842` and
  `...on004842.v1.0.0`);
- a preprint beside its version of record (`10.31219/osf.io/pu5vb` and
  `10.1016/j.neuroimage.2022.119623`).

This CLI rewrites those files in place using the same `dedupe_citations` rules
the pipeline applies, so the corpus converges without waiting for every dataset
to fall out of its freshness window. Idempotent: a second run is a no-op.

`num_citations` and `metadata.total_cumulative_citations` are recomputed from
the surviving records. When a file loses records its stale `confidence_scoring`
block is dropped, mirroring `merge_accession_mentions`, so the next
`score-confidence --skip-existing` re-scores it instead of keeping scores that
refer to citations that no longer exist.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dataset_citations.core.accession_mentions import cites_dataset
from dataset_citations.core.citation_identity import dedupe_citations

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CitationFileUnreadable(Exception):
    """A citation JSON could not be read or parsed."""


def dedupe_citation_file(
    path: Path, *, dry_run: bool = False, when: datetime | None = None
) -> int:
    """Rewrite `path` with duplicates merged. Returns the number dropped.

    Returns 0 and leaves the file untouched when nothing is duplicated, which
    keeps the git diff empty on a steady-state run. Raises
    `CitationFileUnreadable` when the file cannot be read or parsed, so the
    caller can count and report that separately from "nothing to merge".

    Merging duplicates changes citation content as much as a fresh fetch does,
    so `date_last_updated` advances to `when` (default now) whenever a
    duplicate is actually dropped; `date_last_updated` means "last content
    change" (issue #165), and this step was previously the one write path that
    silently left it stale (issue #229).
    """
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("%s: unreadable (%s); skipping", path.name, exc)
        # Raise rather than return 0: a corrupt file and a clean file with no
        # duplicates are not the same outcome, and the caller has to be able
        # to tell them apart to report and exit on it.
        raise CitationFileUnreadable(path.name) from exc

    details = payload.get("citation_details")
    if not isinstance(details, list) or not details:
        return 0

    kept, dropped = dedupe_citations(details)
    if not dropped:
        return 0

    payload["citation_details"] = kept
    payload["num_citations"] = len(kept)
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        metadata["total_cumulative_citations"] = sum(
            int(c.get("cited_by") or 0) for c in kept
        )
        if "num_accession_mentions" in metadata:
            metadata["num_accession_mentions"] = sum(
                1 for c in kept if c.get("discovery_method") == "accession_mention"
            )
        # The two toggle-bucket counters must be re-derived from the surviving
        # records too. AGENTS.md says the dashboard keys on these rather than
        # recomputing them, so leaving them stale after a merge publishes
        # buckets that no longer sum to num_citations. Recompute only when the
        # keys already exist, so a file that never went through find-mentions
        # does not acquire them here.
        if "num_dataset_citations" in metadata:
            metadata["num_dataset_citations"] = sum(1 for c in kept if cites_dataset(c))
        if "num_datapaper_citations" in metadata:
            metadata["num_datapaper_citations"] = sum(
                1 for c in kept if not cites_dataset(c)
            )
    # Scores were computed against the pre-merge citation list; drop them so the
    # scoring step recomputes rather than trusting a stale block.
    payload.pop("confidence_scoring", None)
    payload["date_last_updated"] = (when or datetime.now(UTC)).isoformat()

    if not dry_run:
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    logger.info("%s: merged %d duplicate(s)", path.name, dropped)
    return dropped


def run_dedupe(args: argparse.Namespace) -> None:
    citations_dir = Path(args.citations_dir)
    if not citations_dir.is_dir():
        logger.error("citations-dir %s is not a directory; aborting", citations_dir)
        raise SystemExit(1)

    files = sorted(citations_dir.glob("*_citations.json"))
    total_dropped = 0
    touched = 0
    unreadable = 0
    for path in files:
        try:
            dropped = dedupe_citation_file(path, dry_run=args.dry_run)
        except CitationFileUnreadable:
            unreadable += 1
            continue
        if dropped:
            touched += 1
            total_dropped += dropped

    verb = "would merge" if args.dry_run else "merged"
    logger.info(
        "dedupe complete: %s %d duplicate record(s) across %d of %d file(s), "
        "%d unreadable",
        verb,
        total_dropped,
        touched,
        len(files),
        unreadable,
    )
    if unreadable:
        # Exit non-zero so the cron's `|| echo WARN` guard actually fires. A
        # corrupt citation JSON would otherwise be skipped forever with no
        # signal beyond one ERROR line in a multi-thousand-line nightly log.
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge duplicate citing works inside citation JSON files."
    )
    parser.add_argument(
        "--citations-dir",
        default="citations/json_opencite",
        help="Directory of *_citations.json files (default: citations/json_opencite)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be merged without writing any file.",
    )
    run_dedupe(parser.parse_args())


if __name__ == "__main__":
    main()
