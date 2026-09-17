"""Batch CLI for LLM-based anchor adjudication.

Reads a dataset-IDs file, builds (or refreshes) a per-dataset judgment
sidecar at `<output-dir>/<dataset_id>.json`. Phase 2 of epic #76.

The CLI is the entry point the phase-4 cron pipeline will call; phase 3
reads the sidecars to filter umbrella / methodology / irrelevant anchors
out of the citation pipeline.

Copyright (c) 2026 Seyed Yahya Shirazi (neuromechanist)
All rights reserved.

Author: Seyed Yahya Shirazi
GitHub: https://github.com/neuromechanist
Email: shirazi@ieee.org
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from dataset_citations.backends import OpenCiteBackend
from dataset_citations.quality.anchor_judgment import (
    is_judgment_fresh,
    judge_dataset_anchors,
    load_judgment_sidecar,
    save_judgment_sidecar,
)
from dataset_citations.quality.dataset_metadata import DatasetMetadataRetriever
from dataset_citations.quality.llm_client import OllamaJudgmentClient
from dataset_citations.sources import BidsMetadataSource, NemarMetadataSource

logger = logging.getLogger(__name__)


def _read_dataset_ids(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def _sidecar_path(output_dir: str, dataset_id: str) -> Path:
    return Path(output_dir) / f"{dataset_id}.json"


def _judged_anchor_keys(sidecar: Path) -> set[str]:
    """Anchor identifiers this sidecar already carries a judgment for."""
    try:
        payload = load_judgment_sidecar(sidecar)
    except (OSError, ValueError, TypeError):
        return set()
    keys: set[str] = set()
    for judgment in payload.get("judgments") or []:
        identifier = judgment.get("anchor_identifier")
        if isinstance(identifier, str) and identifier.strip():
            keys.add(identifier.strip().casefold())
    return keys


def _recorded_anchor_keys(citations_dir: str | None, dataset_id: str) -> set[str]:
    """Anchor identifiers the last pipeline run recorded for this dataset.

    Read from the committed citation JSON's `metadata.anchors[]`, which lists
    EVERY anchor the pipeline resolved. Using it here keeps the coverage check
    free: no GitHub or data.nemar.org round trip just to decide whether to skip.
    Returns an empty set when there is nothing to compare against, which makes
    the caller fall back to the old file-exists behavior.
    """
    if not citations_dir:
        return set()
    path = Path(citations_dir) / f"{dataset_id}_citations.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return set()
    keys: set[str] = set()
    for anchor in metadata.get("anchors") or []:
        identifier = anchor.get("identifier") if isinstance(anchor, dict) else None
        if isinstance(identifier, str) and identifier.strip():
            keys.add(identifier.strip().casefold())
    return keys


def _should_skip(
    sidecar: Path,
    *,
    skip_existing: bool,
    max_age_days: int,
    citations_dir: str | None = None,
    dataset_id: str | None = None,
) -> tuple[bool, str]:
    """Return (skip, reason). Reason is for logging only.

    `--skip-existing` used to skip on the mere existence of the sidecar, which
    froze a dataset's judgments at whatever its anchor set was the first time
    it ran. Anchors added afterwards were never judged, so they fell through
    the pipeline's "fetch all when unjudged" path and pulled in the citers of
    BIDS/methods papers. That is how 52% of recorded anchors ended up with a
    null classification while every dataset still had a sidecar (#180).

    So an existing sidecar is only a reason to skip when it actually covers
    every anchor the last pipeline run recorded.
    """
    if not sidecar.exists():
        return False, ""
    if skip_existing:
        if dataset_id:
            recorded = _recorded_anchor_keys(citations_dir, dataset_id)
            missing = recorded - _judged_anchor_keys(sidecar)
            if missing:
                return False, ""
        return True, "exists"
    if max_age_days > 0:
        try:
            payload = load_judgment_sidecar(sidecar)
        except (OSError, ValueError, TypeError):
            # TypeError covers a sidecar whose JSON root is not an object;
            # ValueError still covers JSONDecodeError. A malformed sidecar must
            # fall through to a re-judge, never crash the cron's judge step.
            return False, ""
        if is_judgment_fresh(payload, max_age_days=max_age_days):
            return True, f"fresh<={max_age_days}d"
    return False, ""


def _summarize_judgments(payload: dict) -> str:
    """One-line summary of the per-dataset run for the operator log.

    Counts judgments by classification and errors; the cron run lives or
    dies on these numbers so they are surfaced for every dataset.
    """
    judgments = payload.get("judgments") or []
    total = len(judgments)
    errors = sum(1 for j in judgments if j.get("error"))
    by_class: dict[str, int] = {}
    for j in judgments:
        cls = j.get("classification") or ""
        if cls:
            by_class[cls] = by_class.get(cls, 0) + 1
    class_summary = ", ".join(f"{k}={v}" for k, v in sorted(by_class.items()))
    return f"{total} anchors judged ({errors} errors)" + (
        f"; {class_summary}" if class_summary else ""
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "LLM-judge DOI anchors per dataset and write a sidecar JSON to "
            "<output-dir>/<dataset_id>.json. Phase 2 of epic #76."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset-list-file",
        required=True,
        help="Path to a newline-separated list of dataset IDs to judge.",
    )
    parser.add_argument(
        "--output-dir",
        default="citations/anchor_judgments",
        help="Directory to write sidecar JSON files (default: citations/anchor_judgments).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip datasets whose sidecar already exists (regardless of age).",
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=0,
        help=(
            "Skip datasets whose sidecar's judged_at is within this many days. "
            "Set to 0 (default) to re-judge every dataset; --skip-existing "
            "takes precedence when both are set."
        ),
    )
    parser.add_argument(
        "--ollama-base-url",
        default=None,
        help="Override OLLAMA_BASE_URL env var.",
    )
    parser.add_argument(
        "--ollama-model",
        default=None,
        help="Override OLLAMA_MODEL env var.",
    )
    parser.add_argument(
        "--github-token",
        default=None,
        help=(
            "GitHub token for metadata + source lookups. Falls back to the "
            "GITHUB_TOKEN env var; public-API limits apply when neither is set."
        ),
    )
    parser.add_argument(
        "--citations-dir",
        default="citations/json_opencite",
        help=(
            "Citation JSON directory, read only to decide whether an existing "
            "sidecar still covers the dataset's anchors. With --skip-existing, "
            "a dataset whose citation JSON records an anchor the sidecar has no "
            "judgment for is re-judged instead of skipped (default: "
            "citations/json_opencite)."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    dataset_ids = _read_dataset_ids(args.dataset_list_file)
    if not dataset_ids:
        logger.error("dataset list file %s is empty", args.dataset_list_file)
        return 1

    github_token = args.github_token or os.getenv("GITHUB_TOKEN")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    nemar_source = NemarMetadataSource(github_token=github_token)
    bids_source = BidsMetadataSource(github_token=github_token)
    metadata_retriever = DatasetMetadataRetriever(github_token=github_token)
    backend = OpenCiteBackend(max_results_per_doi=1)

    # Health-check the Ollama daemon BEFORE any per-dataset work so an
    # unreachable GPU host fails fast (exit 2). Without this the loop would
    # write a sidecar per dataset where every judgment is an error, which
    # phase 3 would have to special-case.
    with OllamaJudgmentClient(
        base_url=args.ollama_base_url, model=args.ollama_model
    ) as client:
        if not client.health_check():
            logger.error(
                "ollama daemon at %s is not reachable; aborting before any judgments",
                client.base_url,
            )
            return 2

        logger.info(
            "judging anchors for %d dataset(s) -> %s (model=%s, base_url=%s)",
            len(dataset_ids),
            args.output_dir,
            client.model,
            client.base_url,
        )

        skipped = 0
        judged = 0
        write_failures = 0
        for dataset_id in dataset_ids:
            sidecar = _sidecar_path(args.output_dir, dataset_id)
            skip, reason = _should_skip(
                sidecar,
                skip_existing=args.skip_existing,
                max_age_days=args.max_age_days,
                citations_dir=getattr(args, "citations_dir", None),
                dataset_id=dataset_id,
            )
            if skip:
                logger.info("%s: skipping (%s)", dataset_id, reason)
                skipped += 1
                continue

            payload = judge_dataset_anchors(
                dataset_id,
                nemar_source=nemar_source,
                bids_source=bids_source,
                metadata_retriever=metadata_retriever,
                backend=backend,
                client=client,
            )
            try:
                save_judgment_sidecar(sidecar, payload)
            except OSError as exc:
                logger.error("failed to write %s: %s", sidecar, exc)
                write_failures += 1
                continue
            judged += 1
            logger.info("%s: %s", dataset_id, _summarize_judgments(payload))

    logger.info(
        "anchor judgment run complete: %d judged, %d skipped, %d write failures, %d total",
        judged,
        skipped,
        write_failures,
        len(dataset_ids),
    )
    if write_failures and judged == 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
