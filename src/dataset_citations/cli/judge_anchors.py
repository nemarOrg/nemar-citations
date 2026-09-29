"""Batch CLI for LLM-based anchor adjudication.

Reads a dataset-IDs file, builds (or refreshes) a per-dataset judgment
sidecar at `<output-dir>/<dataset_id>.json`. Phase 2 of epic #76.

`scripts/hallu_cron_pipeline.sh` runs this before `update`. The pipeline reads
the sidecars through the fail-closed anchor gate (`core.anchor_gate`): only
anchors judged `data_paper` by the trusted judge model (plus the dataset's own
concept DOI) contribute citations; everything else, including an anchor with
no successful judgment, is context only.

Exit codes: 0 ok; 1 empty dataset list; 2 the judge is unusable (failed health
check, circuit breaker tripped, or a dependency failing on at least 5 and more
than `--max-failure-share`, default 10%, of its fresh calls) or a sidecar could
not be written. A 2 stops the cron before
`update`, so nothing built on a broken judge is published.

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
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from dataset_citations.backends import OpenCiteBackend
from dataset_citations.quality.anchor_judgment import (
    PERMANENT_LOOKUP_ERROR_PREFIX,
    is_judgment_fresh,
    judge_dataset_anchors,
    load_judgment_sidecar,
    save_judgment_sidecar,
)
from dataset_citations.quality.anchor_judgment_io import canonical_anchor_key
from dataset_citations.quality.dataset_metadata import DatasetMetadataRetriever
from dataset_citations.quality.llm_client import (
    PROMPT_VERSION,
    ClaudeCliJudgmentClient,
)
from dataset_citations.sources import BidsMetadataSource, NemarMetadataSource
from dataset_citations.sources.doi import is_own_dataset_doi

logger = logging.getLogger(__name__)

# A dependency (the judge, opencite lookups, the anchor source) is treated as
# down when at least this many of its calls failed AND they are more than
# `--max-failure-share` of its calls. The floor keeps a couple of flaky calls on
# a quiet night from stalling the pipeline; the share catches an outage that
# starts late in a long run, which a "more failures than successes" rule misses.
# Anchors that failed the same way on their previous run are left out of both
# counts: on a quiet night they can be the only calls made, and a few anchors
# that always fail would otherwise stop the cron every night.
_MIN_FAILURES = 5
_DEFAULT_MAX_FAILURE_SHARE = 0.10
# Consecutive failed judge calls (across datasets) that stop the run at once,
# leaving the remaining sidecars untouched instead of writing error records
# over them.
_CIRCUIT_BREAKER = 10
# An anchor opencite has no record of is retried on this back-off, not nightly.
_UNRESOLVABLE_RETRY_DAYS = 30


def _read_dataset_ids(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def _sidecar_path(output_dir: str, dataset_id: str) -> Path:
    return Path(output_dir) / f"{dataset_id}.json"


def _load_previous(sidecar: Path) -> dict | None:
    """The sidecar on disk, or None when absent or unreadable."""
    if not sidecar.exists():
        return None
    try:
        return load_judgment_sidecar(sidecar)
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("%s: unreadable sidecar (%s); re-judging", sidecar, exc)
        return None


def _judged_relations(payload: dict) -> dict[str, Any]:
    """Anchor key -> source_relation for every judgment the sidecar carries."""
    out: dict[str, Any] = {}
    for judgment in payload.get("judgments") or []:
        if not isinstance(judgment, dict):
            continue
        identifier = judgment.get("anchor_identifier")
        if isinstance(identifier, str) and identifier.strip():
            key = canonical_anchor_key(
                identifier, judgment.get("anchor_identifier_type") or "doi"
            )
            if key is not None:
                out[key] = judgment.get("source_relation")
    return out


def _recorded_relations(
    citations_dir: str | None, dataset_id: str
) -> dict[str, Any] | None:
    """Anchor key -> source_relation for the anchors the judge is responsible for.

    Read from the committed citation JSON's `metadata.anchors[]` (no GitHub or
    data.nemar.org round trip just to decide whether to skip). The dataset's own
    concept DOI and non-DOI anchors are excluded: the judge never judges them,
    so counting them made every sidecar look incomplete and re-ran ~96% of
    datasets every night. None when there is nothing to compare against.
    """
    if not citations_dir:
        return None
    path = Path(citations_dir) / f"{dataset_id}_citations.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict):
        return None
    out: dict[str, Any] = {}
    for anchor in metadata.get("anchors") or []:
        if not isinstance(anchor, dict):
            continue
        identifier = anchor.get("identifier")
        if not isinstance(identifier, str) or not identifier.strip():
            continue
        if (anchor.get("identifier_type") or "doi").lower() != "doi":
            continue
        if is_own_dataset_doi(identifier, dataset_id):
            continue
        key = canonical_anchor_key(identifier, "doi")
        if key is not None:
            out.setdefault(key, anchor.get("source_relation"))
    return out


def _retry_due(judgment: dict, now: datetime) -> bool:
    """True when an errored judgment should be retried on this run."""
    error = judgment.get("error")
    if not error:
        return False
    if not str(error).startswith(PERMANENT_LOOKUP_ERROR_PREFIX):
        return True
    try:
        when = datetime.fromisoformat(str(judgment.get("judged_at")))
    except ValueError:
        return True
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return now - when > timedelta(days=_UNRESOLVABLE_RETRY_DAYS)


def _needs_rejudge(payload: dict, model: str, now: datetime | None = None) -> str:
    """Why a readable sidecar cannot be trusted as-is ('' when it can).

    A sidecar written by another judge model or an older prompt is stale by
    definition (issue #241: every Gemma verdict is re-judged by Claude). An
    errored judgment is retried, except that an anchor opencite has no record
    of waits `_UNRESOLVABLE_RETRY_DAYS` between attempts.
    """
    if payload.get("judgment_model") != model:
        return f"model {payload.get('judgment_model')!r} != {model!r}"
    if payload.get("prompt_version") != PROMPT_VERSION:
        return f"prompt version {payload.get('prompt_version')!r} != {PROMPT_VERSION}"
    now = now or datetime.now(UTC)
    if any(
        _retry_due(j, now)
        for j in payload.get("judgments") or []
        if isinstance(j, dict)
    ):
        return "errored judgments due for a retry"
    return ""


def _should_skip(
    payload: dict | None,
    *,
    skip_existing: bool,
    max_age_days: int,
    model: str,
    citations_dir: str | None = None,
    dataset_id: str | None = None,
) -> tuple[bool, str]:
    """Return (skip, reason) for the sidecar `payload`. Reason is for logging only.

    A sidecar is never skipped when it is missing or unreadable (`payload` is
    None), or when `_needs_rejudge` says it is stale.

    `--skip-existing` used to skip on the mere existence of the sidecar, which
    froze a dataset's judgments at whatever its anchor set was the first time
    it ran (#180). So an existing sidecar is only a reason to skip when it
    covers every anchor the last pipeline run recorded, under the same
    DataCite relation: a relabel (e.g. `References` to `IsDescribedBy` after an
    enrichment sweep) changes the judge's input, so it is re-judged. Under the
    fail-closed gate (#241) an anchor left unjudged would lose its citers.
    """
    if payload is None or _needs_rejudge(payload, model):
        return False, ""
    if skip_existing:
        if dataset_id:
            recorded = _recorded_relations(citations_dir, dataset_id)
            if recorded:
                judged = _judged_relations(payload)
                stale = [
                    key
                    for key, relation in recorded.items()
                    if key not in judged or judged[key] != relation
                ]
                if stale:
                    return False, ""
        return True, "covered"
    if max_age_days > 0 and is_judgment_fresh(payload, max_age_days=max_age_days):
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


def _unhealthy(failures: int, attempts: int, max_share: float) -> bool:
    return failures >= _MIN_FAILURES and failures > max_share * attempts


@dataclass
class _Tally:
    processed: int = 0
    skipped: int = 0
    written: int = 0
    unchanged: int = 0
    write_failures: int = 0
    source_failures: int = 0
    judged: int = 0
    failed: int = 0
    lookup_failed: int = 0
    unresolvable: int = 0
    repeat_failures: int = 0
    reused: int = 0


def run(
    args: argparse.Namespace,
    *,
    client: ClaudeCliJudgmentClient,
    nemar_source: NemarMetadataSource,
    bids_source: BidsMetadataSource,
    metadata_retriever: DatasetMetadataRetriever,
    backend: OpenCiteBackend,
) -> int:
    """Judge every listed dataset with the given dependencies; return the exit code.

    Split from `main` so tests drive the real loop, skip logic, and exit policy
    with test doubles at the network boundary only.
    """
    dataset_ids = _read_dataset_ids(args.dataset_list_file)
    if not dataset_ids:
        logger.error("dataset list file %s is empty", args.dataset_list_file)
        return 1
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # One real judgment BEFORE any per-dataset work, so a logged-out CLI, a
    # missing binary, or an unknown model fails fast (exit 2) instead of
    # writing an errored judgment for every anchor.
    if not client.health_check():
        logger.error(
            "anchor judge (%s via %s) failed its health check; aborting before "
            "any judgments",
            client.model,
            client.claude_bin,
        )
        return 2

    logger.info(
        "judging anchors for %d dataset(s) -> %s (model=%s, workers=%d)",
        len(dataset_ids),
        args.output_dir,
        client.model,
        args.workers,
    )

    tally = _Tally()
    streak = 0
    tripped = False
    for dataset_id in dataset_ids:
        sidecar = _sidecar_path(args.output_dir, dataset_id)
        previous = _load_previous(sidecar)
        skip, reason = _should_skip(
            previous,
            skip_existing=args.skip_existing,
            max_age_days=args.max_age_days,
            model=client.model,
            citations_dir=args.citations_dir,
            dataset_id=dataset_id,
        )
        if skip:
            logger.info("%s: skipping (%s)", dataset_id, reason)
            tally.skipped += 1
            continue

        tally.processed += 1
        result = judge_dataset_anchors(
            dataset_id,
            nemar_source=nemar_source,
            bids_source=bids_source,
            metadata_retriever=metadata_retriever,
            backend=backend,
            client=client,
            previous=previous,
            max_workers=args.workers,
            datasets_dir=args.datasets_dir,
        )
        if result is None:
            tally.source_failures += 1
            continue

        tally.judged += result.judged
        tally.failed += result.failed
        tally.lookup_failed += result.lookup_failed
        tally.unresolvable += result.unresolvable
        tally.repeat_failures += result.repeat_failures
        tally.reused += result.reused
        streak = 0 if result.judged else streak + result.failed
        if streak >= _CIRCUIT_BREAKER:
            logger.error(
                "%d consecutive judge calls failed; stopping at %s without "
                "writing its sidecar or judging the rest",
                streak,
                dataset_id,
            )
            tripped = True
            break
        if previous == result.payload:
            tally.unchanged += 1
            continue
        try:
            save_judgment_sidecar(sidecar, result.payload)
        except OSError as exc:
            logger.error("failed to write %s: %s", sidecar, exc)
            tally.write_failures += 1
            continue
        tally.written += 1
        logger.info("%s: %s", dataset_id, _summarize_judgments(result.payload))

    logger.info(
        "anchor judgment run complete: %d dataset(s) processed, %d skipped, "
        "%d sidecars written, %d unchanged, %d source failures (sidecar left "
        "untouched), %d write failures; judge calls: %d ok, %d failed; lookups: "
        "%d failed, %d still unresolvable; %d anchors failed again as on their "
        "previous run; %d verdicts reused",
        tally.processed,
        tally.skipped,
        tally.written,
        tally.unchanged,
        tally.source_failures,
        tally.write_failures,
        tally.judged,
        tally.failed,
        tally.lookup_failed,
        tally.unresolvable,
        tally.repeat_failures,
        tally.reused,
    )
    return _exit_code(tally, tripped=tripped, max_share=args.max_failure_share)


def _exit_code(tally: _Tally, *, tripped: bool, max_share: float) -> int:
    """2 when the run cannot be trusted to feed `update`, else 0.

    Only fresh attempts count: repeats of an anchor's previous failure and
    anchors still unresolvable after their back-off are left out of both the
    failures and the attempts.
    """
    calls = tally.judged + tally.failed
    lookups = calls + tally.lookup_failed
    problems = {
        "circuit breaker tripped": tripped,
        "sidecar writes failed": tally.write_failures > 0,
        "judge calls failing": _unhealthy(tally.failed, calls, max_share),
        "paper lookups failing": _unhealthy(tally.lookup_failed, lookups, max_share),
        "anchor source failing": _unhealthy(
            tally.source_failures, tally.processed, max_share
        ),
    }
    down = [name for name, bad in problems.items() if bad]
    if down:
        # Abort the cron before `update` so a broken dependency is loud; the
        # gate fails closed, so nothing inflates meanwhile.
        logger.error("anchor judgment run is not trustworthy: %s", "; ".join(down))
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
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
        help=(
            "Skip datasets whose sidecar was judged by the current model and "
            "prompt, has no errored judgment due for a retry, and covers every "
            "anchor recorded in --citations-dir under the same relation "
            "(regardless of age)."
        ),
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=0,
        help=(
            "Skip datasets whose trusted sidecar's judged_at is within this many "
            "days. Set to 0 (default) to re-judge every dataset; --skip-existing "
            "takes precedence when both are set."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Judge model id (default: ANCHOR_JUDGE_MODEL, else claude-sonnet-5-5).",
    )
    parser.add_argument(
        "--claude-bin",
        default=None,
        help="Path to the claude CLI (default: CLAUDE_BIN, else `claude` on PATH).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent judge calls within one dataset (default: 4).",
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
            "sidecar still covers the dataset's anchors (default: "
            "citations/json_opencite)."
        ),
    )
    parser.add_argument(
        "--datasets-dir",
        default="datasets",
        help=(
            "Dataset metadata cached by retrieve-metadata, read for the dataset "
            "description instead of refetching it from GitHub (default: datasets)."
        ),
    )
    parser.add_argument(
        "--max-failure-share",
        type=float,
        default=_DEFAULT_MAX_FAILURE_SHARE,
        help=(
            "Exit 2 when the judge, paper lookups, or the anchor source fail on "
            "at least 5 fresh attempts and more than this share of them "
            f"(default: {_DEFAULT_MAX_FAILURE_SHARE}). Raise it to push a run "
            "past a known, contained failure."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    github_token = args.github_token or os.getenv("GITHUB_TOKEN")
    return run(
        args,
        client=ClaudeCliJudgmentClient(model=args.model, claude_bin=args.claude_bin),
        nemar_source=NemarMetadataSource(github_token=github_token),
        bids_source=BidsMetadataSource(github_token=github_token),
        metadata_retriever=DatasetMetadataRetriever(github_token=github_token),
        backend=OpenCiteBackend(max_results_per_doi=1),
    )


if __name__ == "__main__":
    raise SystemExit(main())
