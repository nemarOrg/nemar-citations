"""Throwaway probe: classify anchor DOIs for ~10 hand-picked datasets.

Acceptance gate for epic #76 (LLM anchor adjudication). For each dataset in
`PROBE_DATASETS`, the script:

  1. Extracts anchor DOIs from the dataset's metadata source
     (`NemarMetadataSource` for nm-/on-, `BidsMetadataSource` for ds-).
  2. Fetches the dataset description via `DatasetMetadataRetriever`.
  3. Fetches each anchor paper's title + abstract via the new
     `OpenCiteBackend.get_paper(doi)` sync facade.
  4. Asks the Claude judge (`ClaudeCliJudgmentClient`, the `claude` CLI in
     headless mode; model from ANCHOR_JUDGE_MODEL, default claude-sonnet-5-5)
     to classify each anchor as one of `data_paper` / `umbrella` /
     `methodology` / `related_work` / `irrelevant`.
  5. Prints one row per anchor + a per-classification tally.

Optional `--output` dumps the full per-anchor payload (including the prompt
and the raw model JSON) so the spot-check evidence can be archived to
`.context/research.md`.

This script does NOT write any sidecar JSON to `citations/anchor_judgments/`;
that's phase 2 (#86).

Usage (anywhere the `claude` CLI is logged in, e.g. on hallu):
    uv run python scripts/probe_anchor_judgment.py \
        --output .context/probe_anchor_judgment_$(date +%Y-%m-%d).json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dataset_citations.backends import OpenCiteBackend
from dataset_citations.quality.dataset_metadata import (
    DatasetMetadataRetriever,
    extract_dataset_text,
)
from dataset_citations.quality.llm_client import (
    ClaudeCliJudgmentClient,
    LlmJudgmentError,
    build_anchor_prompt,
)
from dataset_citations.sources import (
    BidsMetadataSource,
    FetchError,
    FetchSuccess,
    NemarMetadataSource,
)
from dataset_citations.sources.models import DoiReference

logger = logging.getLogger("probe_anchor_judgment")

# Hand-picked datasets covering the acceptance-gate cases from epic #76:
#   - HBN siblings (ds005505, ds005516): IsDerivedFrom anchor is the HBN
#     umbrella paper; expect umbrella for the HBN papers, data_paper for
#     the dataset preprint.
#   - ds000246: methodology anchor (Brainstorm / similar EEG/MEG tool).
#   - ds002034: clean dataset whose anchor is its own data paper.
# `ds002718` is intentionally absent: it is the dataset_id used in the
# few-shot Example 2 inside build_anchor_prompt, so including it here
# would let the model echo the example instead of judging the case.
# nm-/on- slots use real catalog IDs from api.nemar.org/datasets so the
# nemar metadata source has something to extract.
PROBE_DATASETS: tuple[str, ...] = (
    "ds005505",
    "ds005516",
    "ds000117",
    "ds000246",
    "ds002034",
    "ds001785",
    "ds001971",
    "nm000104",
    "on001787",
    "on002181",
)


def _pick_source(
    dataset_id: str,
    *,
    nemar_source: NemarMetadataSource,
    bids_source: BidsMetadataSource,
) -> Any:
    lowered = dataset_id.lower()
    if lowered.startswith(("nm", "on")):
        return nemar_source
    return bids_source


def _probe_anchor(
    *,
    dataset_id: str,
    dataset_description: str,
    ref: DoiReference,
    backend: OpenCiteBackend,
    client: ClaudeCliJudgmentClient,
) -> dict[str, Any]:
    """Fetch the anchor paper + ask the LLM. Returns a result dict.

    On failures (paper not found, LLM malformed output) the dict still
    contains the anchor metadata + an `error` field so the probe can keep
    going and the human reviewer sees the full picture.
    """
    record: dict[str, Any] = {
        "dataset_id": dataset_id,
        "anchor_doi": ref.identifier,
        "anchor_relation": ref.relation_type,
    }
    paper_result = backend.get_paper(ref.identifier)
    if isinstance(paper_result, FetchError):
        record["error"] = (
            f"paper_lookup_failed:{paper_result.reason}:{paper_result.detail}"
        )
        record["classification"] = None
        record["reason"] = None
        return record
    assert isinstance(paper_result, FetchSuccess)  # noqa: S101 - upstream contract guard
    paper = paper_result.value
    record["paper_title"] = paper.title
    record["paper_venue"] = paper.venue
    record["paper_year"] = paper.year
    record["paper_has_abstract"] = bool(paper.abstract)

    prompt = build_anchor_prompt(
        dataset_id=dataset_id,
        dataset_description=dataset_description,
        anchor_doi=ref.identifier,
        anchor_relation=ref.relation_type,
        paper_title=paper.title,
        paper_abstract=paper.abstract,
        paper_venue=paper.venue,
        paper_authors=paper.authors,
        paper_year=paper.year,
    )
    record["prompt"] = prompt

    try:
        judgment = client.judge_anchor(prompt)
    except LlmJudgmentError as exc:
        logger.error(
            "LLM judgment failed for %s / %s: %s", dataset_id, ref.identifier, exc
        )
        record["error"] = f"llm_judgment_failed:{exc}"
        record["raw_response"] = exc.raw_response
        record["classification"] = None
        record["reason"] = None
        return record

    record["classification"] = judgment["classification"]
    record["reason"] = judgment["reason"]
    record["raw_response"] = judgment["raw_response"]
    return record


def _probe_dataset(
    dataset_id: str,
    *,
    nemar_source: NemarMetadataSource,
    bids_source: BidsMetadataSource,
    metadata_retriever: DatasetMetadataRetriever,
    backend: OpenCiteBackend,
    client: ClaudeCliJudgmentClient,
) -> list[dict[str, Any]]:
    source = _pick_source(
        dataset_id, nemar_source=nemar_source, bids_source=bids_source
    )
    refs_result = source.get_doi_references(dataset_id)
    if isinstance(refs_result, FetchError):
        logger.warning(
            "skipping %s: source returned %s (%s)",
            dataset_id,
            refs_result.reason,
            refs_result.detail,
        )
        return [
            {
                "dataset_id": dataset_id,
                "error": f"source_failed:{refs_result.reason}:{refs_result.detail}",
            }
        ]
    assert isinstance(refs_result, FetchSuccess)  # noqa: S101 - upstream contract guard
    refs: list[DoiReference] = [
        r for r in refs_result.value if r.identifier_type == "doi"
    ]
    if not refs:
        logger.info("no DOI anchors for %s", dataset_id)
        return [{"dataset_id": dataset_id, "error": "no_doi_anchors"}]

    metadata = metadata_retriever.get_dataset_metadata(dataset_id)
    dataset_description = extract_dataset_text(metadata)

    records: list[dict[str, Any]] = []
    for ref in refs:
        logger.info("  anchor %s (%s)", ref.identifier, ref.relation_type)
        records.append(
            _probe_anchor(
                dataset_id=dataset_id,
                dataset_description=dataset_description,
                ref=ref,
                backend=backend,
                client=client,
            )
        )
    return records


def _print_row(rec: dict[str, Any]) -> None:
    if rec.get("error"):
        print(
            f"  [SKIP] {rec.get('anchor_doi', '<no-anchor>')}: {rec['error']}",
            file=sys.stderr,
        )
        return
    title = (rec.get("paper_title") or "")[:80]
    reason = (rec.get("reason") or "")[:120]
    print(
        f"  {rec['anchor_doi']:<40}  rel={rec['anchor_relation']:<14}  "
        f"-> {rec['classification']:<13}  {title}\n"
        f"      reason: {reason}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(PROBE_DATASETS),
        help="Override the hand-picked PROBE_DATASETS list.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to dump full per-anchor JSON (prompt + raw response).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Judge model id (default: ANCHOR_JUDGE_MODEL, else claude-sonnet-5-5).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    nemar_source = NemarMetadataSource()
    bids_source = BidsMetadataSource()
    metadata_retriever = DatasetMetadataRetriever()
    backend = OpenCiteBackend(max_results_per_doi=1)

    client = ClaudeCliJudgmentClient(model=args.model)
    if not client.health_check():
        logger.error("anchor judge %s failed its health check; aborting", client.model)
        return 2

    logger.info(
        "probing %d datasets against judge model %s",
        len(args.datasets),
        client.model,
    )

    all_records: list[dict[str, Any]] = []
    for dataset_id in args.datasets:
        print(f"\n=== {dataset_id} ===")
        records = _probe_dataset(
            dataset_id,
            nemar_source=nemar_source,
            bids_source=bids_source,
            metadata_retriever=metadata_retriever,
            backend=backend,
            client=client,
        )
        for rec in records:
            _print_row(rec)
        all_records.extend(records)

    tally: dict[str, int] = {}
    for rec in all_records:
        cls = rec.get("classification")
        if cls:
            tally[cls] = tally.get(cls, 0) + 1
    error_count = sum(1 for r in all_records if r.get("error"))

    print("\n=== Summary ===")
    print(f"anchors classified: {sum(tally.values())}")
    for cls in sorted(tally):
        print(f"  {cls:<14} {tally[cls]}")
    print(f"errors/skips: {error_count}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "datasets": args.datasets,
            "model": client.model,
            "tally": tally,
            "error_count": error_count,
            "records": all_records,
        }
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        logger.info("wrote %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
