"""Find dataset accession-mention citations via OpenAlex and merge them in.

For each dataset that already has a citation JSON, full-text search OpenAlex
for the dataset accession (`ds*`/`on*`/`nm*`, plus the OpenNeuro `ds-` alias
carried in the catalog `source_id`) and fold the matching papers into
`citation_details[]` tagged `discovery_method="accession_mention"`. This
automates the old "search the accession by hand, keep high-confidence hits"
workflow; the downstream score step ranks the merged citations. Issue #169.

Runs after `dataset-citations-update` and before
`dataset-citations-score-confidence` in the hallu cron.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from dataset_citations.backends.accession_search import AccessionSearchBackend
from dataset_citations.core.accession_mentions import merge_accession_mentions
from dataset_citations.core.citation_utils import write_citation_json_if_changed
from dataset_citations.core.run_state import (
    checked_within,
    load_state,
    save_state,
    stalest_first,
    stamp_checked,
)
from dataset_citations.sources.accession import accession_search_terms
from dataset_citations.sources.models import FetchSuccess
from dataset_citations.sources.nemar_catalog import get_or_fetch_catalog

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Per-citations-dir cache of the last time each dataset's accession search ran.
# Separate from update's `.fetch_state.json` so the two stages' freshness
# windows advance independently: a dataset can be citation-fresh but
# mention-stale, and re-running one stage must not reset the other's clock.
_MENTION_STATE_FILENAME = ".mention_state.json"


def _load_source_id_map(
    cache_path: Path | None, max_age_seconds: int
) -> dict[str, str]:
    """Return a {dataset_id: source_id} map for catalog datasets that have one.

    `source_id` is the OpenNeuro `ds-` number for `on-*` datasets, which is the
    accession people actually cite. On catalog failure we log and return {};
    the search then falls back to the dataset id alone.
    """
    result = get_or_fetch_catalog(
        cache_path=cache_path, max_age_seconds=max_age_seconds
    )
    if not isinstance(result, FetchSuccess):
        logger.warning(
            "could not load catalog (%s): %s; searching dataset ids only",
            result.reason,
            result.detail,
        )
        return {}
    return {row.dataset_id: row.source_id for row in result.value if row.source_id}


def _dataset_ids(args: argparse.Namespace, json_dir: str) -> list[str]:
    if args.dataset_list_file:
        with open(args.dataset_list_file, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return sorted(
        os.path.basename(p).removesuffix("_citations.json")
        for p in glob.glob(os.path.join(json_dir, "*_citations.json"))
    )


def run_find_mentions(args: argparse.Namespace) -> None:
    """Merge accession-mention citations into each dataset's citation JSON.

    Raises SystemExit(2) if every processed dataset failed to write.
    """
    json_dir = args.citations_dir
    source_ids = _load_source_id_map(args.catalog_cache, args.catalog_cache_max_age)
    backend = AccessionSearchBackend(max_results=args.max_results)

    dataset_ids = _dataset_ids(args, json_dir)
    total_candidates = len(dataset_ids)

    # Rolling freshness gate (issue #197). A full pass over the corpus costs far
    # more OpenAlex requests than the daily credit budget allows (measured
    # 2026-07-03: ~2.6 requests per dataset, so ~1,979 for a 750-dataset corpus
    # against a ~1,000-request daily budget), and an exhausted budget makes
    # opencite sleep on a 24h Retry-After while holding the cron's flock.
    # Refreshing only the stale slice each night keeps a run inside one day's
    # budget and still covers everything every --max-age-days.
    state_path = os.path.join(json_dir, _MENTION_STATE_FILENAME)
    mention_state = load_state(state_path)
    max_age_seconds = args.max_age_days * 86400 if args.max_age_days > 0 else None
    skipped_fresh = 0
    if max_age_seconds is not None:
        stale = [
            d
            for d in dataset_ids
            if not checked_within(d, mention_state, max_age_seconds)
        ]
        skipped_fresh = len(dataset_ids) - len(stale)
        dataset_ids = stale

    # Stalest first so a --max-datasets truncation always advances the oldest
    # entries; without it the same head of the list (alphabetical for the
    # default directory glob) would be re-searched every night and the tail
    # would never refresh.
    dataset_ids = stalest_first(dataset_ids, mention_state)

    deferred = 0
    if args.max_datasets > 0 and len(dataset_ids) > args.max_datasets:
        deferred = len(dataset_ids) - args.max_datasets
        dataset_ids = dataset_ids[: args.max_datasets]

    logger.info(
        "searching accession mentions for %d of %d dataset(s) "
        "(%d fresh within %dd, %d deferred by --max-datasets=%d)",
        len(dataset_ids),
        total_candidates,
        skipped_fresh,
        args.max_age_days,
        deferred,
        args.max_datasets,
    )

    processed = 0
    updated = 0
    no_terms = 0
    missing = 0
    write_failures = 0
    degraded = 0
    total_mentions = 0
    try:
        for dataset_id in dataset_ids:
            filepath = os.path.join(json_dir, f"{dataset_id}_citations.json")
            if not os.path.isfile(filepath):
                missing += 1
                continue
            terms = accession_search_terms(dataset_id, source_ids.get(dataset_id))
            if not terms:
                no_terms += 1
                # Stamped even though no request was made: the id itself does
                # not match the ds/on/nm + 6-digit accession pattern, so no
                # search is possible until the id changes. Leaving it unstamped
                # would let it occupy a --max-datasets slot every night and
                # starve datasets that can actually be searched.
                stamp_checked(mention_state, dataset_id)
                continue
            try:
                with open(filepath, encoding="utf-8") as f:
                    citation_json = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("%s: could not read (%s); skipping", dataset_id, e)
                continue
            processed += 1
            result = backend.search(terms)
            total_mentions += len(result.citations)
            merged = merge_accession_mentions(
                citation_json, result.citations, terms, when=datetime.now(UTC)
            )
            try:
                if write_citation_json_if_changed(filepath, merged):
                    updated += 1
            except OSError as e:
                logger.error("Failed to write %s: %s", filepath, e)
                write_failures += 1
                continue
            if result.degraded:
                # A rate-limited or timed-out term is logged and skipped inside
                # the backend rather than raised, so an empty result does not
                # imply "no papers mention this accession". Stamping here would
                # hide the dataset for a full --max-age-days window on the
                # strength of a search that never actually ran, which is a
                # silent loss of citation coverage. Leave it stale so
                # stalest-first retries it tomorrow.
                degraded += 1
                logger.warning(
                    "%s: search degraded (%d/%d term(s) failed: %s); not marking "
                    "checked, will retry next run",
                    dataset_id,
                    len(result.failed_terms),
                    len(terms),
                    ",".join(result.failed_terms),
                )
                continue
            stamp_checked(mention_state, dataset_id)
            logger.info(
                "%s: %d mention(s) for %s",
                dataset_id,
                len(result.citations),
                ",".join(terms),
            )
    finally:
        # In a `finally` so an exception escaping the loop (opencite raising
        # after its retries are exhausted, a KeyboardInterrupt) still keeps the
        # datasets already completed. Without this the whole run's progress
        # would be discarded and the next run would redo every dataset, which
        # is the exact behaviour issue #197 removes.
        save_state(state_path, mention_state)

    logger.info(
        "accession-mention run complete: %d processed, %d updated, %d mentions found, "
        "%d no-terms, %d missing-json, %d write failures, %d degraded (will retry).",
        processed,
        updated,
        total_mentions,
        no_terms,
        missing,
        write_failures,
        degraded,
    )
    if processed and write_failures == processed:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find dataset accession-mention citations via OpenAlex full-text "
            "search and merge them into the per-dataset citation JSON."
        ),
    )
    parser.add_argument(
        "--citations-dir",
        default="citations/json_opencite",
        help="Directory of <id>_citations.json files (default: citations/json_opencite).",
    )
    parser.add_argument(
        "--dataset-list-file",
        default=None,
        help="Optional file of dataset IDs (one per line). Defaults to all JSONs in --citations-dir.",
    )
    parser.add_argument(
        "--catalog-cache",
        type=Path,
        default=Path.home() / ".cache" / "dataset_citations" / "catalog.json",
        help="Cached api.nemar.org/datasets response (for the on-* -> ds-* alias).",
    )
    parser.add_argument(
        "--catalog-cache-max-age",
        type=int,
        default=3600,
        help="Seconds before the catalog cache is considered stale (default: 3600).",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=200,
        help="Max OpenAlex works to pull per accession term (default: 200).",
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=7,
        help=(
            "Skip datasets whose accession search ran within this many days "
            "(default: 7). 0 disables the freshness gate and searches every "
            "dataset, which costs far more than one day of OpenAlex credits."
        ),
    )
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=0,
        help=(
            "Hard cap on datasets searched per run, stalest first (default: 0 "
            "= unlimited). Use it to keep a cold-start pass inside the daily "
            "OpenAlex budget; the remainder rolls over to subsequent runs."
        ),
    )
    args = parser.parse_args()
    run_find_mentions(args)


if __name__ == "__main__":
    main()
