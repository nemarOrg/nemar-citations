"""Update citation lists for the datasets in a dataset-IDs file.

Runs the opencite-backed citation discovery pipeline against each dataset
listed in the input file and writes a citation JSON per dataset to
`<output-dir>/json_opencite/<id>_citations.json`.

Copyright (c) 2024 Seyed Yahya Shirazi (neuromechanist)
All rights reserved.

Author: Seyed Yahya Shirazi
GitHub: https://github.com/neuromechanist
Email: shirazi@ieee.org
"""

import argparse
import json
import logging
import os
from pathlib import Path

from dataset_citations.backends import OpenCiteBackend
from dataset_citations.core.citation_utils import strip_volatile_timestamps
from dataset_citations.core.opencite_pipeline import (
    fetch_dataset_citations_via_opencite,
)
from dataset_citations.core.run_state import (
    checked_within,
    load_state,
    save_state,
    stamp_checked,
)
from dataset_citations.sources.doi import normalize_doi
from dataset_citations.sources.models import FetchSuccess
from dataset_citations.sources.nemar_catalog import get_or_fetch_catalog

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Statuses that indicate a genuine API failure rather than a stable outcome.
# A file carrying one of these should be re-fetched on the next run (retry);
# any other status (success, partial, no_doi_references, no_data_paper_anchor,
# unsupported_prefix:*) is stable and safe to skip within the freshness window.
#
# `not_found` is deliberately NOT here (issue #217). It means the anchor DOI
# resolved no record in OpenAlex -- a property of the DOI, not of the API. Some
# anchors are simply unindexed (conference proceedings) or malformed at the
# source (`10.3389/fnhum.2022.xxxxx` is a literal placeholder in a dataset's
# metadata), so they return `not_found` on every run, forever. Counting that as
# an API failure wedged the nightly pipeline: the 36 affected datasets were
# never skipped as fresh, so on most nights they were the ONLY datasets
# processed, which made `successes == 0 and api_failures == processed` true and
# aborted the run at exit 3 before scoring, embeddings or the commit. Treating
# it as a stable outcome lets it retry once per freshness window instead.
_API_FAILURE_STATUSES = frozenset({"rate_limit", "auth", "network", "parse", "other"})

# Per-output-dir cache of the last time each dataset was fetched. Kept OUT of
# the committed citation JSON (and gitignored) so the freshness gate has a
# timestamp that advances every run without churning the diff. `date_last_updated`
# inside the citation JSON now means "last content change" (issue #165), so it
# can no longer double as the freshness signal.
# Smallest processed batch that can be read as evidence of an upstream outage.
# Below this a run is too small for "every dataset returned not_found" to mean
# anything; the permanently-unindexed set alone is ~36 datasets.
_OUTAGE_MIN_BATCH = 50

_FETCH_STATE_FILENAME = ".fetch_state.json"


def _read_json_or_none(filepath: str) -> dict | None:
    """Load JSON from `filepath`, returning None on any read/parse failure.

    A missing file returns None silently (an expected case for both new
    datasets and an absent fetch-state cache). A file that EXISTS but cannot
    be read/parsed is logged at WARNING: a corrupt citation JSON would
    otherwise be silently treated as absent and re-fetched every run with no
    trace in the cron log.
    """
    if not os.path.isfile(filepath):
        return None
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read %s (%s); treating as absent", filepath, e)
        return None
    return data if isinstance(data, dict) else None


def _has_stable_status(filepath: str) -> bool:
    """Return True if the on-disk citation JSON has a non-transient status.

    A missing/unparseable file, or one whose `metadata.fetch_status` is a
    transient API failure, returns False so the dataset is re-fetched.
    """
    payload = _read_json_or_none(filepath)
    if payload is None:
        return False
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return False
    status = metadata.get("fetch_status")
    return isinstance(status, str) and status not in _API_FAILURE_STATUSES


def _load_catalog_doi_map(
    cache_path: Path | None, max_age_seconds: int
) -> dict[str, str]:
    """Return a {dataset_id: catalog_doi} map for catalog-indexed datasets.

    The catalog is fetched from `api.nemar.org/datasets` (or reused from the
    shared cache populated by the discover step earlier in the same workflow
    run). DOIs are run through `normalize_doi` (lowercased + prefix-stripped)
    so the map values match what the pipeline's `_merge_anchors` produces
    when deduping anchors. Empty / missing entries are omitted: a missing
    catalog DOI is not an error; the pipeline simply doesn't seed an extra
    anchor for that dataset.

    On fetch failure we log and return an empty map rather than aborting:
    the rest of the pipeline still produces citation JSON from the source-
    derived anchors. The catalog DOI is purely additive.
    """
    result = get_or_fetch_catalog(
        cache_path=cache_path, max_age_seconds=max_age_seconds
    )
    if not isinstance(result, FetchSuccess):
        logger.warning(
            "could not load catalog (%s): %s; proceeding without catalog DOIs",
            result.reason,
            result.detail,
        )
        return {}
    return {row.dataset_id: normalize_doi(row.doi) for row in result.value if row.doi}


def run_opencite_backend(args: argparse.Namespace) -> None:
    """Run the opencite pipeline against each dataset in the input list.

    Writes results to `<output-dir>/json_opencite/<id>_citations.json`.
    Returns normally on success or partial-success. Raises non-zero
    `SystemExit` so automation (cron / CI) can detect degraded runs:
      - exit 2 when every single dataset failed to write (disk full, read-only).
      - exit 3 when zero datasets returned citing works AND every empty
        result was an API failure (rate_limit, auth, network, etc.). A run
        where datasets legitimately lack DOIs (`no_doi_references`) is
        considered success and does NOT trigger this.
    """
    json_dir = os.path.join(args.output_dir, "json_opencite")
    os.makedirs(json_dir, exist_ok=True)

    with open(args.dataset_list_file, encoding="utf-8") as f:
        dataset_ids = [line.strip() for line in f if line.strip()]

    # OPENCITE_CONCURRENCY caps the anchor fan-out inside one dataset. The CI
    # workflow sets it to 1 to avoid Semantic Scholar 429s when canonical
    # anchor papers are shared across datasets (issue #49). Construct ONE
    # backend instance and pass it to every per-dataset call so the
    # CitationExplorer session and rate-limit headers are reused.
    try:
        concurrency = max(1, int(os.environ.get("OPENCITE_CONCURRENCY", "4")))
    except ValueError:
        concurrency = 4
    backend = OpenCiteBackend(concurrency=concurrency)

    # A missing datasets-dir means every ds-* dataset silently falls back to a
    # live GitHub fetch for DOI extraction -- the cold-start rate-limit storm
    # #174 fixes. Surface it loudly rather than let a misconfigured path stall
    # the run invisibly.
    if not os.path.isdir(args.datasets_dir):
        logger.warning(
            "datasets-dir %s not found; ds-* DOI extraction will fall back to "
            "GitHub (rate-limit risk)",
            args.datasets_dir,
        )

    # nm-*/on-* DOIs normally come from data.nemar.org, but when that returns
    # 5xx the pipeline falls back to GitHub. Unauthenticated that is 60 req/hr,
    # and a single 403 makes PyGithub sleep until the primary reset (~56 min),
    # which is what stalled the 2026-08-01 run for hours (issue #201). Every
    # other CLI in the pipeline already reads this env var; this one did not.
    github_token = args.github_token or os.getenv("GITHUB_TOKEN")
    if not github_token:
        logger.warning(
            "no GitHub token (--github-token / GITHUB_TOKEN); the data.nemar.org "
            "fallback will use the 60 req/hr anonymous pool and can stall on "
            "long PyGithub backoffs"
        )

    # Load the catalog once so each dataset can be seeded with its own
    # NEMAR-minted DOI as an additional opencite anchor. The catalog cache
    # is shared with the discover step in the same workflow run, so this
    # is typically a no-op disk read; the explicit fetch is a safety net
    # for when update.py is invoked outside the workflow.
    catalog_dois = _load_catalog_doi_map(args.catalog_cache, args.catalog_cache_max_age)
    logger.info(
        "Running opencite backend for %d dataset(s) -> %s "
        "(concurrency=%d, catalog rows known=%d)",
        len(dataset_ids),
        json_dir,
        concurrency,
        len(catalog_dois),
    )

    # A run where every processed dataset stubs out with an _API_FAILURE_STATUSES
    # value (and nothing was written) exits non-zero below, so automation can
    # detect a fully-degraded run instead of silently shipping empty JSONs.
    max_age_seconds = args.max_age_days * 86400 if args.max_age_days > 0 else None
    fetch_state_path = os.path.join(args.output_dir, _FETCH_STATE_FILENAME)
    fetch_state = load_state(fetch_state_path)
    successes = 0
    stub_only = 0
    write_failures = 0
    api_failures = 0
    not_found_failures = 0
    skipped_fresh = 0
    unchanged = 0
    for dataset_id in dataset_ids:
        filepath = os.path.join(json_dir, f"{dataset_id}_citations.json")
        # Freshness gate: skip a re-fetch when the on-disk result is stable
        # (not a transient API failure) AND we checked it within the window.
        # The last-checked signal comes from the gitignored fetch-state cache,
        # NOT date_last_updated, which now only advances on real content change.
        if (
            max_age_seconds is not None
            and _has_stable_status(filepath)
            and checked_within(dataset_id, fetch_state, max_age_seconds)
        ):
            logger.info(
                "%s: skipping (checked within %d days, stable status)",
                dataset_id,
                args.max_age_days,
            )
            skipped_fresh += 1
            continue
        payload = fetch_dataset_citations_via_opencite(
            dataset_id,
            backend=backend,
            catalog_doi=catalog_dois.get(dataset_id),
            use_checkpoint=True,
            local_metadata_dir=args.datasets_dir,
            github_token=github_token,
        )
        # Content-idempotent write. The opencite payload never carries the
        # confidence_scoring block (the separate score step adds it), so we
        # carry the existing block onto a comparison copy to avoid a spurious
        # "changed" verdict every run. When nothing changed, leave the file
        # (and its scores + timestamps) byte-for-byte untouched. When citations
        # DID change, write the fresh payload WITHOUT the now-stale scores so
        # the score step re-scores it (its --skip-existing skips files that
        # already have confidence_scoring). See issue #165.
        existing = _read_json_or_none(filepath)
        compare = payload
        if existing is not None and "confidence_scoring" in existing:
            compare = {**payload, "confidence_scoring": existing["confidence_scoring"]}
        if existing is not None and strip_volatile_timestamps(
            existing
        ) == strip_volatile_timestamps(compare):
            unchanged += 1
        else:
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
            except OSError as e:
                logger.error("Failed to write %s: %s", filepath, e)
                write_failures += 1
                continue
        # Record the fetch attempt so the freshness gate can skip it next run.
        stamp_checked(fetch_state, dataset_id)
        if payload["num_citations"] > 0:
            successes += 1
        else:
            stub_only += 1
            status = payload.get("metadata", {}).get("fetch_status", "empty")
            if status in _API_FAILURE_STATUSES:
                api_failures += 1
            if status == "not_found":
                not_found_failures += 1
            logger.info("%s: 0 citations (status=%s)", dataset_id, status)

    save_state(fetch_state_path, fetch_state)
    logger.info(
        "opencite run complete: %d with citations, %d empty/stub "
        "(%d API-failure), %d write failures, %d unchanged, %d skipped (fresh), "
        "%d total.",
        successes,
        stub_only,
        api_failures,
        write_failures,
        unchanged,
        skipped_fresh,
        len(dataset_ids),
    )
    total = len(dataset_ids)
    # The exit-2 / exit-3 sentinels must compare against the count of datasets
    # we actually attempted, not the input list size: a maintenance run that
    # skips most datasets as fresh would otherwise never trip exit-3 even
    # when every processed dataset failed.
    processed = total - skipped_fresh
    if processed and write_failures == processed:
        # Every write attempted failed (disk full, read-only output). Exit 2.
        raise SystemExit(2)
    if processed and successes == 0 and api_failures == processed:
        # Zero successes among processed datasets and every stub was an API
        # failure (not just a dataset without DOIs). Exit 3 so cron can detect
        # the degraded run before downstream steps regenerate empty CSVs.
        raise SystemExit(3)
    if (
        processed >= _OUTAGE_MIN_BATCH
        and processed > total // 2
        and successes == 0
        and not_found_failures == processed
    ):
        # `not_found` is no longer an API-failure status (#217), but it can
        # still be produced BY an outage rather than by a bad DOI: opencite's
        # classify_error buckets any error text containing "not found" here
        # (ahead of its network branch), and _is_unresolved_seed reports it
        # whenever every source fails to resolve an anchor, transiently or not.
        #
        # The two cases separate by scale. The datasets with permanently
        # unindexed anchors are a small, stable minority (36 of ~740). A run
        # that processed MOST of the corpus and got nothing but not_found is an
        # outage, so fail loudly rather than publish an empty night. The
        # majority threshold is what keeps this from re-creating the #217 wedge,
        # where the same 36 datasets were the entire processed set, and the
        # absolute floor keeps a deliberately small run (a two-dataset
        # re-check, a test) from looking like a corpus-wide outage.
        logger.error(
            "every one of %d processed datasets (of %d total) returned "
            "not_found; treating as an upstream outage rather than data",
            processed,
            total,
        )
        raise SystemExit(3)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Update dataset citations via the opencite pipeline. Anchors on "
            "DOIs surfaced by .nemar/metadata.json (nm datasets) or "
            "dataset_description.json (legacy ds datasets), looks up citing "
            "works via opencite, and writes a JSON per dataset to "
            "<output-dir>/json_opencite/."
        ),
    )
    parser.add_argument(
        "--dataset-list-file",
        required=True,
        help="Path to the file containing the list of dataset IDs (one per line).",
    )
    parser.add_argument(
        "--github-token",
        type=str,
        help="GitHub token for API access (uses environment variable GITHUB_TOKEN if not provided)",
    )
    parser.add_argument(
        "--output-dir",
        default="citations",
        help="Directory to save output files (default: citations/).",
    )
    parser.add_argument(
        "--datasets-dir",
        default="datasets",
        help=(
            "Directory of retrieve-metadata output (<id>_datasets.json). When a "
            "ds-* dataset's cached dataset_description is present here, DOI "
            "extraction parses it instead of refetching from GitHub, avoiding "
            "the cold-start GitHub rate limit (issue #174). Default: datasets/."
        ),
    )
    parser.add_argument(
        "--catalog-cache",
        type=Path,
        default=Path.home() / ".cache" / "dataset_citations" / "catalog.json",
        help=(
            "Path to a cached api.nemar.org/datasets response. Shared with "
            "dataset-citations-discover so the catalog is fetched once per "
            "workflow run."
        ),
    )
    parser.add_argument(
        "--catalog-cache-max-age",
        type=int,
        default=3600,
        help="Seconds before the catalog cache is considered stale (default: 3600).",
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=7,
        help=(
            "Skip a re-fetch when the existing JSON has a stable (non-transient) "
            "fetch_status AND the dataset was last fetched within this many days, "
            "per the gitignored .fetch_state.json cache (default: 7, matching the "
            "weekly cron cadence). Set to 0 to force re-fetch every dataset on "
            "every invocation."
        ),
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logger.info("Created output directory: %s", args.output_dir)

    run_opencite_backend(args)


if __name__ == "__main__":
    main()
