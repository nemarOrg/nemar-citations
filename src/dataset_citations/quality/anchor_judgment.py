"""Per-dataset LLM anchor adjudication storage + orchestration.

Phase 2 of epic #76. Productizes the phase-1 probe (`scripts/probe_anchor_judgment.py`)
into a real per-dataset sidecar file at `citations/anchor_judgments/<id>.json`
and the assembly function the batch CLI loops over.

Sidecar schema (locked; phase 3 reads this):

  {
    "dataset_id": "<id>",
    "judged_at": "<ISO-8601 UTC, most recent judgment in this file>",
    "judgment_model": "<judge model id, e.g. claude-sonnet-5-5>",
    "prompt_version": 2,
    "judgments": [
      {
        "anchor_identifier": "10.xxxx/yyyy",
        "anchor_identifier_type": "doi",
        "source_relation": "IsDerivedFrom",
        "classification": "umbrella",
        "reason": "<the LLM's reason string>",
        "paper_title": "<openalex title; null if get_paper failed>",
        "paper_year": 2024,
        "paper_venue": "<openalex venue; nullable>",
        "judged_at": "<ISO-8601 UTC>",
        "error": null
      }
    ]
  }

Classification values MUST belong to `ALLOWED_CLASSIFICATIONS` from
`quality.llm_client`; the LLM client enforces this on parse so the
validation lives there, not here.

Copyright (c) 2026 Seyed Yahya Shirazi (neuromechanist)
All rights reserved.

Author: Seyed Yahya Shirazi
GitHub: https://github.com/neuromechanist
Email: shirazi@ieee.org
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from dataset_citations.backends.opencite_backend import OpenCiteBackend
from dataset_citations.quality.anchor_judgment_io import canonical_anchor_key
from dataset_citations.quality.dataset_metadata import (
    DatasetMetadataRetriever,
    _org_for_dataset,
    extract_dataset_text,
    load_dataset_metadata,
)
from dataset_citations.quality.llm_client import (
    ALLOWED_CLASSIFICATIONS,
    PROMPT_VERSION,
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

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class JudgmentRecord:
    """One per-anchor judgment row inside a sidecar's `judgments[]` list.

    Fields mirror the locked sidecar schema; the `to_dict()` method emits
    them in the documented order so manual file inspection is stable across
    runs. `error` is the only optional-meaning field: None on success, a
    short string when judgment failed (paper lookup, LLM transport, or
    LLM parse). When `error` is non-None, `classification` and `reason`
    may be empty strings and `paper_*` fields may be None.
    """

    anchor_identifier: str
    anchor_identifier_type: str
    source_relation: str
    classification: str
    reason: str
    paper_title: str | None
    paper_year: int | None
    paper_venue: str | None
    judged_at: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_identifier": self.anchor_identifier,
            "anchor_identifier_type": self.anchor_identifier_type,
            "source_relation": self.source_relation,
            "classification": self.classification,
            "reason": self.reason,
            "paper_title": self.paper_title,
            "paper_year": self.paper_year,
            "paper_venue": self.paper_venue,
            "judged_at": self.judged_at,
            "error": self.error,
        }


def _pick_source(
    dataset_id: str,
    *,
    nemar_source: NemarMetadataSource,
    bids_source: BidsMetadataSource,
) -> NemarMetadataSource | BidsMetadataSource:
    """Route a dataset id to the right DOI source.

    Delegates to `quality.dataset_metadata._org_for_dataset` so the
    metadata fetch and the DOI-ref fetch agree on which org owns the
    dataset (commit cbf595f tightened the prefix check from a loose
    `startswith` to require a digit after `nm`/`on`; future ids like
    `nmr-phantom` must NOT route to nemarDatasets).
    """
    if _org_for_dataset(dataset_id) == "nemarDatasets":
        return nemar_source
    return bids_source


def _utcnow_iso() -> str:
    """Return current time as a timezone-aware ISO-8601 UTC string."""
    return datetime.now(UTC).isoformat()


# A lookup error that will not clear on tomorrow's retry: opencite has no
# record of the anchor DOI (unindexed, or a typo such as `10.38119/openneuro.*`).
# The judge CLI retries these on a back-off instead of nightly, and they never
# count toward its judge-health exit code.
PERMANENT_LOOKUP_ERROR_PREFIX = "paper_lookup_failed:not_found:"
# The dataset's description and README could not be read (a rate-limited
# GitHub call returns an empty shell). Judging without them would bias every
# verdict toward related_work and then reuse that verdict forever, so the
# anchors are recorded as errored and retried instead.
METADATA_UNAVAILABLE_ERROR = "dataset_metadata_unavailable"


@dataclass(frozen=True, slots=True)
class DatasetJudgmentRun:
    """Outcome of judging one dataset.

    `payload` is the sidecar dict to write. `judged` / `failed` count the LLM
    calls made in this run. `lookup_failed` counts anchors that could not be
    judged for a transient reason (an opencite error other than not_found, or
    unreadable dataset metadata); `unresolvable` counts anchors opencite has no
    record of; `reused` counts anchors whose previous verdict was kept without
    a call.
    """

    payload: dict[str, Any]
    judged: int
    failed: int
    lookup_failed: int
    unresolvable: int
    reused: int


def _error_record(
    ref: DoiReference,
    *,
    judged_at: str,
    error: str,
    paper: Any | None = None,
) -> JudgmentRecord:
    return JudgmentRecord(
        anchor_identifier=ref.identifier,
        anchor_identifier_type=ref.identifier_type,
        source_relation=ref.relation_type,
        classification="",
        reason="",
        paper_title=paper.title if paper else None,
        paper_year=paper.year if paper else None,
        paper_venue=paper.venue if paper else None,
        judged_at=judged_at,
        error=error,
    )


# JudgmentRecord field order, used to rebuild records read back from disk.
_RECORD_FIELDS = (
    "anchor_identifier",
    "anchor_identifier_type",
    "source_relation",
    "classification",
    "reason",
    "paper_title",
    "paper_year",
    "paper_venue",
    "judged_at",
    "error",
)


def _previous_records(
    previous: dict[str, Any] | None, model: str
) -> dict[str, JudgmentRecord]:
    """Records from the previous sidecar that this run may build on, by anchor.

    Only a sidecar written by `model` under the current `PROMPT_VERSION`
    counts: switching the judge (Gemma to Claude, issue #241) or revising the
    prompt must re-judge every anchor rather than let old verdicts keep
    counting. Both successes and errors are returned; the caller reuses a
    success and uses an unchanged error to avoid rewriting it every night.
    """
    if (
        not previous
        or previous.get("judgment_model") != model
        or previous.get("prompt_version") != PROMPT_VERSION
    ):
        return {}
    out: dict[str, JudgmentRecord] = {}
    for entry in previous.get("judgments") or []:
        if not isinstance(entry, dict):
            continue
        if not entry.get("error") and (
            entry.get("classification") not in ALLOWED_CLASSIFICATIONS
        ):
            continue
        identifier = entry.get("anchor_identifier")
        identifier_type = entry.get("anchor_identifier_type")
        if not isinstance(identifier, str) or not isinstance(identifier_type, str):
            continue
        key = canonical_anchor_key(identifier, identifier_type)
        if key is None or not isinstance(entry.get("judged_at"), str):
            continue
        try:
            out.setdefault(
                key, JudgmentRecord(**{f: entry.get(f) for f in _RECORD_FIELDS})
            )
        except TypeError:
            continue
    return out


def _unique_doi_refs(refs: list[DoiReference]) -> list[DoiReference]:
    """DOI anchors, one per identifier (first wins, like the pipeline's merge).

    A DOI listed under two relations would otherwise be judged twice, and the
    copy whose relation did not match the reused record would be re-judged
    every night.
    """
    seen: set[str] = set()
    out: list[DoiReference] = []
    for ref in refs:
        if ref.identifier_type != "doi":
            continue
        key = canonical_anchor_key(ref.identifier, "doi")
        if key is None or key in seen:
            continue
        seen.add(key)
        out.append(ref)
    return out


def _usable_metadata(metadata: Any) -> bool:
    """True when the metadata carries a dataset name or a README to judge by."""
    if not isinstance(metadata, dict):
        return False
    description = metadata.get("dataset_description")
    name = description.get("Name") if isinstance(description, dict) else None
    readme = metadata.get("readme_content")
    return bool(
        (isinstance(name, str) and name.strip())
        or (isinstance(readme, str) and readme.strip())
    )


def _dataset_metadata(
    dataset_id: str,
    retriever: DatasetMetadataRetriever,
    datasets_dir: Path | str | None,
) -> dict[str, Any] | None:
    """The dataset's description and README, or None when neither is readable.

    Prefers the copy `retrieve-metadata` cached in `datasets_dir` earlier in the
    same run (no GitHub call, issue #95), and falls back to a live fetch.
    """
    if datasets_dir is not None:
        path = Path(datasets_dir) / f"{dataset_id}_datasets.json"
        if path.exists():
            try:
                cached = load_dataset_metadata(str(path))
            except (OSError, ValueError) as exc:
                logger.warning(
                    "%s: unreadable cached metadata %s: %s", dataset_id, path, exc
                )
            else:
                if _usable_metadata(cached):
                    return cached
    fetched = retriever.get_dataset_metadata(dataset_id)
    return fetched if _usable_metadata(fetched) else None


def judge_dataset_anchors(
    dataset_id: str,
    *,
    nemar_source: NemarMetadataSource,
    bids_source: BidsMetadataSource,
    metadata_retriever: DatasetMetadataRetriever,
    backend: OpenCiteBackend,
    client: ClaudeCliJudgmentClient,
    previous: dict[str, Any] | None = None,
    max_workers: int = 1,
    datasets_dir: Path | str | None = None,
) -> DatasetJudgmentRun | None:
    """Judge one dataset's DOI anchors; return the run, or None on source failure.

    None means "do not touch the sidecar": when the anchor source cannot be
    read (rate limit, outage) there is nothing to judge, and overwriting the
    existing sidecar with an empty one would silently discard good judgments
    (issue #241).

    `previous` is the sidecar currently on disk. An anchor whose previous
    verdict (same model, same prompt version) was made under the same
    `source_relation` is reused without a call. When an anchor cannot be judged
    now, its previous verdict is kept instead of an error, so a flaky night
    never downgrades an anchor that was already judged; an unchanged error
    keeps its old record so the sidecar does not churn. Anchors are not judged
    at all when the dataset's description and README are unavailable. Paper
    lookups run sequentially (opencite shares one rate limiter per process);
    the LLM calls run on up to `max_workers` threads.
    """
    source = _pick_source(
        dataset_id, nemar_source=nemar_source, bids_source=bids_source
    )
    refs_result = source.get_doi_references(dataset_id)
    if isinstance(refs_result, FetchError):
        logger.warning(
            "%s: source returned %s (%s); leaving the existing sidecar untouched",
            dataset_id,
            refs_result.reason,
            refs_result.detail,
        )
        return None
    assert isinstance(refs_result, FetchSuccess)  # noqa: S101 - upstream contract guard
    refs = _unique_doi_refs(refs_result.value)
    prior = _previous_records(previous, client.model)
    if not refs and previous and prior:
        # A stale or partial metadata read can list no anchors for a moment.
        # Verdicts for anchors that are not listed are never consulted by the
        # gate, so keeping them costs nothing, while dropping them would force
        # a full re-judge when the anchors reappear.
        logger.warning(
            "%s: the source lists no DOI anchors but the sidecar holds %d "
            "trusted judgment(s); keeping them",
            dataset_id,
            len(prior),
        )
        return DatasetJudgmentRun(
            payload=previous,
            judged=0,
            failed=0,
            lookup_failed=0,
            unresolvable=0,
            reused=len(prior),
        )

    records: list[JudgmentRecord | None] = [None] * len(refs)
    todo: list[int] = []
    for i, ref in enumerate(refs):
        record = prior.get(canonical_anchor_key(ref.identifier, "doi") or "")
        if (
            record is not None
            and not record.error
            and record.source_relation == ref.relation_type
        ):
            records[i] = record
        else:
            todo.append(i)
    reused = len(refs) - len(todo)

    def _unjudged(i: int, error: str, judged_at: str, paper: Any = None):
        """A previous success if any, else an unchanged previous error, else new."""
        ref = refs[i]
        record = prior.get(canonical_anchor_key(ref.identifier, "doi") or "")
        if record is not None and (not record.error or record.error == error):
            return record
        return _error_record(ref, judged_at=judged_at, error=error, paper=paper)

    prompts: dict[int, tuple[Any, str, str]] = {}
    lookup_failed = 0
    unresolvable = 0
    if todo:
        metadata = _dataset_metadata(dataset_id, metadata_retriever, datasets_dir)
        if metadata is None:
            logger.warning(
                "%s: dataset description and README unavailable; %d anchor(s) "
                "left for a later run instead of being judged blind",
                dataset_id,
                len(todo),
            )
            for i in todo:
                records[i] = _unjudged(i, METADATA_UNAVAILABLE_ERROR, _utcnow_iso())
            lookup_failed += len(todo)
            todo = []
        else:
            dataset_description = extract_dataset_text(metadata)
        for i in todo:
            ref = refs[i]
            judged_at = _utcnow_iso()
            paper_result = backend.get_paper(ref.identifier)
            if isinstance(paper_result, FetchError):
                error = (
                    f"paper_lookup_failed:{paper_result.reason}:{paper_result.detail}"
                )
                if error.startswith(PERMANENT_LOOKUP_ERROR_PREFIX):
                    unresolvable += 1
                else:
                    lookup_failed += 1
                records[i] = _unjudged(i, error, judged_at)
                continue
            assert isinstance(paper_result, FetchSuccess)  # noqa: S101 - upstream contract guard
            paper = paper_result.value
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
            prompts[i] = (paper, prompt, judged_at)

    judged = 0
    failed = 0
    if prompts:
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            futures = {
                i: pool.submit(_judge_prompt, client, prompt)
                for i, (_, prompt, _) in prompts.items()
            }
        for i, future in futures.items():
            ref = refs[i]
            paper, _, judged_at = prompts[i]
            verdict = future.result()
            if isinstance(verdict, LlmJudgmentError):
                failed += 1
                logger.warning(
                    "LLM judgment failed for %s / %s: %s",
                    dataset_id,
                    ref.identifier,
                    verdict,
                )
                records[i] = _unjudged(
                    i, f"llm_judgment_failed:{verdict}", judged_at, paper
                )
                continue
            judged += 1
            records[i] = JudgmentRecord(
                anchor_identifier=ref.identifier,
                anchor_identifier_type=ref.identifier_type,
                source_relation=ref.relation_type,
                classification=verdict["classification"],
                reason=verdict["reason"],
                paper_title=paper.title,
                paper_year=paper.year,
                paper_venue=paper.venue,
                judged_at=judged_at,
                error=None,
            )

    final = [r for r in records if r is not None]
    judged_at_latest = max((r.judged_at for r in final), default=None)
    if judged_at_latest is None:
        # No anchors: keep the previous timestamp for an unchanged empty sidecar.
        judged_at_latest = (
            previous.get("judged_at")
            if previous
            and not previous.get("judgments")
            and isinstance(previous.get("judged_at"), str)
            else _utcnow_iso()
        )
    return DatasetJudgmentRun(
        payload={
            "dataset_id": dataset_id,
            # The most recent judgment in the file, reused ones included.
            "judged_at": judged_at_latest,
            "judgment_model": client.model,
            "prompt_version": PROMPT_VERSION,
            "judgments": [r.to_dict() for r in final],
        },
        judged=judged,
        failed=failed,
        lookup_failed=lookup_failed,
        unresolvable=unresolvable,
        reused=reused,
    )


def _judge_prompt(
    client: ClaudeCliJudgmentClient, prompt: str
) -> dict[str, Any] | LlmJudgmentError:
    """Run one judgment on a worker thread; return the error instead of raising."""
    try:
        return client.judge_anchor(prompt)
    except LlmJudgmentError as exc:
        return exc


def save_judgment_sidecar(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically write a sidecar JSON to `path`.

    Writes to a sibling temp file in the same directory, fsyncs, then
    `os.replace`s into place. The sibling-directory choice keeps the
    rename a single filesystem operation (no cross-device copy).
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # mkstemp leaves the file for us to close and rename; we clean up on failure below.
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=".judgment-",
        suffix=".json.tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, target)
    except Exception:
        # Best-effort cleanup of the temp file on any write/replace failure.
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_path)
        raise


def load_judgment_sidecar(path: str | Path) -> dict[str, Any]:
    """Load a sidecar JSON from disk. Raises on missing file / invalid JSON.

    Returns the parsed dict as-is; callers that need typed access build
    `JudgmentRecord` instances from the `judgments[]` entries themselves.
    """
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: sidecar root is not a JSON object")
    return payload


def is_judgment_fresh(payload: dict[str, Any], *, max_age_days: int) -> bool:
    """Return True iff the sidecar's `judged_at` is within `max_age_days`.

    Strict (`<`), like `core/run_state.py::checked_within` since #80: a
    sidecar exactly `max_age_days` old is stale, so a cron whose period equals
    the window does not land on the boundary and skip forever.

    Robust against missing fields and unparseable timestamps: any failure
    to determine freshness returns False (re-judge).
    """
    if max_age_days <= 0:
        return False
    raw = payload.get("judged_at")
    if not isinstance(raw, str):
        return False
    try:
        when = datetime.fromisoformat(raw)
    except ValueError:
        return False
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    age = datetime.now(UTC) - when
    return age < timedelta(days=max_age_days)
