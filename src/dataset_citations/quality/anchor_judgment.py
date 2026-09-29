"""Per-dataset LLM anchor adjudication storage + orchestration.

Phase 2 of epic #76. Productizes the phase-1 probe (`scripts/probe_anchor_judgment.py`)
into a real per-dataset sidecar file at `citations/anchor_judgments/<id>.json`
and the assembly function the batch CLI loops over.

Sidecar schema (locked; phase 3 reads this):

  {
    "dataset_id": "<id>",
    "judged_at": "<ISO-8601 UTC, most recent judgment in this file>",
    "judgment_model": "<judge model id, e.g. claude-sonnet-5-5>",
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
)
from dataset_citations.quality.llm_client import (
    ALLOWED_CLASSIFICATIONS,
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


@dataclass(frozen=True, slots=True)
class DatasetJudgmentRun:
    """Outcome of judging one dataset.

    `payload` is the sidecar dict to write. `judged` / `failed` count the LLM
    calls made in this run, `lookup_failed` the anchors opencite could not
    resolve (no call made), and `reused` the anchors whose previous same-model
    judgment was kept without a call.
    """

    payload: dict[str, Any]
    judged: int
    failed: int
    lookup_failed: int
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


def _reusable_judgments(
    previous: dict[str, Any] | None, model: str
) -> dict[str, JudgmentRecord]:
    """Successful judgments from the previous sidecar made by `model`.

    Judgments by any other model are never reused: switching the judge (gemma
    to Claude, issue #241) must re-judge every anchor rather than let the old
    model's verdicts keep counting.
    """
    if not previous or previous.get("judgment_model") != model:
        return {}
    out: dict[str, JudgmentRecord] = {}
    for entry in previous.get("judgments") or []:
        if not isinstance(entry, dict) or entry.get("error"):
            continue
        if entry.get("classification") not in ALLOWED_CLASSIFICATIONS:
            continue
        identifier = entry.get("anchor_identifier")
        identifier_type = entry.get("anchor_identifier_type")
        if not isinstance(identifier, str) or not isinstance(identifier_type, str):
            continue
        key = canonical_anchor_key(identifier, identifier_type)
        if key is None:
            continue
        try:
            out[key] = JudgmentRecord(**{f: entry.get(f) for f in _RECORD_FIELDS})
        except TypeError:
            continue
    return out


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
) -> DatasetJudgmentRun | None:
    """Judge one dataset's DOI anchors; return the run, or None on source failure.

    None means "do not touch the sidecar": when the anchor source cannot be
    read (rate limit, outage) there is nothing to judge, and overwriting the
    existing sidecar with an empty one would silently discard good judgments
    (301 datasets lost theirs this way on 2026-09-26, issue #241).

    `previous` is the sidecar currently on disk. An anchor whose previous
    judgment is a success by the same model under the same `source_relation`
    is reused without a call. When a new call fails, a previous same-model
    success for that anchor is kept instead of the error, so a flaky night
    never downgrades an anchor that was already judged. Paper lookups run
    sequentially (opencite shares one rate limiter per process); the LLM calls
    run on up to `max_workers` threads.
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
    refs: list[DoiReference] = [
        r for r in refs_result.value if r.identifier_type == "doi"
    ]
    if not refs:
        logger.info("%s: no DOI anchors; writing empty judgments sidecar", dataset_id)
        return DatasetJudgmentRun(
            payload={
                "dataset_id": dataset_id,
                "judged_at": _utcnow_iso(),
                "judgment_model": client.model,
                "judgments": [],
            },
            judged=0,
            failed=0,
            lookup_failed=0,
            reused=0,
        )

    reusable = _reusable_judgments(previous, client.model)
    records: list[JudgmentRecord | None] = [None] * len(refs)
    fallback: dict[int, JudgmentRecord] = {}
    todo: list[int] = []
    for i, ref in enumerate(refs):
        prior = reusable.get(canonical_anchor_key(ref.identifier, "doi") or "")
        if prior is not None and prior.source_relation == ref.relation_type:
            records[i] = prior
        else:
            todo.append(i)
            if prior is not None:
                fallback[i] = prior
    reused = len(refs) - len(todo)

    prompts: dict[int, tuple[Any, str, str]] = {}
    lookup_failed = 0
    if todo:
        metadata = metadata_retriever.get_dataset_metadata(dataset_id)
        dataset_description = extract_dataset_text(metadata)
        for i in todo:
            ref = refs[i]
            judged_at = _utcnow_iso()
            paper_result = backend.get_paper(ref.identifier)
            if isinstance(paper_result, FetchError):
                lookup_failed += 1
                records[i] = fallback.get(i) or _error_record(
                    ref,
                    judged_at=judged_at,
                    error=f"paper_lookup_failed:{paper_result.reason}:{paper_result.detail}",
                )
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
                records[i] = fallback.get(i) or _error_record(
                    ref,
                    judged_at=judged_at,
                    error=f"llm_judgment_failed:{verdict}",
                    paper=paper,
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
    return DatasetJudgmentRun(
        payload={
            "dataset_id": dataset_id,
            # The most recent judgment in the file, reused ones included.
            "judged_at": max(r.judged_at for r in final),
            "judgment_model": client.model,
            "judgments": [r.to_dict() for r in final],
        },
        judged=judged,
        failed=failed,
        lookup_failed=lookup_failed,
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
    # delete=False so we can close and rename; we clean up on failure below.
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

    Mirrors the freshness semantics from `core/run_state.py::checked_within`
    (which currently uses `<=`; issue #80 tracks the off-by-one fix). We
    keep `<=` here so phase-2 freshness behaves identically to the existing
    citation freshness gate; once #80 lands, both gates should be updated
    together.

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
    return age <= timedelta(days=max_age_days)
