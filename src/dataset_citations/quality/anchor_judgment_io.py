"""Read-side helpers for the phase 2 anchor-judgment sidecar.

Phase 2 (#86) writes a per-dataset JSON file at
`citations/anchor_judgments/<dataset_id>.json` describing the LLM's
classification of every DOI/PMID/arXiv anchor the pipeline extracts. Phase 3
(this module + `core/opencite_pipeline.py`) reads those judgments and uses
them to bucket anchors into "fetch citations from this anchor" (the single
`data_paper` per dataset) vs "record as context only" (umbrella, methodology,
related_work, irrelevant).

Why a separate file from phase 2's writer:
  * Phase 2 ships its own `quality/anchor_judgment.py` module that owns the
    write path (sidecar schema, batch runner, retry policy). Splitting the
    read-only helper into its own file means phase 3 can land before phase 2
    without merge conflict on a shared module, and the read API stays small
    and obviously side-effect free.
  * The locked sidecar schema (documented inline below) is the contract; this
    module never talks to the judge and never mutates anything on disk.

The single public function `load_judgment_lookup` returns a mapping from
canonicalized anchor identifier (DOI/PMID/arXiv in the same shape used by
`DoiReference.identifier`) to one of the five classifications in
`llm_client.ALLOWED_CLASSIFICATIONS`. Anchors with a non-null `error` field
are logged at WARN and treated as if they were absent from the sidecar, so
the fail-closed anchor gate (`core.anchor_gate`, issue #241) keeps them as
context only: a failed judgment never lets an anchor's citers count.

Out-of-taxonomy classifications are also logged at WARN and dropped, so a
single malformed sidecar entry can't poison the whole dataset's run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dataset_citations.quality.llm_client import ALLOWED_CLASSIFICATIONS
from dataset_citations.sources.doi import normalize_doi

logger = logging.getLogger(__name__)

DEFAULT_JUDGMENTS_DIR = Path("citations/anchor_judgments")


@dataclass(frozen=True, slots=True)
class JudgmentSidecar:
    """Parsed view of a `citations/anchor_judgments/<id>.json` sidecar.

    `lookup` maps the canonicalized identifier (matching the shape used by
    `DoiReference.identifier`) to the classification string. `model` is the
    `judgment_model` field from the sidecar header, surfaced into schema-v2's
    `metadata.anchor_judgment_model` so consumers can tell which LLM made
    each call. `context_details` carries the per-anchor metadata
    (paper_title, reason, source_relation, identifier_type) that the
    pipeline folds into schema-v2.1's `metadata.anchors[]` (every anchor,
    with a `kept` flag). `present` distinguishes "sidecar exists and
    parsed cleanly" from "sidecar missing"; the pipeline names the missing
    sidecar in its per-dataset unjudged-anchor warning.
    """

    present: bool
    model: str | None = None
    lookup: dict[str, str] = field(default_factory=dict)
    context_details: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_judgment_sidecar(
    dataset_id: str,
    *,
    judgments_dir: Path | str = DEFAULT_JUDGMENTS_DIR,
) -> JudgmentSidecar:
    """Read the sidecar for `dataset_id`; return an empty record on miss.

    Returns a `JudgmentSidecar` with `present=False` and empty fields when:
      * the sidecar file does not exist
      * the file exists but does not parse as JSON
      * the file parses but lacks the expected top-level shape

    Out-of-taxonomy classifications are dropped with a WARN; the rest of the
    sidecar still applies. Anchors with `error != null` are similarly
    dropped with a WARN, so the anchor gate treats them as unjudged.
    """
    path = Path(judgments_dir) / f"{dataset_id}.json"
    if not path.exists():
        return JudgmentSidecar(present=False)

    try:
        with path.open(encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "could not read anchor-judgment sidecar for %s at %s: %s",
            dataset_id,
            path,
            exc,
        )
        return JudgmentSidecar(present=False)

    if not isinstance(raw, dict):
        logger.warning(
            "anchor-judgment sidecar for %s at %s is not a JSON object; ignoring",
            dataset_id,
            path,
        )
        return JudgmentSidecar(present=False)

    model = (
        raw.get("judgment_model")
        if isinstance(raw.get("judgment_model"), str)
        else None
    )
    judgments = raw.get("judgments")
    if not isinstance(judgments, list):
        logger.warning(
            "anchor-judgment sidecar for %s at %s has no 'judgments' list; ignoring",
            dataset_id,
            path,
        )
        return JudgmentSidecar(present=False, model=model)

    lookup: dict[str, str] = {}
    context_details: dict[str, dict[str, Any]] = {}
    for entry in judgments:
        if not isinstance(entry, dict):
            logger.warning(
                "skipping non-dict judgment entry in %s: %r",
                path,
                entry,
            )
            continue
        identifier = entry.get("anchor_identifier")
        identifier_type = entry.get("anchor_identifier_type")
        if not isinstance(identifier, str) or not isinstance(identifier_type, str):
            logger.warning(
                "skipping judgment entry with missing identifier in %s: %r",
                path,
                entry,
            )
            continue
        error = entry.get("error")
        if error:
            logger.warning(
                "skipping errored judgment for %s anchor %s in %s: %s",
                dataset_id,
                identifier,
                path,
                error,
            )
            continue
        classification = entry.get("classification")
        if classification not in ALLOWED_CLASSIFICATIONS:
            logger.warning(
                "dropping out-of-taxonomy classification %r for %s anchor %s in %s",
                classification,
                dataset_id,
                identifier,
                path,
            )
            continue
        key = _canonical_anchor_key(identifier, identifier_type)
        if key is None:
            logger.warning(
                "skipping judgment entry with unsupported identifier_type %r in %s",
                identifier_type,
                path,
            )
            continue
        lookup[key] = classification
        context_details[key] = {
            "anchor_identifier": identifier,
            "anchor_identifier_type": identifier_type,
            "source_relation": entry.get("source_relation"),
            "classification": classification,
            "reason": entry.get("reason"),
            "paper_title": entry.get("paper_title"),
            "paper_year": entry.get("paper_year"),
            "paper_venue": entry.get("paper_venue"),
        }

    return JudgmentSidecar(
        present=True,
        model=model,
        lookup=lookup,
        context_details=context_details,
    )


def load_judgment_lookup(
    dataset_id: str,
    *,
    judgments_dir: Path | str = DEFAULT_JUDGMENTS_DIR,
) -> dict[str, str]:
    """Return `{canonical_anchor_identifier: classification}` for `dataset_id`.

    Returns an empty dict when the sidecar is missing or malformed. This is
    the spec-requested signature; callers that also need the model name or
    the per-anchor context detail should use `load_judgment_sidecar`.
    """
    return load_judgment_sidecar(dataset_id, judgments_dir=judgments_dir).lookup


def canonical_anchor_key(identifier: str, identifier_type: str) -> str | None:
    """Canonicalize an `(identifier, identifier_type)` pair for sidecar lookup.

    Public re-export of the internal canonicalizer so the pipeline can build
    the same lookup key from a `DoiReference` it already has in hand. Returns
    `None` for unsupported `identifier_type` values, matching the loader's
    behavior.
    """
    return _canonical_anchor_key(identifier, identifier_type)


def _canonical_anchor_key(identifier: str, identifier_type: str) -> str | None:
    """Build the lookup key shared between the sidecar and the pipeline.

    The shape must match `DoiReference.identifier`:
      * DOIs: bare lowercased DOI (no prefix), as produced by `normalize_doi`.
      * PMIDs: `pmid:<digits>`.
      * arXiv: `arxiv:<id>`.
    """
    kind = identifier_type.lower()
    raw = identifier.strip()
    if not raw:
        return None
    if kind == "doi":
        return normalize_doi(raw)
    if kind == "pmid":
        body = raw.split(":", 1)[1] if raw.lower().startswith("pmid:") else raw
        return f"pmid:{body.strip().lower()}"
    if kind == "arxiv":
        body = raw.split(":", 1)[1] if raw.lower().startswith("arxiv:") else raw
        return f"arxiv:{body.strip().lower()}"
    return None
