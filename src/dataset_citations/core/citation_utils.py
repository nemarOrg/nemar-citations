"""
Utility functions for handling citation data in JSON format.

This module provides helpers to read, compare, and stamp the canonical
opencite citation JSON (schema v2.1). The legacy pre-opencite write path
(`save_citation_json` / `create_citation_json_structure`) and the pickle->json
`migrate` helper were removed in epic #180 phase 5 along with `citations/json/`
and `citations/pickle/`; `citations/json_opencite/` is now the sole source.

Copyright (c) 2024 Seyed Yahya Shirazi (neuromechanist)
All rights reserved.

Author: Seyed Yahya Shirazi
GitHub: https://github.com/neuromechanist
Email: shirazi@ieee.org
"""

import contextlib
import copy
import json
import logging
import os
import tempfile
from typing import Any, Literal

logger = logging.getLogger(__name__)

DiscoveryBackend = Literal["opencite"]
SCHEMA_VERSION_V2 = "2.0"
# Schema v2.1 (epic #180 phase 3): the per-dataset citation JSON is fully
# self-describing. `metadata.anchors[]` is the superset of every anchor (kept +
# context, replacing the old `context_anchors[]`); `metadata.searched_dois[]`,
# `keywords`, `methods_description`, and `funding` are added. Additive over v2.0
# except for the `context_anchors[]` -> `anchors[]` rename.
SCHEMA_VERSION_V2_1 = "2.1"

# Timestamp fields that advance on every pipeline run regardless of whether the
# citation content changed. They are excluded from content-equality checks so a
# re-fetch / re-score that produced identical data does not rewrite the file and
# churn the nightly git diff (issue #165).
_VOLATILE_TIMESTAMP_PATHS: tuple[tuple[str, ...], ...] = (
    ("date_last_updated",),
    ("metadata", "fetch_date"),
    ("confidence_scoring", "scoring_date"),
)


def strip_volatile_timestamps(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of ``payload`` with per-run timestamp fields removed.

    The result is only ever used to compare two citation payloads for
    substantive equality; it is never written to disk. Removing
    ``date_last_updated``, ``metadata.fetch_date`` and
    ``confidence_scoring.scoring_date`` lets the update step detect when a
    re-fetch produced identical content and leave the file (and its timestamps)
    untouched, so ``date_last_updated`` advances only on a real change
    (issue #165).
    """
    clone = copy.deepcopy(payload)
    for *parents, leaf in _VOLATILE_TIMESTAMP_PATHS:
        node: Any = clone
        for key in parents:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(leaf, None)
    return clone


def write_citation_json_if_changed(filepath: str, payload: dict[str, Any]) -> bool:
    """Write `payload` as pretty JSON unless only volatile timestamps differ.

    Compares against the file already on disk after stripping volatile
    timestamps; when the substantive content matches, the existing file is left
    byte-for-byte untouched so the nightly pipeline emits no spurious diff
    (issue #165).

    Returns:
        True if the file was written, False if left unchanged.

    Raises:
        OSError: If the file exists but cannot be written.
    """
    if os.path.isfile(filepath):
        try:
            with open(filepath, encoding="utf-8") as f:
                existing = json.load(f)
        except (OSError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict) and strip_volatile_timestamps(
            existing
        ) == strip_volatile_timestamps(payload):
            return False
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return True


def write_json_atomic(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    """Write `payload` as pretty JSON (trailing newline) without a torn file.

    Writes a sibling temp file, fsyncs it, then `os.replace`s it into place, so
    a crash or a full disk leaves the previous file intact instead of a
    truncated one. Raises OSError on failure after removing the temp file.
    """
    target = os.fspath(path)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(target)}.",
        suffix=".tmp",
        dir=os.path.dirname(target) or ".",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_path)
        raise


def add_discovery_provenance(
    citation_json: dict[str, Any],
    *,
    discovery_backend: DiscoveryBackend,
    schema_version: str = SCHEMA_VERSION_V2_1,
    anchor_judgment_model: str | None = None,
    anchors: list | None = None,
    searched_dois: list | None = None,
) -> dict[str, Any]:
    """Stamp a citation JSON with Phase 3 discovery-provenance markers.

    Mutates and returns the input dict. Always overwrites the top-level
    `metadata.schema_version` and `metadata.discovery_backend` keys: callers
    are explicitly declaring the schema/backend identity of this output, so
    any prior values were either stale or wrong. Per-citation
    `discovery_backend` is set only when missing (`entry.setdefault`) so the
    orchestrator can pre-populate the field with finer-grained labels in the
    future without this helper clobbering them.

    Per-citation `source_doi` and `source_relation` are NOT set here. The
    opencite orchestrator writes them per-citation while building the JSON
    because it alone knows which anchor surfaced each citing work. The
    helper is only responsible for the top-level provenance markers and the
    per-citation `discovery_backend` label.

    Phase 3 (#87) extends the schema with anchor-adjudication provenance:
      * `metadata.anchor_judgment_model` carries the judge model from the
        sidecar (e.g. ``"claude-sonnet-5-5"``) so consumers can tell which model
        judged each dataset's anchors. Falsy values are normalized to ``None``
        and persisted, since "no usable sidecar" is itself a meaningful signal
        for downstream code.
      * `metadata.anchors` (schema v2.1, epic #180) is the list of EVERY anchor
        extracted for the dataset, kept or not. Each entry is a dict with
        ``identifier``, ``identifier_type``, ``source_relation``,
        ``classification`` (the judge's verdict; ``None`` if unjudged),
        ``kept`` (``True`` = fetched for citations, ``False`` = context-only),
        ``kept_reason`` (the anchor gate's reason: ``own_doi`` /
        ``dataset_record`` / ``judged_data_paper`` / ``judged_not_data_paper``
        / ``never_anchor`` / ``unjudged`` / ``awaiting_fetch``, see
        `core.anchor_gate`), ``paper_title``, ``paper_year``, ``paper_venue``,
        ``reason``, and ``judgment_model``. This REPLACES the v2.0
        ``context_anchors`` key (which was only the ``kept=False`` subset);
        consumers select context anchors via
        ``[a for a in anchors if not a["kept"]]``.
      * `metadata.searched_dois` (schema v2.1) is the flat list of DOI anchors
        actually sent to the citation backend (the ``kept`` DOIs), the citation
        analog of ``metadata.searched_accessions`` (which the later
        find-mentions step writes for the accession full-text search). The two
        are disjoint provenance keys and never mix.
    """
    metadata = citation_json.setdefault("metadata", {})
    metadata["schema_version"] = schema_version
    metadata["discovery_backend"] = discovery_backend
    metadata["anchor_judgment_model"] = anchor_judgment_model or None
    metadata["anchors"] = list(anchors) if anchors else []
    metadata["searched_dois"] = list(searched_dois) if searched_dois else []
    for entry in citation_json.get("citation_details", []) or []:
        entry.setdefault("discovery_backend", discovery_backend)
    return citation_json


def stamp_dataset_metadata(
    citation_json: dict[str, Any],
    *,
    keywords: list | None = None,
    methods_description: str | None = None,
    funding: list | None = None,
) -> dict[str, Any]:
    """Write the schema-v2.1 descriptive dataset fields into ``metadata``.

    Separate from `add_discovery_provenance` because these are dataset-describing
    facts (pulled from ``.nemar/metadata.json``), not discovery provenance. Keys
    are always written (empty defaults for legacy ds-* datasets that carry none)
    so a consumer gating on schema v2.1 can assume their presence. Deterministic
    and idempotent: repeated calls with the same inputs produce an identical dict,
    so the issue-#165 content-equality write gate never churns on them.

    Mutates and returns the input dict.
    """
    metadata = citation_json.setdefault("metadata", {})
    metadata["keywords"] = list(keywords) if keywords else []
    metadata["methods_description"] = methods_description or None
    metadata["funding"] = list(funding) if funding else []
    return citation_json


def load_citation_json(filepath: str) -> dict[str, Any]:
    """
    Load citation data from JSON file.

    Args:
        filepath (str): Path to JSON file

    Returns:
        Dict[str, Any]: Citation data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)  # type: ignore
    except FileNotFoundError:
        logger.error(f"Citation JSON file not found: {filepath}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {filepath}: {e}")
        raise


def get_citation_summary_from_json(json_filepath: str) -> dict[str, Any]:
    """
    Extract summary information from a citation JSON file.

    Args:
        json_filepath (str): Path to citation JSON file

    Returns:
        Dict[str, Any]: Summary with keys: dataset_id, num_citations,
                       total_cumulative_citations, date_last_updated
    """
    citation_data = load_citation_json(json_filepath)

    return {
        "dataset_id": citation_data.get("dataset_id"),
        "num_citations": citation_data.get("num_citations", 0),
        "total_cumulative_citations": citation_data.get("metadata", {}).get(
            "total_cumulative_citations", 0
        ),
        "date_last_updated": citation_data.get("date_last_updated"),
    }


def load_citations_from_json(file_path: str) -> dict[str, Any]:
    """
    Load citation data from a JSON file.

    Args:
        file_path: Path to the citation JSON file

    Returns:
        Dictionary containing citation data structure

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        logger.debug(f"Loaded citations from {file_path}")
        return data

    except FileNotFoundError:
        logger.error(f"Citation file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in citation file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading citation file {file_path}: {e}")
        raise
