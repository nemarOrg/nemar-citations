"""Fetch and parse `.nemar/metadata.json` for nm-prefixed nemarDatasets repos.

The file follows a DataCite-style schema (v2.0) with a `related_identifiers`
block. We read anchors from a fixed allow-list of `relation_type` values:
References, IsDerivedFrom, IsIdenticalTo, IsVersionOf, IsDescribedBy.
`IsDescribedBy` is how a data paper is linked (e.g. 10.1038/sdata.2015.1
describes on000117). The relation is only a hint; the anchor judgment and gate
(`core.anchor_gate`) decide which anchors count. We skip URL-typed identifiers
(they exist for human navigation, not literature lookup).

Two fetch paths are supported. The default path GETs
`https://data.nemar.org/<id>/metadata.json`, which is the same neuroschema
document served from the worker without authentication or GitHub rate-limit
pressure. The legacy path pulls the file from `nemarDatasets/<id>/.nemar/metadata.json`
via the GitHub API and is retained as a fallback for IDs not yet covered
by the data.nemar.org route.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, cast, get_args

import requests
from github.GithubException import GithubException

from dataset_citations.sources.doi import (
    is_openneuro_dataset_doi,
    normalize_doi,
    validate_identifier,
)
from dataset_citations.sources.models import (
    DoiReference,
    FetchError,
    FetchResult,
    FetchSuccess,
    RelationType,
)
from dataset_citations.utils.github_client import build_github

logger = logging.getLogger(__name__)

_CITATION_RELATIONS: frozenset[RelationType] = frozenset(get_args(RelationType))

DEFAULT_DATA_API_BASE = "https://data.nemar.org"
DEFAULT_DATA_API_TIMEOUT = 15.0


@dataclass(frozen=True, slots=True)
class NemarDatasetMetadata:
    """Descriptive dataset fields pulled from `.nemar/metadata.json` (schema v2.1).

    `keywords` are flattened to plain strings; `funding` entries are normalized to
    `{funder_name, award_number?, award_title?}`. `methods_description` is only
    carried by the GitHub schema-2.0 doc, not the data.nemar.org neuroschema
    (schema 0.3.0), so it is `None` whenever rich metadata falls back to the data
    API. Empty instance = no rich metadata available (e.g. legacy ds-* datasets).
    """

    keywords: tuple[str, ...] = ()
    methods_description: str | None = None
    funding: tuple[dict[str, str], ...] = ()


EMPTY_NEMAR_DATASET_METADATA = NemarDatasetMetadata()


class NemarMetadataSource:
    """Reads `.nemar/metadata.json` and returns DOI references for opencite.

    The default fetch path uses `data.nemar.org/<id>/metadata.json`. If
    that returns 404, the source falls back to the GitHub git-tree read
    on `nemarDatasets/<id>/.nemar/metadata.json`. Set `prefer_data_api=False`
    to skip the data API entirely (useful for offline development and the
    test environment).
    """

    def __init__(
        self,
        github_token: str | None = None,
        *,
        prefer_data_api: bool = True,
        data_api_base: str = DEFAULT_DATA_API_BASE,
        data_api_timeout: float = DEFAULT_DATA_API_TIMEOUT,
    ) -> None:
        self.github = build_github(github_token)
        self.prefer_data_api = prefer_data_api
        self.data_api_base = data_api_base.rstrip("/")
        self.data_api_timeout = data_api_timeout

    def get_doi_references(
        self, dataset_id: str, org: str = "nemarDatasets"
    ) -> FetchResult[list[DoiReference]]:
        """Return DOI/PMID anchors for one nm-dataset, or a FetchError.

        Empty success (`FetchSuccess([])`) means the metadata file was parsed
        but contained no DOI-typed related_identifiers in our allow-list.
        That is distinct from `FetchError("not_found", ...)` (no metadata
        file via either route).

        Order of attempts:
          1. `data.nemar.org/<id>/metadata.json` (unless `prefer_data_api=False`).
          2. GitHub git tree on `<org>/<dataset_id>/.nemar/metadata.json`.
        Transient or absence-style errors from path 1 (not_found, rate_limit,
        network) fall back to path 2 so a 30-second worker blip doesn't drop
        every dataset in the window. Parse errors and unclassified failures
        short-circuit so the operator can investigate a malformed payload.
        """
        if self.prefer_data_api:
            data_api_result = self._fetch_from_data_api(dataset_id)
            if isinstance(data_api_result, FetchSuccess):
                return FetchSuccess(parse_nemar_metadata(data_api_result.value))
            if data_api_result.reason not in {"not_found", "rate_limit", "network"}:
                # Parse / auth / unknown errors should not silently fall back;
                # surface them so the operator can investigate.
                return data_api_result
            logger.info(
                "data.nemar.org %s for %s (%s); falling back to GitHub",
                data_api_result.reason,
                dataset_id,
                data_api_result.detail,
            )
        return self._fetch_from_github(dataset_id, org)

    def _fetch_from_data_api(self, dataset_id: str) -> FetchResult[dict[str, Any]]:
        url = f"{self.data_api_base}/{dataset_id}/metadata.json"
        try:
            response = requests.get(url, timeout=self.data_api_timeout)
        except requests.exceptions.RequestException as exc:
            return FetchError("network", f"GET {url}: {exc}")

        if response.status_code == 404:
            return FetchError("not_found", f"{url} returned 404")
        if response.status_code == 429:
            return FetchError("rate_limit", f"{url} returned 429")
        if response.status_code != 200:
            return FetchError(
                "other",
                f"GET {url}: HTTP {response.status_code} {response.text[:200]}",
            )

        try:
            payload = response.json()
        except ValueError as exc:
            return FetchError("parse", f"invalid JSON from {url}: {exc}")
        if not isinstance(payload, dict):
            return FetchError("parse", f"{url} root is not an object")
        return FetchSuccess(payload)

    def _fetch_from_github(
        self, dataset_id: str, org: str
    ) -> FetchResult[list[DoiReference]]:
        result = self._github_payload(dataset_id, org)
        if isinstance(result, FetchSuccess):
            return FetchSuccess(parse_nemar_metadata(result.value))
        return result

    def _github_payload(self, dataset_id: str, org: str) -> FetchResult[dict[str, Any]]:
        """Read the raw `.nemar/metadata.json` dict from GitHub, or a FetchError.

        Shared by the DOI-reference path (`_fetch_from_github` maps the payload
        through `parse_nemar_metadata`) and the rich-metadata path
        (`get_dataset_metadata` maps it through `parse_nemar_dataset_metadata`),
        so the GitHub file is decoded once per concern with one set of error
        rules. Only reads `self.github`.
        """
        try:
            repo = self.github.get_repo(f"{org}/{dataset_id}")
        except GithubException as e:
            return _github_error(e, f"{org}/{dataset_id}")

        try:
            content = repo.get_contents(".nemar/metadata.json")
        except GithubException as e:
            if getattr(e, "status", None) == 404:
                return FetchError("not_found", ".nemar/metadata.json missing")
            return _github_error(e, ".nemar/metadata.json")

        if isinstance(content, list):
            return FetchError("parse", "expected file, got directory listing")
        # `decoded_content` asserts the blob is base64-encoded; a >1MB
        # .nemar/metadata.json comes back with encoding=None and the property
        # raises AssertionError, which hasattr() propagates. Guard it so one
        # oversized blob is a per-dataset parse failure, not a pipeline abort
        # (same fix as bids_metadata.get_doi_references; see issue #168).
        try:
            raw = content.decoded_content
        except (AssertionError, ValueError, AttributeError) as e:
            return FetchError("parse", f"could not decode .nemar/metadata.json: {e}")
        if raw is None:
            return FetchError("parse", ".nemar/metadata.json has no decodable content")
        try:
            payload: Any = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            return FetchError("parse", f"invalid JSON: {e}")

        if not isinstance(payload, dict):
            return FetchError("parse", "metadata.json root is not an object")
        return FetchSuccess(payload)

    def get_dataset_metadata(
        self, dataset_id: str, org: str = "nemarDatasets"
    ) -> NemarDatasetMetadata:
        """Return keywords / methods_description / funding for one nm/on dataset.

        Prefers the GitHub `.nemar/metadata.json` (schema 2.0): it carries the
        latest producer-side fields, including `methods_description`, which the
        data.nemar.org neuroschema (schema 0.3.0) does NOT expose and which ships
        stale tag-versioned data. Falls back to the data.nemar.org payload for
        keywords + funding when GitHub is unavailable (methods_description stays
        None in that case). Returns `EMPTY_NEMAR_DATASET_METADATA` on total
        failure, logged: rich metadata is provenance, never a hard dependency.

        Distinct from `get_doi_references`, which prefers the data API for the
        anchor extraction; the source-of-truth preference is deliberately
        inverted here because only GitHub has the freshest descriptive fields.
        """
        github_payload = self._github_payload(dataset_id, org)
        if isinstance(github_payload, FetchSuccess):
            return parse_nemar_dataset_metadata(github_payload.value)

        logger.info(
            "GitHub .nemar/metadata.json unavailable for %s (%s: %s); "
            "falling back to data.nemar.org for rich metadata",
            dataset_id,
            github_payload.reason,
            github_payload.detail,
        )
        if self.prefer_data_api:
            data_api = self._fetch_from_data_api(dataset_id)
            if isinstance(data_api, FetchSuccess):
                return parse_nemar_dataset_metadata(data_api.value)
            # Both sources failed: rich metadata is dropped for this dataset, a
            # production degradation that should stand out in a 594-dataset run.
            logger.warning(
                "rich metadata unavailable for %s from GitHub and data.nemar.org "
                "(%s); keywords/funding will be absent from output",
                dataset_id,
                data_api.reason,
            )
        return EMPTY_NEMAR_DATASET_METADATA


def parse_nemar_metadata(payload: dict[str, Any]) -> list[DoiReference]:
    """Pure function: convert parsed `.nemar/metadata.json` into DoiReferences.

    Exposed for unit testing with fixture files. The GitHub-fetching class
    delegates here after decoding bytes to JSON.
    """
    related = payload.get("related_identifiers") or []
    refs: list[DoiReference] = []
    for entry in related:
        ref = _entry_to_reference(entry)
        if ref is not None:
            refs.append(ref)
    return refs


def parse_nemar_dataset_metadata(payload: dict[str, Any]) -> NemarDatasetMetadata:
    """Pure function: extract keywords / methods_description / funding (schema v2.1).

    Tolerates both `.nemar/metadata.json` shapes:
      * GitHub schema 2.0: `funding_references[]`, carries `methods_description`.
      * data.nemar.org neuroschema 0.3.0: `funding[]`, no `methods_description`.
    `keywords` entries are `{"term": str}` in both shapes (a bare string is also
    accepted); each is flattened to its term. Exposed for unit testing against
    fixture files; `get_dataset_metadata` delegates here after a fetch.
    """
    methods = payload.get("methods_description")
    methods_description = (
        methods if isinstance(methods, str) and methods.strip() else None
    )
    raw_funding = payload.get("funding_references")
    if not isinstance(raw_funding, list):
        raw_funding = payload.get("funding")
    return NemarDatasetMetadata(
        keywords=_parse_keywords(payload.get("keywords")),
        methods_description=methods_description,
        funding=_parse_funding(raw_funding),
    )


def _parse_keywords(raw: Any) -> tuple[str, ...]:
    # `None` = key absent (the common, expected case). A present-but-non-list
    # value is a producer-side schema mismatch worth a WARN: it would otherwise
    # silently zero out keywords for every affected dataset.
    if raw is None:
        return ()
    if not isinstance(raw, list):
        logger.warning("keywords is not a list (got %s); dropping", type(raw).__name__)
        return ()
    terms: list[str] = []
    for entry in raw:
        term = entry.get("term") if isinstance(entry, dict) else entry
        if isinstance(term, str) and term.strip():
            terms.append(term.strip())
    return tuple(terms)


def _parse_funding(raw: Any) -> tuple[dict[str, str], ...]:
    """Normalize funding entries to `{funder_name, award_number?, award_title?}`.

    Drops entries without a non-empty string `funder_name` and omits null/empty
    optional keys so the serialized JSON stays minimal and deterministic. A
    present-but-non-list `raw` is a schema mismatch and logs a WARN (vs `None`,
    which just means the key is absent).
    """
    if raw is None:
        return ()
    if not isinstance(raw, list):
        logger.warning("funding is not a list (got %s); dropping", type(raw).__name__)
        return ()
    records: list[dict[str, str]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        funder_name = entry.get("funder_name")
        if not isinstance(funder_name, str) or not funder_name.strip():
            continue
        record: dict[str, str] = {"funder_name": funder_name.strip()}
        for key in ("award_number", "award_title"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                record[key] = value.strip()
        records.append(record)
    return tuple(records)


def _entry_to_reference(entry: Any) -> DoiReference | None:
    if not isinstance(entry, dict):
        return None
    relation = entry.get("relation_type")
    if relation not in _CITATION_RELATIONS:
        return None
    identifier_type = entry.get("identifier_type")
    raw = entry.get("identifier")
    if not isinstance(raw, str) or not raw.strip():
        return None
    if identifier_type != "DOI":
        # URL / handle / other types: skip for opencite lookup.
        return None

    normalized = normalize_doi(raw)
    if not validate_identifier(normalized):
        logger.warning("skipping malformed DOI %r in .nemar/metadata.json", raw)
        return None
    if is_openneuro_dataset_doi(normalized):
        # OpenNeuro DatasetDOIs are not indexed in OpenAlex; skip from lookup set.
        return None
    return DoiReference(
        identifier=normalized,
        identifier_type="doi",
        relation_type=cast(RelationType, relation),
        source="nemar_metadata",
        source_field="related_identifiers",
    )


def _github_error(exc: GithubException, path: str) -> FetchError:
    status = getattr(exc, "status", None)
    if status == 404:
        return FetchError("not_found", f"{path} not found")
    if status == 401 or status == 403:
        return FetchError("auth", f"{path}: {exc}")
    if status == 429:
        return FetchError("rate_limit", f"{path}: {exc}")
    return FetchError("other", f"{path}: {exc}")
