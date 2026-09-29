"""Tests for nemar_metadata parsing.

Loads real .nemar/metadata.json snapshots from tests/test_data/sources/nemar/
and runs them through the pure parse_nemar_metadata function. No mocks, no
network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import get_args
from unittest import TestCase

from dataset_citations.sources.models import FetchError, FetchSuccess, RelationType
from dataset_citations.sources.nemar_metadata import (
    EMPTY_NEMAR_DATASET_METADATA,
    NemarMetadataSource,
    parse_nemar_dataset_metadata,
    parse_nemar_metadata,
)

FIXTURE_DIR = Path(__file__).parent / "test_data" / "sources" / "nemar"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text())


class ParseNm000103(TestCase):
    """nm000103 (Healthy Brain Network EEG) carries References, IsDescribedBy,
    and IsIdenticalTo entries (verified against the checked-in fixture)."""

    def setUp(self) -> None:
        self.payload = load_fixture("nm000103.metadata.json")
        self.refs = parse_nemar_metadata(self.payload)

    def test_emits_only_citation_relation_types(self) -> None:
        # nm000103's IsDescribedBy entries are URL-typed, so they're dropped by
        # the URL filter (not the relation allow-list, which now permits
        # IsDescribedBy since #181). IsSupplementedBy is not a consumed relation.
        rel_types = {r.relation_type for r in self.refs}
        self.assertNotIn("IsSupplementedBy", rel_types)
        self.assertTrue(
            rel_types.issubset(
                {
                    "References",
                    "IsDerivedFrom",
                    "IsIdenticalTo",
                    "IsVersionOf",
                    "IsDescribedBy",
                }
            )
        )

    def test_skips_url_identifiers(self) -> None:
        # All emitted references must be DOI-typed; URL entries are dropped.
        for ref in self.refs:
            self.assertEqual(ref.identifier_type, "doi")

    def test_includes_known_references_dois(self) -> None:
        identifiers = {r.identifier for r in self.refs}
        self.assertIn("10.1038/sdata.2017.181", identifiers)
        self.assertIn("10.1038/sdata.2017.40", identifiers)

    def test_isidenticalto_zenodo_kept(self) -> None:
        ids = [r for r in self.refs if r.identifier == "10.5281/zenodo.17306881"]
        self.assertEqual(len(ids), 1)
        self.assertEqual(ids[0].relation_type, "IsIdenticalTo")

    def test_source_field(self) -> None:
        for ref in self.refs:
            self.assertEqual(ref.source, "nemar_metadata")
            self.assertEqual(ref.source_field, "related_identifiers")


class ParseNm000115(TestCase):
    """Second fixture for relation_type variety."""

    def setUp(self) -> None:
        self.payload = load_fixture("nm000115.metadata.json")
        self.refs = parse_nemar_metadata(self.payload)

    def test_emits_at_least_one_reference(self) -> None:
        self.assertGreater(len(self.refs), 0)
        # nm000115's IsDescribedBy entries are URL-typed (dropped); we expect a
        # DOI-typed References/related entry.
        rel_types = {r.relation_type for r in self.refs}
        self.assertTrue(rel_types & {"References", "IsDerivedFrom", "IsIdenticalTo"})


class RelationTypeMembers(TestCase):
    """Pin the canonical RelationType members. The allow-lists derive from
    get_args(RelationType), so this is the one place that asserts the literal's
    exact contents and would catch a typo (e.g. "IsDescribedBY") or an
    accidental removal that the derived tests cannot."""

    def test_exact_members(self) -> None:
        self.assertEqual(
            set(get_args(RelationType)),
            {
                "References",
                "IsDerivedFrom",
                "IsIdenticalTo",
                "IsVersionOf",
                "IsDescribedBy",
                "IsSupplementTo",
            },
        )


class ParseSyntheticEdgeCases(TestCase):
    """Hand-crafted edge cases that aren't easy to find in the real fixtures."""

    def test_missing_related_identifiers(self) -> None:
        refs = parse_nemar_metadata({"version": "2.0"})
        self.assertEqual(refs, [])

    def test_empty_related_identifiers(self) -> None:
        refs = parse_nemar_metadata({"related_identifiers": []})
        self.assertEqual(refs, [])

    def test_drops_url_identifier_regardless_of_relation(self) -> None:
        # A URL-typed identifier is dropped even when its relation is consumed
        # (IsDescribedBy is allowed since #181); only the DOI survives.
        refs = parse_nemar_metadata(
            {
                "related_identifiers": [
                    {
                        "identifier": "https://example.com",
                        "identifier_type": "URL",
                        "relation_type": "IsDescribedBy",
                    },
                    {
                        "identifier": "10.1038/foo.bar",
                        "identifier_type": "DOI",
                        "relation_type": "References",
                    },
                ]
            }
        )
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].identifier, "10.1038/foo.bar")

    def test_keeps_isdescribedby_data_paper_doi(self) -> None:
        # The data paper is linked via IsDescribedBy (e.g. 10.1038/sdata.2015.1
        # describes on000117). Since #181 the DOI is kept so anchor judgment can
        # bucket it; before, this relation was filtered and the paper was lost.
        refs = parse_nemar_metadata(
            {
                "related_identifiers": [
                    {
                        "identifier": "10.1038/sdata.2015.1",
                        "identifier_type": "DOI",
                        "relation_type": "IsDescribedBy",
                    },
                ]
            }
        )
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].identifier, "10.1038/sdata.2015.1")
        self.assertEqual(refs[0].relation_type, "IsDescribedBy")

    def test_skips_openneuro_dataset_doi(self) -> None:
        refs = parse_nemar_metadata(
            {
                "related_identifiers": [
                    {
                        "identifier": "10.18112/openneuro.ds000117.v1.0.0",
                        "identifier_type": "DOI",
                        "relation_type": "IsIdenticalTo",
                    },
                ]
            }
        )
        self.assertEqual(refs, [])

    def test_skips_malformed_identifier(self) -> None:
        refs = parse_nemar_metadata(
            {
                "related_identifiers": [
                    {
                        "identifier": "10.1038/foo\nbar",
                        "identifier_type": "DOI",
                        "relation_type": "References",
                    },
                ]
            }
        )
        self.assertEqual(refs, [])

    def test_handles_non_dict_entry(self) -> None:
        refs = parse_nemar_metadata(
            {
                "related_identifiers": [
                    "not a dict",
                    None,
                    {
                        "identifier": "10.1038/sdata.2015.1",
                        "identifier_type": "DOI",
                        "relation_type": "References",
                    },
                ]
            }
        )
        self.assertEqual(len(refs), 1)

    def test_isversionof_kept(self) -> None:
        # IsVersionOf is in the citation-relations allow-list; should survive.
        refs = parse_nemar_metadata(
            {
                "related_identifiers": [
                    {
                        "identifier": "10.1038/older.2014.5",
                        "identifier_type": "DOI",
                        "relation_type": "IsVersionOf",
                    },
                ]
            }
        )
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].relation_type, "IsVersionOf")


# --- Real substitute objects (NOT Mocks) for the GitHub fallback path
# (_fetch_from_github), mirroring the shapes it touches. See issue #168.


class _RaisingContent:
    """PyGithub ContentFile stand-in for a >1MB blob: decoded_content asserts
    encoding == "base64" and raises AssertionError when encoding is None."""

    @property
    def decoded_content(self) -> bytes:
        raise AssertionError("unsupported encoding: None")


class _NoneContent:
    @property
    def decoded_content(self) -> bytes | None:
        return None


class _BytesContent:
    def __init__(self, data: bytes) -> None:
        self._data = data

    @property
    def decoded_content(self) -> bytes:
        return self._data


class _FakeRepo:
    def __init__(self, content: object) -> None:
        self._content = content

    def get_contents(self, path: str) -> object:
        return self._content


class _FakeGithub:
    def __init__(self, content: object) -> None:
        self._content = content

    def get_repo(self, full_name: str) -> _FakeRepo:
        return _FakeRepo(self._content)


class FetchFromGithubDecodeGuard(TestCase):
    """_fetch_from_github must turn an undecodable .nemar/metadata.json into a
    per-dataset FetchError, never let AssertionError abort judge-anchors for
    nm-* / on-* datasets. Issue #168."""

    def _source_returning(self, content: object) -> NemarMetadataSource:
        # Bypass __init__ (which builds a real Github client); _fetch_from_github
        # only reads self.github.
        src = NemarMetadataSource.__new__(NemarMetadataSource)
        src.github = _FakeGithub(content)  # type: ignore[attr-defined]
        return src

    def test_undecodable_blob_returns_parse_error_not_raises(self) -> None:
        result = self._source_returning(_RaisingContent())._fetch_from_github(
            "nm000999", "nemarDatasets"
        )
        assert isinstance(result, FetchError)
        self.assertEqual(result.reason, "parse")

    def test_none_blob_returns_parse_error(self) -> None:
        result = self._source_returning(_NoneContent())._fetch_from_github(
            "nm000999", "nemarDatasets"
        )
        assert isinstance(result, FetchError)
        self.assertEqual(result.reason, "parse")

    def test_valid_blob_returns_success(self) -> None:
        payload = json.dumps(
            {
                "related_identifiers": [
                    {
                        "identifier": "10.1038/sdata.2015.1",
                        "identifier_type": "DOI",
                        "relation_type": "References",
                    }
                ]
            }
        ).encode("utf-8")
        result = self._source_returning(_BytesContent(payload))._fetch_from_github(
            "nm000999", "nemarDatasets"
        )
        assert isinstance(result, FetchSuccess)
        self.assertEqual(len(result.value), 1)
        self.assertEqual(result.value[0].identifier, "10.1038/sdata.2015.1")


class ParseDatasetMetadata(TestCase):
    """parse_nemar_dataset_metadata tolerates both payload shapes (schema v2.1)."""

    def test_github_shape_has_methods_and_funding_references(self) -> None:
        # nm000103 is the GitHub .nemar/metadata.json shape: funding_references[]
        # + methods_description present.
        meta = parse_nemar_dataset_metadata(load_fixture("nm000103.metadata.json"))
        self.assertIn("EEG", meta.keywords)
        self.assertTrue(all(isinstance(k, str) for k in meta.keywords))
        self.assertIsNotNone(meta.methods_description)
        # funding normalized to {funder_name, award_number?, award_title?}.
        funders = {f["funder_name"] for f in meta.funding}
        self.assertIn("NIH", funders)
        nih = next(f for f in meta.funding if f["funder_name"] == "NIH")
        self.assertEqual(nih["award_number"], "R01MH125934")
        self.assertEqual(nih["award_title"], "BIDS data preparation")

    def test_data_api_shape_uses_funding_and_drops_methods(self) -> None:
        # nm000104 is the data.nemar.org neuroschema shape: funding[] (not
        # funding_references) and NO methods_description.
        meta = parse_nemar_dataset_metadata(load_fixture("nm000104.data_api.json"))
        self.assertTrue(meta.keywords)
        self.assertIsNone(meta.methods_description)
        self.assertTrue(meta.funding)
        self.assertTrue(all("funder_name" in f for f in meta.funding))

    def test_empty_payload_yields_empty_metadata(self) -> None:
        meta = parse_nemar_dataset_metadata({})
        self.assertEqual(meta, EMPTY_NEMAR_DATASET_METADATA)

    def test_keywords_flattened_and_funding_without_funder_name_dropped(self) -> None:
        meta = parse_nemar_dataset_metadata(
            {
                "keywords": [{"term": "EEG"}, {"term": "  "}, {"no_term": "x"}, "MEG"],
                "methods_description": "   ",
                "funding_references": [
                    {"funder_name": "NIH", "award_number": "R01", "award_title": None},
                    {"award_number": "no-funder"},  # dropped: no funder_name
                    {"funder_name": "   "},  # dropped: blank funder_name
                ],
            }
        )
        # Bare-string keyword accepted; blank/term-less entries dropped.
        self.assertEqual(meta.keywords, ("EEG", "MEG"))
        # Blank methods_description -> None.
        self.assertIsNone(meta.methods_description)
        # Only the well-formed funder survives; null award_title omitted.
        self.assertEqual(meta.funding, ({"funder_name": "NIH", "award_number": "R01"},))

    def test_wrong_type_keywords_and_funding_warn(self) -> None:
        # A present-but-non-list shape is a schema mismatch -> WARN, not silent.
        # `funding` (not `funding_references`) is used so the wrong-type value is
        # the one that actually reaches _parse_funding.
        with self.assertLogs(
            "dataset_citations.sources.nemar_metadata", level="WARNING"
        ) as logs:
            meta = parse_nemar_dataset_metadata({"keywords": "EEG, MEG", "funding": 7})
        self.assertEqual(meta.keywords, ())
        self.assertEqual(meta.funding, ())
        msgs = " ".join(r.getMessage() for r in logs.records)
        self.assertIn("keywords is not a list", msgs)
        self.assertIn("funding is not a list", msgs)


class GetDatasetMetadataFallback(TestCase):
    """NemarMetadataSource.get_dataset_metadata fetch-fallback chain.

    Uses a real NemarMetadataSource subclass with canned fetch results (no
    unittest.mock, per .rules/testing.md) to exercise the GitHub-preferred
    ladder offline.
    """

    class _CannedSource(NemarMetadataSource):
        def __init__(self, github_result, data_api_result, *, prefer_data_api=True):
            self._github_result = github_result
            self._data_api_result = data_api_result
            self.prefer_data_api = prefer_data_api

        def _github_payload(self, dataset_id, org):  # type: ignore[override]
            return self._github_result

        def _fetch_from_data_api(self, dataset_id):  # type: ignore[override]
            return self._data_api_result

    def test_github_success_is_preferred(self) -> None:
        src = self._CannedSource(
            FetchSuccess(load_fixture("nm000103.metadata.json")),
            FetchError("not_found", "should not be consulted"),
        )
        meta = src.get_dataset_metadata("nm000103")
        # GitHub shape carries methods_description.
        self.assertIsNotNone(meta.methods_description)
        self.assertIn("EEG", meta.keywords)

    def test_github_fail_falls_back_to_data_api(self) -> None:
        src = self._CannedSource(
            FetchError("not_found", ".nemar/metadata.json missing"),
            FetchSuccess(load_fixture("nm000104.data_api.json")),
        )
        meta = src.get_dataset_metadata("nm000104")
        # data.nemar.org shape: keywords + funding present, methods absent.
        self.assertTrue(meta.keywords)
        self.assertTrue(meta.funding)
        self.assertIsNone(meta.methods_description)

    def test_both_fail_returns_empty_with_warning(self) -> None:
        src = self._CannedSource(
            FetchError("not_found", "github gone"),
            FetchError("rate_limit", "data api 429"),
        )
        with self.assertLogs(
            "dataset_citations.sources.nemar_metadata", level="WARNING"
        ) as logs:
            meta = src.get_dataset_metadata("nm000999")
        self.assertEqual(meta, EMPTY_NEMAR_DATASET_METADATA)
        self.assertTrue(
            any("rich metadata unavailable" in r.getMessage() for r in logs.records)
        )

    def test_prefer_data_api_false_skips_data_api(self) -> None:
        # data_api_result WOULD parse to non-empty, but prefer_data_api=False
        # must skip it entirely -> EMPTY (proves the guard).
        src = self._CannedSource(
            FetchError("not_found", "github gone"),
            FetchSuccess(load_fixture("nm000104.data_api.json")),
            prefer_data_api=False,
        )
        self.assertEqual(
            src.get_dataset_metadata("nm000104"), EMPTY_NEMAR_DATASET_METADATA
        )


class ParseOn007615(TestCase):
    """on007615 links its data paper as `IsSupplementTo` (the dataset supplements
    the paper), the relation older enrichments used. The anchor must reach the
    judge; a live data.nemar.org snapshot recorded 2026-09-29."""

    def test_supplement_to_data_paper_is_an_anchor(self) -> None:
        refs = parse_nemar_metadata(load_fixture("on007615.metadata.json"))
        by_doi = {r.identifier: r.relation_type for r in refs}
        self.assertEqual(by_doi.get("10.3389/fnhum.2025.1647425"), "IsSupplementTo")
        # URL-typed landing pages under IsDescribedBy are still skipped.
        self.assertFalse(any(r.identifier.startswith("http") for r in refs))
