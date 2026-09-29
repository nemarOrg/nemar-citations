"""Tests for phase 3 pipeline integration with the anchor-judgment sidecar.

Test doubles at the network boundary only: a `_StubBackend` (matching the
pattern in `tests/test_core_opencite_pipeline.py`) plus a `_StubSource`
returning hand-built `DoiReference` records, and on-disk sidecar JSONs written
into a per-test tempdir. The pipeline and the gate run for real; no
`unittest.mock` per `.rules/testing.md`.

The locked sidecar shape lives in epic #76's phase 2 contract (see issue
#86). Tests below construct the JSON manually so a phase 2 schema rename
breaks this file loudly instead of silently.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from dataset_citations.backends.opencite_backend import OpenCiteBackend
from dataset_citations.core.opencite_pipeline import (
    fetch_dataset_citations_via_opencite,
)
from dataset_citations.quality.anchor_judgment_io import (
    JudgmentSidecar,
    canonical_anchor_key,
    load_judgment_lookup,
    load_judgment_sidecar,
)
from dataset_citations.quality.llm_client import (
    ALLOWED_CLASSIFICATIONS,
    trusted_judge_model,
)
from dataset_citations.sources.models import (
    Author,
    CitingWork,
    DoiReference,
    FetchSuccess,
)
from dataset_citations.sources.nemar_metadata import NemarDatasetMetadata

WHEN = datetime(2026, 5, 22, tzinfo=UTC)
# Sidecars written by the trusted judge are the ones the gate honors.
JUDGE = trusted_judge_model()


def _make_work(
    title: str,
    *,
    doi: str | None,
    source_doi: str,
    source_relation: str = "References",
    citation_count: int = 7,
    year: int = 2024,
) -> CitingWork:
    return CitingWork(
        title=title,
        doi=doi,
        pmid=None,
        openalex_id=None,
        year=year,
        authors=(Author(name="A. Researcher"),),
        venue="Journal of Tests",
        abstract=None,
        citation_count=citation_count,
        source_doi=source_doi,
        source_relation=source_relation,  # type: ignore[arg-type]
    )


class _StubBackend(OpenCiteBackend):
    """OpenCiteBackend test double; records which anchors it received."""

    def __init__(self, batch_outcome):
        self._batch_outcome = batch_outcome
        self.calls: list[list[str]] = []
        self._config = None  # type: ignore[assignment]
        self._max_results_per_doi = 100
        self._concurrency = 1

    def get_citing_works_batch(self, refs):  # type: ignore[override]
        self.calls.append([r.identifier for r in refs])
        return {r.identifier: self._batch_outcome[r.identifier] for r in refs}


class _ExplodingBackend(OpenCiteBackend):
    """Backend that fails the test loudly if any anchor reaches it."""

    def __init__(self) -> None:
        self._config = None  # type: ignore[assignment]
        self._max_results_per_doi = 100
        self._concurrency = 1

    def get_citing_works_batch(self, refs):  # type: ignore[override]
        raise AssertionError(
            f"backend must not be called; received {[r.identifier for r in refs]}"
        )


class _StubSource:
    def __init__(self, outcome):
        self._outcome = outcome

    def get_doi_references(self, dataset_id):
        return self._outcome


class _RichStubSource(_StubSource):
    """`_StubSource` that also exposes the schema-v2.1 rich-metadata capability."""

    def __init__(self, outcome, metadata: NemarDatasetMetadata):
        super().__init__(outcome)
        self._metadata = metadata

    def get_dataset_metadata(self, dataset_id):
        return self._metadata


def _context_anchors(out: dict) -> list[dict]:
    """The kept=False subset of metadata.anchors[] (the old context_anchors)."""
    return [a for a in out["metadata"]["anchors"] if not a["kept"]]


def _write_sidecar(
    judgments_dir: Path,
    dataset_id: str,
    judgments: list[dict],
    *,
    model: str = JUDGE,
) -> Path:
    """Write a sidecar JSON in the locked phase 2 shape."""
    judgments_dir.mkdir(parents=True, exist_ok=True)
    path = judgments_dir / f"{dataset_id}.json"
    payload = {
        "dataset_id": dataset_id,
        "judged_at": "2026-05-22T00:00:00Z",
        "judgment_model": model,
        "judgments": judgments,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _judgment(
    identifier: str,
    classification: str,
    *,
    identifier_type: str = "doi",
    source_relation: str = "IsDerivedFrom",
    reason: str = "stub reason",
    paper_title: str | None = "stub title",
    paper_year: int | None = 2024,
    error: str | None = None,
) -> dict:
    return {
        "anchor_identifier": identifier,
        "anchor_identifier_type": identifier_type,
        "source_relation": source_relation,
        "classification": classification,
        "reason": reason,
        "paper_title": paper_title,
        "paper_year": paper_year,
        "paper_venue": "Journal of Stubs",
        "judged_at": "2026-05-22T00:00:00Z",
        "error": error,
    }


class JudgmentLookupTests(TestCase):
    """Direct tests for the read-side sidecar helper."""

    def test_missing_sidecar_returns_empty(self) -> None:
        with TemporaryDirectory() as tmp:
            judgments_dir = Path(tmp)
            sidecar = load_judgment_sidecar("nm000999", judgments_dir=judgments_dir)
            self.assertFalse(sidecar.present)
            self.assertEqual(sidecar.lookup, {})
            # The spec-named helper returns the same empty dict.
            self.assertEqual(
                load_judgment_lookup("nm000999", judgments_dir=judgments_dir),
                {},
            )

    def test_taxonomy_enforced_via_allowed_classifications(self) -> None:
        # If phase 1 adds a new class, this test should still pass: the loader
        # always trusts ALLOWED_CLASSIFICATIONS. Use a literal-bad string so
        # we never accidentally test against a future-valid name.
        bogus = "not_a_real_class"
        self.assertNotIn(bogus, ALLOWED_CLASSIFICATIONS)

        with TemporaryDirectory() as tmp:
            judgments_dir = Path(tmp)
            _write_sidecar(
                judgments_dir,
                "nm000001",
                [
                    _judgment("10.1234/data", "data_paper"),
                    _judgment("10.1234/bad", bogus),
                ],
            )
            sidecar = load_judgment_sidecar("nm000001", judgments_dir=judgments_dir)
            self.assertTrue(sidecar.present)
            self.assertEqual(sidecar.lookup, {"10.1234/data": "data_paper"})

    def test_errored_judgment_dropped(self) -> None:
        with TemporaryDirectory() as tmp:
            judgments_dir = Path(tmp)
            _write_sidecar(
                judgments_dir,
                "nm000002",
                [
                    _judgment(
                        "10.1234/timeout",
                        "data_paper",
                        error="llm_judgment_failed:claude CLI timed out",
                    ),
                    _judgment("10.1234/ok", "umbrella"),
                ],
            )
            sidecar = load_judgment_sidecar("nm000002", judgments_dir=judgments_dir)
            self.assertEqual(sidecar.lookup, {"10.1234/ok": "umbrella"})

    def test_canonical_anchor_key_doi_normalization(self) -> None:
        # The pipeline uses canonical_anchor_key to bridge DoiReference and
        # the sidecar; make sure the normalization rules round-trip.
        self.assertEqual(
            canonical_anchor_key("https://doi.org/10.1234/Abc", "doi"),
            "10.1234/abc",
        )
        self.assertEqual(canonical_anchor_key("pmid:12345", "pmid"), "pmid:12345")
        self.assertEqual(canonical_anchor_key("12345", "pmid"), "pmid:12345")


class PipelineBucketingTests(TestCase):
    """End-to-end pipeline tests with on-disk sidecars."""

    def setUp(self) -> None:
        self._tmp_ctx = TemporaryDirectory()
        self.judgments_dir = Path(self._tmp_ctx.name)

    def tearDown(self) -> None:
        self._tmp_ctx.cleanup()

    def test_data_paper_anchor_alone_is_fetched(self) -> None:
        """Anchors classified `data_paper` reach the backend; others don't."""
        data_paper_ref = DoiReference(
            identifier="10.1038/data.paper",
            identifier_type="doi",
            relation_type="IsDerivedFrom",
            source="nemar_metadata",
        )
        umbrella_ref = DoiReference(
            identifier="10.1038/hbn.umbrella",
            identifier_type="doi",
            relation_type="IsDerivedFrom",
            source="nemar_metadata",
        )
        nemar = _StubSource(FetchSuccess([data_paper_ref, umbrella_ref]))
        backend = _StubBackend(
            {
                data_paper_ref.identifier: FetchSuccess(
                    [
                        _make_work(
                            "Cites the data paper",
                            doi="10.5/cites",
                            source_doi=data_paper_ref.identifier,
                        )
                    ]
                )
            }
        )
        _write_sidecar(
            self.judgments_dir,
            "nm000010",
            [
                _judgment(data_paper_ref.identifier, "data_paper"),
                _judgment(
                    umbrella_ref.identifier,
                    "umbrella",
                    paper_title="HBN umbrella paper",
                    reason="describes broader initiative",
                ),
            ],
        )
        out = fetch_dataset_citations_via_opencite(
            "nm000010",
            backend=backend,
            nemar_source=nemar,
            fetch_date=WHEN,
            judgments_dir=self.judgments_dir,
        )
        # Only the data_paper anchor reached the backend.
        self.assertEqual(backend.calls, [[data_paper_ref.identifier]])
        self.assertEqual(out["num_citations"], 1)
        self.assertEqual(out["metadata"]["anchor_count"], 1)
        self.assertEqual(out["metadata"]["fetch_status"], "success")
        self.assertEqual(out["metadata"]["anchor_judgment_model"], JUDGE)
        # anchors[] is the full superset: both anchors, with kept flags.
        anchors = out["metadata"]["anchors"]
        self.assertEqual(len(anchors), 2)
        kept = {a["identifier"]: a for a in anchors if a["kept"]}
        self.assertEqual(set(kept), {data_paper_ref.identifier})
        self.assertEqual(
            kept[data_paper_ref.identifier]["classification"], "data_paper"
        )
        # context_details is populated for data_paper too, so the kept anchor
        # carries its own paper title (default "stub title" from _judgment()).
        self.assertEqual(kept[data_paper_ref.identifier]["paper_title"], "stub title")
        # searched_dois is the flat list of the fetched DOI anchor(s).
        self.assertEqual(out["metadata"]["searched_dois"], [data_paper_ref.identifier])
        # The context subset (kept=False) is the umbrella anchor.
        context = _context_anchors(out)
        self.assertEqual(len(context), 1)
        self.assertEqual(context[0]["classification"], "umbrella")
        self.assertEqual(context[0]["identifier"], umbrella_ref.identifier)
        self.assertEqual(context[0]["paper_title"], "HBN umbrella paper")
        self.assertEqual(context[0]["source_relation"], "IsDerivedFrom")
        # paper_year + paper_venue forwarded from the sidecar so consumers can
        # render anchors without re-opening it. _judgment() defaults these to
        # 2024 / Journal of Stubs. judgment_model is stamped per anchor too.
        self.assertEqual(context[0]["paper_year"], 2024)
        self.assertEqual(context[0]["paper_venue"], "Journal of Stubs")
        self.assertEqual(context[0]["judgment_model"], JUDGE)
        # _StubSource has no get_dataset_metadata -> empty rich-metadata keys.
        self.assertEqual(out["metadata"]["keywords"], [])
        self.assertIsNone(out["metadata"]["methods_description"])
        self.assertEqual(out["metadata"]["funding"], [])

    def test_no_sidecar_fetches_nothing(self) -> None:
        """Without a sidecar no anchor is fetched: the gate fails closed (#241).

        The pre-#241 fallback fetched every anchor here, which is how a dead
        judge turned each related-work paper's citers into citations.
        """
        ref_a = DoiReference(
            identifier="10.1234/a",
            identifier_type="doi",
            relation_type="References",
            source="nemar_metadata",
        )
        ref_b = DoiReference(
            identifier="10.1234/b",
            identifier_type="doi",
            relation_type="IsDerivedFrom",
            source="nemar_metadata",
        )
        nemar = _StubSource(FetchSuccess([ref_a, ref_b]))
        with self.assertLogs(
            "dataset_citations.core.opencite_pipeline", level="WARNING"
        ) as logs:
            out = fetch_dataset_citations_via_opencite(
                "nm000011",
                backend=_ExplodingBackend(),
                nemar_source=nemar,
                fetch_date=WHEN,
                judgments_dir=self.judgments_dir,  # empty tempdir
            )
        warns = [r.getMessage() for r in logs.records]
        self.assertEqual(len(warns), 1)
        self.assertIn(
            "2/2 anchors have no successful judgment (no usable sidecar: missing)",
            warns[0],
        )
        self.assertEqual(out["num_citations"], 0)
        self.assertEqual(out["metadata"]["fetch_status"], "no_data_paper_anchor")
        self.assertIsNone(out["metadata"]["anchor_judgment_model"])
        anchors = out["metadata"]["anchors"]
        self.assertEqual(len(anchors), 2)
        self.assertTrue(all(not a["kept"] for a in anchors))
        self.assertTrue(all(a["kept_reason"] == "unjudged" for a in anchors))
        self.assertTrue(all(a["classification"] is None for a in anchors))
        self.assertEqual(out["metadata"]["searched_dois"], [])

    def test_out_of_taxonomy_entry_dropped_with_warning(self) -> None:
        """One bad entry doesn't sink the rest of the sidecar."""
        data_paper_ref = DoiReference(
            identifier="10.1234/good-data",
            identifier_type="doi",
            relation_type="References",
            source="nemar_metadata",
        )
        umbrella_ref = DoiReference(
            identifier="10.1234/umbrella",
            identifier_type="doi",
            relation_type="IsDerivedFrom",
            source="nemar_metadata",
        )
        broken_ref = DoiReference(
            identifier="10.1234/broken",
            identifier_type="doi",
            relation_type="IsDerivedFrom",
            source="nemar_metadata",
        )
        nemar = _StubSource(FetchSuccess([data_paper_ref, umbrella_ref, broken_ref]))
        backend = _StubBackend(
            {
                data_paper_ref.identifier: FetchSuccess(
                    [
                        _make_work(
                            "good",
                            doi="10.5/good",
                            source_doi=data_paper_ref.identifier,
                        )
                    ]
                ),
            }
        )
        _write_sidecar(
            self.judgments_dir,
            "nm000012",
            [
                _judgment(data_paper_ref.identifier, "data_paper"),
                _judgment(umbrella_ref.identifier, "umbrella"),
                _judgment(broken_ref.identifier, "garbage_label"),
            ],
        )
        with self.assertLogs(
            "dataset_citations.quality.anchor_judgment_io",
            level=logging.WARNING,
        ) as logs:
            out = fetch_dataset_citations_via_opencite(
                "nm000012",
                backend=backend,
                nemar_source=nemar,
                fetch_date=WHEN,
                judgments_dir=self.judgments_dir,
            )
        self.assertTrue(any("out-of-taxonomy" in r.getMessage() for r in logs.records))
        # Only the data paper is fetched. The garbage_label entry is dropped
        # from the lookup, so that anchor gates as unjudged: context only.
        self.assertEqual(backend.calls, [[data_paper_ref.identifier]])
        self.assertEqual(out["num_citations"], 1)
        reasons = {
            a["identifier"]: a["kept_reason"] for a in out["metadata"]["anchors"]
        }
        self.assertEqual(
            reasons,
            {
                data_paper_ref.identifier: "judged_data_paper",
                umbrella_ref.identifier: "judged_not_data_paper",
                broken_ref.identifier: "unjudged",
            },
        )

    def test_all_anchors_non_data_paper_zero_citations(self) -> None:
        """Every anchor classified as non-data_paper -> backend never called."""
        refs = [
            DoiReference(
                identifier=f"10.1234/{label}",
                identifier_type="doi",
                relation_type="IsDerivedFrom",
                source="nemar_metadata",
            )
            for label in ("umbrella", "method", "related", "junk")
        ]
        nemar = _StubSource(FetchSuccess(refs))
        _write_sidecar(
            self.judgments_dir,
            "nm000013",
            [
                _judgment(refs[0].identifier, "umbrella", paper_title="Umbrella"),
                _judgment(
                    refs[1].identifier,
                    "methodology",
                    paper_title="MNE-Python",
                ),
                _judgment(
                    refs[2].identifier,
                    "related_work",
                    paper_title="Related",
                ),
                _judgment(refs[3].identifier, "irrelevant", paper_title="Junk"),
            ],
        )
        out = fetch_dataset_citations_via_opencite(
            "nm000013",
            backend=_ExplodingBackend(),
            nemar_source=nemar,
            fetch_date=WHEN,
            judgments_dir=self.judgments_dir,
        )
        self.assertEqual(out["num_citations"], 0)
        self.assertEqual(out["metadata"]["fetch_status"], "no_data_paper_anchor")
        # anchor_count is the kept count on every path (the sweep's definition).
        self.assertEqual(out["metadata"]["anchor_count"], 0)
        # All four anchors are present, none kept (none reached the backend).
        self.assertEqual(len(out["metadata"]["anchors"]), 4)
        self.assertEqual(len(_context_anchors(out)), 4)
        self.assertEqual(out["metadata"]["searched_dois"], [])
        # The model name is still surfaced on the stub payload.
        self.assertEqual(out["metadata"]["anchor_judgment_model"], JUDGE)

    def test_another_datasets_doi_respects_judgment(self) -> None:
        """Another dataset's NEMAR DOI is not this dataset's own DOI, so it
        needs a judgment like any anchor; judged umbrella, it is not fetched."""
        source_ref = DoiReference(
            identifier="10.1038/source-paper",
            identifier_type="doi",
            relation_type="IsDerivedFrom",
            source="nemar_metadata",
        )
        catalog_doi = "10.82901/nemar.nm000777"
        nemar = _StubSource(FetchSuccess([source_ref]))
        backend = _StubBackend(
            {
                source_ref.identifier: FetchSuccess(
                    [
                        _make_work(
                            "source paper citation",
                            doi="10.5/source",
                            source_doi=source_ref.identifier,
                        )
                    ]
                ),
            }
        )
        _write_sidecar(
            self.judgments_dir,
            "nm000014",
            [
                _judgment(source_ref.identifier, "data_paper"),
                _judgment(
                    catalog_doi,
                    "umbrella",
                    source_relation="References",
                    paper_title="catalog ref classified as umbrella",
                ),
            ],
        )
        out = fetch_dataset_citations_via_opencite(
            "nm000014",
            backend=backend,
            nemar_source=nemar,
            catalog_doi=catalog_doi,
            fetch_date=WHEN,
            judgments_dir=self.judgments_dir,
        )
        # Only the source_ref reaches the backend.
        self.assertEqual(backend.calls, [[source_ref.identifier]])
        self.assertEqual(out["metadata"]["anchor_count"], 1)
        catalog_records = [
            a for a in out["metadata"]["anchors"] if a["identifier"] == catalog_doi
        ]
        self.assertEqual(len(catalog_records), 1)
        self.assertEqual(catalog_records[0]["classification"], "umbrella")
        self.assertFalse(catalog_records[0]["kept"])

    def test_partial_sidecar_warns_per_dataset(self) -> None:
        """When the sidecar covers some but not all anchors, the uncovered
        anchors stay context only AND a single per-dataset WARN logs the gap
        so operators see the drift."""
        judged_ref = DoiReference(
            identifier="10.1038/judged-paper",
            identifier_type="doi",
            relation_type="References",
            source="nemar_metadata",
        )
        unjudged_ref = DoiReference(
            identifier="10.1038/unjudged-paper",
            identifier_type="doi",
            relation_type="References",
            source="nemar_metadata",
        )
        nemar = _StubSource(FetchSuccess([judged_ref, unjudged_ref]))
        backend = _StubBackend(
            {
                judged_ref.identifier: FetchSuccess(
                    [
                        _make_work(
                            "Cites judged",
                            doi="10.5/cj",
                            source_doi=judged_ref.identifier,
                        )
                    ]
                ),
            }
        )
        _write_sidecar(
            self.judgments_dir,
            "nm000015",
            [_judgment(judged_ref.identifier, "data_paper")],
        )
        with self.assertLogs(
            "dataset_citations.core.opencite_pipeline", level="WARNING"
        ) as logs:
            out = fetch_dataset_citations_via_opencite(
                "nm000015",
                backend=backend,
                nemar_source=nemar,
                fetch_date=WHEN,
                judgments_dir=self.judgments_dir,
            )
        # Only the judged data paper reached the backend.
        self.assertEqual(backend.calls, [[judged_ref.identifier]])
        # Exactly one WARN per dataset, naming the gap count.
        warns = [
            r for r in logs.records if "have no successful judgment" in r.getMessage()
        ]
        self.assertEqual(len(warns), 1)
        self.assertIn("1/2 anchors", warns[0].getMessage())
        context = _context_anchors(out)
        self.assertEqual([a["identifier"] for a in context], [unjudged_ref.identifier])
        self.assertEqual(out["metadata"]["searched_dois"], [judged_ref.identifier])

    def test_never_anchor_blocks_a_judged_data_paper(self) -> None:
        """A standards paper judged `data_paper` is still never fetched (#241).

        Both routes: a DOI on the curated list (EEG-BIDS, which gemma really
        did call a data paper) and a BIDS-spec title on an unlisted DOI.
        """
        eeg_bids = DoiReference(
            identifier="10.1038/s41597-019-0104-8",
            identifier_type="doi",
            relation_type="IsDescribedBy",
            source="nemar_metadata",
        )
        future_spec = DoiReference(
            identifier="10.9999/nirs-bids",
            identifier_type="doi",
            relation_type="IsDescribedBy",
            source="nemar_metadata",
        )
        nemar = _StubSource(FetchSuccess([eeg_bids, future_spec]))
        _write_sidecar(
            self.judgments_dir,
            "nm000030",
            [
                _judgment(eeg_bids.identifier, "data_paper"),
                _judgment(
                    future_spec.identifier,
                    "data_paper",
                    paper_title="NIRS-BIDS: an extension to the brain imaging "
                    "data structure for near-infrared spectroscopy",
                ),
            ],
        )
        out = fetch_dataset_citations_via_opencite(
            "nm000030",
            backend=_ExplodingBackend(),
            nemar_source=nemar,
            fetch_date=WHEN,
            judgments_dir=self.judgments_dir,
        )
        self.assertEqual(out["num_citations"], 0)
        for anchor in out["metadata"]["anchors"]:
            self.assertFalse(anchor["kept"])
            self.assertEqual(anchor["kept_reason"], "never_anchor")
            # The judge's verdict is still recorded for auditing.
            self.assertEqual(anchor["classification"], "data_paper")

    def test_own_concept_doi_is_kept_without_judgment(self) -> None:
        """The dataset's own NEMAR concept DOI needs no judgment: citing it is
        citing the dataset. Other anchors still need one."""
        own = "10.82901/nemar.nm000031"
        related = DoiReference(
            identifier="10.1016/related.2014",
            identifier_type="doi",
            relation_type="References",
            source="nemar_metadata",
        )
        nemar = _StubSource(FetchSuccess([related]))
        backend = _StubBackend(
            {
                own: FetchSuccess(
                    [_make_work("Uses the data", doi="10.5/u", source_doi=own)]
                )
            }
        )
        out = fetch_dataset_citations_via_opencite(
            "nm000031",
            backend=backend,
            nemar_source=nemar,
            catalog_doi=own,
            fetch_date=WHEN,
            judgments_dir=self.judgments_dir,  # no sidecar at all
        )
        self.assertEqual(backend.calls, [[own]])
        self.assertEqual(out["num_citations"], 1)
        reasons = {
            a["identifier"]: a["kept_reason"] for a in out["metadata"]["anchors"]
        }
        self.assertEqual(reasons, {related.identifier: "unjudged", own: "own_doi"})

    def test_citations_older_than_their_anchor_are_dropped(self) -> None:
        """nm000275's failure mode: a 2015 paper cannot cite a 2019 data paper.

        Works with an unknown year (0) are kept; there is nothing to compare.
        """
        data_paper = DoiReference(
            identifier="10.1038/s41597-019-0027-4",
            identifier_type="doi",
            relation_type="IsDescribedBy",
            source="nemar_metadata",
        )
        nemar = _StubSource(FetchSuccess([data_paper]))
        backend = _StubBackend(
            {
                data_paper.identifier: FetchSuccess(
                    [
                        _make_work(
                            "The PREP pipeline",
                            doi="10.3389/fninf.2015.00016",
                            source_doi=data_paper.identifier,
                            year=2015,
                        ),
                        _make_work(
                            "Same-year citer",
                            doi="10.5/same",
                            source_doi=data_paper.identifier,
                            year=2019,
                        ),
                        _make_work(
                            "Unknown year",
                            doi="10.5/unknown",
                            source_doi=data_paper.identifier,
                            year=0,
                        ),
                    ]
                )
            }
        )
        _write_sidecar(
            self.judgments_dir,
            "nm000032",
            [_judgment(data_paper.identifier, "data_paper", paper_year=2019)],
        )
        out = fetch_dataset_citations_via_opencite(
            "nm000032",
            backend=backend,
            nemar_source=nemar,
            fetch_date=WHEN,
            judgments_dir=self.judgments_dir,
        )
        titles = sorted(c["title"] for c in out["citation_details"])
        self.assertEqual(titles, ["Same-year citer", "Unknown year"])
        self.assertEqual(out["num_citations"], 2)

    def test_rich_metadata_capability_lands_in_metadata(self) -> None:
        """A source exposing get_dataset_metadata populates the v2.1 keys."""
        data_paper_ref = DoiReference(
            identifier="10.1038/data.paper",
            identifier_type="doi",
            relation_type="IsDerivedFrom",
            source="nemar_metadata",
        )
        metadata = NemarDatasetMetadata(
            keywords=("EEG", "BIDS"),
            methods_description="collected during cognitive tasks",
            funding=({"funder_name": "NIH", "award_number": "R01MH1"},),
        )
        nemar = _RichStubSource(FetchSuccess([data_paper_ref]), metadata)
        backend = _StubBackend(
            {
                data_paper_ref.identifier: FetchSuccess(
                    [
                        _make_work(
                            "cites",
                            doi="10.5/c",
                            source_doi=data_paper_ref.identifier,
                        )
                    ]
                )
            }
        )
        _write_sidecar(
            self.judgments_dir,
            "nm000016",
            [_judgment(data_paper_ref.identifier, "data_paper")],
        )
        out = fetch_dataset_citations_via_opencite(
            "nm000016",
            backend=backend,
            nemar_source=nemar,
            fetch_date=WHEN,
            judgments_dir=self.judgments_dir,
        )
        self.assertEqual(out["metadata"]["keywords"], ["EEG", "BIDS"])
        self.assertEqual(
            out["metadata"]["methods_description"], "collected during cognitive tasks"
        )
        self.assertEqual(
            out["metadata"]["funding"],
            [{"funder_name": "NIH", "award_number": "R01MH1"}],
        )

    def test_rich_metadata_present_even_with_no_doi_references(self) -> None:
        """A DOI-less dataset still carries its rich metadata (parsed before
        the no_doi_references early-out)."""
        metadata = NemarDatasetMetadata(keywords=("MEG",), methods_description=None)
        nemar = _RichStubSource(FetchSuccess([]), metadata)
        out = fetch_dataset_citations_via_opencite(
            "nm000017",
            backend=_ExplodingBackend(),
            nemar_source=nemar,
            fetch_date=WHEN,
            judgments_dir=self.judgments_dir,
        )
        self.assertEqual(out["metadata"]["fetch_status"], "no_doi_references")
        self.assertEqual(out["metadata"]["keywords"], ["MEG"])
        self.assertEqual(out["metadata"]["anchors"], [])
        self.assertEqual(out["metadata"]["searched_dois"], [])

    def test_no_data_paper_stub_carries_rich_metadata(self) -> None:
        """The no_data_paper_anchor stub still emits the dataset's rich metadata."""
        umbrella_ref = DoiReference(
            identifier="10.1/umbrella",
            identifier_type="doi",
            relation_type="IsDerivedFrom",
            source="nemar_metadata",
        )
        metadata = NemarDatasetMetadata(
            keywords=("EEG",),
            methods_description="collected during tasks",
            funding=({"funder_name": "NIH"},),
        )
        nemar = _RichStubSource(FetchSuccess([umbrella_ref]), metadata)
        _write_sidecar(
            self.judgments_dir,
            "nm000018",
            [_judgment(umbrella_ref.identifier, "umbrella")],
        )
        out = fetch_dataset_citations_via_opencite(
            "nm000018",
            backend=_ExplodingBackend(),
            nemar_source=nemar,
            fetch_date=WHEN,
            judgments_dir=self.judgments_dir,
        )
        self.assertEqual(out["metadata"]["fetch_status"], "no_data_paper_anchor")
        self.assertEqual(out["metadata"]["keywords"], ["EEG"])
        self.assertEqual(
            out["metadata"]["methods_description"], "collected during tasks"
        )
        self.assertEqual(out["metadata"]["funding"], [{"funder_name": "NIH"}])
        self.assertEqual(out["metadata"]["searched_dois"], [])
        self.assertEqual(len(_context_anchors(out)), 1)

    def test_kept_pmid_anchor_excluded_from_searched_dois(self) -> None:
        """A kept PMID anchor appears in anchors[] but not in searched_dois
        (which is DOI-only). The judge only judges DOIs, so a PMID is kept
        only through an identity relation."""
        pmid_ref = DoiReference(
            identifier="pmid:12345",
            identifier_type="pmid",
            relation_type="IsIdenticalTo",
            source="nemar_metadata",
        )
        doi_ref = DoiReference(
            identifier="10.1/d",
            identifier_type="doi",
            relation_type="IsDerivedFrom",
            source="nemar_metadata",
        )
        nemar = _StubSource(FetchSuccess([pmid_ref, doi_ref]))
        backend = _StubBackend(
            {
                pmid_ref.identifier: FetchSuccess(
                    [_make_work("p", doi="10.5/p", source_doi=pmid_ref.identifier)]
                ),
                doi_ref.identifier: FetchSuccess(
                    [_make_work("d", doi="10.5/d", source_doi=doi_ref.identifier)]
                ),
            }
        )
        _write_sidecar(
            self.judgments_dir,
            "nm000019",
            [_judgment(doi_ref.identifier, "data_paper")],
        )
        out = fetch_dataset_citations_via_opencite(
            "nm000019",
            backend=backend,
            nemar_source=nemar,
            fetch_date=WHEN,
            judgments_dir=self.judgments_dir,
        )
        kept = {a["identifier"] for a in out["metadata"]["anchors"] if a["kept"]}
        self.assertEqual(kept, {pmid_ref.identifier, doi_ref.identifier})
        # searched_dois is DOI-only: the kept PMID anchor is excluded.
        self.assertEqual(out["metadata"]["searched_dois"], [doi_ref.identifier])


class TrustedJudgeAndSafetyTests(TestCase):
    """#241 review: only the trusted judge counts, a broken sidecar never
    wipes a dataset, and the gate's remaining rules hold end to end."""

    def setUp(self) -> None:
        self._tmp_ctx = TemporaryDirectory()
        self.judgments_dir = Path(self._tmp_ctx.name)

    def tearDown(self) -> None:
        self._tmp_ctx.cleanup()

    def _ref(self, identifier: str, relation: str = "IsDescribedBy") -> DoiReference:
        return DoiReference(
            identifier=identifier,
            identifier_type="doi",
            relation_type=relation,  # type: ignore[arg-type]
            source="nemar_metadata",
        )

    def test_a_retired_judges_verdicts_do_not_count(self) -> None:
        ref = self._ref("10.1038/data.paper")
        _write_sidecar(
            self.judgments_dir,
            "nm000040",
            [_judgment(ref.identifier, "data_paper")],
            model="gemma4:31b",
        )
        with self.assertLogs("dataset_citations", "WARNING"):
            out = fetch_dataset_citations_via_opencite(
                "nm000040",
                backend=_ExplodingBackend(),
                nemar_source=_StubSource(FetchSuccess([ref])),
                fetch_date=WHEN,
                judgments_dir=self.judgments_dir,
            )
        self.assertEqual(out["metadata"]["fetch_status"], "no_data_paper_anchor")
        self.assertIsNone(out["metadata"]["anchor_judgment_model"])
        [anchor] = out["metadata"]["anchors"]
        self.assertEqual((anchor["kept"], anchor["kept_reason"]), (False, "unjudged"))
        self.assertIsNone(anchor["classification"])

    def test_judge_model_names_the_trusted_judge(self) -> None:
        ref = self._ref("10.1038/data.paper")
        _write_sidecar(
            self.judgments_dir,
            "nm000041",
            [_judgment(ref.identifier, "data_paper")],
            model="claude-opus-5-5",
        )
        backend = _StubBackend(
            {
                ref.identifier: FetchSuccess(
                    [_make_work("c", doi="10.5/c", source_doi=ref.identifier)]
                )
            }
        )
        out = fetch_dataset_citations_via_opencite(
            "nm000041",
            backend=backend,
            nemar_source=_StubSource(FetchSuccess([ref])),
            fetch_date=WHEN,
            judgments_dir=self.judgments_dir,
            judge_model="claude-opus-5-5",
        )
        self.assertEqual(out["num_citations"], 1)
        self.assertEqual(out["metadata"]["anchor_judgment_model"], "claude-opus-5-5")

    def test_unreadable_sidecar_fetches_nothing_and_says_so(self) -> None:
        ref = self._ref("10.1038/data.paper")
        (self.judgments_dir / "nm000042.json").write_text(
            "<<<<<<< HEAD\n{}", encoding="utf-8"
        )
        with self.assertLogs("dataset_citations", "ERROR"):
            out = fetch_dataset_citations_via_opencite(
                "nm000042",
                backend=_ExplodingBackend(),
                nemar_source=_StubSource(FetchSuccess([ref])),
                fetch_date=WHEN,
                judgments_dir=self.judgments_dir,
            )
        self.assertEqual(out["metadata"]["fetch_status"], "judgment_unreadable")
        self.assertEqual(out["num_citations"], 0)

    def test_unjudged_identity_relation_is_fetched(self) -> None:
        """nm000114's figshare deposit: no judgment, but IsIdenticalTo."""
        deposit = self._ref("10.6084/m9.figshare.1", relation="IsIdenticalTo")
        backend = _StubBackend(
            {
                deposit.identifier: FetchSuccess(
                    [
                        _make_work(
                            "Uses the deposit",
                            doi="10.5/d",
                            source_doi=deposit.identifier,
                            source_relation="IsIdenticalTo",
                        )
                    ]
                )
            }
        )
        out = fetch_dataset_citations_via_opencite(
            "nm000043",
            backend=backend,
            nemar_source=_StubSource(FetchSuccess([deposit])),
            fetch_date=WHEN,
            judgments_dir=self.judgments_dir,  # no sidecar
        )
        self.assertEqual(backend.calls, [[deposit.identifier]])
        self.assertEqual(out["num_citations"], 1)
        [anchor] = out["metadata"]["anchors"]
        self.assertEqual(anchor["kept_reason"], "dataset_record")

    def test_own_record_is_not_its_own_citer(self) -> None:
        own = "10.82901/nemar.on004554"
        paper = self._ref("10.3934/mbe.2023507")
        _write_sidecar(
            self.judgments_dir,
            "on004554",
            [_judgment(paper.identifier, "data_paper", paper_year=2023)],
        )
        backend = _StubBackend(
            {
                paper.identifier: FetchSuccess(
                    [
                        _make_work(
                            "Forced Picture Naming Task",
                            doi="10.82901/nemar.on004554.v1.0.0",
                            source_doi=paper.identifier,
                            year=2026,
                        ),
                        _make_work(
                            "A real citer", doi="10.5/r", source_doi=paper.identifier
                        ),
                    ]
                ),
                own: FetchSuccess([]),
            }
        )
        out = fetch_dataset_citations_via_opencite(
            "on004554",
            backend=backend,
            nemar_source=_StubSource(FetchSuccess([paper])),
            catalog_doi=own,
            fetch_date=WHEN,
            judgments_dir=self.judgments_dir,
        )
        self.assertEqual(
            [c["title"] for c in out["citation_details"]], ["A real citer"]
        )

    def test_temporal_guard_uses_the_anchor_a_work_can_cite(self) -> None:
        """A 2016 work reached through a 2019 anchor first and a 2015 anchor
        second is kept, recorded under the 2015 anchor it can cite."""
        new = self._ref("10.1/new")
        old = self._ref("10.1/old")
        _write_sidecar(
            self.judgments_dir,
            "nm000044",
            [
                _judgment(new.identifier, "data_paper", paper_year=2019),
                _judgment(old.identifier, "data_paper", paper_year=2015),
            ],
        )
        backend = _StubBackend(
            {
                ref.identifier: FetchSuccess(
                    [
                        _make_work(
                            "Both",
                            doi="10.5/both",
                            source_doi=ref.identifier,
                            year=2016,
                        )
                    ]
                )
                for ref in (new, old)
            }
        )
        out = fetch_dataset_citations_via_opencite(
            "nm000044",
            backend=backend,
            nemar_source=_StubSource(FetchSuccess([new, old])),
            fetch_date=WHEN,
            judgments_dir=self.judgments_dir,
        )
        [record] = out["citation_details"]
        self.assertEqual(record["source_doi"], old.identifier)


class SidecarShapeRegressionTests(TestCase):
    """Pin the schema contract phase 4 will consume."""

    def test_judgment_sidecar_is_frozen_dataclass(self) -> None:
        # If phase 4 starts mutating sidecar.lookup, this test fails loudly.
        sidecar = JudgmentSidecar(present=True, model="m", lookup={"a": "b"})
        with self.assertRaises(dataclasses.FrozenInstanceError):
            sidecar.lookup = {}  # type: ignore[misc]
