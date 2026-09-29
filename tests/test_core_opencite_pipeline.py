"""Tests for src/dataset_citations/core/opencite_pipeline.py.

The opencite backend is replaced in unit tests by a subclass that returns
hardcoded `FetchSuccess` / `FetchError` results. This is real code (no
unittest.mock), keeps the no-mocks rule, and lets us exercise the dedup,
error-propagation, and schema-shape branches without hitting the network.

Integration tests run live opencite lookups and are gated behind
RUN_INTEGRATION_TESTS=1.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless

from dataset_citations.backends.opencite_backend import OpenCiteBackend
from dataset_citations.core.opencite_pipeline import (
    fetch_dataset_citations_via_opencite,
)
from dataset_citations.sources.models import (
    Author,
    CitingWork,
    DoiReference,
    FetchError,
    FetchSuccess,
)


def _make_work(
    title: str,
    *,
    doi: str | None,
    source_doi: str,
    source_relation: str = "References",
    citation_count: int = 10,
) -> CitingWork:
    return CitingWork(
        title=title,
        doi=doi,
        pmid=None,
        openalex_id=None,
        year=2020,
        authors=(Author(name="A. Researcher"),),
        venue="Journal of Tests",
        abstract=None,
        citation_count=citation_count,
        source_doi=source_doi,
        source_relation=source_relation,  # type: ignore[arg-type]
    )


class _StubBackend(OpenCiteBackend):
    """OpenCiteBackend test double that returns a preset batch result without
    touching the network."""

    def __init__(self, batch_outcome: dict[str, FetchSuccess | FetchError]):
        # Skip parent init so we don't try to read OPENCITE config.
        self._batch_outcome = batch_outcome
        self._config = None  # type: ignore[assignment]
        self._max_results_per_doi = 100
        self._concurrency = 1

    def get_citing_works_batch(self, refs):  # type: ignore[override]
        return {r.identifier: self._batch_outcome[r.identifier] for r in refs}


class _StubSource:
    """Real source-shaped object returning a fixed FetchResult."""

    def __init__(self, outcome):
        self._outcome = outcome

    def get_doi_references(self, dataset_id):
        return self._outcome


WHEN = datetime(2026, 5, 18, tzinfo=UTC)


def _judged(
    test: TestCase,
    dataset_id: str,
    refs: list[DoiReference],
    *,
    paper_year: int | None = None,
) -> Path:
    """Write a sidecar judging every ref `data_paper`; return its directory.

    The anchor gate fails closed (#241): only an anchor with a successful
    `data_paper` judgment is fetched, so tests that exercise the fetch path
    have to supply one. The tempdir also keeps the test off the repo's real
    `citations/anchor_judgments/`.
    """
    directory = Path(test.enterContext(TemporaryDirectory()))
    judgments = [
        {
            "anchor_identifier": ref.identifier,
            "anchor_identifier_type": ref.identifier_type,
            "source_relation": ref.relation_type,
            "classification": "data_paper",
            "reason": "test fixture",
            "paper_title": f"Data paper {ref.identifier}",
            "paper_year": paper_year,
            "paper_venue": None,
            "judged_at": WHEN.isoformat(),
            "error": None,
        }
        for ref in refs
    ]
    payload = {
        "dataset_id": dataset_id,
        "judged_at": WHEN.isoformat(),
        "judgment_model": "test-judge",
        "judgments": judgments,
    }
    (directory / f"{dataset_id}.json").write_text(json.dumps(payload), "utf-8")
    return directory


class FetchViaOpenCiteTests(TestCase):
    def test_unsupported_prefix_returns_stub(self) -> None:
        out = fetch_dataset_citations_via_opencite(
            "xx123", fetch_date=WHEN, backend=_StubBackend({})
        )
        self.assertEqual(out["num_citations"], 0)
        self.assertIn("unsupported_prefix", out["metadata"]["fetch_status"])

    def test_source_error_propagated(self) -> None:
        bids = _StubSource(FetchError("not_found", "missing"))
        out = fetch_dataset_citations_via_opencite(
            "ds000999",
            backend=_StubBackend({}),
            bids_source=bids,
            fetch_date=WHEN,
        )
        self.assertEqual(out["metadata"]["fetch_status"], "not_found")
        self.assertEqual(out["citation_details"], [])

    def test_no_references_returns_no_doi_status(self) -> None:
        bids = _StubSource(FetchSuccess([]))
        out = fetch_dataset_citations_via_opencite(
            "ds000999",
            backend=_StubBackend({}),
            bids_source=bids,
            fetch_date=WHEN,
        )
        self.assertEqual(out["metadata"]["fetch_status"], "no_doi_references")

    def test_happy_path_produces_schema_v2_1(self) -> None:
        ref = DoiReference(
            identifier="10.1038/sdata.2015.1",
            identifier_type="doi",
            relation_type="References",
            source="openneuro_description",
            source_field="ReferencesAndLinks",
        )
        bids = _StubSource(FetchSuccess([ref]))
        backend = _StubBackend(
            {
                ref.identifier: FetchSuccess(
                    [
                        _make_work(
                            "Method paper",
                            doi="10.1234/method",
                            source_doi=ref.identifier,
                        )
                    ]
                )
            }
        )
        out = fetch_dataset_citations_via_opencite(
            "ds000117",
            backend=backend,
            bids_source=bids,
            fetch_date=WHEN,
            judgments_dir=_judged(self, "ds000117", [ref]),
        )
        self.assertEqual(out["dataset_id"], "ds000117")
        self.assertEqual(out["num_citations"], 1)
        self.assertEqual(out["metadata"]["schema_version"], "2.1")
        self.assertEqual(out["metadata"]["discovery_backend"], "opencite")
        self.assertEqual(out["metadata"]["anchor_count"], 1)
        # Phase 3 review: happy path must be distinguishable from "stub empty".
        self.assertEqual(out["metadata"]["fetch_status"], "success")
        self.assertEqual(out["metadata"]["anchor_errors"], {})
        # Schema v2.1: anchors[] superset + searched_dois present.
        self.assertEqual(len(out["metadata"]["anchors"]), 1)
        self.assertTrue(out["metadata"]["anchors"][0]["kept"])
        self.assertEqual(out["metadata"]["anchors"][0]["classification"], "data_paper")
        self.assertEqual(
            out["metadata"]["anchors"][0]["kept_reason"], "judged_data_paper"
        )
        self.assertEqual(out["metadata"]["searched_dois"], ["10.1038/sdata.2015.1"])
        # Legacy ds-* (BIDS source) has no rich metadata -> empty keys present.
        self.assertEqual(out["metadata"]["keywords"], [])
        self.assertIsNone(out["metadata"]["methods_description"])
        self.assertEqual(out["metadata"]["funding"], [])
        entry = out["citation_details"][0]
        self.assertEqual(entry["title"], "Method paper")
        self.assertEqual(entry["source_doi"], "10.1038/sdata.2015.1")
        self.assertEqual(entry["source_relation"], "References")
        self.assertEqual(entry["discovery_backend"], "opencite")

    def test_dedup_across_anchors(self) -> None:
        ref_a = DoiReference(
            identifier="10.1234/A",
            identifier_type="doi",
            relation_type="References",
            source="nemar_metadata",
        )
        ref_b = DoiReference(
            identifier="10.1234/B",
            identifier_type="doi",
            relation_type="IsDerivedFrom",
            source="nemar_metadata",
        )
        shared_work = _make_work(
            "Same paper cites both", doi="10.5555/cited", source_doi=ref_a.identifier
        )
        unique_to_b = _make_work(
            "Only cites B", doi="10.5555/only-b", source_doi=ref_b.identifier
        )
        nemar = _StubSource(FetchSuccess([ref_a, ref_b]))
        backend = _StubBackend(
            {
                ref_a.identifier: FetchSuccess([shared_work]),
                ref_b.identifier: FetchSuccess([shared_work, unique_to_b]),
            }
        )
        out = fetch_dataset_citations_via_opencite(
            "nm000103",
            backend=backend,
            nemar_source=nemar,
            fetch_date=WHEN,
            judgments_dir=_judged(self, "nm000103", [ref_a, ref_b]),
        )
        self.assertEqual(out["num_citations"], 2)
        titles = {c["title"] for c in out["citation_details"]}
        self.assertEqual(titles, {"Same paper cites both", "Only cites B"})

    def test_all_anchors_rate_limited_surfaces_dominant_reason(self) -> None:
        ref = DoiReference(
            identifier="10.1234/A",
            identifier_type="doi",
            relation_type="References",
            source="nemar_metadata",
        )
        nemar = _StubSource(FetchSuccess([ref]))
        backend = _StubBackend(
            {ref.identifier: FetchError("rate_limit", "429 too many")}
        )
        out = fetch_dataset_citations_via_opencite(
            "nm000103",
            backend=backend,
            nemar_source=nemar,
            fetch_date=WHEN,
            judgments_dir=_judged(self, "nm000103", [ref]),
        )
        self.assertEqual(out["metadata"]["fetch_status"], "rate_limit")
        self.assertEqual(
            out["metadata"]["anchor_errors"], {ref.identifier: "rate_limit"}
        )
        self.assertEqual(out["citation_details"], [])

    def test_partial_failure_keeps_anchor_errors(self) -> None:
        """When some anchors succeed and others fail, the successes survive
        and the failing-anchor errors are recorded for downstream debugging."""
        ref_ok = DoiReference(
            identifier="10.1234/OK",
            identifier_type="doi",
            relation_type="References",
            source="nemar_metadata",
        )
        ref_bad = DoiReference(
            identifier="10.1234/BAD",
            identifier_type="doi",
            relation_type="IsDerivedFrom",
            source="nemar_metadata",
        )
        nemar = _StubSource(FetchSuccess([ref_ok, ref_bad]))
        backend = _StubBackend(
            {
                ref_ok.identifier: FetchSuccess(
                    [
                        _make_work(
                            "good paper", doi="10.5/good", source_doi=ref_ok.identifier
                        )
                    ]
                ),
                ref_bad.identifier: FetchError("rate_limit", "429"),
            }
        )
        out = fetch_dataset_citations_via_opencite(
            "nm000103",
            backend=backend,
            nemar_source=nemar,
            fetch_date=WHEN,
            judgments_dir=_judged(self, "nm000103", [ref_ok, ref_bad]),
        )
        self.assertEqual(out["num_citations"], 1)
        self.assertEqual(out["metadata"]["fetch_status"], "partial")
        self.assertEqual(
            out["metadata"]["anchor_errors"], {ref_bad.identifier: "rate_limit"}
        )

    def test_dominant_error_reason_tiebreak(self) -> None:
        """When all anchors fail with mixed reasons, _dominant_error_reason
        picks the most frequent. The tie-break path was previously untested."""
        refs = [
            DoiReference(
                identifier=f"10.1234/{label}",
                identifier_type="doi",
                relation_type="References",
                source="nemar_metadata",
            )
            for label in ("A", "B", "C")
        ]
        nemar = _StubSource(FetchSuccess(refs))
        backend = _StubBackend(
            {
                refs[0].identifier: FetchError("rate_limit", "429"),
                refs[1].identifier: FetchError("rate_limit", "429"),
                refs[2].identifier: FetchError("not_found", "404"),
            }
        )
        out = fetch_dataset_citations_via_opencite(
            "nm000103",
            backend=backend,
            nemar_source=nemar,
            fetch_date=WHEN,
            judgments_dir=_judged(self, "nm000103", refs),
        )
        # Two rate_limit vs one not_found; dominant should win.
        self.assertEqual(out["metadata"]["fetch_status"], "rate_limit")

    def test_nm_prefix_constructs_default_source_when_missing(self) -> None:
        """fetch_dataset_citations_via_opencite("nm...") with no nemar_source
        must construct a NemarMetadataSource. We substitute the class via
        setattr so the test stays offline."""
        import dataset_citations.core.opencite_pipeline as pipeline_module

        recorded: list[str] = []

        class RecordingNemarSource:
            def __init__(self, github_token=None):
                recorded.append("constructed")

            def get_doi_references(self, dataset_id):
                return FetchSuccess([])

        original = pipeline_module.NemarMetadataSource
        pipeline_module.NemarMetadataSource = RecordingNemarSource  # type: ignore[assignment,misc]
        try:
            out = fetch_dataset_citations_via_opencite(
                "nm000103", backend=_StubBackend({}), fetch_date=WHEN
            )
        finally:
            pipeline_module.NemarMetadataSource = original  # type: ignore[assignment]
        self.assertEqual(recorded, ["constructed"])
        self.assertEqual(out["metadata"]["fetch_status"], "no_doi_references")


class CatalogDoiSeeding(TestCase):
    """The catalog_doi parameter adds an extra anchor with relation=References."""

    def test_seeds_extra_anchor_when_no_overlap(self) -> None:
        ref = DoiReference(
            identifier="10.1038/source-ref.1",
            identifier_type="doi",
            relation_type="IsDerivedFrom",
            source="nemar_metadata",
        )
        catalog_doi = "10.82901/nemar.nm000999"
        catalog_work = _make_work(
            "Paper citing the dataset",
            doi="10.5555/cites-catalog",
            source_doi=catalog_doi,
        )
        source_work = _make_work(
            "Paper citing methods",
            doi="10.5555/cites-source",
            source_doi=ref.identifier,
        )
        nemar = _StubSource(FetchSuccess([ref]))
        backend = _StubBackend(
            {
                ref.identifier: FetchSuccess([source_work]),
                catalog_doi: FetchSuccess([catalog_work]),
            }
        )
        out = fetch_dataset_citations_via_opencite(
            "nm000999",
            backend=backend,
            nemar_source=nemar,
            catalog_doi=catalog_doi,
            fetch_date=WHEN,
            # Only the source ref is judged: the catalog DOI is the dataset's
            # own concept DOI, which the gate keeps without a judgment.
            judgments_dir=_judged(self, "nm000999", [ref]),
        )
        self.assertEqual(out["metadata"]["anchor_count"], 2)
        reasons = {
            a["identifier"]: a["kept_reason"] for a in out["metadata"]["anchors"]
        }
        self.assertEqual(
            reasons,
            {ref.identifier: "judged_data_paper", catalog_doi: "own_doi"},
        )
        titles = {c["title"] for c in out["citation_details"]}
        self.assertIn("Paper citing the dataset", titles)
        self.assertIn("Paper citing methods", titles)

    def test_catalog_doi_dedup_against_source_ref(self) -> None:
        """When the catalog DOI is already in the source's related_identifiers,
        it must not be added twice and the source's relation_type wins."""
        shared = "10.1038/shared.1"
        ref = DoiReference(
            identifier=shared,
            identifier_type="doi",
            relation_type="IsIdenticalTo",
            source="nemar_metadata",
        )
        nemar = _StubSource(FetchSuccess([ref]))
        backend = _StubBackend({shared: FetchSuccess([])})
        out = fetch_dataset_citations_via_opencite(
            "nm000999",
            backend=backend,
            nemar_source=nemar,
            catalog_doi=shared,
            fetch_date=WHEN,
        )
        # Single anchor preserved; identity comes from the source-derived ref.
        self.assertEqual(out["metadata"]["anchor_count"], 1)

    def test_malformed_catalog_doi_ignored(self) -> None:
        ref = DoiReference(
            identifier="10.1038/keep.1",
            identifier_type="doi",
            relation_type="References",
            source="nemar_metadata",
        )
        nemar = _StubSource(FetchSuccess([ref]))
        backend = _StubBackend({ref.identifier: FetchSuccess([])})
        out = fetch_dataset_citations_via_opencite(
            "nm000999",
            backend=backend,
            nemar_source=nemar,
            catalog_doi="not-a-doi",
            fetch_date=WHEN,
        )
        self.assertEqual(out["metadata"]["anchor_count"], 1)


class CheckpointResume(TestCase):
    """Pipeline integrates with the CheckpointStore."""

    def setUp(self) -> None:
        from pathlib import Path

        self.tmp = Path(__file__).parent / "test_data" / "_tmp_pipeline_ckpt"
        if self.tmp.exists():
            import shutil

            shutil.rmtree(self.tmp)
        self.tmp.mkdir(parents=True, exist_ok=True)

        from dataset_citations.core.checkpoint import CheckpointStore

        self.store = CheckpointStore(base_dir=self.tmp)

    def tearDown(self) -> None:
        import shutil

        if self.tmp.exists():
            shutil.rmtree(self.tmp)

    def test_first_run_writes_checkpoint(self) -> None:
        ref = DoiReference(
            identifier="10.1038/anchor.1",
            identifier_type="doi",
            relation_type="References",
            source="nemar_metadata",
        )
        nemar = _StubSource(FetchSuccess([ref]))
        backend = _StubBackend(
            {
                ref.identifier: FetchSuccess(
                    [_make_work("W", doi="10.5555/w", source_doi=ref.identifier)]
                )
            }
        )
        out = fetch_dataset_citations_via_opencite(
            "nm000001",
            backend=backend,
            nemar_source=nemar,
            fetch_date=WHEN,
            checkpoint_store=self.store,
            use_checkpoint=True,
            judgments_dir=_judged(self, "nm000001", [ref]),
        )
        # On full success, the checkpoint file is removed.
        self.assertEqual(out["metadata"]["fetch_status"], "success")
        self.assertFalse(self.store.path_for("nm000001").exists())

    def test_partial_failure_preserves_checkpoint_for_resume(self) -> None:
        ref_ok = DoiReference(
            identifier="10.1038/ok.1",
            identifier_type="doi",
            relation_type="References",
            source="nemar_metadata",
        )
        ref_fail = DoiReference(
            identifier="10.1038/fail.1",
            identifier_type="doi",
            relation_type="References",
            source="nemar_metadata",
        )
        nemar = _StubSource(FetchSuccess([ref_ok, ref_fail]))
        backend = _StubBackend(
            {
                ref_ok.identifier: FetchSuccess(
                    [_make_work("OK", doi="10.5555/ok", source_doi=ref_ok.identifier)]
                ),
                ref_fail.identifier: FetchError("rate_limit", "429"),
            }
        )
        out = fetch_dataset_citations_via_opencite(
            "nm000001",
            backend=backend,
            nemar_source=nemar,
            fetch_date=WHEN,
            checkpoint_store=self.store,
            use_checkpoint=True,
            judgments_dir=_judged(self, "nm000001", [ref_ok, ref_fail]),
        )
        self.assertEqual(out["metadata"]["fetch_status"], "partial")
        # The successful anchor must be recorded so the next run skips it.
        loaded = self.store.load("nm000001")
        self.assertTrue(loaded.is_success(ref_ok.identifier))
        self.assertFalse(loaded.is_success(ref_fail.identifier))

    def test_second_run_reuses_checkpoint_and_skips_backend(self) -> None:
        ref = DoiReference(
            identifier="10.1038/anchor.1",
            identifier_type="doi",
            relation_type="References",
            source="nemar_metadata",
        )
        # Pre-populate the checkpoint with a successful result.
        self.store.record_anchor(
            "nm000001",
            ref.identifier,
            FetchSuccess(
                [_make_work("Cached", doi="10.5555/cached", source_doi=ref.identifier)]
            ),
        )

        # A backend that would explode if called: we expect zero calls.
        class _ExplodingBackend(_StubBackend):
            def __init__(self) -> None:
                super().__init__({})

            def get_citing_works_batch(self, refs):  # type: ignore[override]
                raise AssertionError(
                    "backend should not be called when all anchors checkpointed"
                )

        nemar = _StubSource(FetchSuccess([ref]))
        out = fetch_dataset_citations_via_opencite(
            "nm000001",
            backend=_ExplodingBackend(),
            nemar_source=nemar,
            fetch_date=WHEN,
            checkpoint_store=self.store,
            use_checkpoint=True,
            judgments_dir=_judged(self, "nm000001", [ref]),
        )
        self.assertEqual(out["num_citations"], 1)
        self.assertEqual(out["citation_details"][0]["title"], "Cached")


@skipUnless(
    os.getenv("RUN_INTEGRATION_TESTS"),
    "live opencite lookup; set RUN_INTEGRATION_TESTS=1 to enable",
)
class FetchViaOpenCiteIntegration(TestCase):
    def test_ds000117_live(self) -> None:
        out = fetch_dataset_citations_via_opencite("ds000117")
        # Either we get real citations or a transient FetchError; both are
        # acceptable for the integration test as long as the schema is right.
        self.assertEqual(out["dataset_id"], "ds000117")
        self.assertEqual(out["metadata"]["schema_version"], "2.1")
        self.assertEqual(out["metadata"]["discovery_backend"], "opencite")
        if out["num_citations"] > 0:
            entry = out["citation_details"][0]
            self.assertTrue(entry.get("title"))
            self.assertEqual(entry.get("discovery_backend"), "opencite")
