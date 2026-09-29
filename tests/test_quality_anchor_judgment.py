"""Tests for `dataset_citations.quality.anchor_judgment`.

No mocks. The LLM and opencite layers are replaced with real subclasses that
return hand-built records (the pattern used by
`tests/test_quality_llm_client.py::_RecordedClient` and
`tests/test_core_opencite_pipeline.py::_StubBackend`). The fake judge answers
in the real `claude -p --output-format json` shape, recorded on hallu.

The live test against the real `claude` CLI and opencite is gated behind
RUN_CLAUDE_JUDGE_TESTS=1.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import TestCase, skipUnless

from dataset_citations.backends.opencite_backend import OpenCiteBackend
from dataset_citations.quality.anchor_judgment import (
    DatasetJudgmentRun,
    JudgmentRecord,
    is_judgment_fresh,
    judge_dataset_anchors,
    load_judgment_sidecar,
    save_judgment_sidecar,
)
from dataset_citations.quality.dataset_metadata import DatasetMetadataRetriever
from dataset_citations.quality.llm_client import ClaudeCliJudgmentClient
from dataset_citations.sources.models import (
    Author,
    CitingWork,
    DoiReference,
    FetchError,
    FetchSuccess,
)

_RECORDED_OK = json.loads(
    (Path(__file__).parent / "test_data" / "claude_cli_judgment_ok.json").read_text(
        "utf-8"
    )
)


def _cli_output(classification: str, reason: str = "ok") -> str:
    """The recorded CLI success output carrying the given verdict."""
    payload = dict(_RECORDED_OK)
    payload["structured_output"] = {"classification": classification, "reason": reason}
    return json.dumps(payload)


class _FakeClient(ClaudeCliJudgmentClient):
    """Real subclass that answers from a table instead of spawning the CLI.

    `verdicts` maps an anchor DOI (found in the prompt) to a classification;
    `default` answers everything else. A classification of "garbage_label" is
    returned as-is so the client's taxonomy validation rejects it. Thread-safe,
    so the concurrent path can be exercised.
    """

    def __init__(
        self,
        default: str = "data_paper",
        verdicts: dict[str, str] | None = None,
        model: str = "test-model",
    ) -> None:
        super().__init__(model=model, claude_bin="unused", timeout=5)
        self._default = default
        self._verdicts = verdicts or {}
        self._lock = threading.Lock()
        self.calls: list[str] = []

    def _run_cli(self, prompt: str) -> str:
        doi = next((d for d in self._verdicts if f"DOI: {d}" in prompt), None)
        with self._lock:
            self.calls.append(doi or "?")
        return _cli_output(
            self._verdicts.get(doi, self._default) if doi else self._default
        )


class _NeverCalledClient(_FakeClient):
    def _run_cli(self, prompt: str) -> str:
        raise AssertionError("the judge must not be called")


class _StubBackend(OpenCiteBackend):
    """OpenCiteBackend test double that returns preset `get_paper` results."""

    def __init__(self, papers: dict[str, FetchSuccess | FetchError]) -> None:
        # Skip parent init so we don't read OPENCITE config.
        self._papers = papers
        self._config = None  # type: ignore[assignment]
        self._max_results_per_doi = 100
        self._concurrency = 1

    def get_paper(self, doi: str):  # type: ignore[override]
        return self._papers[doi]


class _StubSource:
    """Real source-shaped object returning a fixed FetchResult."""

    def __init__(self, outcome) -> None:
        self._outcome = outcome

    def get_doi_references(self, dataset_id):
        return self._outcome


class _StubMetadataRetriever:
    """Real retriever-shaped object returning a canned dataset metadata dict."""

    def __init__(self, text: str = "Stub dataset description.") -> None:
        self._text = text

    def get_dataset_metadata(self, dataset_id: str) -> dict:
        return {
            "dataset_id": dataset_id,
            "dataset_description": {"Name": self._text},
            "readme_content": None,
            "github_info": {"description": None},
        }


def _make_work(
    *,
    title: str = "Sample paper",
    doi: str = "10.1234/example",
    venue: str | None = "Journal of Tests",
    year: int | None = 2024,
    abstract: str | None = "A stub abstract.",
    source_doi: str | None = None,
) -> CitingWork:
    return CitingWork(
        title=title,
        doi=doi,
        pmid=None,
        openalex_id=None,
        year=year,
        authors=(Author(name="A. Researcher"),),
        venue=venue,
        abstract=abstract,
        citation_count=10,
        source_doi=source_doi or doi,
        source_relation="References",
    )


def _ref(identifier: str, relation: str = "References") -> DoiReference:
    return DoiReference(
        identifier=identifier,
        identifier_type="doi",
        relation_type=relation,  # type: ignore[arg-type]
        source="nemar_metadata",
    )


def _papers(*refs: DoiReference) -> _StubBackend:
    return _StubBackend(
        {
            r.identifier: FetchSuccess(
                _make_work(title=f"Paper {r.identifier}", source_doi=r.identifier)
            )
            for r in refs
        }
    )


def _judge(
    refs_outcome,
    backend: _StubBackend,
    client: ClaudeCliJudgmentClient,
    *,
    dataset_id: str = "nm000104",
    previous: dict | None = None,
    max_workers: int = 1,
) -> DatasetJudgmentRun | None:
    source = _StubSource(refs_outcome)
    return judge_dataset_anchors(
        dataset_id,
        nemar_source=source,  # type: ignore[arg-type]
        bids_source=source,  # type: ignore[arg-type]
        metadata_retriever=_StubMetadataRetriever(),  # type: ignore[arg-type]
        backend=backend,
        client=client,
        previous=previous,
        max_workers=max_workers,
    )


def _previous(model: str, *records: JudgmentRecord) -> dict:
    return {
        "dataset_id": "nm000104",
        "judged_at": "2026-09-01T00:00:00+00:00",
        "judgment_model": model,
        "judgments": [r.to_dict() for r in records],
    }


def _record(identifier: str, classification: str, relation: str = "References"):
    return JudgmentRecord(
        anchor_identifier=identifier,
        anchor_identifier_type="doi",
        source_relation=relation,
        classification=classification,
        reason="earlier verdict",
        paper_title=f"Paper {identifier}",
        paper_year=2019,
        paper_venue="Journal of Tests",
        judged_at="2026-09-01T00:00:00+00:00",
        error=None,
    )


class JudgeDatasetAnchorsTests(TestCase):
    def test_happy_path_produces_locked_schema(self) -> None:
        ref = _ref("10.1234/example", "IsDerivedFrom")
        run = _judge(
            FetchSuccess([ref]),
            _StubBackend(
                {ref.identifier: FetchSuccess(_make_work(source_doi=ref.identifier))}
            ),
            _FakeClient("umbrella"),
        )
        assert run is not None
        self.assertEqual((run.judged, run.failed, run.reused), (1, 0, 0))
        payload = run.payload
        self.assertEqual(payload["dataset_id"], "nm000104")
        self.assertEqual(payload["judgment_model"], "test-model")
        self.assertIn("judged_at", payload)
        self.assertEqual(len(payload["judgments"]), 1)

        judgment = payload["judgments"][0]
        self.assertEqual(judgment["anchor_identifier"], "10.1234/example")
        self.assertEqual(judgment["anchor_identifier_type"], "doi")
        self.assertEqual(judgment["source_relation"], "IsDerivedFrom")
        self.assertEqual(judgment["classification"], "umbrella")
        self.assertEqual(judgment["reason"], "ok")
        self.assertEqual(judgment["paper_title"], "Sample paper")
        self.assertEqual(judgment["paper_year"], 2024)
        self.assertEqual(judgment["paper_venue"], "Journal of Tests")
        self.assertIsNone(judgment["error"])

    def test_ds_prefix_routes_to_bids_source(self) -> None:
        """ds-prefixed ids must consult `bids_source`, not `nemar_source`."""
        ref = _ref("10.5555/bids")
        run = judge_dataset_anchors(
            "ds000117",
            nemar_source=_StubSource(FetchError("not_found", "not for ds")),  # type: ignore[arg-type]
            bids_source=_StubSource(FetchSuccess([ref])),  # type: ignore[arg-type]
            metadata_retriever=_StubMetadataRetriever(),  # type: ignore[arg-type]
            backend=_papers(ref),
            client=_FakeClient("data_paper"),
        )
        assert run is not None
        self.assertEqual(run.payload["judgments"][0]["classification"], "data_paper")

    def test_source_error_returns_none_so_the_sidecar_is_untouched(self) -> None:
        """#241: a failed source lookup used to overwrite good judgments with
        an empty list. Now the caller gets None and writes nothing."""
        with self.assertLogs("dataset_citations.quality.anchor_judgment", "WARNING"):
            run = _judge(
                FetchError("rate_limit", "github 403"),
                _StubBackend({}),
                _NeverCalledClient(),
            )
        self.assertIsNone(run)

    def test_no_doi_anchors_produces_empty_judgments(self) -> None:
        run = _judge(FetchSuccess([]), _StubBackend({}), _NeverCalledClient())
        assert run is not None
        self.assertEqual(run.payload["judgments"], [])

    def test_paper_lookup_failure_records_error(self) -> None:
        ref = _ref("10.1234/missing")
        run = _judge(
            FetchSuccess([ref]),
            _StubBackend({ref.identifier: FetchError("not_found", "openalex 404")}),
            _NeverCalledClient(),
        )
        assert run is not None
        self.assertEqual((run.judged, run.failed, run.lookup_failed), (0, 0, 1))
        judgment = run.payload["judgments"][0]
        self.assertIn("paper_lookup_failed", judgment["error"])
        self.assertIsNone(judgment["paper_title"])
        self.assertEqual(judgment["classification"], "")

    def test_llm_judgment_failure_records_error(self) -> None:
        """An out-of-taxonomy answer records `error` but keeps the paper bib
        fields the lookup did fetch."""
        ref = _ref("10.1234/example")
        run = _judge(
            FetchSuccess([ref]),
            _StubBackend(
                {ref.identifier: FetchSuccess(_make_work(source_doi=ref.identifier))}
            ),
            _FakeClient("garbage_label"),
        )
        assert run is not None
        self.assertEqual((run.judged, run.failed), (0, 1))
        judgment = run.payload["judgments"][0]
        self.assertIn("llm_judgment_failed", judgment["error"])
        self.assertEqual(judgment["paper_title"], "Sample paper")
        self.assertEqual(judgment["paper_year"], 2024)
        self.assertEqual(judgment["classification"], "")

    def test_concurrent_judgments_keep_anchor_order(self) -> None:
        refs = [_ref(f"10.1234/{c}") for c in "ABCDE"]
        verdicts = {r.identifier: "methodology" for r in refs}
        verdicts[refs[2].identifier] = "data_paper"
        client = _FakeClient(verdicts=verdicts)
        run = _judge(FetchSuccess(refs), _papers(*refs), client, max_workers=4)
        assert run is not None
        self.assertEqual(run.judged, 5)
        self.assertEqual(
            [j["anchor_identifier"] for j in run.payload["judgments"]],
            [r.identifier for r in refs],
        )
        self.assertEqual(
            [j["classification"] for j in run.payload["judgments"]],
            ["methodology", "methodology", "data_paper", "methodology", "methodology"],
        )


class ReuseAndCarryForwardTests(TestCase):
    """How a run treats the sidecar already on disk (#241)."""

    def test_same_model_same_relation_is_reused_without_a_call(self) -> None:
        ref = _ref("10.1234/A")
        previous = _previous("test-model", _record(ref.identifier, "data_paper"))
        run = _judge(
            FetchSuccess([ref]), _papers(ref), _NeverCalledClient(), previous=previous
        )
        assert run is not None
        self.assertEqual((run.judged, run.reused), (0, 1))
        judgment = run.payload["judgments"][0]
        self.assertEqual(judgment["classification"], "data_paper")
        self.assertEqual(judgment["judged_at"], "2026-09-01T00:00:00+00:00")

    def test_only_new_anchors_are_judged(self) -> None:
        old, new = _ref("10.1234/old"), _ref("10.1234/new")
        previous = _previous("test-model", _record(old.identifier, "methodology"))
        client = _FakeClient(verdicts={new.identifier: "data_paper"})
        run = _judge(
            FetchSuccess([old, new]), _papers(old, new), client, previous=previous
        )
        assert run is not None
        self.assertEqual(client.calls, [new.identifier])
        self.assertEqual(
            [j["classification"] for j in run.payload["judgments"]],
            ["methodology", "data_paper"],
        )

    def test_changed_relation_is_rejudged(self) -> None:
        # An enrichment sweep relabeling References -> IsDescribedBy changes
        # the judge's input, so the old verdict is not reused.
        ref = _ref("10.1234/A", "IsDescribedBy")
        previous = _previous("test-model", _record(ref.identifier, "related_work"))
        client = _FakeClient("data_paper")
        run = _judge(FetchSuccess([ref]), _papers(ref), client, previous=previous)
        assert run is not None
        self.assertEqual((run.judged, run.reused), (1, 0))
        self.assertEqual(run.payload["judgments"][0]["classification"], "data_paper")
        self.assertEqual(
            run.payload["judgments"][0]["source_relation"], "IsDescribedBy"
        )

    def test_other_models_verdicts_are_never_reused(self) -> None:
        ref = _ref("10.1234/A")
        previous = _previous("gemma4:e4b", _record(ref.identifier, "data_paper"))
        client = _FakeClient("related_work")
        run = _judge(FetchSuccess([ref]), _papers(ref), client, previous=previous)
        assert run is not None
        self.assertEqual((run.judged, run.reused), (1, 0))
        self.assertEqual(run.payload["judgments"][0]["classification"], "related_work")

    def test_failed_rejudge_keeps_the_previous_same_model_verdict(self) -> None:
        ref = _ref("10.1234/A", "IsDescribedBy")
        previous = _previous(
            "test-model", _record(ref.identifier, "data_paper", relation="References")
        )
        run = _judge(
            FetchSuccess([ref]),
            _papers(ref),
            _FakeClient("garbage_label"),
            previous=previous,
        )
        assert run is not None
        self.assertEqual(run.failed, 1)
        judgment = run.payload["judgments"][0]
        self.assertIsNone(judgment["error"])
        self.assertEqual(judgment["classification"], "data_paper")

    def test_failed_judge_does_not_resurrect_another_models_verdict(self) -> None:
        ref = _ref("10.1234/A")
        previous = _previous("gemma4:e4b", _record(ref.identifier, "data_paper"))
        run = _judge(
            FetchSuccess([ref]),
            _papers(ref),
            _FakeClient("garbage_label"),
            previous=previous,
        )
        assert run is not None
        judgment = run.payload["judgments"][0]
        self.assertIn("llm_judgment_failed", judgment["error"])
        self.assertEqual(judgment["classification"], "")


class SidecarRoundTripTests(TestCase):
    def test_round_trip_preserves_locked_fields(self) -> None:
        record = JudgmentRecord(
            anchor_identifier="10.1234/example",
            anchor_identifier_type="doi",
            source_relation="IsDerivedFrom",
            classification="umbrella",
            reason="broad initiative",
            paper_title="A paper",
            paper_year=2024,
            paper_venue="Journal of Tests",
            judged_at="2026-05-22T12:00:00+00:00",
            error=None,
        )
        payload = {
            "dataset_id": "nm000104",
            "judged_at": "2026-05-22T12:00:00+00:00",
            "judgment_model": "test-model",
            "judgments": [record.to_dict()],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nm000104.json"
            save_judgment_sidecar(path, payload)
            loaded = load_judgment_sidecar(path)
        self.assertEqual(loaded["dataset_id"], "nm000104")
        self.assertEqual(loaded["judgment_model"], "test-model")
        self.assertEqual(len(loaded["judgments"]), 1)
        judgment = loaded["judgments"][0]
        for field in (
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
        ):
            self.assertIn(field, judgment, f"{field} missing after round-trip")
        self.assertEqual(judgment["classification"], "umbrella")
        self.assertIsNone(judgment["error"])

    def test_save_writes_atomically_no_partial(self) -> None:
        """`save_judgment_sidecar` uses os.replace so the target file is
        either the new payload or the previous one — never a half-written
        file. We verify the cleanup-after-failure path keeps the target
        intact."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ds000117.json"
            # Seed an initial payload on disk.
            save_judgment_sidecar(path, {"dataset_id": "ds000117", "judgments": []})

            # Save again with a non-serializable value to force a failure
            # mid-write. The pre-existing file must remain valid JSON.
            bad_payload = {"dataset_id": "ds000117", "judgments": [{"x": object()}]}
            with self.assertRaises(TypeError):
                save_judgment_sidecar(path, bad_payload)

            # File still parses and matches the prior good payload.
            survivor = load_judgment_sidecar(path)
            self.assertEqual(survivor["dataset_id"], "ds000117")
            # No `.tmp` litter left behind in the parent directory.
            leftovers = [p for p in Path(tmp).iterdir() if p.suffix == ".tmp"]
            self.assertEqual(leftovers, [])

    def test_load_rejects_non_object_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(["not", "an", "object"]))
            with self.assertRaises(TypeError):
                load_judgment_sidecar(path)


class IsJudgmentFreshTests(TestCase):
    def test_fresh_payload_returns_true(self) -> None:
        payload = {
            "judged_at": datetime.now(UTC).isoformat(),
        }
        self.assertTrue(is_judgment_fresh(payload, max_age_days=7))

    def test_stale_payload_returns_false(self) -> None:
        stale = datetime.now(UTC) - timedelta(days=30)
        payload = {"judged_at": stale.isoformat()}
        self.assertFalse(is_judgment_fresh(payload, max_age_days=7))

    def test_zero_max_age_days_returns_false(self) -> None:
        payload = {"judged_at": datetime.now(UTC).isoformat()}
        self.assertFalse(is_judgment_fresh(payload, max_age_days=0))

    def test_negative_max_age_days_returns_false(self) -> None:
        payload = {"judged_at": datetime.now(UTC).isoformat()}
        self.assertFalse(is_judgment_fresh(payload, max_age_days=-1))

    def test_missing_judged_at_returns_false(self) -> None:
        self.assertFalse(is_judgment_fresh({}, max_age_days=7))

    def test_malformed_judged_at_returns_false(self) -> None:
        self.assertFalse(is_judgment_fresh({"judged_at": "not-a-date"}, max_age_days=7))

    def test_naive_timestamp_treated_as_utc(self) -> None:
        naive_now = datetime.now(UTC).replace(tzinfo=None).isoformat()
        self.assertTrue(is_judgment_fresh({"judged_at": naive_now}, max_age_days=7))


@skipUnless(
    os.getenv("RUN_CLAUDE_JUDGE_TESTS"),
    "live claude CLI + opencite call; set RUN_CLAUDE_JUDGE_TESTS=1 to enable",
)
class AnchorJudgmentIntegration(TestCase):
    """Live integration test: builds a sidecar for one anchor with the real
    `claude` CLI and a real opencite lookup. Uses a hand-built DoiReference
    to avoid hitting GitHub/NEMAR catalog rate limits."""

    def test_round_trip_via_real_claude(self) -> None:
        ref = DoiReference(
            identifier="10.3389/fnins.2013.00267",
            identifier_type="doi",
            relation_type="IsDerivedFrom",
            source="openneuro_description",
        )

        class _OneRefSource:
            def get_doi_references(self, dataset_id):
                return FetchSuccess([ref])

        run = judge_dataset_anchors(
            "ds000117",
            nemar_source=_OneRefSource(),  # type: ignore[arg-type]
            bids_source=_OneRefSource(),  # type: ignore[arg-type]
            metadata_retriever=DatasetMetadataRetriever(),
            backend=OpenCiteBackend(max_results_per_doi=1),
            client=ClaudeCliJudgmentClient(),
        )
        assert run is not None
        self.assertEqual(run.payload["dataset_id"], "ds000117")
        self.assertEqual(run.payload["judgments"][0]["classification"], "methodology")
