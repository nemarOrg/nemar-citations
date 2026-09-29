"""Tests for `dataset-citations-gate-anchors` (issue #241).

Real data: `tests/test_data/gate_nm000275_citations.json` is a trimmed copy of
the committed nm000275 citation file from the 2026-09-22 fetch (every anchor
fetched because the judge was down), including the PREP pipeline paper that
entered through the 2014 "Kinesthesia" anchor. The sidecar is written per test
in the locked phase 2 shape. The parity tests run the real fetch pipeline with
test doubles at the network boundary and check the sweep agrees with it.
"""

from __future__ import annotations

import io
import json
import shutil
from contextlib import redirect_stdout
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from dataset_citations.backends.opencite_backend import OpenCiteBackend
from dataset_citations.cli.gate_anchors import (
    build_parser,
    gate_citation_file,
    main,
    run_gate,
)
from dataset_citations.core.opencite_pipeline import (
    fetch_dataset_citations_via_opencite,
)
from dataset_citations.quality.llm_client import trusted_judge_model
from dataset_citations.sources import (
    EMPTY_NEMAR_DATASET_METADATA,
    NemarMetadataSource,
)
from dataset_citations.sources.models import (
    Author,
    CitingWork,
    DoiReference,
    FetchError,
    FetchSuccess,
)

FIXTURE = Path(__file__).parent / "test_data" / "gate_nm000275_citations.json"
WHEN = datetime(2026, 9, 30, 3, 0, tzinfo=UTC)
JUDGE = trusted_judge_model()

DATA_PAPER = "10.1038/s41597-019-0027-4"
RAW_DEPOSIT = "10.6084/m9.figshare.6427334.v5"
KINESTHESIA = "10.1016/j.neuroimage.2014.01.015"
OWN_DOI = "10.82901/nemar.nm000275"


def _judgment(
    identifier: str,
    classification: str,
    year: int,
    *,
    title: str | None = None,
    relation: str = "References",
) -> dict:
    return {
        "anchor_identifier": identifier,
        "anchor_identifier_type": "doi",
        "source_relation": relation,
        "classification": classification,
        "reason": "test fixture",
        "paper_title": title or f"Paper {identifier}",
        "paper_year": year,
        "paper_venue": None,
        "judged_at": WHEN.isoformat(),
        "error": None,
    }


def _write_sidecar(
    directory: Path, dataset_id: str, judgments: list[dict], model: str = JUDGE
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{dataset_id}.json").write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "judged_at": WHEN.isoformat(),
                "judgment_model": model,
                "judgments": judgments,
            }
        ),
        "utf-8",
    )


class GateCitationFileTests(TestCase):
    def setUp(self) -> None:
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.citations = self.root / "json_opencite"
        self.citations.mkdir()
        self.path = self.citations / "nm000275_citations.json"
        shutil.copy(FIXTURE, self.path)
        self.judgments = self.root / "anchor_judgments"
        self.judgments.mkdir()

    def _write_sidecar(self, judgments: list[dict], model: str = JUDGE) -> None:
        _write_sidecar(self.judgments, "nm000275", judgments, model)

    def _judged(self) -> None:
        self._write_sidecar(
            [
                _judgment(DATA_PAPER, "data_paper", 2019),
                _judgment(RAW_DEPOSIT, "data_paper", 2019),
                _judgment(KINESTHESIA, "related_work", 2014),
            ]
        )

    def _gate(self, **kwargs):
        return gate_citation_file(
            self.path, judgments_dir=self.judgments, when=WHEN, **kwargs
        )

    def _payload(self) -> dict:
        return json.loads(self.path.read_text("utf-8"))

    def _edit(self, fn) -> None:
        payload = self._payload()
        fn(payload)
        self.path.write_text(json.dumps(payload), "utf-8")

    def _citer(self, **fields) -> dict:
        record = dict(self._payload()["citation_details"][1])
        record.update(fields)
        return record

    def test_related_work_citers_are_dropped(self) -> None:
        self._judged()
        outcome = self._gate()
        self.assertTrue(outcome.changed)
        self.assertEqual(outcome.dropped, 2)
        self.assertEqual(outcome.dropped_unkept, 2)
        payload = self._payload()
        titles = [c["title"] for c in payload["citation_details"]]
        self.assertFalse(any("PREP pipeline" in t for t in titles))
        self.assertEqual(
            {c["source_doi"] for c in payload["citation_details"]},
            {DATA_PAPER, RAW_DEPOSIT},
        )
        self.assertEqual(payload["num_citations"], 3)
        self.assertEqual(payload["metadata"]["num_datapaper_citations"], 3)
        self.assertEqual(payload["metadata"]["num_dataset_citations"], 0)
        self.assertEqual(
            payload["metadata"]["total_cumulative_citations"],
            sum(c["cited_by"] for c in payload["citation_details"]),
        )
        self.assertEqual(payload["date_last_updated"], WHEN.isoformat())
        self.assertEqual(payload["metadata"]["fetch_status"], "success")

    def test_anchor_records_explain_the_gate(self) -> None:
        self._judged()
        self._gate()
        metadata = self._payload()["metadata"]
        reasons = {a["identifier"]: a["kept_reason"] for a in metadata["anchors"]}
        self.assertEqual(reasons[DATA_PAPER], "judged_data_paper")
        self.assertEqual(reasons[RAW_DEPOSIT], "judged_data_paper")
        self.assertEqual(reasons[KINESTHESIA], "judged_not_data_paper")
        self.assertEqual(reasons[OWN_DOI], "own_doi")
        # Anchors the sidecar never judged stay context only.
        self.assertEqual(reasons["10.1038/srep21353"], "unjudged")
        self.assertEqual(
            sorted(metadata["searched_dois"]),
            sorted([DATA_PAPER, RAW_DEPOSIT, OWN_DOI]),
        )
        self.assertEqual(metadata["anchor_count"], 3)
        self.assertEqual(metadata["anchor_judgment_model"], JUDGE)

    def test_second_run_is_a_no_op(self) -> None:
        self._judged()
        self._gate()
        written = self.path.read_bytes()
        outcome = self._gate()
        self.assertEqual((outcome.changed, outcome.dropped), (False, 0))
        self.assertEqual(self.path.read_bytes(), written)

    def test_citers_older_than_the_data_paper_are_dropped(self) -> None:
        def early(payload: dict) -> None:
            record = next(
                c for c in payload["citation_details"] if c["source_doi"] == DATA_PAPER
            )
            record["year"] = 2016

        self._edit(early)
        self._judged()
        outcome = self._gate()
        self.assertEqual((outcome.dropped, outcome.dropped_early), (3, 1))

    def test_a_flagged_mention_older_than_its_anchor_is_kept(self) -> None:
        flagged = self._citer(
            title="Names nm000275 in its methods",
            doi="10.5555/named",
            year=2012,
            source_doi=DATA_PAPER,
            mentions_accession=True,
        )
        self._edit(lambda p: p["citation_details"].append(flagged))
        self._judged()
        self._gate()
        titles = [c["title"] for c in self._payload()["citation_details"]]
        self.assertIn(flagged["title"], titles)

    def test_own_doi_citers_survive_and_count_as_citing_the_dataset(self) -> None:
        own_citer = self._citer(
            title="Reuses the nm000275 recordings",
            doi="10.5555/own",
            source_doi=OWN_DOI,
        )
        self._edit(lambda p: p["citation_details"].append(own_citer))
        self._judged()
        self._gate()
        after = self._payload()
        self.assertIn(
            own_citer["title"], [c["title"] for c in after["citation_details"]]
        )
        self.assertEqual(after["metadata"]["num_dataset_citations"], 1)

    def test_the_datasets_own_record_is_not_its_citer(self) -> None:
        own_record = self._citer(
            title="nm000275 (dataset record)",
            doi="10.82901/nemar.nm000275.v1.0.0",
            source_doi=DATA_PAPER,
        )
        self._edit(lambda p: p["citation_details"].append(own_record))
        self._judged()
        outcome = self._gate()
        self.assertEqual(outcome.dropped_records, 1)

    def test_no_judgments_keeps_only_mentions_and_own_doi(self) -> None:
        """Fail closed: with no sidecar nothing but the dataset itself counts."""

        def shape(payload: dict) -> None:
            payload["metadata"]["num_accession_mentions"] = 0
            mention = {
                "title": "Driver drowsiness from openly shared EEG (uses nm000275)",
                "doi": "10.5555/mention",
                "year": 2025,
                "cited_by": 1,
                "source_doi": None,
                "source_relation": None,
                "discovery_method": "accession_mention",
                "matched_accession": "nm000275",
            }
            flagged = dict(payload["citation_details"][3], mentions_accession=True)
            payload["citation_details"] = [
                *payload["citation_details"][:3],
                flagged,
                mention,
            ]
            payload["confidence_scoring"] = {"stale": True}

        self._edit(shape)
        outcome = self._gate()
        self.assertEqual(outcome.dropped, 3)
        self.assertEqual(outcome.sidecar_status, "missing")
        after = self._payload()
        self.assertEqual(len(after["citation_details"]), 2)
        self.assertEqual(after["metadata"]["num_dataset_citations"], 2)
        self.assertEqual(after["metadata"]["num_accession_mentions"], 1)
        self.assertNotIn("confidence_scoring", after)
        kept = [a["identifier"] for a in after["metadata"]["anchors"] if a["kept"]]
        self.assertEqual(kept, [OWN_DOI])
        # The own DOI survived, so the file still has a kept anchor.
        self.assertEqual(after["metadata"]["fetch_status"], "success")

    def test_a_retired_judges_sidecar_counts_as_missing(self) -> None:
        self._judged()
        self._write_sidecar(
            [_judgment(DATA_PAPER, "data_paper", 2019)], model="gemma4:e4b"
        )
        outcome = self._gate()
        self.assertEqual(outcome.sidecar_status, "untrusted_model")
        after = self._payload()
        self.assertIsNone(after["metadata"]["anchor_judgment_model"])
        reasons = {
            a["identifier"]: a["kept_reason"] for a in after["metadata"]["anchors"]
        }
        self.assertEqual(reasons[DATA_PAPER], "unjudged")

    def test_a_corrupt_sidecar_leaves_the_file_untouched(self) -> None:
        (self.judgments / "nm000275.json").write_text("<<<<<<< HEAD", "utf-8")
        before = self.path.read_bytes()
        with self.assertLogs("dataset_citations", "ERROR"):
            outcome = self._gate()
        self.assertEqual(outcome.sidecar_status, "unreadable")
        self.assertFalse(outcome.changed)
        self.assertEqual(self.path.read_bytes(), before)

    def test_no_kept_anchor_left_marks_the_file(self) -> None:
        def only_papers(payload: dict) -> None:
            payload["metadata"]["anchors"] = [
                a for a in payload["metadata"]["anchors"] if a["identifier"] != OWN_DOI
            ]

        self._edit(only_papers)
        self._write_sidecar([_judgment(KINESTHESIA, "related_work", 2014)])
        self._gate()
        metadata = self._payload()["metadata"]
        self.assertEqual(metadata["fetch_status"], "no_data_paper_anchor")
        self.assertEqual((metadata["anchor_count"], metadata["searched_dois"]), (0, []))

    def test_a_failure_status_is_left_alone(self) -> None:
        self._edit(lambda p: p["metadata"].update(fetch_status="rate_limit"))
        self._write_sidecar([])
        self._gate()
        self.assertEqual(self._payload()["metadata"]["fetch_status"], "rate_limit")

    def test_a_newly_qualifying_anchor_awaits_a_fetch(self) -> None:
        """The sweep only removes: a data paper judged after the fetch is
        recorded, but it is not kept or searched until the refetch."""

        def context_only(payload: dict) -> None:
            for anchor in payload["metadata"]["anchors"]:
                if anchor["identifier"] == "10.1038/srep21353":
                    anchor["kept"] = False

        self._edit(context_only)
        self._write_sidecar(
            [
                _judgment(DATA_PAPER, "data_paper", 2019),
                _judgment("10.1038/srep21353", "data_paper", 2016),
            ]
        )
        self._gate()
        metadata = self._payload()["metadata"]
        anchor = next(
            a for a in metadata["anchors"] if a["identifier"] == "10.1038/srep21353"
        )
        self.assertEqual(
            (anchor["kept"], anchor["kept_reason"], anchor["classification"]),
            (False, "awaiting_fetch", "data_paper"),
        )
        self.assertNotIn("10.1038/srep21353", metadata["searched_dois"])

    def test_never_anchor_is_dropped_even_when_judged_data_paper(self) -> None:
        leak = self._citer(
            title="Uses EEG-BIDS",
            doi="10.5555/leak",
            source_doi="10.1038/s41597-019-0104-8",
        )

        def add(payload: dict) -> None:
            payload["metadata"]["anchors"].append(
                {
                    "identifier": "10.1038/s41597-019-0104-8",
                    "identifier_type": "doi",
                    "source_relation": "References",
                    "kept": True,
                }
            )
            payload["citation_details"].append(leak)

        self._edit(add)
        self._write_sidecar(
            [
                _judgment(DATA_PAPER, "data_paper", 2019),
                _judgment("10.1038/s41597-019-0104-8", "data_paper", 2019),
            ]
        )
        self._gate()
        after = self._payload()
        self.assertNotIn(
            "Uses EEG-BIDS", [c["title"] for c in after["citation_details"]]
        )
        reason = next(
            a["kept_reason"]
            for a in after["metadata"]["anchors"]
            if a["identifier"] == "10.1038/s41597-019-0104-8"
        )
        self.assertEqual(reason, "never_anchor")

    def test_title_rule_verdict_is_stable_across_runs(self) -> None:
        """A spec title recorded on the anchor keeps blocking it after the
        sweep, even with no judgment to supply the title again."""

        def spec(payload: dict) -> None:
            payload["metadata"]["anchors"].append(
                {
                    "identifier": "10.9999/nirs-bids",
                    "identifier_type": "doi",
                    "source_relation": "IsDescribedBy",
                    "kept": True,
                    "paper_title": "NIRS-BIDS: an extension to the brain imaging "
                    "data structure for near-infrared spectroscopy",
                }
            )

        self._edit(spec)
        self._judged()
        for _ in range(3):
            self._gate()
            anchor = next(
                a
                for a in self._payload()["metadata"]["anchors"]
                if a["identifier"] == "10.9999/nirs-bids"
            )
            self.assertEqual(anchor["kept_reason"], "never_anchor")

    def test_a_file_without_anchors_is_rebuilt_and_gated(self) -> None:
        """Schema 2.0 files (ds004944, ds005234) have no anchors[]; their
        gemma-era citations used to pass the sweep untouched."""

        def legacy(payload: dict) -> None:
            payload["metadata"].pop("anchors")
            payload["metadata"]["schema_version"] = "2.0"
            payload["metadata"]["context_anchors"] = []

        self._edit(legacy)
        self._write_sidecar([_judgment(DATA_PAPER, "data_paper", 2019)])
        outcome = self._gate()
        self.assertTrue(outcome.needs_judgment)
        after = self._payload()
        self.assertEqual(
            {c["source_doi"] for c in after["citation_details"]}, {DATA_PAPER}
        )
        reasons = {
            a["identifier"]: a["kept_reason"] for a in after["metadata"]["anchors"]
        }
        self.assertEqual(
            reasons,
            {
                KINESTHESIA: "unjudged",
                DATA_PAPER: "judged_data_paper",
                RAW_DEPOSIT: "unjudged",
            },
        )

    def test_dry_run_writes_nothing(self) -> None:
        self._judged()
        before = self.path.read_bytes()
        outcome = self._gate(dry_run=True)
        self.assertTrue(outcome.changed)
        self.assertEqual(self.path.read_bytes(), before)


class RunGateTests(TestCase):
    def setUp(self) -> None:
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.citations = self.root / "json_opencite"
        self.citations.mkdir()
        self.judgments = self.root / "anchor_judgments"

    def _add(self, dataset_id: str, *, judged: bool = True) -> Path:
        payload = json.loads(FIXTURE.read_text("utf-8"))
        payload["dataset_id"] = dataset_id
        path = self.citations / f"{dataset_id}_citations.json"
        path.write_text(json.dumps(payload), "utf-8")
        if judged:
            _write_sidecar(
                self.judgments, dataset_id, [_judgment(DATA_PAPER, "data_paper", 2019)]
            )
        return path

    def _run(self, *extra: str) -> int | None:
        args = build_parser().parse_args(
            [
                "--citations-dir",
                str(self.citations),
                "--judgments-dir",
                str(self.judgments),
                *extra,
            ]
        )
        try:
            run_gate(args)
        except SystemExit as exc:
            return int(exc.code or 0)
        return 0

    def test_all_judged_corpus_is_gated(self) -> None:
        path = self._add("nm000275")
        self.assertEqual(self._run(), 0)
        # The two citers of the judged data paper survive.
        self.assertEqual(json.loads(path.read_text("utf-8"))["num_citations"], 2)

    def test_unreadable_citation_file_aborts_before_any_write(self) -> None:
        good = self._add("nm000275")
        before = good.read_bytes()
        (self.citations / "nm000001_citations.json").write_text("{not json", "utf-8")
        with self.assertLogs("dataset_citations.cli.gate_anchors", "ERROR"):
            self.assertEqual(self._run(), 1)
        self.assertEqual(good.read_bytes(), before)

    def test_unreadable_sidecar_aborts_before_any_write(self) -> None:
        good = self._add("nm000275")
        before = good.read_bytes()
        self._add("nm000276", judged=False)
        (self.judgments / "nm000276.json").write_text("[", "utf-8")
        with self.assertLogs("dataset_citations", "ERROR"):
            self.assertEqual(self._run(), 1)
        self.assertEqual(good.read_bytes(), before)

    def test_missing_or_empty_judgments_dir_aborts(self) -> None:
        self._add("nm000275", judged=False)
        with self.assertLogs("dataset_citations.cli.gate_anchors", "ERROR"):
            self.assertEqual(self._run(), 1)
        self.judgments.mkdir()
        with self.assertLogs("dataset_citations.cli.gate_anchors", "ERROR"):
            self.assertEqual(self._run(), 1)

    def test_most_sidecars_missing_aborts_unless_allowed(self) -> None:
        judged = self._add("nm000275")
        before = judged.read_bytes()
        self._add("nm000276", judged=False)
        self._add("nm000277", judged=False)
        with self.assertLogs("dataset_citations.cli.gate_anchors", "ERROR"):
            self.assertEqual(self._run(), 1)
        self.assertEqual(judged.read_bytes(), before)
        with self.assertLogs("dataset_citations.cli.gate_anchors", "WARNING"):
            self.assertEqual(self._run("--max-missing-share", "1"), 0)
        self.assertNotEqual(judged.read_bytes(), before)

    def test_missing_citations_dir_exits_one(self) -> None:
        self.citations.rmdir()
        with self.assertLogs("dataset_citations.cli.gate_anchors", "ERROR"):
            self.assertEqual(self._run(), 1)

    def test_help_lists_the_flags(self) -> None:
        buf = io.StringIO()
        with self.assertRaises(SystemExit), redirect_stdout(buf):
            main(["--help"])
        for flag in (
            "--citations-dir",
            "--judgments-dir",
            "--judge-model",
            "--max-missing-share",
            "--dry-run",
        ):
            self.assertIn(flag, buf.getvalue())


class _Source(NemarMetadataSource):
    def __init__(self, refs: list[DoiReference]) -> None:
        super().__init__(prefer_data_api=False)
        self._refs = refs

    def get_doi_references(self, dataset_id: str):  # type: ignore[override]
        return FetchSuccess(self._refs)

    def get_dataset_metadata(self, dataset_id: str):  # type: ignore[override]
        # The real method reads GitHub; the descriptive fields are not under test.
        return EMPTY_NEMAR_DATASET_METADATA


class _Backend(OpenCiteBackend):
    def __init__(self, outcomes: dict) -> None:
        self._outcomes = outcomes
        self._config = None  # type: ignore[assignment]
        self._max_results_per_doi = 100
        self._concurrency = 1

    def get_citing_works_batch(self, refs):  # type: ignore[override]
        return {r.identifier: self._outcomes[r.identifier] for r in refs}


def _work(n: int, source_doi: str, relation: str, year: int = 2024) -> CitingWork:
    return CitingWork(
        title=f"Citing work {n}",
        doi=f"10.5555/w{n}",
        pmid=None,
        openalex_id=None,
        year=year,
        authors=(Author(name="A. Researcher"),),
        venue="Journal of Tests",
        abstract=None,
        citation_count=n,
        source_doi=source_doi,
        source_relation=relation,  # type: ignore[arg-type]
    )


class SweepMatchesFetchTests(TestCase):
    """The sweep must be a no-op on anything the fetch path just wrote, or
    every weekly refetch followed by a sweep would churn the file (B1)."""

    def _roundtrip(self, refs, judgments, outcomes, *, catalog_doi=None) -> None:
        root = Path(self.enterContext(TemporaryDirectory()))
        judgments_dir = root / "anchor_judgments"
        _write_sidecar(judgments_dir, "nm000500", judgments)
        payload = fetch_dataset_citations_via_opencite(
            "nm000500",
            backend=_Backend(outcomes),
            nemar_source=_Source(refs),
            catalog_doi=catalog_doi,
            fetch_date=WHEN,
            judgments_dir=judgments_dir,
        )
        path = root / "nm000500_citations.json"
        path.write_text(json.dumps(payload), "utf-8")
        outcome = gate_citation_file(path, judgments_dir=judgments_dir, when=WHEN)
        self.assertEqual((outcome.changed, outcome.dropped), (False, 0), payload)

    def _ref(self, doi: str, relation: str = "IsDescribedBy") -> DoiReference:
        return DoiReference(
            identifier=doi,
            identifier_type="doi",
            relation_type=relation,  # type: ignore[arg-type]
            source="nemar_metadata",
        )

    def test_every_gate_shape_round_trips(self) -> None:
        paper, other = self._ref("10.1/paper"), self._ref("10.1/other", "References")
        deposit = self._ref("10.6084/m9.figshare.9", "IsIdenticalTo")
        own = "10.82901/nemar.nm000500"
        works = {
            paper.identifier: FetchSuccess(
                [_work(1, paper.identifier, "IsDescribedBy")]
            ),
            other.identifier: FetchSuccess([_work(2, other.identifier, "References")]),
            deposit.identifier: FetchSuccess(
                [_work(3, deposit.identifier, "IsIdenticalTo")]
            ),
            own: FetchSuccess([_work(4, own, "References")]),
        }
        cases = {
            "all kept": (
                [paper, other],
                [
                    _judgment(paper.identifier, "data_paper", 2019),
                    _judgment(other.identifier, "data_paper", 2018),
                ],
                None,
            ),
            "mixed": (
                [paper, other],
                [
                    _judgment(paper.identifier, "data_paper", 2019),
                    _judgment(other.identifier, "methodology", 2010),
                ],
                None,
            ),
            "none kept": (
                [paper, other],
                [_judgment(paper.identifier, "umbrella", 2019)],
                None,
            ),
            "own DOI only": ([other], [], own),
            "identity record unjudged": ([deposit, other], [], None),
        }
        for name, (refs, judgments, catalog_doi) in cases.items():
            with self.subTest(name):
                self._roundtrip(refs, judgments, works, catalog_doi=catalog_doi)

    def test_failure_stub_round_trips(self) -> None:
        paper = self._ref("10.1/paper")
        self._roundtrip(
            [paper],
            [_judgment(paper.identifier, "data_paper", 2019)],
            {paper.identifier: FetchError("rate_limit", "429")},
        )
