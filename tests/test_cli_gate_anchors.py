"""Tests for `dataset-citations-gate-anchors` (issue #241).

Real data: `tests/test_data/gate_nm000275_citations.json` is a trimmed copy of
the committed nm000275 citation file from the inflated 2026-09-26 run (every
anchor fetched because the judge was down), including the PREP pipeline paper
that entered through the 2014 "Kinesthesia" anchor. The sidecar is written per
test in the locked phase 2 shape.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from dataset_citations.cli.gate_anchors import gate_citation_file, run_gate

FIXTURE = Path(__file__).parent / "test_data" / "gate_nm000275_citations.json"
WHEN = datetime(2026, 9, 30, 3, 0, tzinfo=UTC)

DATA_PAPER = "10.1038/s41597-019-0027-4"
RAW_DEPOSIT = "10.6084/m9.figshare.6427334.v5"
KINESTHESIA = "10.1016/j.neuroimage.2014.01.015"
OWN_DOI = "10.82901/nemar.nm000275"


def _judgment(identifier: str, classification: str, year: int) -> dict:
    return {
        "anchor_identifier": identifier,
        "anchor_identifier_type": "doi",
        "source_relation": "References",
        "classification": classification,
        "reason": "test fixture",
        "paper_title": f"Paper {identifier}",
        "paper_year": year,
        "paper_venue": None,
        "judged_at": WHEN.isoformat(),
        "error": None,
    }


class GateCitationFileTests(TestCase):
    def setUp(self) -> None:
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.citations = self.root / "json_opencite"
        self.citations.mkdir()
        self.path = self.citations / "nm000275_citations.json"
        shutil.copy(FIXTURE, self.path)
        self.judgments = self.root / "anchor_judgments"
        self.judgments.mkdir()

    def _write_sidecar(self, judgments: list[dict]) -> None:
        (self.judgments / "nm000275.json").write_text(
            json.dumps(
                {
                    "dataset_id": "nm000275",
                    "judged_at": WHEN.isoformat(),
                    "judgment_model": "claude-sonnet-5-5",
                    "judgments": judgments,
                }
            ),
            "utf-8",
        )

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

    def test_related_work_citers_are_dropped(self) -> None:
        self._judged()
        outcome = self._gate()
        self.assertTrue(outcome.changed)
        self.assertEqual(outcome.dropped, 2)
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
        self.assertEqual(metadata["anchor_judgment_model"], "claude-sonnet-5-5")

    def test_second_run_is_a_no_op(self) -> None:
        self._judged()
        self._gate()
        written = self.path.read_bytes()
        outcome = self._gate()
        self.assertEqual((outcome.changed, outcome.dropped), (False, 0))
        self.assertEqual(self.path.read_bytes(), written)

    def test_citers_older_than_the_data_paper_are_dropped(self) -> None:
        payload = self._payload()
        early = next(
            c for c in payload["citation_details"] if c["source_doi"] == DATA_PAPER
        )
        early["year"] = 2016
        self.path.write_text(json.dumps(payload), "utf-8")
        self._judged()
        outcome = self._gate()
        self.assertEqual(outcome.dropped, 3)
        self.assertNotIn(
            early["title"], [c["title"] for c in self._payload()["citation_details"]]
        )

    def test_no_judgments_keeps_only_mentions_and_own_doi(self) -> None:
        """Fail closed: with no sidecar nothing but the dataset itself counts."""
        payload = self._payload()
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
        self.path.write_text(json.dumps(payload), "utf-8")

        outcome = self._gate()
        self.assertEqual(outcome.dropped, 3)
        after = self._payload()
        self.assertEqual(
            [c["title"] for c in after["citation_details"]],
            [flagged["title"], mention["title"]],
        )
        self.assertEqual(after["metadata"]["num_dataset_citations"], 2)
        self.assertEqual(after["metadata"]["num_accession_mentions"], 1)
        self.assertNotIn("confidence_scoring", after)
        kept = [a["identifier"] for a in after["metadata"]["anchors"] if a["kept"]]
        self.assertEqual(kept, [OWN_DOI])

    def test_never_anchor_is_dropped_even_when_judged_data_paper(self) -> None:
        payload = self._payload()
        payload["metadata"]["anchors"].append(
            {
                "identifier": "10.1038/s41597-019-0104-8",
                "identifier_type": "doi",
                "source_relation": "References",
            }
        )
        leak = dict(payload["citation_details"][1])
        leak.update(
            title="Uses EEG-BIDS",
            doi="10.5555/leak",
            source_doi="10.1038/s41597-019-0104-8",
        )
        payload["citation_details"].append(leak)
        self.path.write_text(json.dumps(payload), "utf-8")
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

    def test_dry_run_writes_nothing(self) -> None:
        self._judged()
        before = self.path.read_bytes()
        outcome = self._gate(dry_run=True)
        self.assertTrue(outcome.changed)
        self.assertEqual(self.path.read_bytes(), before)


class RunGateTests(TestCase):
    def test_unreadable_file_exits_one(self) -> None:
        with TemporaryDirectory() as tmp:
            citations = Path(tmp)
            (citations / "nm000001_citations.json").write_text("{not json", "utf-8")
            args = argparse.Namespace(
                citations_dir=str(citations),
                judgments_dir=str(citations),
                dry_run=False,
            )
            with (
                self.assertLogs("dataset_citations.cli.gate_anchors", "ERROR"),
                self.assertRaises(SystemExit) as ctx,
            ):
                run_gate(args)
            self.assertEqual(ctx.exception.code, 1)
