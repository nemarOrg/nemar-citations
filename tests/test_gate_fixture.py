"""The gated nm000275 fixture used by the dashboard's data-papers tests.

`tests/test_data/gate_nm000275_gated_citations.json` is the real pre-gate
nm000275 fixture run through the real anchor gate
(`cli.gate_anchors.gate_citation_file`). No Claude judgment exists for this
dataset yet, so the judge sidecar holds three verdicts written by hand; the
titles, venues, and years are the real registry records of those DOIs. This test
rebuilds the file the same way and asserts it matches, so the fixture cannot
drift from the gate (issue #250).

To regenerate after a deliberate gate change, write the rebuilt payload over the
committed file:

    uv run python tests/test_gate_fixture.py
"""

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from dataset_citations.cli.gate_anchors import gate_citation_file
from dataset_citations.quality.llm_client import trusted_judge_model

DATA = Path(__file__).parent / "test_data"
PRE_GATE = DATA / "gate_nm000275_citations.json"
GATED = DATA / "gate_nm000275_gated_citations.json"
WHEN = datetime(2026, 9, 30, 3, 0, tzinfo=UTC)
DATASET_ID = "nm000275"


def _judgment(
    identifier: str,
    classification: str,
    title: str,
    year: int,
    venue: str,
    relation: str = "References",
) -> dict:
    return {
        "anchor_identifier": identifier,
        "anchor_identifier_type": "doi",
        "source_relation": relation,
        "classification": classification,
        "reason": "hand-written verdict for a test fixture",
        "paper_title": title,
        "paper_year": year,
        "paper_venue": venue,
        "judged_at": WHEN.isoformat(),
        "error": None,
    }


def _verdicts() -> list[dict]:
    return [
        _judgment(
            "10.1038/s41597-019-0027-4",
            "data_paper",
            "Multi-channel EEG recordings during a sustained-attention driving task",
            2019,
            "Scientific Data",
        ),
        _judgment(
            "10.6084/m9.figshare.6427334.v5",
            "data_paper",
            "Multi-channel EEG recordings during a sustained-attention driving task (raw dataset)",
            2019,
            "figshare",
            relation="IsDerivedFrom",
        ),
        _judgment(
            "10.1016/j.neuroimage.2014.01.015",
            "related_work",
            "Kinesthesia in a sustained-attention driving task",
            2014,
            "NeuroImage",
        ),
    ]


def rebuild() -> dict:
    """The pre-gate fixture gated by the real gate under the hand verdicts."""
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        citations = root / "json_opencite"
        judgments = root / "anchor_judgments"
        citations.mkdir()
        judgments.mkdir()
        work = citations / f"{DATASET_ID}_citations.json"
        shutil.copy(PRE_GATE, work)
        (judgments / f"{DATASET_ID}.json").write_text(
            json.dumps(
                {
                    "dataset_id": DATASET_ID,
                    "judged_at": WHEN.isoformat(),
                    "judgment_model": trusted_judge_model(),
                    "judgments": _verdicts(),
                }
            ),
            "utf-8",
        )
        gate_citation_file(work, judgments_dir=judgments, when=WHEN, quiet=True)
        return json.loads(work.read_text("utf-8"))


def regenerate() -> None:
    """Overwrite the committed fixture with a fresh rebuild."""
    GATED.write_text(
        json.dumps(rebuild(), indent=2, ensure_ascii=False) + "\n", "utf-8"
    )


class GatedFixtureTests(TestCase):
    def test_committed_fixture_matches_a_rebuild_through_the_real_gate(self) -> None:
        committed = json.loads(GATED.read_text("utf-8"))
        self.assertEqual(committed, rebuild())

    def test_the_fixture_is_the_shape_the_manifest_reads(self) -> None:
        anchors = json.loads(GATED.read_text("utf-8"))["metadata"]["anchors"]
        reasons = {a["identifier"]: a["kept_reason"] for a in anchors}
        self.assertEqual(
            sorted(i for i, r in reasons.items() if r == "judged_data_paper"),
            ["10.1038/s41597-019-0027-4", "10.6084/m9.figshare.6427334.v5"],
        )
        self.assertEqual(reasons["10.82901/nemar.nm000275"], "own_doi")
        self.assertEqual(
            reasons["10.1016/j.neuroimage.2014.01.015"], "judged_not_data_paper"
        )
        self.assertTrue(all(r for r in reasons.values()))


if __name__ == "__main__":
    regenerate()
