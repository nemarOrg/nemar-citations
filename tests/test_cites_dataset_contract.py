"""`cites_dataset` against the fixture the dashboard's `citesDataset` also reads.

`web/test/gate.test.ts` runs the same cases through the TypeScript rule, so the
two bucketing rules cannot drift apart silently (#241 review).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import TestCase

from dataset_citations.core.accession_mentions import cites_dataset

CASES = Path(__file__).parent / "test_data" / "cites_dataset_cases.json"


class CitesDatasetContractTests(TestCase):
    def test_shared_cases(self) -> None:
        cases = json.loads(CASES.read_text("utf-8"))["cases"]
        self.assertGreater(len(cases), 0)
        for case in cases:
            with self.subTest(case["citation"]):
                self.assertEqual(cites_dataset(case["citation"]), case["cites_dataset"])
