"""Tests for the OpenAlex accession-search backend mappers. Issue #169.

The pure mappers (work_to_citation, reconstruct_abstract) are tested against a
realistic OpenAlex work JSON, no network. The live search is covered by a
gated integration test (RUN_INTEGRATION_TESTS=1).
"""

from __future__ import annotations

import os
import unittest
from unittest import TestCase

from dataset_citations.backends.accession_search import (
    dedup_citations,
    reconstruct_abstract,
    work_to_citation,
)

# Trimmed but real-shaped OpenAlex /works result.
_WORK = {
    "id": "https://openalex.org/W3035020590",
    "doi": "https://doi.org/10.3389/fnins.2020.610388",
    "title": "From BIDS-Formatted EEG Data to Sensor-Space Group Results",
    "publication_year": 2020,
    "cited_by_count": 42,
    "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/33390894"},
    "authorships": [
        {"author": {"display_name": "Cyril Pernet"}},
        {"author": {"display_name": "Marijn van Vliet"}},
    ],
    "primary_location": {"source": {"display_name": "Frontiers in Neuroscience"}},
    "abstract_inverted_index": {"We": [0], "used": [1], "ds002718": [2]},
}


class ReconstructAbstractTests(TestCase):
    def test_rebuilds_word_order(self) -> None:
        self.assertEqual(
            reconstruct_abstract({"hello": [1], "world": [2], "say": [0]}),
            "say hello world",
        )

    def test_none_and_empty(self) -> None:
        self.assertIsNone(reconstruct_abstract(None))
        self.assertIsNone(reconstruct_abstract({}))


class WorkToCitationTests(TestCase):
    def setUp(self) -> None:
        self.c = work_to_citation(_WORK, "ds002718")

    def test_bare_doi_lowercased(self) -> None:
        self.assertEqual(self.c["doi"], "10.3389/fnins.2020.610388")

    def test_openalex_id_shortened(self) -> None:
        self.assertEqual(self.c["openalex_id"], "W3035020590")

    def test_pmid_shortened(self) -> None:
        self.assertEqual(self.c["pmid"], "33390894")

    def test_authors_joined(self) -> None:
        self.assertEqual(self.c["author"], "Cyril Pernet, Marijn van Vliet")

    def test_venue_and_year_and_cited_by(self) -> None:
        self.assertEqual(self.c["venue"], "Frontiers in Neuroscience")
        self.assertEqual(self.c["year"], 2020)
        self.assertEqual(self.c["cited_by"], 42)

    def test_abstract_reconstructed(self) -> None:
        self.assertEqual(self.c["abstract"], "We used ds002718")

    def test_tagged_as_accession_mention(self) -> None:
        self.assertEqual(self.c["discovery_method"], "accession_mention")
        self.assertEqual(self.c["matched_accession"], "ds002718")
        self.assertIsNone(self.c["source_doi"])
        self.assertIsNone(self.c["source_relation"])

    def test_missing_fields_default_safely(self) -> None:
        c = work_to_citation({"title": "bare"}, "nm000001")
        self.assertEqual(c["author"], "n/a")
        self.assertEqual(c["venue"], "n/a")
        self.assertEqual(c["year"], 0)
        self.assertIsNone(c["doi"])
        self.assertIsNone(c["abstract"])


class DedupCitationsTests(TestCase):
    """Cross-term dedup: a paper matching both the on- and ds- accession of the
    same dataset must appear once (the key correctness contract for on-*)."""

    def _hit(self, doi: str, oaid: str, title: str = "t") -> dict:
        return {"doi": doi, "openalex_id": oaid, "title": title}

    def test_paper_in_two_terms_appears_once(self) -> None:
        on_hits = [self._hit("10.1/p1", "W1"), self._hit("10.1/p2", "W2")]
        ds_hits = [self._hit("10.1/p2", "W2"), self._hit("10.1/p3", "W3")]
        result = dedup_citations([on_hits, ds_hits])
        self.assertEqual([c["doi"] for c in result], ["10.1/p1", "10.1/p2", "10.1/p3"])

    def test_stable_sort_independent_of_input_order(self) -> None:
        a = [self._hit("10.1/c", "W3"), self._hit("10.1/a", "W1")]
        b = [self._hit("10.1/b", "W2")]
        self.assertEqual(
            [c["doi"] for c in dedup_citations([a, b])],
            [c["doi"] for c in dedup_citations([b, a])],
        )

    def test_dedupes_case_insensitive_doi(self) -> None:
        result = dedup_citations(
            [[self._hit("10.1/X", "W1")], [self._hit("10.1/x", "W9")]]
        )
        self.assertEqual(len(result), 1)


@unittest.skipUnless(
    os.environ.get("RUN_INTEGRATION_TESTS") == "1",
    "live OpenAlex call; set RUN_INTEGRATION_TESTS=1 to enable",
)
class LiveAccessionSearchTests(TestCase):
    def test_finds_known_accession(self) -> None:
        from dataset_citations.backends.accession_search import AccessionSearchBackend

        result = AccessionSearchBackend(max_results=5).search(["ds002718"])
        self.assertEqual(result.failed_terms, [])
        self.assertGreater(len(result.citations), 0)
        for c in result.citations:
            self.assertEqual(c["discovery_method"], "accession_mention")
