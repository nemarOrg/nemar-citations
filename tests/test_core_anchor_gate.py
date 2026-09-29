"""Unit tests for `core.anchor_gate` (issue #241): the fail-closed gate."""

from __future__ import annotations

from unittest import TestCase

from dataset_citations.core.anchor_gate import (
    DROPPED_NEVER_ANCHOR,
    DROPPED_NOT_DATA_PAPER,
    DROPPED_UNJUDGED,
    KEPT_DATA_PAPER,
    KEPT_DATASET_RECORD,
    KEPT_OWN_DOI,
    GateDecision,
    drop_pre_anchor_citations,
    drop_self_citations,
    gate_anchor,
)
from dataset_citations.sources.doi import is_own_dataset_doi


def _gate(identifier: str, classification: str | None, **kwargs):
    return gate_anchor(
        dataset_id=kwargs.get("dataset_id", "nm000275"),
        identifier=identifier,
        identifier_type=kwargs.get("identifier_type", "doi"),
        classification=classification,
        paper_title=kwargs.get("paper_title"),
        source_relation=kwargs.get("source_relation"),
    )


class GateAnchorTests(TestCase):
    def test_own_concept_doi_is_kept_without_judgment(self) -> None:
        decision = _gate("10.82901/nemar.nm000275", None)
        self.assertTrue(decision.kept)
        self.assertEqual(decision.reason, KEPT_OWN_DOI)

    def test_own_concept_doi_is_kept_whatever_the_judge_says(self) -> None:
        for label in ("irrelevant", "umbrella", "related_work"):
            decision = _gate("10.82901/nemar.nm000275", label)
            self.assertEqual(decision, GateDecision(kept=True, reason=KEPT_OWN_DOI))

    def test_unjudged_dataset_record_is_kept_by_its_relation(self) -> None:
        # nm000114's figshare deposit, nm000110's PhysioNet record.
        for relation in ("IsIdenticalTo", "IsVersionOf"):
            decision = _gate("10.6084/m9.figshare.1", None, source_relation=relation)
            self.assertEqual(
                decision, GateDecision(kept=True, reason=KEPT_DATASET_RECORD)
            )

    def test_a_judgment_overrides_the_identity_relation(self) -> None:
        decision = _gate(
            "10.6084/m9.figshare.1", "related_work", source_relation="IsIdenticalTo"
        )
        self.assertEqual(decision.reason, DROPPED_NOT_DATA_PAPER)

    def test_never_anchor_beats_the_identity_relation(self) -> None:
        # A standards paper mislabeled by the enrichment.
        decision = _gate("10.21105/joss.01896", None, source_relation="IsVersionOf")
        self.assertEqual(decision.reason, DROPPED_NEVER_ANCHOR)

    def test_other_relations_stay_unjudged(self) -> None:
        for relation in ("References", "IsDescribedBy", "IsDerivedFrom", None):
            decision = _gate("10.1/x", None, source_relation=relation)
            self.assertEqual(decision.reason, DROPPED_UNJUDGED, relation)

    def test_another_datasets_nemar_doi_needs_a_judgment(self) -> None:
        decision = _gate("10.82901/nemar.nm000103", None)
        self.assertFalse(decision.kept)
        self.assertEqual(decision.reason, DROPPED_UNJUDGED)

    def test_judged_data_paper_is_kept(self) -> None:
        decision = _gate("10.1038/s41597-019-0027-4", "data_paper")
        self.assertTrue(decision.kept)
        self.assertEqual(decision.reason, KEPT_DATA_PAPER)

    def test_every_other_classification_is_context(self) -> None:
        for label in ("umbrella", "methodology", "related_work", "irrelevant"):
            decision = _gate("10.1016/j.neuroimage.2014.01.015", label)
            self.assertFalse(decision.kept, label)
            self.assertEqual(decision.reason, DROPPED_NOT_DATA_PAPER)

    def test_unjudged_anchor_is_context(self) -> None:
        # The fail-open fallback this replaces fetched exactly this case.
        decision = _gate("10.1016/j.neuroimage.2014.01.015", None)
        self.assertFalse(decision.kept)
        self.assertEqual(decision.reason, DROPPED_UNJUDGED)

    def test_never_anchor_overrides_a_data_paper_verdict_and_says_so(self) -> None:
        with self.assertLogs("dataset_citations.core.anchor_gate", "INFO") as logs:
            decision = _gate("10.21105/joss.01896", "data_paper")
        self.assertFalse(decision.kept)
        self.assertEqual(decision.reason, DROPPED_NEVER_ANCHOR)
        self.assertIn("10.21105/joss.01896", logs.output[0])

    def test_spec_title_blocks_non_doi_anchors_too(self) -> None:
        decision = _gate(
            "pmid:31239423",
            "data_paper",
            identifier_type="pmid",
            paper_title="EEG-BIDS, an extension to the brain imaging data "
            "structure for electroencephalography",
        )
        self.assertFalse(decision.kept)
        self.assertEqual(decision.reason, DROPPED_NEVER_ANCHOR)


class DropPreAnchorCitationsTests(TestCase):
    def test_drops_only_works_older_than_their_anchor(self) -> None:
        details = [
            {"title": "PREP", "year": 2015, "source_doi": "10.1/data"},
            {"title": "same year", "year": 2019, "source_doi": "10.1/data"},
            {"title": "later", "year": 2023, "source_doi": "10.1/data"},
            {"title": "unknown year", "year": 0, "source_doi": "10.1/data"},
            {"title": "other anchor", "year": 2001, "source_doi": "10.1/other"},
            {"title": "mention", "year": 2010, "discovery_method": "accession_mention"},
        ]
        kept, dropped = drop_pre_anchor_citations(details, {"10.1/data": 2019})
        self.assertEqual(dropped, 1)
        self.assertEqual(
            [d["title"] for d in kept],
            ["same year", "later", "unknown year", "other anchor", "mention"],
        )

    def test_versioned_source_doi_matches_its_anchor(self) -> None:
        details = [
            {"title": "old", "year": 2010, "source_doi": "10.6084/m9.figshare.1.v5"}
        ]
        kept, dropped = drop_pre_anchor_citations(
            details, {"10.6084/m9.figshare.1.v5": 2019}
        )
        self.assertEqual((kept, dropped), ([], 1))

    def test_unknown_anchor_year_keeps_everything(self) -> None:
        details = [{"title": "old", "year": 1990, "source_doi": "10.1/data"}]
        self.assertEqual(drop_pre_anchor_citations(details, {}), (details, 0))

    def test_one_year_of_slack_for_preprints(self) -> None:
        # A 2018 paper citing the bioRxiv version of a 2019 journal paper.
        details = [{"title": "preprint citer", "year": 2018, "source_doi": "10.1/d"}]
        self.assertEqual(
            drop_pre_anchor_citations(details, {"10.1/d": 2019}), (details, 0)
        )

    def test_a_citation_that_names_the_accession_is_never_dropped(self) -> None:
        details = [
            {
                "title": "names ds002778",
                "year": 2012,
                "source_doi": "10.1/d",
                "mentions_accession": True,
            }
        ]
        self.assertEqual(
            drop_pre_anchor_citations(details, {"10.1/d": 2019}), (details, 0)
        )


class DropSelfCitationsTests(TestCase):
    def test_the_datasets_own_record_is_not_its_citer(self) -> None:
        details = [
            {"title": "own record", "doi": "10.82901/nemar.on004554.v1.0.0"},
            {"title": "own concept", "doi": "10.82901/NEMAR.ON004554"},
            {"title": "another dataset", "doi": "10.82901/nemar.on004555"},
            {"title": "a paper", "doi": "10.3934/mbe.2023507"},
            {"title": "no doi", "doi": None},
        ]
        kept, dropped = drop_self_citations(details, "on004554")
        self.assertEqual(dropped, 2)
        self.assertEqual(
            [d["title"] for d in kept], ["another dataset", "a paper", "no doi"]
        )


class OwnDatasetDoiTests(TestCase):
    def test_concept_doi_in_any_spelling(self) -> None:
        for spelling in (
            "10.82901/nemar.nm000275",
            "10.82901/NEMAR.NM000275",
            "https://doi.org/10.82901/nemar.nm000275",
        ):
            self.assertTrue(is_own_dataset_doi(spelling, "nm000275"), spelling)

    def test_other_ids_and_prefixes_do_not_match(self) -> None:
        self.assertFalse(is_own_dataset_doi("10.82901/nemar.nm000276", "nm000275"))
        self.assertFalse(is_own_dataset_doi("10.82901/nemar.nm0002750", "nm000275"))
        self.assertFalse(
            is_own_dataset_doi("10.18112/openneuro.ds002778.v1.0.5", "on002778")
        )
