"""Tests for the never-anchor guard (issue #241).

Titles below are the real Crossref titles of the papers, so the title rule is
exercised against what opencite actually returns for them.
"""

from __future__ import annotations

import json
from importlib.resources import files
from unittest import TestCase

from dataset_citations.quality.never_anchor import (
    is_never_anchor,
    is_spec_title,
    never_anchor_dois,
    parse_never_anchor_list,
)


class NeverAnchorListTests(TestCase):
    def test_packaged_list_is_normalized_and_unique(self) -> None:
        raw = json.loads(
            files("dataset_citations.quality")
            .joinpath("never_anchor_dois.json")
            .read_text("utf-8")
        )
        listed = [entry["doi"] for entry in raw["dois"]]
        self.assertEqual(len(listed), len(set(listed)), "duplicate DOI in the list")
        for doi in listed:
            self.assertEqual(doi, doi.strip().lower(), f"{doi!r} is not normalized")
        self.assertEqual(len(never_anchor_dois()), len(listed))

    def test_listed_doi_blocks_in_any_spelling(self) -> None:
        for spelling in (
            "10.1038/s41597-019-0104-8",
            "10.1038/S41597-019-0104-8",
            "doi:10.1038/s41597-019-0104-8",
            "https://doi.org/10.1038/s41597-019-0104-8",
        ):
            self.assertTrue(is_never_anchor(spelling), spelling)

    def test_data_paper_doi_is_not_blocked(self) -> None:
        # nm000275's Scientific Data descriptor.
        self.assertFalse(
            is_never_anchor(
                "10.1038/s41597-019-0027-4",
                "Multi-channel EEG recordings during a sustained-attention "
                "driving task",
            )
        )

    def test_a_related_data_descriptor_is_left_to_the_judge(self) -> None:
        # MIPDB is nm000153's own data paper, and only related work for the
        # Healthy Brain Network datasets; the program paper is the umbrella.
        self.assertFalse(
            is_never_anchor(
                "10.1038/sdata.2017.40",
                "A resource for assessing information processing in the "
                "developing brain using EEG and eye tracking",
            )
        )
        self.assertTrue(is_never_anchor("10.1038/sdata.2017.181"))

    def test_nirs_bids_is_listed(self) -> None:
        self.assertIn("10.1038/s41597-024-04136-9", never_anchor_dois())


class ParseListTests(TestCase):
    """A broken list must fail loudly, never read as empty."""

    def test_valid_list_is_normalized(self) -> None:
        raw = json.dumps({"dois": [{"doi": " https://doi.org/10.1/ABC "}]})
        self.assertEqual(parse_never_anchor_list(raw), frozenset({"10.1/abc"}))

    def test_broken_lists_raise(self) -> None:
        for raw, error in (
            ('{"dois": []}', ValueError),
            ('{"dois": [{"doi": "  "}]}', ValueError),
            ('{"entries": []}', KeyError),
            ('{"dois": [{"label": "no doi"}]}', KeyError),
            ("not json", ValueError),
        ):
            with self.subTest(raw=raw), self.assertRaises(error):
                parse_never_anchor_list(raw)


class SpecTitleTests(TestCase):
    def test_bids_spec_and_tool_titles_match(self) -> None:
        for title in (
            "EEG-BIDS, an extension to the brain imaging data structure for "
            "electroencephalography",
            "MEG-BIDS, the brain imaging data structure extended to "
            "magnetoencephalography",
            "iEEG-BIDS, extending the Brain Imaging Data Structure specification "
            "to human intracranial electrophysiology",
            "Motion-BIDS: an extension to the brain imaging data structure to "
            "organize motion data for reproducible research",
            "The brain imaging data structure, a format for organizing and "
            "describing outputs of neuroimaging experiments",
            "MNE-BIDS: Organizing electrophysiological data into the BIDS format "
            "and facilitating their analysis",
            "BIDS apps: Improving ease of use, accessibility, and reproducibility "
            "of neuroimaging data analysis methods",
            # The spec wording with no name prefix.
            "An extension to the brain imaging data structure for "
            "near-infrared spectroscopy",
            # A future extension nobody has added to the DOI list yet.
            "NIRS-BIDS: an extension to the brain imaging data structure for "
            "near-infrared spectroscopy",
        ):
            self.assertTrue(is_spec_title(title), title)
            # The title alone is enough, even for a DOI not on the list.
            self.assertTrue(is_never_anchor("10.9999/not-on-the-list", title))

    def test_data_papers_that_use_bids_do_not_match(self) -> None:
        for title in (
            "BIDS-formatted EEG recordings during a visual oddball task",
            "An open EEG dataset organized in the Brain Imaging Data Structure",
            "A multi-subject, multi-modal human neuroimaging dataset",
            "HBN-EEG: The FAIR implementation of the Healthy Brain Network (HBN) "
            "electroencephalography dataset",
            # Review of #243: ordinary data-descriptor wording near the phrase.
            "An extended EEG dataset of visual working memory, organized in the "
            "Brain Imaging Data Structure",
            "A large-scale EEG dataset with extended metadata in the Brain Imaging "
            "Data Structure (BIDS)",
            "A multi-subject EEG dataset organized following the Brain Imaging Data "
            "Structure extension for EEG",
            "A dataset extending the Brain Imaging Data Structure with HED annotations",
            # Tool names outside the title rule; PyBIDS is caught by its DOI
            # (see test_pybids_is_caught_by_its_doi).
            "PyBIDS: Python tools for BIDS datasets",
            "BIDS-MATLAB: a MATLAB toolbox for BIDS datasets",
            None,
            "",
        ):
            self.assertFalse(is_spec_title(title), title)

    def test_pybids_is_caught_by_its_doi(self) -> None:
        self.assertTrue(
            is_never_anchor(
                "10.21105/joss.01294", "PyBIDS: Python tools for BIDS datasets"
            )
        )
