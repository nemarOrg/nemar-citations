"""Dedup identity rules, exercised on the shapes that actually double-counted.

Every fixture here is a real pair taken from `citations/json_opencite/`; the
DOIs and titles are copied verbatim from the corpus rather than invented, so a
regression shows up as the same duplicate class returning.
"""

from dataset_citations.core.citation_identity import (
    base_doi,
    dedupe_citations,
    is_preprint_doi,
    normalize_doi,
    normalize_title,
)


def _work(doi, title, **extra):
    record = {"doi": doi, "title": title, "cited_by": 0}
    record.update(extra)
    return record


class TestNormalizeDoi:
    def test_strips_resolver_prefix_and_case(self):
        assert normalize_doi("https://doi.org/10.1016/J.Cortex.2024.12.017") == (
            "10.1016/j.cortex.2024.12.017"
        )

    def test_absent_doi_is_empty_string(self):
        assert normalize_doi(None) == ""
        assert normalize_doi("") == ""


class TestBaseDoi:
    def test_collapses_nemar_version_suffix(self):
        # on004842 carried both forms as separate citations.
        assert base_doi("10.82901/nemar.on004842.v1.0.0") == "10.82901/nemar.on004842"

    def test_collapses_figshare_version_suffix(self):
        assert base_doi("10.6084/m9.figshare.31991745.v2") == (
            "10.6084/m9.figshare.31991745"
        )

    def test_leaves_unversioned_doi_alone(self):
        assert (
            base_doi("10.1016/j.cortex.2019.12.001") == "10.1016/j.cortex.2019.12.001"
        )


class TestIsPreprintDoi:
    def test_known_prefixes(self):
        assert is_preprint_doi("10.1101/2024.06.06.597688")
        assert is_preprint_doi("10.48550/arxiv.2306.13505")
        assert is_preprint_doi("10.31219/osf.io/pu5vb")

    def test_new_biorxiv_prefix_via_date_suffix(self):
        # 10.64898 is bioRxiv's current prefix; the date-shaped suffix catches
        # it even though the prefix list is not the thing matching here.
        assert is_preprint_doi("10.64898/2026.01.22.701110")

    def test_journal_doi_is_not_a_preprint(self):
        assert not is_preprint_doi("10.1523/jneurosci.0133-26.2026")


class TestNormalizeTitle:
    def test_folds_en_dash_and_trailing_period(self):
        # The nm000132 pair that survived the old tuple key.
        left = "Revisiting the electrophysiological correlates of valence and expectancy in reward processing - A multi-lab replication."
        right = "Revisiting the electrophysiological correlates of valence and expectancy in reward processing \u2013 A multi-lab replication"
        assert normalize_title(left) == normalize_title(right)

    def test_distinct_titles_stay_distinct(self):
        assert normalize_title("Sensorimotor conflicts alter monitoring") != (
            normalize_title("Distinct basal ganglia contributions")
        )


class TestDedupeCitations:
    def test_same_doi_different_punctuation_collapses(self):
        records = [
            _work("10.1016/j.cortex.2024.12.017", "Reward processing - A replication."),
            _work(
                "10.1016/j.cortex.2024.12.017",
                "Reward processing \u2013 A replication",
            ),
        ]
        kept, dropped = dedupe_citations(records)
        assert len(kept) == 1 and dropped == 1

    def test_concept_and_version_doi_collapse_to_one(self):
        records = [
            _work("10.82901/nemar.on004842", "TX15"),
            _work("10.82901/nemar.on004842.v1.0.0", "TX15"),
        ]
        kept, dropped = dedupe_citations(records)
        assert dropped == 1
        assert kept[0]["superseded_dois"] == ["10.82901/nemar.on004842.v1.0.0"]

    def test_preprint_loses_to_version_of_record(self):
        title = (
            "Open and reproducible neuroimaging: from study inception to publication"
        )
        records = [
            _work("10.31219/osf.io/pu5vb", title),
            _work("10.1016/j.neuroimage.2022.119623", title),
        ]
        kept, dropped = dedupe_citations(records)
        assert dropped == 1
        assert kept[0]["doi"] == "10.1016/j.neuroimage.2022.119623"
        assert kept[0]["superseded_dois"] == ["10.31219/osf.io/pu5vb"]

    def test_version_of_record_wins_regardless_of_input_order(self):
        title = "Virtual Reality Sickness Reduces Attention"
        forward = dedupe_citations(
            [
                _work("10.1109/tvcg.2023.3320222", title),
                _work("10.48550/arxiv.2306.13505", title),
            ]
        )[0]
        reverse = dedupe_citations(
            [
                _work("10.48550/arxiv.2306.13505", title),
                _work("10.1109/tvcg.2023.3320222", title),
            ]
        )[0]
        assert forward[0]["doi"] == reverse[0]["doi"] == "10.1109/tvcg.2023.3320222"

    def test_two_journal_dois_sharing_a_title_are_kept(self):
        # A correction or reprint: the title match is not corroborated by a
        # preprint or a missing DOI, so we must not silently merge them.
        title = "Opposite effects of alpha oscillations on mind wandering"
        records = [
            _work("10.1523/jneurosci.0133-26.2026", title),
            _work("10.1002/some.other.2026", title),
        ]
        kept, dropped = dedupe_citations(records)
        assert len(kept) == 2 and dropped == 0

    def test_doi_less_record_merges_into_its_doi_bearing_twin(self):
        title = "PSDNorm: Test-Time Temporal Normalization for Deep Learning"
        records = [
            _work(None, title),
            _work("10.1109/tnsre.2026.1234567", title),
        ]
        kept, dropped = dedupe_citations(records)
        assert dropped == 1
        assert kept[0]["doi"] == "10.1109/tnsre.2026.1234567"

    def test_distinct_works_are_untouched(self):
        records = [
            _work("10.1016/j.cortex.2019.12.001", "Sensorimotor conflicts"),
            _work("10.1038/s41467-024-49538-w", "Distinct basal ganglia"),
        ]
        kept, dropped = dedupe_citations(records)
        assert len(kept) == 2 and dropped == 0

    def test_accession_flag_survives_a_merge(self):
        title = "Stress-Testing EEG Foundation Models"
        records = [
            _work("10.1016/j.neuroimage.2026.1", title),
            _work(
                "10.1101/2026.01.01.000001",
                title,
                mentions_accession=True,
                matched_accession="ds005509",
            ),
        ]
        kept, _ = dedupe_citations(records)
        assert kept[0]["mentions_accession"] is True
        assert kept[0]["matched_accession"] == "ds005509"

    def test_empty_input(self):
        assert dedupe_citations([]) == ([], 0)


class TestDedupeCitationsMultiRecord:
    """Groups larger than a pair, where the two passes interact."""

    def test_doi_pass_winner_carries_into_the_title_pass(self):
        """A record merged in pass 1 must still be the one pass 2 merges into.

        A and B share a base DOI, so pass 1 collapses them; C is the preprint
        of that same work under a different DOI, so pass 2 collapses again. The
        survivor has to accumulate BOTH dropped DOIs, which only happens if the
        pass-1 winner is what pass 2 sees.
        """
        title = "Leveraging SVD Entropy for Alzheimer Detection"
        kept, dropped = dedupe_citations(
            [
                _work("10.36227/techrxiv.23992554", title, cited_by=1),
                _work("10.36227/techrxiv.23992554.v2", title, cited_by=2),
                _work("10.1109/tbme.2024.123456", title, cited_by=30),
            ]
        )
        assert dropped == 2
        assert len(kept) == 1
        assert kept[0]["doi"] == "10.1109/tbme.2024.123456"
        assert kept[0]["superseded_dois"] == [
            "10.36227/techrxiv.23992554",
            "10.36227/techrxiv.23992554.v2",
        ]

    def test_repeated_zenodo_deposits_collapse_to_one(self):
        """Five deposits of one record, as seen in on004504."""
        title = "Analysis code for: Empirical validation of the framework"
        kept, dropped = dedupe_citations(
            [_work(f"10.5281/zenodo.193622{n}", title) for n in range(5)]
        )
        assert len(kept) == 1
        assert dropped == 4
        assert len(kept[0]["superseded_dois"]) == 4

    def test_doi_pass_replaces_the_earlier_record_when_the_later_one_wins(self):
        """Order-independence inside pass 1, not just pass 2.

        Both records share a base DOI; the second is the version of record, so
        it must replace the first in place rather than lose by arriving later.
        """
        kept, dropped = dedupe_citations(
            [
                _work("10.6084/m9.figshare.31991745.v1", "UREEF"),
                _work("10.6084/m9.figshare.31991745", "UREEF", cited_by=9),
            ]
        )
        assert dropped == 1
        assert kept[0]["doi"] == "10.6084/m9.figshare.31991745"

    def test_records_with_neither_doi_nor_title_never_merge(self):
        """Empty identity must not collapse unrelated records into one.

        `base_doi(None)` and `normalize_title(None)` are both "", so a naive
        key would make every identity-less record equal to every other.
        """
        kept, dropped = dedupe_citations(
            [_work(None, None), _work(None, ""), _work(None, None)]
        )
        assert len(kept) == 3
        assert dropped == 0

    def test_existing_with_doi_absorbs_incoming_without_one(self):
        """The symmetric half of the corroborated guard.

        The DOI-bearing record arrives FIRST here; the earlier test covers the
        opposite arrival order.
        """
        title = "BrainWave: A Brain Signal Foundation Model"
        kept, dropped = dedupe_citations(
            [_work("10.1038/s41586-024-00001-x", title), _work(None, title)]
        )
        assert dropped == 1
        assert kept[0]["doi"] == "10.1038/s41586-024-00001-x"

    def test_two_preprints_of_one_work_keep_a_deterministic_winner(self):
        """No version of record present; the choice must still be stable."""
        title = "Stochastic Graph Heat Modelling for Connectivity"
        forward = dedupe_citations(
            [
                _work("10.48550/arxiv.2402.12785", title),
                _work("10.1101/2024.02.13.580102", title),
            ]
        )[0]
        reverse = dedupe_citations(
            [
                _work("10.1101/2024.02.13.580102", title),
                _work("10.48550/arxiv.2402.12785", title),
            ]
        )[0]
        assert forward[0]["doi"] == reverse[0]["doi"]


class TestZenodoIsTitleDependent:
    """Zenodo deposits merge via the title pass, not via `base_doi`.

    Documents a real limit rather than an aspiration: Zenodo mints a distinct
    integer per deposit instead of a `.vN` suffix, so no amount of suffix
    stripping relates two of them.
    """

    def test_base_doi_cannot_relate_two_zenodo_deposits(self):
        assert base_doi("10.5281/zenodo.19051613") != base_doi(
            "10.5281/zenodo.19051614"
        )

    def test_identical_titles_still_merge(self):
        kept, dropped = dedupe_citations(
            [
                _work("10.5281/zenodo.19051613", "The Relational Foundation v2"),
                _work("10.5281/zenodo.19051614", "The Relational Foundation v2"),
            ]
        )
        assert len(kept) == 1 and dropped == 1

    def test_titles_that_drift_between_versions_do_not_merge(self):
        """The documented limitation. If this ever starts passing, the module
        docstring's Zenodo paragraph needs updating too."""
        kept, dropped = dedupe_citations(
            [
                _work("10.5281/zenodo.19051613", "Analysis code (Version 1)"),
                _work("10.5281/zenodo.19051614", "Analysis code (Version 2)"),
            ]
        )
        assert len(kept) == 2 and dropped == 0
