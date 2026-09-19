"""Backfill CLI: rewrites existing citation JSONs with duplicates merged.

Fixtures mirror the real file shape (schema v2.1) rather than a reduced stub,
so the count recomputation and the confidence_scoring drop are exercised the
way the corpus exercises them.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from dataset_citations.cli.dedupe_citations import (
    CitationFileUnreadable,
    dedupe_citation_file,
)

_OLD = "2026-09-16T00:00:00+00:00"


def _write(tmp_path: Path, name: str, details: list[dict], **extra) -> Path:
    payload = {
        "dataset_id": name,
        "num_citations": len(details),
        "date_last_updated": _OLD,
        "metadata": {
            "schema_version": "2.1",
            "discovery_backend": "opencite",
            "total_cumulative_citations": sum(
                int(d.get("cited_by") or 0) for d in details
            ),
        },
        "citation_details": details,
    }
    payload.update(extra)
    path = tmp_path / f"{name}_citations.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def preprint_pair(tmp_path):
    title = "Open and reproducible neuroimaging"
    return _write(
        tmp_path,
        "nm000132",
        [
            {"doi": "10.31219/osf.io/pu5vb", "title": title, "cited_by": 3},
            {"doi": "10.1016/j.neuroimage.2022.119623", "title": title, "cited_by": 40},
        ],
        confidence_scoring={"mean_confidence": 0.7},
    )


class TestDedupeCitationFile:
    def test_merges_and_recomputes_counts(self, preprint_pair):
        dropped = dedupe_citation_file(preprint_pair)
        assert dropped == 1
        payload = json.loads(preprint_pair.read_text())
        assert payload["num_citations"] == 1
        # The cumulative total must follow the survivor, not the sum of both.
        assert payload["metadata"]["total_cumulative_citations"] == 40
        assert payload["citation_details"][0]["doi"] == (
            "10.1016/j.neuroimage.2022.119623"
        )

    def test_drops_stale_confidence_block(self, preprint_pair):
        dedupe_citation_file(preprint_pair)
        payload = json.loads(preprint_pair.read_text())
        assert "confidence_scoring" not in payload

    def test_dry_run_leaves_the_file_byte_identical(self, preprint_pair):
        before = preprint_pair.read_bytes()
        assert dedupe_citation_file(preprint_pair, dry_run=True) == 1
        assert preprint_pair.read_bytes() == before

    def test_clean_file_is_not_rewritten(self, tmp_path):
        path = _write(
            tmp_path,
            "on000001",
            [
                {"doi": "10.1016/a", "title": "A", "cited_by": 1},
                {"doi": "10.1016/b", "title": "B", "cited_by": 2},
            ],
        )
        before = path.read_bytes()
        assert dedupe_citation_file(path) == 0
        assert path.read_bytes() == before

    def test_second_run_is_a_noop(self, preprint_pair):
        assert dedupe_citation_file(preprint_pair) == 1
        assert dedupe_citation_file(preprint_pair) == 0

    def test_empty_citation_list_is_safe(self, tmp_path):
        path = _write(tmp_path, "on000002", [])
        assert dedupe_citation_file(path) == 0

    def test_unreadable_file_raises_rather_than_reporting_zero(self, tmp_path):
        """A corrupt file must be distinguishable from a clean one.

        Returning 0 made "could not parse this file" identical to "no
        duplicates here", so a truncated citation JSON was skipped forever
        with no aggregate signal and the CLI still exited 0.
        """
        path = tmp_path / "broken_citations.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(CitationFileUnreadable):
            dedupe_citation_file(path)

    def test_bucket_counters_follow_the_surviving_records(self, tmp_path):
        """num_dataset_citations / num_datapaper_citations must be re-derived.

        The dashboard keys on these rather than recomputing them (AGENTS.md),
        so leaving them stale after a merge publishes buckets that no longer
        sum to num_citations.
        """
        title = "Shared work"
        path = _write(
            tmp_path,
            "on004148",
            [
                {
                    "doi": "10.1016/j.neuroimage.2022.1",
                    "title": title,
                    "cited_by": 5,
                    "source_relation": "References",
                },
                {
                    "doi": "10.1101/2022.01.01.000001",
                    "title": title,
                    "cited_by": 1,
                    "source_relation": "References",
                },
                {
                    "doi": "10.1016/j.other.2022.2",
                    "title": "Another work",
                    "cited_by": 2,
                    "discovery_method": "accession_mention",
                },
            ],
        )
        payload = json.loads(path.read_text())
        payload["metadata"]["num_dataset_citations"] = 1
        payload["metadata"]["num_datapaper_citations"] = 2
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        assert dedupe_citation_file(path) == 1
        after = json.loads(path.read_text())
        buckets = (
            after["metadata"]["num_dataset_citations"]
            + after["metadata"]["num_datapaper_citations"]
        )
        assert buckets == after["num_citations"] == 2

    def test_bucket_counters_are_not_invented_when_absent(self, tmp_path):
        """A file that never went through find-mentions must not gain them."""
        title = "Shared work"
        path = _write(
            tmp_path,
            "on000003",
            [
                {"doi": "10.1016/a.1", "title": title, "cited_by": 5},
                {"doi": "10.1101/2022.01.01.000002", "title": title, "cited_by": 1},
            ],
        )
        assert dedupe_citation_file(path) == 1
        after = json.loads(path.read_text())
        assert "num_dataset_citations" not in after["metadata"]
        assert "num_datapaper_citations" not in after["metadata"]


class TestDateLastUpdated:
    """Issue #229: merging duplicates changes content and must advance the stamp."""

    def test_merging_a_duplicate_advances_date_last_updated(self, preprint_pair):
        when = datetime(2026, 9, 20, 3, 0, 0, tzinfo=UTC)
        dedupe_citation_file(preprint_pair, when=when)
        payload = json.loads(preprint_pair.read_text())
        assert payload["date_last_updated"] == when.isoformat()

    def test_clean_file_leaves_date_last_updated_untouched(self, tmp_path):
        path = _write(
            tmp_path,
            "on000004",
            [
                {"doi": "10.1016/a", "title": "A", "cited_by": 1},
                {"doi": "10.1016/b", "title": "B", "cited_by": 2},
            ],
        )
        when = datetime(2026, 9, 20, 3, 0, 0, tzinfo=UTC)
        assert dedupe_citation_file(path, when=when) == 0
        payload = json.loads(path.read_text())
        assert payload["date_last_updated"] == _OLD

    def test_dry_run_does_not_advance_the_stamp_on_disk(self, preprint_pair):
        when = datetime(2026, 9, 20, 3, 0, 0, tzinfo=UTC)
        dedupe_citation_file(preprint_pair, dry_run=True, when=when)
        payload = json.loads(preprint_pair.read_text())
        assert payload["date_last_updated"] == _OLD

    def test_defaults_to_now_when_when_is_omitted(self, preprint_pair):
        before = datetime.now(UTC)
        dedupe_citation_file(preprint_pair)
        payload = json.loads(preprint_pair.read_text())
        stamped = datetime.fromisoformat(payload["date_last_updated"])
        assert stamped >= before
