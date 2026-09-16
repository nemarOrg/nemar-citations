"""Backfill CLI: rewrites existing citation JSONs with duplicates merged.

Fixtures mirror the real file shape (schema v2.1) rather than a reduced stub,
so the count recomputation and the confidence_scoring drop are exercised the
way the corpus exercises them.
"""

import json
from pathlib import Path

import pytest

from dataset_citations.cli.dedupe_citations import dedupe_citation_file


def _write(tmp_path: Path, name: str, details: list[dict], **extra) -> Path:
    payload = {
        "dataset_id": name,
        "num_citations": len(details),
        "date_last_updated": "2026-09-16T00:00:00+00:00",
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

    def test_unreadable_file_is_skipped_not_raised(self, tmp_path):
        path = tmp_path / "broken_citations.json"
        path.write_text("{not json", encoding="utf-8")
        assert dedupe_citation_file(path) == 0
