"""Tests for run_find_mentions orchestration. Issue #169.

No network, no Mocks: an empty catalog cache file short-circuits the catalog
fetch, and a real substitute backend (`_FakeBackend`) returns canned mentions in
place of OpenAlex. Real files on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import TestCase

from dataset_citations.backends.accession_search import AccessionSearchResult
from dataset_citations.cli import find_mentions as cli


class _FakeBackend:
    """Real substitute for AccessionSearchBackend (no network)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def search(self, terms: list[str]) -> AccessionSearchResult:
        return AccessionSearchResult(citations=[], failed_terms=[])


def _args(
    citations_dir: Path,
    *,
    dataset_list_file: str | None = None,
    max_results: int = 0,
    max_age_days: int = 0,
    max_datasets: int = 0,
) -> argparse.Namespace:
    cache = citations_dir.parent / "catalog.json"
    cache.write_text("[]")  # empty catalog -> _load_source_id_map hits no network
    return argparse.Namespace(
        citations_dir=str(citations_dir),
        dataset_list_file=dataset_list_file,
        catalog_cache=cache,
        catalog_cache_max_age=3600,
        max_results=max_results,
        # 0 = freshness gate off, so the pre-existing cases below still exercise
        # every seeded dataset regardless of the state cache (issue #197).
        max_age_days=max_age_days,
        max_datasets=max_datasets,
    )


def _seed(citations_dir: Path, dataset_id: str) -> Path:
    citations_dir.mkdir(parents=True, exist_ok=True)
    path = citations_dir / f"{dataset_id}_citations.json"
    path.write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "num_citations": 0,
                "date_last_updated": "2026-06-18T00:00:00+00:00",
                "metadata": {"fetch_status": "success"},
                "citation_details": [],
            },
            indent=2,
        )
    )
    return path


class RunFindMentionsTests(TestCase):
    def setUp(self) -> None:
        self._orig_backend = cli.AccessionSearchBackend
        cli.AccessionSearchBackend = _FakeBackend  # type: ignore[misc,assignment]

    def tearDown(self) -> None:
        cli.AccessionSearchBackend = self._orig_backend  # type: ignore[misc]

    def test_processes_valid_datasets_via_glob(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            _seed(cdir, "ds002718")
            _seed(cdir, "nm000207")
            cli.run_find_mentions(_args(cdir))
            # Processed datasets gain the searched_accessions marker.
            for did, acc in (("ds002718", "ds002718"), ("nm000207", "nm000207")):
                data = json.loads((cdir / f"{did}_citations.json").read_text())
                self.assertEqual(data["metadata"]["searched_accessions"], [acc])

    def test_dataset_list_file_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            _seed(cdir, "ds002718")
            _seed(cdir, "ds000247")  # present but NOT in the list -> untouched
            list_file = Path(tmp) / "list.txt"
            list_file.write_text("ds002718\n")
            cli.run_find_mentions(_args(cdir, dataset_list_file=str(list_file)))
            listed = json.loads((cdir / "ds002718_citations.json").read_text())
            self.assertIn("searched_accessions", listed["metadata"])
            unlisted = json.loads((cdir / "ds000247_citations.json").read_text())
            self.assertNotIn("searched_accessions", unlisted["metadata"])

    def test_skips_missing_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            cdir.mkdir(parents=True)
            list_file = Path(tmp) / "list.txt"
            list_file.write_text("ds999999\n")  # no JSON on disk
            # Should return normally (nothing processed), no exit.
            cli.run_find_mentions(_args(cdir, dataset_list_file=str(list_file)))
            self.assertFalse((cdir / "ds999999_citations.json").exists())

    def test_skips_dataset_with_no_valid_accession(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            _seed(cdir, "experiment-1")  # fails the accession regex
            cli.run_find_mentions(_args(cdir))
            data = json.loads((cdir / "experiment-1_citations.json").read_text())
            # Skipped before merge -> no marker written.
            self.assertNotIn("searched_accessions", data["metadata"])

    def test_all_writes_fail_exits_2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            path = _seed(cdir, "ds002718")
            os.chmod(path, stat.S_IRUSR)  # read-only file -> open('w') raises
            try:
                with self.assertRaises(SystemExit) as ctx:
                    cli.run_find_mentions(_args(cdir))
                self.assertEqual(ctx.exception.code, 2)
            finally:
                os.chmod(path, stat.S_IRWXU)

    def test_idempotent_second_run_leaves_file_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            path = _seed(cdir, "ds002718")
            cli.run_find_mentions(_args(cdir))
            after_first = path.read_text()
            cli.run_find_mentions(_args(cdir))
            self.assertEqual(path.read_text(), after_first)

    def test_real_mention_advances_date_last_updated_on_disk(self) -> None:
        """End-to-end regression guard for issue #229.

        `_FakeBackend` above always returns zero citations, so none of the
        other tests in this file exercise a real content change through
        `run_find_mentions` -> `merge_accession_mentions(..., when=...)` ->
        `write_citation_json_if_changed`. Without this, a regression that
        dropped the `when=` argument at the call site (find_mentions.py) would
        pass every other test here.
        """

        class _FoundBackend:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def search(self, terms: list[str]) -> AccessionSearchResult:
                return AccessionSearchResult(
                    citations=[
                        {
                            "title": "A paper naming the dataset",
                            "author": "Someone",
                            "venue": "A Journal",
                            "year": 2026,
                            "url": "https://doi.org/10.1/found",
                            "cited_by": 0,
                            "abstract": "",
                            "doi": "10.1/found",
                            "pmid": None,
                            "openalex_id": "W1",
                            "source_doi": None,
                            "source_relation": None,
                            "discovery_backend": "openalex",
                            "discovery_method": "accession_mention",
                            "matched_accession": "ds002718",
                        }
                    ],
                    failed_terms=[],
                )

        cli.AccessionSearchBackend = _FoundBackend  # type: ignore[misc]
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            path = _seed(cdir, "ds002718")
            cli.run_find_mentions(_args(cdir))
            data = json.loads(path.read_text())
            self.assertEqual(data["num_citations"], 1)
            self.assertNotEqual(data["date_last_updated"], "2026-06-18T00:00:00+00:00")
            stamped = datetime.fromisoformat(data["date_last_updated"])
            self.assertGreater(stamped, datetime(2026, 6, 18, tzinfo=UTC))


class FreshnessGateTests(TestCase):
    """Rolling `--max-age-days` / `--max-datasets` gating (issue #197).

    Real state files on disk; the freshness decision is real clock arithmetic.
    """

    def test_state_file_written_and_gitignored_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            _seed(cdir, "ds002718")
            cli.run_find_mentions(_args(cdir))
            self.assertTrue((cdir / ".mention_state.json").is_file())

    def test_fresh_dataset_is_skipped_on_second_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            _seed(cdir, "ds002718")
            cli.run_find_mentions(_args(cdir, max_age_days=7))
            state = json.loads((cdir / ".mention_state.json").read_text())
            self.assertIn("ds002718", state)

            # Second run inside the window must not re-stamp (no re-search).
            cli.run_find_mentions(_args(cdir, max_age_days=7))
            self.assertEqual(
                json.loads((cdir / ".mention_state.json").read_text()), state
            )

    def test_stale_dataset_is_reprocessed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            _seed(cdir, "ds002718")
            stale = (datetime.now(UTC) - timedelta(days=30)).isoformat()
            (cdir / ".mention_state.json").write_text(json.dumps({"ds002718": stale}))
            cli.run_find_mentions(_args(cdir, max_age_days=7))
            state = json.loads((cdir / ".mention_state.json").read_text())
            self.assertNotEqual(state["ds002718"], stale)

    def test_max_datasets_caps_the_run_and_rolls_over(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            for i in range(5):
                _seed(cdir, f"ds00271{i}")
            cli.run_find_mentions(_args(cdir, max_age_days=7, max_datasets=2))
            first = json.loads((cdir / ".mention_state.json").read_text())
            self.assertEqual(len(first), 2)

            # The remaining three roll over to subsequent runs.
            cli.run_find_mentions(_args(cdir, max_age_days=7, max_datasets=2))
            second = json.loads((cdir / ".mention_state.json").read_text())
            self.assertEqual(len(second), 4)
            cli.run_find_mentions(_args(cdir, max_age_days=7, max_datasets=2))
            third = json.loads((cdir / ".mention_state.json").read_text())
            self.assertEqual(len(third), 5)

    def test_max_age_zero_disables_the_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            _seed(cdir, "ds002718")
            cli.run_find_mentions(_args(cdir, max_age_days=0))
            first = json.loads((cdir / ".mention_state.json").read_text())
            cli.run_find_mentions(_args(cdir, max_age_days=0))
            second = json.loads((cdir / ".mention_state.json").read_text())
            # Gate off -> searched again -> stamp advances.
            self.assertNotEqual(first["ds002718"], second["ds002718"])

    def test_state_persists_when_the_search_raises(self) -> None:
        """A mid-loop failure (the OpenAlex budget running out) must not
        discard the progress already made, or the next run redoes everything.
        """

        class _RaisingBackend:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.calls = 0

            def search(self, terms: list[str]) -> AccessionSearchResult:
                self.calls += 1
                if self.calls > 1:
                    raise RuntimeError("connection reset")
                return AccessionSearchResult(citations=[], failed_terms=[])

        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            for i in range(3):
                _seed(cdir, f"ds00271{i}")
            original = cli.AccessionSearchBackend
            cli.AccessionSearchBackend = _RaisingBackend  # type: ignore[misc]
            try:
                with self.assertRaises(RuntimeError):
                    cli.run_find_mentions(_args(cdir, max_age_days=7))
            finally:
                cli.AccessionSearchBackend = original  # type: ignore[misc]

            state = json.loads((cdir / ".mention_state.json").read_text())
            self.assertEqual(len(state), 1)  # the one that succeeded

    def test_write_failure_does_not_stamp_and_retries_next_run(self) -> None:
        """A dataset whose write failed must NOT be recorded as checked.

        Regression guard for the silent-skip failure mode: if `stamp_checked`
        ever moved above the write's try/except, a transient write error would
        mark the dataset checked and drop it from citation coverage for a full
        --max-age-days window with nothing in the cron log to show for it.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            path = _seed(cdir, "ds002718")
            _seed(cdir, "ds002719")
            os.chmod(path, stat.S_IRUSR)  # read-only -> open('w') raises
            try:
                cli.run_find_mentions(_args(cdir, max_age_days=7))
                state = json.loads((cdir / ".mention_state.json").read_text())
                self.assertNotIn("ds002718", state)  # failed write -> unstamped
                self.assertIn("ds002719", state)  # its neighbour still recorded
            finally:
                os.chmod(path, stat.S_IRWXU)

            # Still stale, so the next run picks it up again.
            cli.run_find_mentions(_args(cdir, max_age_days=7))
            state = json.loads((cdir / ".mention_state.json").read_text())
            self.assertIn("ds002718", state)

    def test_unreadable_citation_json_does_not_stamp(self) -> None:
        """A corrupt citation JSON must stay stale rather than be marked done."""
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            cdir.mkdir(parents=True)
            (cdir / "ds002718_citations.json").write_text("{not valid json")
            _seed(cdir, "ds002719")
            cli.run_find_mentions(_args(cdir, max_age_days=7))
            state = json.loads((cdir / ".mention_state.json").read_text())
            self.assertNotIn("ds002718", state)
            self.assertIn("ds002719", state)

    def test_no_terms_dataset_is_stamped(self) -> None:
        """Locks in the deliberate choice at find_mentions.py: a dataset with no
        usable accession IS stamped, so it stops consuming a --max-datasets slot
        every night even though no request was made for it.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            _seed(cdir, "experiment-1")  # fails the accession regex
            cli.run_find_mentions(_args(cdir, max_age_days=7))
            state = json.loads((cdir / ".mention_state.json").read_text())
            self.assertIn("experiment-1", state)

            # And is therefore filtered out as fresh on the next run.
            cli.run_find_mentions(_args(cdir, max_age_days=7))
            self.assertEqual(
                json.loads((cdir / ".mention_state.json").read_text()), state
            )


class DegradedSearchTests(TestCase):
    """A rate-limited search must NOT be recorded as a completed one.

    `AccessionSearchBackend` logs and skips a failed term rather than raising
    (accession_search.py `_search_all`), so an empty result is ambiguous.
    Stamping on a degraded search would hide the dataset for a full
    --max-age-days window with no citation coverage and no error anywhere.
    """

    class _DegradedBackend:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def search(self, terms: list[str]) -> AccessionSearchResult:
            # Every term failed: exactly what an exhausted OpenAlex budget
            # produces once opencite's retries are used up.
            return AccessionSearchResult(citations=[], failed_terms=list(terms))

    def setUp(self) -> None:
        self._orig = cli.AccessionSearchBackend
        cli.AccessionSearchBackend = self._DegradedBackend  # type: ignore[misc]

    def tearDown(self) -> None:
        cli.AccessionSearchBackend = self._orig  # type: ignore[misc]

    def test_degraded_search_is_not_stamped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            _seed(cdir, "ds002718")
            cli.run_find_mentions(_args(cdir, max_age_days=7))
            state = json.loads((cdir / ".mention_state.json").read_text())
            self.assertEqual(state, {})

    def test_degraded_dataset_is_retried_next_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            _seed(cdir, "ds002718")
            cli.run_find_mentions(_args(cdir, max_age_days=7))
            # Backend recovers; the dataset is still stale so it gets searched.
            cli.AccessionSearchBackend = _FakeBackend  # type: ignore[misc]
            cli.run_find_mentions(_args(cdir, max_age_days=7))
            state = json.loads((cdir / ".mention_state.json").read_text())
            self.assertIn("ds002718", state)

    def test_partial_term_failure_also_withholds_the_stamp(self) -> None:
        class _PartialBackend:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def search(self, terms: list[str]) -> AccessionSearchResult:
                return AccessionSearchResult(citations=[], failed_terms=terms[:1])

        cli.AccessionSearchBackend = _PartialBackend  # type: ignore[misc]
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            _seed(cdir, "ds002718")
            cli.run_find_mentions(_args(cdir, max_age_days=7))
            state = json.loads((cdir / ".mention_state.json").read_text())
            self.assertNotIn("ds002718", state)
