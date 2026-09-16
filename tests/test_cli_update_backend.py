"""Tests for `dataset-citations-update` (opencite-only after Phase 4).

Uses monkeypatch (stdlib setattr) to substitute one real function for another,
matching the project's "no Mock objects" rule.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from datetime import UTC
from pathlib import Path
from unittest import TestCase

from dataset_citations.cli import update as cli_update


def _make_args(
    list_file: Path,
    out_dir: str,
    *,
    catalog_doi_map: dict[str, str] | None = None,
    max_age_days: int = 0,
    github_token: str | None = None,
) -> argparse.Namespace:
    """Build the argparse.Namespace `run_opencite_backend` expects.

    Writes an empty catalog cache so `_load_catalog_doi_map` returns an
    empty map without hitting the network. Pass `catalog_doi_map` to
    inject a non-empty mapping for tests that exercise catalog-DOI
    seeding.
    """
    cache_path = Path(out_dir) / "catalog.json"
    if catalog_doi_map:
        rows = [
            {
                "dataset_id": did,
                "doi": doi,
                "concept_doi": doi,
                "source": "openneuro" if did.startswith(("ds", "on")) else "nemar",
                "source_id": None,
                "github_repo": f"nemarDatasets/{did}",
                "modalities": "eeg",
                "name": None,
                "visibility": "public",
            }
            for did, doi in catalog_doi_map.items()
        ]
        cache_path.write_text(json.dumps(rows))
    else:
        cache_path.write_text("[]")
    return argparse.Namespace(
        dataset_list_file=str(list_file),
        output_dir=out_dir,
        # Distinct from output_dir and intentionally empty: these tests stub the
        # pipeline or use unsupported prefixes, so DOI extraction never reads the
        # cache. The cache-hit path is covered in test_sources_bids_metadata.py.
        datasets_dir=os.path.join(out_dir, "datasets"),
        catalog_cache=cache_path,
        catalog_cache_max_age=3600,
        max_age_days=max_age_days,
        github_token=github_token,
    )


class CliParserTests(TestCase):
    def test_real_help_text_documents_opencite_pipeline(self) -> None:
        """The CLI is now single-path; help text must describe the opencite
        pipeline and accept --dataset-list-file. The legacy --backend flag
        must NOT be present (Phase 4 retired scholarly)."""
        buf = io.StringIO()
        old_argv = sys.argv
        sys.argv = ["dataset-citations-update", "--help"]
        try:
            with self.assertRaises(SystemExit), redirect_stdout(buf):
                cli_update.main()
        finally:
            sys.argv = old_argv
        help_text = buf.getvalue()
        self.assertIn("--dataset-list-file", help_text)
        self.assertIn("opencite", help_text)
        self.assertNotIn("--backend", help_text)
        self.assertNotIn("scholarly", help_text)

    def test_main_invokes_run_opencite_backend(self) -> None:
        """main() always calls run_opencite_backend; no backend dispatch."""
        with tempfile.TemporaryDirectory() as tmp:
            list_file = Path(tmp) / "list.txt"
            list_file.write_text("ds999999\n")

            captured: dict[str, object] = {}

            def recording_run_opencite(args):
                captured["called"] = True
                captured["dataset_list_file"] = args.dataset_list_file
                captured["output_dir"] = args.output_dir

            original = cli_update.run_opencite_backend
            cli_update.run_opencite_backend = recording_run_opencite  # type: ignore[assignment]
            old_argv = sys.argv
            sys.argv = [
                "dataset-citations-update",
                "--dataset-list-file",
                str(list_file),
                "--output-dir",
                tmp,
            ]
            try:
                cli_update.main()
            finally:
                cli_update.run_opencite_backend = original  # type: ignore[assignment]
                sys.argv = old_argv

            self.assertTrue(captured.get("called"))
            self.assertEqual(captured.get("output_dir"), tmp)
            self.assertEqual(captured.get("dataset_list_file"), str(list_file))


class RunOpenciteBackendTests(TestCase):
    """Direct tests for run_opencite_backend's write/dispatch behavior."""

    def test_unsupported_prefix_still_writes_stub(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            list_file = Path(out_dir) / "list.txt"
            list_file.write_text("xx123\n")
            args = _make_args(list_file, out_dir)
            cli_update.run_opencite_backend(args)
            stub = Path(out_dir) / "json_opencite" / "xx123_citations.json"
            self.assertTrue(stub.exists())
            payload = json.loads(stub.read_text())
            self.assertEqual(payload["num_citations"], 0)
            self.assertIn("unsupported_prefix", payload["metadata"]["fetch_status"])
            self.assertFalse((Path(out_dir) / "json").exists())

    def test_all_api_failures_raises_systemexit_3(self) -> None:
        """When every dataset stubs out with an API-failure status (and no
        write succeeded), exit 3 so cron / CI catches the degraded run.

        Phase 4 review finding: a fully-rate-limited run would otherwise
        proceed silently and downstream steps would regenerate empty CSVs.
        Substitute the pipeline call with a recording function via setattr.
        """
        from dataset_citations.cli import update as cli_module

        with tempfile.TemporaryDirectory() as out_dir:
            list_file = Path(out_dir) / "list.txt"
            list_file.write_text("ds111111\nds222222\n")

            def stub_pipeline(dataset_id, **_):
                return {
                    "dataset_id": dataset_id,
                    "num_citations": 0,
                    "date_last_updated": "2026-01-01T00:00:00",
                    "metadata": {
                        "schema_version": "2.0",
                        "discovery_backend": "opencite",
                        "fetch_status": "rate_limit",
                    },
                    "citation_details": [],
                }

            original = cli_module.fetch_dataset_citations_via_opencite
            cli_module.fetch_dataset_citations_via_opencite = stub_pipeline  # type: ignore[assignment]
            args = _make_args(list_file, out_dir)
            try:
                with self.assertRaises(SystemExit) as ctx:
                    cli_update.run_opencite_backend(args)
                self.assertEqual(ctx.exception.code, 3)
            finally:
                cli_module.fetch_dataset_citations_via_opencite = original  # type: ignore[assignment]

    def test_no_doi_references_does_not_exit_nonzero(self) -> None:
        """A run where every dataset legitimately has no DOI references
        should NOT exit non-zero; that's a normal outcome (e.g., new nm
        datasets without metadata.json yet)."""
        from dataset_citations.cli import update as cli_module

        with tempfile.TemporaryDirectory() as out_dir:
            list_file = Path(out_dir) / "list.txt"
            list_file.write_text("nm999999\n")

            def stub_pipeline(dataset_id, **_):
                return {
                    "dataset_id": dataset_id,
                    "num_citations": 0,
                    "date_last_updated": "2026-01-01T00:00:00",
                    "metadata": {
                        "schema_version": "2.0",
                        "discovery_backend": "opencite",
                        "fetch_status": "no_doi_references",
                    },
                    "citation_details": [],
                }

            original = cli_module.fetch_dataset_citations_via_opencite
            cli_module.fetch_dataset_citations_via_opencite = stub_pipeline  # type: ignore[assignment]
            args = _make_args(list_file, out_dir)
            try:
                # Should return normally, no SystemExit.
                cli_update.run_opencite_backend(args)
            finally:
                cli_module.fetch_dataset_citations_via_opencite = original  # type: ignore[assignment]

    def test_not_found_only_run_does_not_exit_nonzero(self) -> None:
        """A run where every dataset's anchors are unindexed must NOT exit 3.

        End-to-end regression for #217, driving `run_opencite_backend` rather
        than asserting on the constant. 36 datasets have anchor DOIs that are
        permanently absent from OpenAlex (`on004019`'s is the placeholder
        `10.3389/fnhum.2022.xxxxx`), so on most nights they were the only
        datasets processed. While `not_found` counted as an API failure that
        made `successes == 0 and api_failures == processed` true, aborting the
        nightly pipeline before scoring, embeddings and the commit.

        The bug can return through the successes/api_failures/processed
        bookkeeping, not just through the frozenset, which is why this drives
        the real exit path.
        """
        from dataset_citations.cli import update as cli_module

        with tempfile.TemporaryDirectory() as out_dir:
            list_file = Path(out_dir) / "list.txt"
            list_file.write_text("on004019\non006593\n")

            def stub_pipeline(dataset_id, **_):
                return {
                    "dataset_id": dataset_id,
                    "num_citations": 0,
                    "date_last_updated": "2026-01-01T00:00:00",
                    "metadata": {
                        "schema_version": "2.1",
                        "discovery_backend": "opencite",
                        "fetch_status": "not_found",
                    },
                    "citation_details": [],
                }

            original = cli_module.fetch_dataset_citations_via_opencite
            cli_module.fetch_dataset_citations_via_opencite = stub_pipeline  # type: ignore[assignment]
            args = _make_args(list_file, out_dir)
            try:
                # Must return normally: no SystemExit at all.
                cli_update.run_opencite_backend(args)
            finally:
                cli_module.fetch_dataset_citations_via_opencite = original  # type: ignore[assignment]

    def test_rate_limit_still_exits_3_after_the_not_found_change(self) -> None:
        """Removing `not_found` must not blunt outage detection.

        The exit-3 sentinel still has to fire when the failures are genuinely
        transient, or #217's fix would trade one silent failure for another.
        """
        from dataset_citations.cli import update as cli_module

        with tempfile.TemporaryDirectory() as out_dir:
            list_file = Path(out_dir) / "list.txt"
            list_file.write_text("on004019\non006593\n")

            def stub_pipeline(dataset_id, **_):
                return {
                    "dataset_id": dataset_id,
                    "num_citations": 0,
                    "date_last_updated": "2026-01-01T00:00:00",
                    "metadata": {
                        "schema_version": "2.1",
                        "discovery_backend": "opencite",
                        "fetch_status": "rate_limit",
                    },
                    "citation_details": [],
                }

            original = cli_module.fetch_dataset_citations_via_opencite
            cli_module.fetch_dataset_citations_via_opencite = stub_pipeline  # type: ignore[assignment]
            args = _make_args(list_file, out_dir)
            try:
                with self.assertRaises(SystemExit) as ctx:
                    cli_update.run_opencite_backend(args)
                self.assertEqual(ctx.exception.code, 3)
            finally:
                cli_module.fetch_dataset_citations_via_opencite = original  # type: ignore[assignment]

    def test_total_write_failure_raises_systemexit(self) -> None:
        """When every write fails, surface non-zero exit.

        Force-fail writes by making the json_opencite output directory
        read-only after run_opencite_backend creates it. We re-enter the
        function with the directory already in place but non-writable.
        """
        import os
        import stat

        with tempfile.TemporaryDirectory() as out_dir:
            list_file = Path(out_dir) / "list.txt"
            list_file.write_text("xx123\nxx456\n")
            # Pre-create the json_opencite directory and make it read-only;
            # makedirs(exist_ok=True) tolerates an existing dir, then each
            # file write inside fails with PermissionError (OSError subclass).
            json_dir = Path(out_dir) / "json_opencite"
            json_dir.mkdir()
            os.chmod(json_dir, stat.S_IRUSR | stat.S_IXUSR)
            args = _make_args(list_file, out_dir)
            try:
                with self.assertRaises(SystemExit) as ctx:
                    cli_update.run_opencite_backend(args)
                self.assertEqual(ctx.exception.code, 2)
            finally:
                # Restore write perm so tempfile cleanup works.
                os.chmod(json_dir, stat.S_IRWXU)


class FreshnessGateTests(TestCase):
    """Direct tests for the decoupled freshness helpers (issue #165).

    `_has_stable_status` reads the on-disk JSON; `checked_within` reads the
    fetch-state cache. Together they gate a re-fetch. Pure functions, real
    fixtures on disk, no stubs.
    """

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    def test_has_stable_status_false_when_missing(self) -> None:
        from dataset_citations.cli.update import _has_stable_status

        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(_has_stable_status(str(Path(tmp) / "absent.json")))

    def test_has_stable_status_true_for_success(self) -> None:
        from dataset_citations.cli.update import _has_stable_status

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nm000104_citations.json"
            self._write_json(path, {"metadata": {"fetch_status": "success"}})
            self.assertTrue(_has_stable_status(str(path)))

    def test_has_stable_status_true_for_no_doi_references(self) -> None:
        # The key fix: an empty dataset (no DOI refs) is a STABLE outcome, so it
        # should be skippable within the window instead of refetched nightly.
        from dataset_citations.cli.update import _has_stable_status

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nm000104_citations.json"
            self._write_json(path, {"metadata": {"fetch_status": "no_doi_references"}})
            self.assertTrue(_has_stable_status(str(path)))

    def test_has_stable_status_true_for_partial(self) -> None:
        # `partial` (some anchors errored, rest succeeded) is a stable outcome.
        from dataset_citations.cli.update import _has_stable_status

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nm000104_citations.json"
            self._write_json(path, {"metadata": {"fetch_status": "partial"}})
            self.assertTrue(_has_stable_status(str(path)))

    def test_has_stable_status_true_for_no_data_paper_anchor(self) -> None:
        from dataset_citations.cli.update import _has_stable_status

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nm000104_citations.json"
            self._write_json(
                path, {"metadata": {"fetch_status": "no_data_paper_anchor"}}
            )
            self.assertTrue(_has_stable_status(str(path)))

    def test_has_stable_status_false_for_transient_failure(self) -> None:
        from dataset_citations.cli.update import _has_stable_status

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nm000104_citations.json"
            self._write_json(path, {"metadata": {"fetch_status": "rate_limit"}})
            self.assertFalse(_has_stable_status(str(path)))

    def test_has_stable_status_false_for_malformed_or_headerless(self) -> None:
        from dataset_citations.cli.update import _has_stable_status

        with tempfile.TemporaryDirectory() as tmp:
            broken = Path(tmp) / "broken.json"
            broken.write_text("not json at all")
            self.assertFalse(_has_stable_status(str(broken)))
            headerless = Path(tmp) / "headerless.json"
            self._write_json(headerless, {"dataset_id": "x", "num_citations": 0})
            self.assertFalse(_has_stable_status(str(headerless)))

    def test_checked_within_true_for_recent(self) -> None:
        from datetime import datetime

        from dataset_citations.core.run_state import checked_within

        state = {"ds1": datetime.now(UTC).isoformat()}
        self.assertTrue(checked_within("ds1", state, 7 * 86400))

    def test_checked_within_false_for_stale(self) -> None:
        from datetime import datetime, timedelta

        from dataset_citations.core.run_state import checked_within

        stale = (datetime.now(UTC) - timedelta(days=30)).isoformat()
        self.assertFalse(checked_within("ds1", {"ds1": stale}, 7 * 86400))

    def test_checked_within_false_for_missing_or_unparseable(self) -> None:
        from dataset_citations.core.run_state import checked_within

        self.assertFalse(checked_within("ds1", {}, 7 * 86400))
        self.assertFalse(checked_within("ds1", {"ds1": "not-a-date"}, 7 * 86400))

    def test_checked_within_treats_naive_timestamp_as_utc(self) -> None:
        from datetime import datetime

        from dataset_citations.core.run_state import checked_within

        naive_now = datetime.now(UTC).replace(tzinfo=None).isoformat()
        self.assertTrue(checked_within("ds1", {"ds1": naive_now}, 7 * 86400))

    def test_fetch_state_roundtrip(self) -> None:
        from dataset_citations.core.run_state import load_state, save_state

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / ".fetch_state.json")
            self.assertEqual(load_state(path), {})  # absent -> empty
            save_state(path, {"ds1": "2026-06-18T00:00:00+00:00"})
            self.assertEqual(load_state(path), {"ds1": "2026-06-18T00:00:00+00:00"})

    def test_load_fetch_state_returns_empty_on_corrupt_file(self) -> None:
        from dataset_citations.core.run_state import load_state

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".fetch_state.json"
            path.write_text("not json at all")
            self.assertEqual(load_state(str(path)), {})

    def test_has_stable_status_true_for_not_found(self) -> None:
        """A DOI absent from OpenAlex is a property of the DOI, not the API.

        Regression for issue #217: while `not_found` counted as a transient API
        failure, the 36 datasets whose anchors are permanently unindexed were
        re-fetched every night, were the only datasets processed on most nights,
        and so tripped the exit-3 guard and aborted the whole nightly pipeline
        before scoring, embeddings or the commit.
        """
        from dataset_citations.cli.update import _has_stable_status

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "on004019_citations.json"
            self._write_json(path, {"metadata": {"fetch_status": "not_found"}})
            self.assertTrue(_has_stable_status(str(path)))

    def test_not_found_is_not_an_api_failure_status(self) -> None:
        """The exit-3 sentinel must not fire on permanently unindexed anchors."""
        from dataset_citations.cli.update import _API_FAILURE_STATUSES

        self.assertNotIn("not_found", _API_FAILURE_STATUSES)
        # The genuinely transient ones stay, or exit 3 stops detecting outages.
        for status in ("rate_limit", "auth", "network", "parse", "other"):
            self.assertIn(status, _API_FAILURE_STATUSES)


class IdempotentWriteTests(TestCase):
    """run_opencite_backend must not rewrite a citation file when only the
    volatile timestamps differ, and must drop stale scores when content
    genuinely changes so the score step re-scores (issue #165).
    """

    def _seed(self, out_dir: str, dataset_id: str, payload: dict) -> Path:
        json_dir = Path(out_dir) / "json_opencite"
        json_dir.mkdir(parents=True, exist_ok=True)
        path = json_dir / f"{dataset_id}_citations.json"
        path.write_text(json.dumps(payload, indent=2))
        return path

    def _run_with_stub(self, out_dir: str, dataset_id: str, payload: dict) -> None:
        from dataset_citations.cli import update as cli_module

        list_file = Path(out_dir) / "list.txt"
        list_file.write_text(f"{dataset_id}\n")

        def stub_pipeline(_did, **_):
            return payload

        original = cli_module.fetch_dataset_citations_via_opencite
        cli_module.fetch_dataset_citations_via_opencite = stub_pipeline  # type: ignore[assignment]
        try:
            cli_module.run_opencite_backend(_make_args(list_file, out_dir))
        finally:
            cli_module.fetch_dataset_citations_via_opencite = original  # type: ignore[assignment]

    def test_unchanged_content_does_not_rewrite_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            existing = {
                "dataset_id": "ds000001",
                "num_citations": 0,
                "date_last_updated": "2026-06-14T10:00:00+00:00",
                "metadata": {
                    "fetch_date": "2026-06-14T10:00:00+00:00",
                    "fetch_status": "no_doi_references",
                },
                "citation_details": [],
            }
            path = self._seed(out_dir, "ds000001", existing)
            # Same content, only the timestamps advance.
            fetched = json.loads(json.dumps(existing))
            fetched["date_last_updated"] = "2026-06-18T10:00:00+00:00"
            fetched["metadata"]["fetch_date"] = "2026-06-18T10:00:00+00:00"
            self._run_with_stub(out_dir, "ds000001", fetched)
            on_disk = json.loads(path.read_text())
            # File left untouched: old timestamp preserved.
            self.assertEqual(on_disk["date_last_updated"], "2026-06-14T10:00:00+00:00")

    def test_changed_content_rewrites_and_advances_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            existing = {
                "dataset_id": "ds000002",
                "num_citations": 1,
                "date_last_updated": "2026-06-14T10:00:00+00:00",
                "metadata": {
                    "fetch_date": "2026-06-14T10:00:00+00:00",
                    "fetch_status": "success",
                },
                "citation_details": [{"title": "A", "doi": "10.1/a"}],
            }
            path = self._seed(out_dir, "ds000002", existing)
            fetched = {
                "dataset_id": "ds000002",
                "num_citations": 2,
                "date_last_updated": "2026-06-18T10:00:00+00:00",
                "metadata": {
                    "fetch_date": "2026-06-18T10:00:00+00:00",
                    "fetch_status": "success",
                },
                "citation_details": [
                    {"title": "A", "doi": "10.1/a"},
                    {"title": "B", "doi": "10.1/b"},
                ],
            }
            self._run_with_stub(out_dir, "ds000002", fetched)
            on_disk = json.loads(path.read_text())
            self.assertEqual(on_disk["num_citations"], 2)
            self.assertEqual(on_disk["date_last_updated"], "2026-06-18T10:00:00+00:00")

    def test_unchanged_preserves_existing_confidence_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            existing = {
                "dataset_id": "ds000003",
                "num_citations": 1,
                "date_last_updated": "2026-06-14T10:00:00+00:00",
                "metadata": {
                    "fetch_date": "2026-06-14T10:00:00+00:00",
                    "fetch_status": "success",
                },
                "citation_details": [{"title": "A", "doi": "10.1/a"}],
                "confidence_scoring": {
                    "model_used": "Qwen/Qwen3-Embedding-0.6B",
                    "scoring_date": "2026-06-15T01:00:00+00:00",
                },
            }
            path = self._seed(out_dir, "ds000003", existing)
            # opencite payload never carries confidence_scoring.
            fetched = {
                "dataset_id": "ds000003",
                "num_citations": 1,
                "date_last_updated": "2026-06-18T10:00:00+00:00",
                "metadata": {
                    "fetch_date": "2026-06-18T10:00:00+00:00",
                    "fetch_status": "success",
                },
                "citation_details": [{"title": "A", "doi": "10.1/a"}],
            }
            self._run_with_stub(out_dir, "ds000003", fetched)
            on_disk = json.loads(path.read_text())
            # Not rewritten -> score block (and its date) preserved.
            self.assertIn("confidence_scoring", on_disk)
            self.assertEqual(
                on_disk["confidence_scoring"]["scoring_date"],
                "2026-06-15T01:00:00+00:00",
            )

    def test_changed_content_drops_stale_confidence_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            existing = {
                "dataset_id": "ds000004",
                "num_citations": 1,
                "date_last_updated": "2026-06-14T10:00:00+00:00",
                "metadata": {
                    "fetch_date": "2026-06-14T10:00:00+00:00",
                    "fetch_status": "success",
                },
                "citation_details": [{"title": "A", "doi": "10.1/a"}],
                "confidence_scoring": {
                    "model_used": "Qwen/Qwen3-Embedding-0.6B",
                    "scoring_date": "2026-06-15T01:00:00+00:00",
                },
            }
            path = self._seed(out_dir, "ds000004", existing)
            fetched = {
                "dataset_id": "ds000004",
                "num_citations": 2,
                "date_last_updated": "2026-06-18T10:00:00+00:00",
                "metadata": {
                    "fetch_date": "2026-06-18T10:00:00+00:00",
                    "fetch_status": "success",
                },
                "citation_details": [
                    {"title": "A", "doi": "10.1/a"},
                    {"title": "B", "doi": "10.1/b"},
                ],
            }
            self._run_with_stub(out_dir, "ds000004", fetched)
            on_disk = json.loads(path.read_text())
            # Citations changed -> stale scores dropped so the score step re-scores.
            self.assertNotIn("confidence_scoring", on_disk)

    def test_fresh_stable_dataset_is_skipped(self) -> None:
        from datetime import datetime

        from dataset_citations.cli import update as cli_module

        with tempfile.TemporaryDirectory() as out_dir:
            existing = {
                "dataset_id": "ds000005",
                "num_citations": 0,
                "date_last_updated": "2026-01-01T00:00:00+00:00",
                "metadata": {
                    "fetch_date": "2026-01-01T00:00:00+00:00",
                    "fetch_status": "no_doi_references",
                },
                "citation_details": [],
            }
            self._seed(out_dir, "ds000005", existing)
            # Mark it as checked just now in the fetch-state cache.
            (Path(out_dir) / ".fetch_state.json").write_text(
                json.dumps({"ds000005": datetime.now(UTC).isoformat()})
            )
            list_file = Path(out_dir) / "list.txt"
            list_file.write_text("ds000005\n")

            calls: list[str] = []

            def stub_pipeline(did, **_):
                calls.append(did)
                return existing

            original = cli_module.fetch_dataset_citations_via_opencite
            cli_module.fetch_dataset_citations_via_opencite = stub_pipeline  # type: ignore[assignment]
            try:
                cli_module.run_opencite_backend(
                    _make_args(list_file, out_dir, max_age_days=7)
                )
            finally:
                cli_module.fetch_dataset_citations_via_opencite = original  # type: ignore[assignment]

            # Stable status + checked within window -> no fetch at all.
            self.assertEqual(calls, [])

    def test_fetch_state_written_to_disk_after_run(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            fetched = {
                "dataset_id": "ds000006",
                "num_citations": 0,
                "date_last_updated": "2026-06-18T10:00:00+00:00",
                "metadata": {
                    "fetch_date": "2026-06-18T10:00:00+00:00",
                    "fetch_status": "no_doi_references",
                },
                "citation_details": [],
            }
            self._run_with_stub(out_dir, "ds000006", fetched)
            state_path = Path(out_dir) / ".fetch_state.json"
            self.assertTrue(state_path.exists())
            self.assertIn("ds000006", json.loads(state_path.read_text()))

    def test_write_failure_does_not_update_fetch_state(self) -> None:
        import os
        import stat

        from dataset_citations.cli import update as cli_module

        with tempfile.TemporaryDirectory() as out_dir:
            list_file = Path(out_dir) / "list.txt"
            list_file.write_text("ds000007\n")
            # Pre-create json_opencite read-only so the write fails (OSError).
            json_dir = Path(out_dir) / "json_opencite"
            json_dir.mkdir()
            os.chmod(json_dir, stat.S_IRUSR | stat.S_IXUSR)

            fetched = {
                "dataset_id": "ds000007",
                "num_citations": 1,
                "date_last_updated": "2026-06-18T10:00:00+00:00",
                "metadata": {
                    "fetch_date": "2026-06-18T10:00:00+00:00",
                    "fetch_status": "success",
                },
                "citation_details": [{"title": "A", "doi": "10.1/a"}],
            }

            def stub_pipeline(_did, **_):
                return fetched

            original = cli_module.fetch_dataset_citations_via_opencite
            cli_module.fetch_dataset_citations_via_opencite = stub_pipeline  # type: ignore[assignment]
            try:
                # Single dataset, write fails -> exit 2; state saved before exit.
                with self.assertRaises(SystemExit):
                    cli_module.run_opencite_backend(_make_args(list_file, out_dir))
            finally:
                cli_module.fetch_dataset_citations_via_opencite = original  # type: ignore[assignment]
                os.chmod(json_dir, stat.S_IRWXU)
            state_path = Path(out_dir) / ".fetch_state.json"
            on_disk = json.loads(state_path.read_text()) if state_path.exists() else {}
            self.assertNotIn("ds000007", on_disk)

    def test_transient_failure_forces_refetch_despite_fresh_state(self) -> None:
        from datetime import datetime

        from dataset_citations.cli import update as cli_module

        with tempfile.TemporaryDirectory() as out_dir:
            existing = {
                "dataset_id": "ds000008",
                "num_citations": 0,
                "date_last_updated": "2026-06-18T10:00:00+00:00",
                "metadata": {
                    "fetch_date": "2026-06-18T10:00:00+00:00",
                    # Transient failure: must NOT be skipped even if checked recently.
                    "fetch_status": "rate_limit",
                },
                "citation_details": [],
            }
            self._seed(out_dir, "ds000008", existing)
            (Path(out_dir) / ".fetch_state.json").write_text(
                json.dumps({"ds000008": datetime.now(UTC).isoformat()})
            )
            list_file = Path(out_dir) / "list.txt"
            list_file.write_text("ds000008\n")

            calls: list[str] = []

            def stub_pipeline(did, **_):
                calls.append(did)
                return existing

            original = cli_module.fetch_dataset_citations_via_opencite
            cli_module.fetch_dataset_citations_via_opencite = stub_pipeline  # type: ignore[assignment]
            try:
                # Single all-rate-limit dataset is a degraded run -> exit 3.
                # The fetch still happened (loop runs before the exit check),
                # and exit-3 firing despite unchanged=1 confirms the unchanged
                # path still flows through the api_failure accounting.
                with self.assertRaises(SystemExit) as ctx:
                    cli_module.run_opencite_backend(
                        _make_args(list_file, out_dir, max_age_days=7)
                    )
                self.assertEqual(ctx.exception.code, 3)
            finally:
                cli_module.fetch_dataset_citations_via_opencite = original  # type: ignore[assignment]

            # rate_limit is transient -> refetched despite a fresh cache entry.
            self.assertEqual(calls, ["ds000008"])


class CatalogDoiWiringTests(TestCase):
    """run_opencite_backend forwards catalog_doi + use_checkpoint=True to
    the pipeline. Substitutes the pipeline call to record arguments."""

    def test_catalog_doi_and_checkpoint_forwarded(self) -> None:
        from dataset_citations.cli import update as cli_module

        with tempfile.TemporaryDirectory() as out_dir:
            list_file = Path(out_dir) / "list.txt"
            list_file.write_text("nm000104\nnm000999\n")

            captured: list[dict] = []

            def stub_pipeline(dataset_id, **kwargs):
                captured.append({"dataset_id": dataset_id, **kwargs})
                return {
                    "dataset_id": dataset_id,
                    "num_citations": 0,
                    "date_last_updated": "2026-01-01T00:00:00",
                    "metadata": {
                        "schema_version": "2.0",
                        "discovery_backend": "opencite",
                        "fetch_status": "no_doi_references",
                    },
                    "citation_details": [],
                }

            original = cli_module.fetch_dataset_citations_via_opencite
            cli_module.fetch_dataset_citations_via_opencite = stub_pipeline  # type: ignore[assignment]
            args = _make_args(
                list_file,
                out_dir,
                catalog_doi_map={"nm000104": "10.82901/nemar.nm000104"},
            )
            try:
                cli_update.run_opencite_backend(args)
            finally:
                cli_module.fetch_dataset_citations_via_opencite = original  # type: ignore[assignment]

            self.assertEqual(len(captured), 2)
            # nm000104 was in the catalog -> catalog_doi forwarded.
            self.assertEqual(captured[0]["catalog_doi"], "10.82901/nemar.nm000104")
            # nm000999 was NOT in the catalog -> catalog_doi=None.
            self.assertIsNone(captured[1]["catalog_doi"])
            # Both calls opt in to checkpoint resume.
            self.assertTrue(captured[0]["use_checkpoint"])
            self.assertTrue(captured[1]["use_checkpoint"])


class GithubTokenPassthroughTests(TestCase):
    """The update stage must authenticate its GitHub fallback (issue #201).

    nm-*/on-* DOIs normally come from data.nemar.org; when that 5xxs the
    pipeline falls back to GitHub. Unauthenticated that is 60 req/hr and a
    single 403 makes PyGithub sleep ~56 minutes, which stalled the 2026-08-01
    run for hours. These assert the token actually reaches the pipeline.
    """

    def _capture_token(self, monkey_env: dict[str, str], **arg_kw) -> list:
        """Run the CLI against a recording substitute and return seen tokens."""
        seen: list = []

        def _recording_pipeline(dataset_id, **kwargs):
            seen.append(kwargs.get("github_token"))
            return {
                "dataset_id": dataset_id,
                "num_citations": 0,
                "date_last_updated": "2026-01-01T00:00:00+00:00",
                "metadata": {"fetch_status": "no_doi_references"},
                "citation_details": [],
            }

        from dataset_citations.cli import update as cli_module

        original = cli_module.fetch_dataset_citations_via_opencite
        old_env = os.environ.get("GITHUB_TOKEN")
        cli_module.fetch_dataset_citations_via_opencite = _recording_pipeline  # type: ignore[assignment]
        try:
            for k, v in monkey_env.items():
                os.environ[k] = v
            if "GITHUB_TOKEN" not in monkey_env:
                os.environ.pop("GITHUB_TOKEN", None)
            with tempfile.TemporaryDirectory() as out_dir:
                list_file = Path(out_dir) / "ids.txt"
                list_file.write_text("ds000117\n")
                cli_module.run_opencite_backend(
                    _make_args(list_file, out_dir, **arg_kw)
                )
        finally:
            cli_module.fetch_dataset_citations_via_opencite = original  # type: ignore[assignment]
            if old_env is None:
                os.environ.pop("GITHUB_TOKEN", None)
            else:
                os.environ["GITHUB_TOKEN"] = old_env
        return seen

    def test_env_token_reaches_the_pipeline(self) -> None:
        seen = self._capture_token({"GITHUB_TOKEN": "env-token-value"})
        self.assertEqual(seen, ["env-token-value"])

    def test_explicit_flag_wins_over_env(self) -> None:
        seen = self._capture_token(
            {"GITHUB_TOKEN": "env-token-value"},
            github_token="flag-token-value",  # noqa: S106 - test fixture, not a secret
        )
        self.assertEqual(seen, ["flag-token-value"])

    def test_absent_token_passes_none(self) -> None:
        seen = self._capture_token({})
        self.assertEqual(seen, [None])
