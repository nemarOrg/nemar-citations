"""Tests for `dataset-citations-judge-anchors`.

Uses real subclasses for the LLM client + opencite backend (matching the
no-mocks rule). The pipeline-level orchestration is substituted via setattr
so the CLI path can be exercised without the network.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import TestCase

from dataset_citations.cli import judge_anchors as cli_judge


def _write_list(tmp: Path, ids: list[str]) -> Path:
    path = tmp / "datasets.txt"
    path.write_text("\n".join(ids) + "\n")
    return path


def _seed_sidecar(
    output_dir: Path,
    dataset_id: str,
    *,
    judged_at: str,
    judgments: list | None = None,
) -> Path:
    """Write a minimal sidecar to disk and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{dataset_id}.json"
    path.write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "judged_at": judged_at,
                "judgment_model": "preexisting",
                "judgments": judgments or [],
            }
        )
    )
    return path


class CliHelpAndParser(TestCase):
    def test_help_text_documents_required_flags(self) -> None:
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        old_argv = sys.argv
        sys.argv = ["dataset-citations-judge-anchors", "--help"]
        try:
            with self.assertRaises(SystemExit), redirect_stdout(buf):
                cli_judge.main()
        finally:
            sys.argv = old_argv
        help_text = buf.getvalue()
        self.assertIn("--dataset-list-file", help_text)
        self.assertIn("--output-dir", help_text)
        self.assertIn("--skip-existing", help_text)
        self.assertIn("--max-age-days", help_text)
        self.assertIn("--ollama-base-url", help_text)
        self.assertIn("--ollama-model", help_text)

    def test_empty_dataset_list_exits_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            list_file = Path(tmp) / "empty.txt"
            list_file.write_text("")
            old_argv = sys.argv
            sys.argv = [
                "dataset-citations-judge-anchors",
                "--dataset-list-file",
                str(list_file),
                "--output-dir",
                str(Path(tmp) / "out"),
            ]
            try:
                rc = cli_judge.main()
            finally:
                sys.argv = old_argv
            self.assertEqual(rc, 1)


class CliOllamaPreflight(TestCase):
    """A down Ollama daemon must abort with exit 2 before any sidecar is touched."""

    def test_ollama_unreachable_returns_two(self) -> None:
        class _DeadClient:
            base_url = "http://dead-host:11434"
            model = "dead-model"

            def __init__(self, base_url=None, model=None):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def health_check(self):
                return False

        original = cli_judge.OllamaJudgmentClient
        cli_judge.OllamaJudgmentClient = _DeadClient  # type: ignore[assignment]
        with tempfile.TemporaryDirectory() as tmp:
            list_file = _write_list(Path(tmp), ["nm000104"])
            output_dir = Path(tmp) / "out"
            old_argv = sys.argv
            sys.argv = [
                "dataset-citations-judge-anchors",
                "--dataset-list-file",
                str(list_file),
                "--output-dir",
                str(output_dir),
            ]
            try:
                rc = cli_judge.main()
            finally:
                cli_judge.OllamaJudgmentClient = original  # type: ignore[assignment]
                sys.argv = old_argv
            self.assertEqual(rc, 2)
            # No sidecar should have been written.
            self.assertFalse(any(output_dir.rglob("*.json")))


class _UpClient:
    """Test double for OllamaJudgmentClient with a healthy daemon."""

    base_url = "http://test"
    model = "test-model"

    def __init__(self, base_url=None, model=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def health_check(self):
        return True


class CliSkipExistingAndMaxAge(TestCase):
    """Verify the skip-existing and max-age-days gating."""

    def _run_with_substitutes(
        self,
        *,
        list_file: Path,
        output_dir: Path,
        cli_args: list[str],
        judge_recorder: list[str],
        judge_payload_factory,
    ) -> int:
        """Run cli_judge.main with substituted client + judge function.

        `judge_payload_factory(dataset_id)` returns the payload the CLI
        will save. `judge_recorder` is appended with each dataset_id the
        substituted function sees.
        """

        def fake_judge(dataset_id, **_):
            judge_recorder.append(dataset_id)
            return judge_payload_factory(dataset_id)

        original_client = cli_judge.OllamaJudgmentClient
        original_judge = cli_judge.judge_dataset_anchors
        cli_judge.OllamaJudgmentClient = _UpClient  # type: ignore[assignment]
        cli_judge.judge_dataset_anchors = fake_judge  # type: ignore[assignment]
        old_argv = sys.argv
        sys.argv = [
            "dataset-citations-judge-anchors",
            "--dataset-list-file",
            str(list_file),
            "--output-dir",
            str(output_dir),
            *cli_args,
        ]
        try:
            return cli_judge.main()
        finally:
            cli_judge.OllamaJudgmentClient = original_client  # type: ignore[assignment]
            cli_judge.judge_dataset_anchors = original_judge  # type: ignore[assignment]
            sys.argv = old_argv

    def test_skip_existing_skips_when_sidecar_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "out"
            # Pre-create one sidecar; that dataset must be skipped.
            _seed_sidecar(
                output_dir,
                "nm000104",
                judged_at=datetime.now(UTC).isoformat(),
            )
            list_file = _write_list(tmp_path, ["nm000104", "nm000999"])

            recorded: list[str] = []
            rc = self._run_with_substitutes(
                list_file=list_file,
                output_dir=output_dir,
                cli_args=["--skip-existing"],
                judge_recorder=recorded,
                judge_payload_factory=lambda did: {
                    "dataset_id": did,
                    "judged_at": datetime.now(UTC).isoformat(),
                    "judgment_model": "test-model",
                    "judgments": [],
                },
            )
            self.assertEqual(rc, 0)
            # Only the non-existing dataset was judged.
            self.assertEqual(recorded, ["nm000999"])
            # Pre-existing sidecar is untouched.
            payload = json.loads((output_dir / "nm000104.json").read_text())
            self.assertEqual(payload["judgment_model"], "preexisting")

    def test_max_age_days_re_runs_stale_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "out"
            # Stale: judged 30 days ago. Fresh threshold is 7 days.
            stale = datetime.now(UTC) - timedelta(days=30)
            _seed_sidecar(output_dir, "nm000104", judged_at=stale.isoformat())
            list_file = _write_list(tmp_path, ["nm000104"])

            recorded: list[str] = []
            rc = self._run_with_substitutes(
                list_file=list_file,
                output_dir=output_dir,
                cli_args=["--max-age-days", "7"],
                judge_recorder=recorded,
                judge_payload_factory=lambda did: {
                    "dataset_id": did,
                    "judged_at": datetime.now(UTC).isoformat(),
                    "judgment_model": "test-model",
                    "judgments": [],
                },
            )
            self.assertEqual(rc, 0)
            # Stale sidecar triggered a re-run.
            self.assertEqual(recorded, ["nm000104"])
            payload = json.loads((output_dir / "nm000104.json").read_text())
            self.assertEqual(payload["judgment_model"], "test-model")

    def test_max_age_days_skips_fresh_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "out"
            fresh = datetime.now(UTC) - timedelta(days=2)
            _seed_sidecar(output_dir, "nm000104", judged_at=fresh.isoformat())
            list_file = _write_list(tmp_path, ["nm000104"])

            recorded: list[str] = []
            rc = self._run_with_substitutes(
                list_file=list_file,
                output_dir=output_dir,
                cli_args=["--max-age-days", "7"],
                judge_recorder=recorded,
                judge_payload_factory=lambda did: {
                    "dataset_id": did,
                    "judged_at": datetime.now(UTC).isoformat(),
                    "judgment_model": "test-model",
                    "judgments": [],
                },
            )
            self.assertEqual(rc, 0)
            # Fresh sidecar means we never invoked the judge function.
            self.assertEqual(recorded, [])


class CliWritesJudgmentRecords(TestCase):
    """End-to-end CLI path: substitute the judge function with a real one
    that returns hand-built records (including one with `error`), confirm
    the sidecar on disk matches."""

    def test_judgment_with_error_field_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "out"
            list_file = _write_list(tmp_path, ["nm000104"])

            def fake_judge(dataset_id, **_):
                return {
                    "dataset_id": dataset_id,
                    "judged_at": datetime.now(UTC).isoformat(),
                    "judgment_model": "test-model",
                    "judgments": [
                        {
                            "anchor_identifier": "10.1234/ok",
                            "anchor_identifier_type": "doi",
                            "source_relation": "References",
                            "classification": "data_paper",
                            "reason": "matches",
                            "paper_title": "A paper",
                            "paper_year": 2020,
                            "paper_venue": "Venue",
                            "judged_at": datetime.now(UTC).isoformat(),
                            "error": None,
                        },
                        {
                            "anchor_identifier": "10.1234/bad",
                            "anchor_identifier_type": "doi",
                            "source_relation": "IsDerivedFrom",
                            "classification": "",
                            "reason": "",
                            "paper_title": None,
                            "paper_year": None,
                            "paper_venue": None,
                            "judged_at": datetime.now(UTC).isoformat(),
                            "error": "paper_lookup_failed:not_found:404",
                        },
                    ],
                }

            # All rebinds + sys.argv mutation go inside the try block so a
            # failure anywhere after the first assignment still triggers
            # the restore in finally. Otherwise a single bad chmod/assign
            # leaks substitutes across tests.
            original_client = cli_judge.OllamaJudgmentClient
            original_judge = cli_judge.judge_dataset_anchors
            old_argv = sys.argv
            try:
                cli_judge.OllamaJudgmentClient = _UpClient  # type: ignore[assignment]
                cli_judge.judge_dataset_anchors = fake_judge  # type: ignore[assignment]
                sys.argv = [
                    "dataset-citations-judge-anchors",
                    "--dataset-list-file",
                    str(list_file),
                    "--output-dir",
                    str(output_dir),
                ]
                rc = cli_judge.main()
            finally:
                cli_judge.OllamaJudgmentClient = original_client  # type: ignore[assignment]
                cli_judge.judge_dataset_anchors = original_judge  # type: ignore[assignment]
                sys.argv = old_argv
            self.assertEqual(rc, 0)
            payload = json.loads((output_dir / "nm000104.json").read_text())
            self.assertEqual(payload["dataset_id"], "nm000104")
            self.assertEqual(len(payload["judgments"]), 2)
            self.assertIsNone(payload["judgments"][0]["error"])
            self.assertIn("paper_lookup_failed", payload["judgments"][1]["error"])

    def test_write_failure_returns_two_when_zero_judged(self) -> None:
        """A read-only output dir makes every save fail; with zero successful
        writes the CLI exits 2 (matches `cli/update.py` semantics)."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "out"
            output_dir.mkdir()
            list_file = _write_list(tmp_path, ["nm000104"])

            def fake_judge(dataset_id, **_):
                return {
                    "dataset_id": dataset_id,
                    "judged_at": datetime.now(UTC).isoformat(),
                    "judgment_model": "test-model",
                    "judgments": [],
                }

            # All mutation (rebinds, sys.argv, chmod) goes inside try so a
            # failing chmod can't leak the rebinds to subsequent tests.
            original_client = cli_judge.OllamaJudgmentClient
            original_judge = cli_judge.judge_dataset_anchors
            old_argv = sys.argv
            try:
                cli_judge.OllamaJudgmentClient = _UpClient  # type: ignore[assignment]
                cli_judge.judge_dataset_anchors = fake_judge  # type: ignore[assignment]
                sys.argv = [
                    "dataset-citations-judge-anchors",
                    "--dataset-list-file",
                    str(list_file),
                    "--output-dir",
                    str(output_dir),
                ]
                os.chmod(output_dir, stat.S_IRUSR | stat.S_IXUSR)
                rc = cli_judge.main()
            finally:
                # Restore write so tempfile cleanup works regardless of result.
                os.chmod(output_dir, stat.S_IRWXU)
                cli_judge.OllamaJudgmentClient = original_client  # type: ignore[assignment]
                cli_judge.judge_dataset_anchors = original_judge  # type: ignore[assignment]
                sys.argv = old_argv
            self.assertEqual(rc, 2)


class SkipExistingRespectsAnchorCoverage(TestCase):
    """`--skip-existing` must not freeze a sidecar at its first anchor set.

    Regression for #180. Skipping on mere file existence meant an anchor added
    after the sidecar was first written never got judged; it fell through the
    pipeline's "fetch all when unjudged" path and pulled in the citers of
    BIDS/methods papers. Measured on the committed corpus, that left 851 of
    1,642 recorded anchors (52%) with a null classification even though every
    dataset had a sidecar.
    """

    def _fixture(self, tmp: Path, judged: list[str], recorded: list[str]) -> Path:
        sidecar_dir = tmp / "anchor_judgments"
        sidecar_dir.mkdir(parents=True)
        (sidecar_dir / "on000001.json").write_text(
            json.dumps(
                {
                    "dataset_id": "on000001",
                    "judged_at": "2026-09-01T00:00:00+00:00",
                    "judgment_model": "gemma4:e4b",
                    "judgments": [
                        {"anchor_identifier": a, "classification": "data_paper"}
                        for a in judged
                    ],
                }
            ),
            encoding="utf-8",
        )
        citations_dir = tmp / "json_opencite"
        citations_dir.mkdir(parents=True)
        (citations_dir / "on000001_citations.json").write_text(
            json.dumps(
                {
                    "dataset_id": "on000001",
                    "metadata": {"anchors": [{"identifier": a} for a in recorded]},
                    "citation_details": [],
                }
            ),
            encoding="utf-8",
        )
        return sidecar_dir / "on000001.json"

    def test_uncovered_anchor_forces_a_rejudge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sidecar = self._fixture(
                root,
                judged=["10.1038/sdata.2015.1"],
                recorded=["10.1038/sdata.2015.1", "10.21105/joss.01896"],
            )
            skip, _ = cli_judge._should_skip(
                sidecar,
                skip_existing=True,
                max_age_days=0,
                citations_dir=str(root / "json_opencite"),
                dataset_id="on000001",
            )
            self.assertFalse(skip)

    def test_full_coverage_still_skips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sidecar = self._fixture(
                root,
                judged=["10.1038/sdata.2015.1", "10.21105/joss.01896"],
                recorded=["10.1038/sdata.2015.1", "10.21105/joss.01896"],
            )
            skip, reason = cli_judge._should_skip(
                sidecar,
                skip_existing=True,
                max_age_days=0,
                citations_dir=str(root / "json_opencite"),
                dataset_id="on000001",
            )
            self.assertTrue(skip)
            self.assertEqual(reason, "exists")

    def test_coverage_match_is_case_insensitive(self) -> None:
        """DOIs are case-insensitive; a case difference is not a missing anchor."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sidecar = self._fixture(
                root,
                judged=["10.1038/SData.2015.1"],
                recorded=["10.1038/sdata.2015.1"],
            )
            skip, _ = cli_judge._should_skip(
                sidecar,
                skip_existing=True,
                max_age_days=0,
                citations_dir=str(root / "json_opencite"),
                dataset_id="on000001",
            )
            self.assertTrue(skip)

    def test_missing_citation_json_falls_back_to_file_exists(self) -> None:
        """No citation JSON means nothing to compare; keep the old behavior."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sidecar = self._fixture(root, judged=["10.1/a"], recorded=["10.1/a"])
            skip, _ = cli_judge._should_skip(
                sidecar,
                skip_existing=True,
                max_age_days=0,
                citations_dir=str(root / "does-not-exist"),
                dataset_id="on000001",
            )
            self.assertTrue(skip)
