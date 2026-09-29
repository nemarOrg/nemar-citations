"""Tests for `dataset-citations-judge-anchors`.

No mocks. `run()` drives the real per-dataset loop, the real
`judge_dataset_anchors`, the skip logic, and the exit policy; only the network
boundary is replaced, by real subclasses of the client, the anchor source, the
metadata retriever, and the opencite backend. The judge answers in the real
`claude -p --output-format json` shape recorded on hallu.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import stat
import tempfile
import threading
from contextlib import redirect_stdout
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import TestCase

from dataset_citations.backends import OpenCiteBackend
from dataset_citations.cli import judge_anchors as cli_judge
from dataset_citations.quality.dataset_metadata import DatasetMetadataRetriever
from dataset_citations.quality.llm_client import (
    PROMPT_VERSION,
    ClaudeCliJudgmentClient,
    LlmJudgmentError,
)
from dataset_citations.sources import BidsMetadataSource, NemarMetadataSource
from dataset_citations.sources.models import (
    Author,
    CitingWork,
    DoiReference,
    FetchError,
    FetchSuccess,
)

_MODEL = "claude-sonnet-5-5"
_RECORDED_OK = json.loads(
    (Path(__file__).parent / "test_data" / "claude_cli_judgment_ok.json").read_text(
        "utf-8"
    )
)


def _cli_output(classification: str, model: str) -> str:
    """The recorded CLI success output carrying `classification`, served by `model`."""
    payload = dict(_RECORDED_OK)
    payload["structured_output"] = {"classification": classification, "reason": "ok"}
    payload["modelUsage"] = {model: next(iter(_RECORDED_OK["modelUsage"].values()))}
    return json.dumps(payload)


class _ScriptedClient(ClaudeCliJudgmentClient):
    """Real client whose process step answers from a script.

    Every judgment returns `default`, except anchors whose DOI is in
    `fail_dois` (or every anchor when `fail_all`), which fail the way a
    non-zero CLI exit does. The health check always passes.
    """

    def __init__(
        self,
        *,
        default: str = "data_paper",
        fail_dois: set[str] | None = None,
        fail_all: bool = False,
        model: str = _MODEL,
    ) -> None:
        super().__init__(model=model, claude_bin="unused", timeout=5)
        self._default = default
        self._fail_dois = fail_dois or set()
        self._fail_all = fail_all
        self._lock = threading.Lock()
        self.calls = 0

    def _run_cli(self, prompt: str) -> str:
        if "healthcheck" not in prompt:
            with self._lock:
                self.calls += 1
            if self._fail_all or any(f"DOI: {d}" in prompt for d in self._fail_dois):
                raise LlmJudgmentError("claude CLI exited 1: stand-in failure")
        return _cli_output(self._default, self.model)


class _MapSource(NemarMetadataSource):
    """Anchor source answering from a dataset id -> DOI list (or FetchError) map."""

    def __init__(self, anchors: dict[str, list[str] | FetchError]) -> None:
        super().__init__(prefer_data_api=False)
        self._anchors = anchors
        self.calls: list[str] = []

    def get_doi_references(self, dataset_id: str):  # type: ignore[override]
        self.calls.append(dataset_id)
        outcome = self._anchors[dataset_id]
        if isinstance(outcome, FetchError):
            return outcome
        return FetchSuccess(
            [
                DoiReference(
                    identifier=doi,
                    identifier_type="doi",
                    relation_type="IsDescribedBy",
                    source="nemar_metadata",
                )
                for doi in outcome
            ]
        )


class _Retriever(DatasetMetadataRetriever):
    """Retriever returning a canned description, or failing the test if called."""

    def __init__(self, *, forbidden: bool = False) -> None:
        super().__init__()
        self._forbidden = forbidden

    def get_dataset_metadata(self, dataset_id: str) -> dict:
        if self._forbidden:
            raise AssertionError("the cached metadata should have been used")
        return {
            "dataset_id": dataset_id,
            "dataset_description": {"Name": "EEG during a driving task"},
            "readme_content": None,
            "github_info": {"description": None},
        }


class _Papers(OpenCiteBackend):
    """opencite backend answering `get_paper` from a DOI -> FetchError map.

    Any DOI not in `errors` resolves to a small paper record.
    """

    def __init__(self, errors: dict[str, FetchError] | None = None) -> None:
        # Skip parent init so we don't read OPENCITE config.
        self._errors = errors or {}
        self._config = None  # type: ignore[assignment]
        self._max_results_per_doi = 1
        self._concurrency = 1

    def get_paper(self, doi: str):  # type: ignore[override]
        if doi in self._errors:
            return self._errors[doi]
        return FetchSuccess(
            CitingWork(
                title=f"Paper {doi}",
                doi=doi,
                pmid=None,
                openalex_id=None,
                year=2019,
                authors=(Author(name="A. Researcher"),),
                venue="Scientific Data",
                abstract="A data descriptor.",
                citation_count=10,
                source_doi=doi,
                source_relation="IsDescribedBy",
            )
        )


def _dois(n: int, prefix: str = "10.1/a") -> list[str]:
    return [f"{prefix}{i}" for i in range(n)]


class _Harness:
    """Temp dirs, a dataset list, and a `run()` call with the given doubles."""

    def __init__(self, root: Path, ids: list[str]) -> None:
        self.root = root
        self.output_dir = root / "anchor_judgments"
        self.citations_dir = root / "json_opencite"
        self.datasets_dir = root / "datasets"
        self.list_file = root / "datasets.txt"
        self.list_file.write_text("\n".join(ids) + "\n", encoding="utf-8")
        self.source = _MapSource({})

    def args(self, *extra: str) -> argparse.Namespace:
        return cli_judge.build_parser().parse_args(
            [
                "--dataset-list-file",
                str(self.list_file),
                "--output-dir",
                str(self.output_dir),
                "--citations-dir",
                str(self.citations_dir),
                "--datasets-dir",
                str(self.datasets_dir),
                "--workers",
                "1",
                *extra,
            ]
        )

    def run(
        self,
        anchors: dict[str, list[str] | FetchError],
        *,
        client: ClaudeCliJudgmentClient | None = None,
        backend: OpenCiteBackend | None = None,
        retriever: DatasetMetadataRetriever | None = None,
        extra: tuple[str, ...] = (),
    ) -> int:
        source = _MapSource(anchors)
        self.source = source
        return cli_judge.run(
            self.args(*extra),
            client=client or _ScriptedClient(),
            nemar_source=source,
            bids_source=BidsMetadataSource(),
            metadata_retriever=retriever or _Retriever(),
            backend=backend or _Papers(),
        )

    def sidecar(self, dataset_id: str) -> Path:
        return self.output_dir / f"{dataset_id}.json"


def _payload(
    *,
    judgments: list[dict] | None = None,
    model: str = _MODEL,
    prompt_version: int | None = PROMPT_VERSION,
    judged_at: str | None = None,
) -> dict:
    payload = {
        "dataset_id": "on000001",
        "judged_at": judged_at or datetime.now(UTC).isoformat(),
        "judgment_model": model,
        "judgments": judgments or [],
    }
    if prompt_version is not None:
        payload["prompt_version"] = prompt_version
    return payload


def _judgment(
    doi: str,
    *,
    relation: str = "IsDescribedBy",
    error: str | None = None,
    judged_at: str | None = None,
) -> dict:
    return {
        "anchor_identifier": doi,
        "anchor_identifier_type": "doi",
        "source_relation": relation,
        "classification": "" if error else "data_paper",
        "reason": "" if error else "ok",
        "judged_at": judged_at or datetime.now(UTC).isoformat(),
        "error": error,
    }


class CliParserAndMain(TestCase):
    def test_help_text_documents_the_flags(self) -> None:
        buf = io.StringIO()
        with self.assertRaises(SystemExit), redirect_stdout(buf):
            cli_judge.main(["--help"])
        help_text = buf.getvalue()
        for flag in (
            "--dataset-list-file",
            "--output-dir",
            "--skip-existing",
            "--max-age-days",
            "--model",
            "--claude-bin",
            "--workers",
            "--citations-dir",
            "--datasets-dir",
        ):
            self.assertIn(flag, help_text)
        self.assertNotIn("ollama", help_text.lower())

    def test_empty_dataset_list_exits_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            list_file = Path(tmp) / "empty.txt"
            list_file.write_text("")
            with self.assertLogs("dataset_citations", "ERROR"):
                rc = cli_judge.main(
                    [
                        "--dataset-list-file",
                        str(list_file),
                        "--output-dir",
                        str(Path(tmp) / "out"),
                    ]
                )
            self.assertEqual(rc, 1)

    def test_unusable_judge_returns_two_before_any_sidecar(self) -> None:
        """The real client pointed at a binary that does not exist, and at one
        that exists but cannot be executed (a 0644 file)."""
        with tempfile.TemporaryDirectory() as tmp:
            list_file = Path(tmp) / "ids.txt"
            list_file.write_text("nm000104\n")
            output_dir = Path(tmp) / "out"
            not_executable = Path(tmp) / "claude"
            not_executable.write_text("#!/bin/sh\necho {}\n")
            not_executable.chmod(0o644)
            for binary in (Path(tmp) / "no-such-claude", not_executable):
                with self.subTest(binary.name):
                    with self.assertLogs("dataset_citations", "ERROR"):
                        rc = cli_judge.main(
                            [
                                "--dataset-list-file",
                                str(list_file),
                                "--output-dir",
                                str(output_dir),
                                "--claude-bin",
                                str(binary),
                            ]
                        )
                    self.assertEqual(rc, 2)
                    self.assertFalse(any(output_dir.rglob("*.json")))


class CliRunWritesSidecars(TestCase):
    def test_judged_and_unresolvable_anchors_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            h = _Harness(Path(tmp), ["nm000275"])
            rc = h.run(
                {"nm000275": ["10.1/ok", "10.38119/typo"]},
                backend=_Papers({"10.38119/typo": FetchError("not_found", "404")}),
            )
            self.assertEqual(rc, 0)
            payload = json.loads(h.sidecar("nm000275").read_text())
            self.assertEqual(payload["judgment_model"], _MODEL)
            self.assertEqual(payload["prompt_version"], PROMPT_VERSION)
            by_doi = {j["anchor_identifier"]: j for j in payload["judgments"]}
            self.assertEqual(by_doi["10.1/ok"]["classification"], "data_paper")
            self.assertIsNone(by_doi["10.1/ok"]["error"])
            self.assertTrue(
                by_doi["10.38119/typo"]["error"].startswith(
                    cli_judge.PERMANENT_LOOKUP_ERROR_PREFIX
                )
            )

    def test_cached_dataset_metadata_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            h = _Harness(Path(tmp), ["nm000275"])
            h.datasets_dir.mkdir()
            (h.datasets_dir / "nm000275_datasets.json").write_text(
                json.dumps(
                    {
                        "dataset_id": "nm000275",
                        "dataset_description": {"Name": "Driving EEG"},
                        "readme_content": None,
                    }
                )
            )
            rc = h.run({"nm000275": ["10.1/ok"]}, retriever=_Retriever(forbidden=True))
            self.assertEqual(rc, 0)
            self.assertTrue(h.sidecar("nm000275").exists())

    def test_unchanged_sidecar_is_not_rewritten(self) -> None:
        """A second night with nothing new reuses every verdict and writes nothing."""
        with tempfile.TemporaryDirectory() as tmp:
            h = _Harness(Path(tmp), ["nm000275"])
            anchors: dict[str, list[str] | FetchError] = {"nm000275": ["10.1/ok"]}
            self.assertEqual(h.run(anchors), 0)
            before = h.sidecar("nm000275").stat().st_mtime_ns
            os.utime(h.sidecar("nm000275"), ns=(before - 10**9, before - 10**9))
            client = _ScriptedClient()
            self.assertEqual(h.run(anchors, client=client), 0)
            self.assertEqual(client.calls, 0)
            self.assertEqual(h.sidecar("nm000275").stat().st_mtime_ns, before - 10**9)

    def test_source_failure_leaves_the_sidecar_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            h = _Harness(Path(tmp), ["nm000275"])
            h.output_dir.mkdir()
            h.sidecar("nm000275").write_text(json.dumps(_payload(model="gemma4:e4b")))
            before = h.sidecar("nm000275").read_text()
            rc = h.run({"nm000275": FetchError("rate_limit", "403")})
            self.assertEqual(rc, 0)
            self.assertEqual(h.source.calls, ["nm000275"])
            self.assertEqual(h.sidecar("nm000275").read_text(), before)

    def test_write_failure_exits_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            h = _Harness(Path(tmp), ["nm000275"])
            h.output_dir.mkdir()
            try:
                os.chmod(h.output_dir, stat.S_IRUSR | stat.S_IXUSR)
                with self.assertLogs("dataset_citations.cli.judge_anchors", "ERROR"):
                    rc = h.run({"nm000275": ["10.1/ok"]})
            finally:
                os.chmod(h.output_dir, stat.S_IRWXU)
            self.assertEqual(rc, 2)


class CliExitPolicy(TestCase):
    """Exit 2 when a dependency fails on at least 5 calls AND over 10% of them."""

    def _rc(self, anchors: dict, **kwargs) -> int:
        with tempfile.TemporaryDirectory() as tmp:
            return _Harness(Path(tmp), list(anchors)).run(anchors, **kwargs)

    def test_a_few_failed_judge_calls_do_not_stall_the_pipeline(self) -> None:
        dois = _dois(4)
        client = _ScriptedClient(fail_dois=set(dois))
        self.assertEqual(self._rc({"nm000001": dois}, client=client), 0)

    def test_judge_failures_over_ten_percent_exit_two(self) -> None:
        # 6 failed of 56 calls (10.7%).
        bad = _dois(6, "10.9/bad")
        client = _ScriptedClient(fail_dois=set(bad))
        with self.assertLogs("dataset_citations", "ERROR"):
            rc = self._rc({"nm000001": _dois(50) + bad}, client=client)
        self.assertEqual(rc, 2)

    def test_judge_failures_under_ten_percent_pass(self) -> None:
        # 6 failed of 66 calls (9.1%).
        bad = _dois(6, "10.9/bad")
        client = _ScriptedClient(fail_dois=set(bad))
        self.assertEqual(self._rc({"nm000001": _dois(60) + bad}, client=client), 0)

    def test_transient_lookup_failures_exit_two(self) -> None:
        bad = _dois(5, "10.9/bad")
        backend = _Papers({d: FetchError("network", "timeout") for d in bad})
        with self.assertLogs("dataset_citations", "ERROR"):
            rc = self._rc({"nm000001": _dois(5) + bad}, backend=backend)
        self.assertEqual(rc, 2)

    def test_unresolvable_anchors_do_not_count(self) -> None:
        bad = _dois(8, "10.9/typo")
        backend = _Papers({d: FetchError("not_found", "404") for d in bad})
        self.assertEqual(self._rc({"nm000001": bad}, backend=backend), 0)

    def test_widespread_source_failures_exit_two(self) -> None:
        anchors: dict[str, list[str] | FetchError] = {
            f"nm00000{i}": FetchError("rate_limit", "403") for i in range(5)
        }
        anchors["nm000009"] = ["10.1/ok"]
        with self.assertLogs("dataset_citations", "ERROR"):
            self.assertEqual(self._rc(anchors), 2)

    def test_circuit_breaker_stops_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ids = ["nm000001", "nm000002", "nm000003", "nm000004"]
            h = _Harness(Path(tmp), ids)
            anchors: dict[str, list[str] | FetchError] = {
                did: _dois(4, f"10.1/{did}-") for did in ids
            }
            with self.assertLogs("dataset_citations.cli.judge_anchors", "ERROR"):
                rc = h.run(anchors, client=_ScriptedClient(fail_all=True))
            self.assertEqual(rc, 2)
            # 4 + 4 failures, then the third dataset reaches 12 and trips it.
            self.assertTrue(h.sidecar("nm000002").exists())
            self.assertFalse(h.sidecar("nm000003").exists())
            self.assertEqual(h.source.calls, ["nm000001", "nm000002", "nm000003"])


class ShouldSkip(TestCase):
    """`_should_skip` decisions on a sidecar payload (#180, #241)."""

    def _skip(self, payload: dict | None, root: Path | None = None, **kw) -> bool:
        skip, _ = cli_judge._should_skip(
            payload,
            skip_existing=kw.pop("skip_existing", True),
            max_age_days=kw.pop("max_age_days", 0),
            model=_MODEL,
            citations_dir=str(root / "json_opencite") if root else None,
            dataset_id="on000001",
        )
        return skip

    def _record(self, root: Path, anchors: list[dict]) -> None:
        citations_dir = root / "json_opencite"
        citations_dir.mkdir(parents=True, exist_ok=True)
        (citations_dir / "on000001_citations.json").write_text(
            json.dumps({"dataset_id": "on000001", "metadata": {"anchors": anchors}})
        )

    def test_missing_sidecar_is_judged(self) -> None:
        self.assertFalse(self._skip(None))

    def test_other_model_or_prompt_is_rejudged(self) -> None:
        self.assertFalse(self._skip(_payload(model="gemma4:e4b")))
        self.assertFalse(self._skip(_payload(prompt_version=None)))
        self.assertFalse(self._skip(_payload(prompt_version=PROMPT_VERSION - 1)))
        self.assertTrue(self._skip(_payload()))

    def test_transient_error_is_retried(self) -> None:
        judgments = [_judgment("10.1/a", error="llm_judgment_failed:timeout")]
        self.assertFalse(self._skip(_payload(judgments=judgments)))

    def test_unresolvable_anchor_waits_for_its_back_off(self) -> None:
        error = f"{cli_judge.PERMANENT_LOOKUP_ERROR_PREFIX}404"
        recent = datetime.now(UTC) - timedelta(days=3)
        old = datetime.now(UTC) - timedelta(days=31)
        self.assertTrue(
            self._skip(
                _payload(
                    judgments=[
                        _judgment("10.1/a", error=error, judged_at=recent.isoformat())
                    ]
                )
            )
        )
        self.assertFalse(
            self._skip(
                _payload(
                    judgments=[
                        _judgment("10.1/a", error=error, judged_at=old.isoformat())
                    ]
                )
            )
        )

    def test_uncovered_anchor_forces_a_rejudge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._record(
                root,
                [
                    {"identifier": "10.1/a", "source_relation": "IsDescribedBy"},
                    {"identifier": "10.1/b", "source_relation": "IsDescribedBy"},
                ],
            )
            self.assertFalse(
                self._skip(_payload(judgments=[_judgment("10.1/a")]), root)
            )

    def test_full_coverage_skips_case_insensitively(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._record(
                root,
                [
                    {
                        "identifier": "10.1038/sdata.2015.1",
                        "source_relation": "IsDescribedBy",
                    }
                ],
            )
            payload = _payload(judgments=[_judgment("10.1038/SData.2015.1")])
            skip, reason = cli_judge._should_skip(
                payload,
                skip_existing=True,
                max_age_days=0,
                model=_MODEL,
                citations_dir=str(root / "json_opencite"),
                dataset_id="on000001",
            )
            self.assertTrue(skip)
            self.assertEqual(reason, "covered")

    def test_relabeled_anchor_is_rejudged(self) -> None:
        """An enrichment sweep relabeling References to IsDescribedBy."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._record(
                root, [{"identifier": "10.1/a", "source_relation": "IsDescribedBy"}]
            )
            payload = _payload(judgments=[_judgment("10.1/a", relation="References")])
            self.assertFalse(self._skip(payload, root))

    def test_own_doi_and_non_doi_anchors_need_no_judgment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._record(
                root,
                [
                    {"identifier": "10.1/a", "source_relation": "IsDescribedBy"},
                    {
                        "identifier": "10.82901/nemar.on000001",
                        "source_relation": "References",
                    },
                    {
                        "identifier": "12345678",
                        "identifier_type": "pmid",
                        "source_relation": "References",
                    },
                ],
            )
            self.assertTrue(self._skip(_payload(judgments=[_judgment("10.1/a")]), root))

    def test_missing_citation_json_skips_a_trusted_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(
                self._skip(_payload(judgments=[_judgment("10.1/a")]), Path(tmp))
            )

    def test_max_age_days(self) -> None:
        fresh = (datetime.now(UTC) - timedelta(days=2)).isoformat()
        stale = (datetime.now(UTC) - timedelta(days=30)).isoformat()
        self.assertTrue(
            self._skip(_payload(judged_at=fresh), skip_existing=False, max_age_days=7)
        )
        self.assertFalse(
            self._skip(_payload(judged_at=stale), skip_existing=False, max_age_days=7)
        )
        self.assertFalse(
            self._skip(_payload(judged_at=fresh), skip_existing=False, max_age_days=0)
        )
