"""Tests for `dataset_citations.quality.llm_client`.

No mocks. Parsing is exercised on real `claude -p --output-format json`
outputs recorded on hallu (`tests/test_data/claude_cli_*.json`), fed through a
real subclass that overrides the process step. The subprocess plumbing itself
runs against small stand-in executables written to a tempdir.

The live test against the real `claude` CLI is gated behind
RUN_CLAUDE_JUDGE_TESTS=1 (it costs about a cent).
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless

from dataset_citations.quality.llm_client import (
    ALLOWED_CLASSIFICATIONS,
    ClaudeCliJudgmentClient,
    LlmJudgmentError,
    build_anchor_prompt,
)

_TEST_DATA = Path(__file__).parent / "test_data"
_OK_OUTPUT = (_TEST_DATA / "claude_cli_judgment_ok.json").read_text("utf-8")
_NOT_LOGGED_IN_OUTPUT = (_TEST_DATA / "claude_cli_not_logged_in.json").read_text(
    "utf-8"
)


def _with_verdict(verdict: object) -> str:
    """The recorded success output with its structured_output replaced."""
    payload = json.loads(_OK_OUTPUT)
    payload["structured_output"] = verdict
    return json.dumps(payload)


class _RecordedClient(ClaudeCliJudgmentClient):
    """Real subclass that returns a recorded CLI stdout instead of a process."""

    def __init__(self, stdout: str) -> None:
        # The recorded outputs were served by claude-sonnet-5-5.
        super().__init__(model="claude-sonnet-5-5", claude_bin="unused", timeout=5)
        self._stdout = stdout

    def _run_cli(self, prompt: str) -> str:
        return self._stdout


def _stand_in(directory: Path, body: str) -> str:
    """Write an executable shell script standing in for the claude binary."""
    path = directory / "claude"
    path.write_text(f"#!/bin/sh\n{body}\n", "utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


class BuildAnchorPromptTests(TestCase):
    def test_prompt_includes_taxonomy_and_dataset(self) -> None:
        prompt = build_anchor_prompt(
            dataset_id="ds005505",
            dataset_description="EEG recordings from HBN.",
            anchor_doi="10.1234/example",
            anchor_relation="IsDerivedFrom",
            paper_title="The Healthy Brain Network",
            paper_abstract="Multi-site initiative collecting brain data.",
            paper_venue="Scientific Data",
            paper_authors=[],
            paper_year=2017,
        )
        # Taxonomy labels appear verbatim so the model can pick from them.
        for label in ALLOWED_CLASSIFICATIONS:
            self.assertIn(label, prompt)
        self.assertIn("ds005505", prompt)
        self.assertIn("10.1234/example", prompt)
        self.assertIn("IsDerivedFrom", prompt)
        self.assertIn("EEG recordings from HBN", prompt)
        self.assertIn("Multi-site initiative", prompt)
        # JSON-only directive is critical for parsing.
        self.assertIn("strict JSON only", prompt)

    def test_prompt_distinguishes_journal_method_papers(self) -> None:
        # Issue #131: e4b misclassified analysis-method papers published as
        # journal articles (e.g. NeuroImage) as data_paper. The taxonomy
        # definition and a dedicated few-shot example must steer methodology.
        prompt = build_anchor_prompt(
            dataset_id="ds004362",
            dataset_description="EEG dataset.",
            anchor_doi="10.1016/j.neuroimage.2020.117465",
            anchor_relation="References",
            paper_title="An automated pipeline for EEG artifact rejection",
            paper_abstract="A general analysis method for EEG.",
        )
        self.assertIn("method/algorithm paper is methodology, NOT data_paper", prompt)
        self.assertIn("analysis-method paper in a journal", prompt)
        # Example 4 used to carry ERP CORE's DOI next to an unrelated title.
        self.assertNotIn("anchor DOI 10.1016/j.neuroimage.2020.117465", prompt)

    def test_prompt_names_standards_and_asks_for_evidence(self) -> None:
        prompt = build_anchor_prompt(
            dataset_id="nm000275",
            dataset_description="EEG during a driving task.",
            anchor_doi="10.1038/s41597-019-0104-8",
            anchor_relation="IsDescribedBy",
            paper_title="EEG-BIDS",
            paper_abstract=None,
        )
        self.assertIn("EEG-BIDS", prompt)
        self.assertIn("OpenNeuro or NEMAR", prompt)
        self.assertIn("Choose it only on concrete evidence", prompt)
        self.assertIn("prefer related_work", prompt)

    def test_prompt_handles_missing_abstract(self) -> None:
        prompt = build_anchor_prompt(
            dataset_id="ds000999",
            dataset_description="A test dataset.",
            anchor_doi="10.0000/none",
            anchor_relation="References",
            paper_title="Some Paper",
            paper_abstract=None,
            paper_venue=None,
            paper_authors=None,
            paper_year=None,
        )
        self.assertIn("[unavailable]", prompt)

    def test_prompt_truncates_long_dataset_description(self) -> None:
        long_text = "x" * 5000
        prompt = build_anchor_prompt(
            dataset_id="ds000999",
            dataset_description=long_text,
            anchor_doi="10.0000/none",
            anchor_relation="References",
            paper_title="t",
            paper_abstract="a",
        )
        # The literal 5000 x's must not appear; truncation marker should.
        self.assertNotIn("x" * 5000, prompt)
        self.assertIn("…", prompt)

    def test_authors_trim_after_five_names(self) -> None:
        def prompt_authors(names: list[str]) -> str:
            prompt = build_anchor_prompt(
                dataset_id="ds000999",
                dataset_description="d",
                anchor_doi="10.0000/none",
                anchor_relation="References",
                paper_title="t",
                paper_abstract="a",
                paper_authors=names,
            )
            return next(
                line for line in prompt.splitlines() if line.startswith("authors:")
            )

        five = [f"A{i}" for i in range(5)]
        self.assertEqual(prompt_authors(five), "authors: A0, A1, A2, A3, A4")
        self.assertEqual(
            prompt_authors([*five, "A5"]), "authors: A0, A1, A2, A3, A4, et al."
        )
        self.assertEqual(prompt_authors([]), "authors: [unavailable]")


class JudgeAnchorParseTests(TestCase):
    """Validate CLI output parsing without spawning a process."""

    def test_recorded_success_parses(self) -> None:
        out = _RecordedClient(_OK_OUTPUT).judge_anchor("prompt")
        self.assertEqual(out["classification"], "methodology")
        self.assertIn("PREP", out["reason"])
        self.assertEqual(out["model"], "claude-sonnet-5-5")
        self.assertEqual(out["raw_response"], _OK_OUTPUT)

    def test_not_logged_in_raises_with_cli_message(self) -> None:
        with self.assertRaises(LlmJudgmentError) as ctx:
            _RecordedClient(_NOT_LOGGED_IN_OUTPUT).judge_anchor("prompt")
        self.assertIn("Not logged in", str(ctx.exception))

    def test_reason_is_stripped(self) -> None:
        client = _RecordedClient(
            _with_verdict({"classification": "data_paper", "reason": "  trimmed  "})
        )
        self.assertEqual(client.judge_anchor("prompt")["reason"], "trimmed")

    def test_unknown_classification_raises(self) -> None:
        client = _RecordedClient(
            _with_verdict({"classification": "nonsense", "reason": "bad label"})
        )
        with self.assertRaises(LlmJudgmentError) as ctx:
            client.judge_anchor("prompt")
        self.assertIn("not in taxonomy", str(ctx.exception))

    def test_missing_or_empty_fields_raise(self) -> None:
        for verdict in (
            {"reason": "no class field"},
            {"classification": "umbrella"},
            {"classification": "irrelevant", "reason": "   "},
            None,
            ["data_paper", "ok"],
        ):
            with self.assertRaises(LlmJudgmentError, msg=repr(verdict)):
                _RecordedClient(_with_verdict(verdict)).judge_anchor("prompt")

    def test_output_served_by_another_model_raises(self) -> None:
        # The sidecar would record this client's model as the judge, so a call
        # the CLI served with a different model must not count as a verdict.
        client = _RecordedClient(_OK_OUTPUT)
        client.model = "claude-opus-5-5"
        with self.assertRaises(LlmJudgmentError) as ctx:
            client.judge_anchor("prompt")
        self.assertIn("instead of 'claude-opus-5-5'", str(ctx.exception))

    def test_non_json_output_raises(self) -> None:
        with self.assertRaises(LlmJudgmentError) as ctx:
            _RecordedClient("this is not json at all").judge_anchor("prompt")
        self.assertEqual(ctx.exception.raw_response, "this is not json at all")


class RunCliProcessTests(TestCase):
    """The real subprocess path, against stand-in executables."""

    def setUp(self) -> None:
        self.dir = Path(self.enterContext(TemporaryDirectory()))

    def test_round_trip_passes_prompt_on_stdin(self) -> None:
        # The stand-in echoes the recorded output only if the prompt arrived
        # on stdin, proving the plumbing (argv, stdin, stdout capture).
        recorded = self.dir / "out.json"
        recorded.write_text(_OK_OUTPUT, "utf-8")
        claude = _stand_in(self.dir, f'grep -q "the prompt" && cat "{recorded}"')
        client = ClaudeCliJudgmentClient(
            model="claude-sonnet-5-5", claude_bin=claude, timeout=10
        )
        self.assertEqual(
            client.judge_anchor("the prompt")["classification"], "methodology"
        )

    def test_missing_binary_raises(self) -> None:
        client = ClaudeCliJudgmentClient(
            claude_bin=str(self.dir / "does-not-exist"), timeout=5
        )
        with self.assertRaises(LlmJudgmentError) as ctx:
            client.judge_anchor("prompt")
        self.assertIn("not found", str(ctx.exception))

    def test_nonzero_exit_without_output_raises(self) -> None:
        claude = _stand_in(self.dir, 'echo "boom" >&2; exit 3')
        client = ClaudeCliJudgmentClient(claude_bin=claude, timeout=5)
        with self.assertRaises(LlmJudgmentError) as ctx:
            client.judge_anchor("prompt")
        self.assertIn("exited 3: boom", str(ctx.exception))

    def test_timeout_raises(self) -> None:
        claude = _stand_in(self.dir, "sleep 5")
        client = ClaudeCliJudgmentClient(claude_bin=claude, timeout=1)
        with self.assertRaises(LlmJudgmentError) as ctx:
            client.judge_anchor("prompt")
        self.assertIn("timed out", str(ctx.exception))

    def test_health_check_reflects_a_real_round_trip(self) -> None:
        recorded = self.dir / "out.json"
        recorded.write_text(_OK_OUTPUT, "utf-8")
        healthy = ClaudeCliJudgmentClient(
            model="claude-sonnet-5-5",
            claude_bin=_stand_in(self.dir, f'cat "{recorded}"'),
            timeout=5,
        )
        self.assertTrue(healthy.health_check())

        logged_out = self.dir / "logged_out.json"
        logged_out.write_text(_NOT_LOGGED_IN_OUTPUT, "utf-8")
        other = Path(self.enterContext(TemporaryDirectory()))
        broken = ClaudeCliJudgmentClient(
            claude_bin=_stand_in(other, f'cat "{logged_out}"'), timeout=5
        )
        with self.assertLogs("dataset_citations.quality.llm_client", "ERROR"):
            self.assertFalse(broken.health_check())

    def test_argv_and_cwd_contract(self) -> None:
        """The flags that make the call safe and pinned must reach the CLI."""
        argv_file = self.dir / "argv.txt"
        cwd_file = self.dir / "cwd.txt"
        recorded = self.dir / "out.json"
        recorded.write_text(_OK_OUTPUT, "utf-8")
        claude = _stand_in(
            self.dir,
            f'for a in "$@"; do printf "%s\\n" "$a"; done > "{argv_file}"; '
            f'pwd -P > "{cwd_file}"; cat "{recorded}"',
        )
        cache = Path(self.enterContext(TemporaryDirectory())).resolve()
        prior = os.environ.get("XDG_CACHE_HOME")
        os.environ["XDG_CACHE_HOME"] = str(cache)
        try:
            ClaudeCliJudgmentClient(
                model="claude-sonnet-5-5", claude_bin=claude, timeout=10
            ).judge_anchor("prompt")
        finally:
            if prior is None:
                os.environ.pop("XDG_CACHE_HOME", None)
            else:
                os.environ["XDG_CACHE_HOME"] = prior
        argv = argv_file.read_text("utf-8").split("\n")[:-1]
        self.assertEqual(argv[argv.index("--model") + 1], "claude-sonnet-5-5")
        self.assertEqual(argv[argv.index("--tools") + 1], "")
        self.assertEqual(argv[argv.index("--setting-sources") + 1], "project")
        schema = json.loads(argv[argv.index("--json-schema") + 1])
        self.assertEqual(
            schema["properties"]["classification"]["enum"],
            sorted(ALLOWED_CLASSIFICATIONS),
        )
        for flag in ("-p", "--no-session-persistence", "--strict-mcp-config"):
            self.assertIn(flag, argv)
        # A private 0700 directory under the user's cache, never a shared /tmp
        # where another user could plant project settings (review of #243).
        workdir = cache / "dataset-citations" / "anchor-judge"
        self.assertEqual(cwd_file.read_text("utf-8").strip(), str(workdir))
        self.assertEqual(stat.S_IMODE(workdir.stat().st_mode), 0o700)

    def test_logged_out_exit_one_surfaces_the_cli_message(self) -> None:
        # The real logged-out shape: is_error JSON on stdout AND a non-zero exit.
        logged_out = self.dir / "logged_out.json"
        logged_out.write_text(_NOT_LOGGED_IN_OUTPUT, "utf-8")
        claude = _stand_in(self.dir, f'cat "{logged_out}"; exit 1')
        client = ClaudeCliJudgmentClient(claude_bin=claude, timeout=5)
        with self.assertRaises(LlmJudgmentError) as ctx:
            client.judge_anchor("prompt")
        self.assertIn("exited 1: Not logged in", str(ctx.exception))

    def test_nonzero_exit_with_a_success_payload_still_raises(self) -> None:
        recorded = self.dir / "out.json"
        recorded.write_text(_OK_OUTPUT, "utf-8")
        claude = _stand_in(self.dir, f'cat "{recorded}"; exit 1')
        client = ClaudeCliJudgmentClient(
            model="claude-sonnet-5-5", claude_bin=claude, timeout=5
        )
        with self.assertRaises(LlmJudgmentError):
            client.judge_anchor("prompt")

    def test_unrunnable_binaries_raise_and_fail_the_health_check(self) -> None:
        not_executable = self.dir / "claude-0644"
        not_executable.write_text("#!/bin/sh\necho hi\n", "utf-8")
        garbage = self.dir / "claude-garbage"
        garbage.write_bytes(b"\x7fELF-not-really\x00\x01")
        garbage.chmod(0o755)
        for binary in (not_executable, garbage):
            client = ClaudeCliJudgmentClient(claude_bin=str(binary), timeout=5)
            with self.assertRaises(LlmJudgmentError, msg=binary.name):
                client.judge_anchor("prompt")
            with self.assertLogs("dataset_citations.quality.llm_client", "ERROR"):
                self.assertFalse(client.health_check())

    def test_undecodable_output_is_a_judgment_error(self) -> None:
        claude = _stand_in(self.dir, "printf '\\377\\376 not json'")
        client = ClaudeCliJudgmentClient(claude_bin=claude, timeout=5)
        with self.assertRaises(LlmJudgmentError):
            client.judge_anchor("prompt")

    def test_lone_surrogate_in_the_prompt_does_not_raise(self) -> None:
        # OpenAlex abstracts can carry unpaired surrogates; they must be
        # replaced, not crash the worker thread.
        recorded = self.dir / "out.json"
        recorded.write_text(_OK_OUTPUT, "utf-8")
        claude = _stand_in(self.dir, f'cat > /dev/null; cat "{recorded}"')
        client = ClaudeCliJudgmentClient(
            model="claude-sonnet-5-5", claude_bin=claude, timeout=5
        )
        self.assertEqual(
            client.judge_anchor("abstract \ud83d here")["classification"],
            "methodology",
        )


class EnvDefaultsTests(TestCase):
    """Verify defaults and env var overrides without spawning anything."""

    _KEYS = ("ANCHOR_JUDGE_MODEL", "CLAUDE_BIN", "ANCHOR_JUDGE_TIMEOUT_SECONDS")

    def setUp(self) -> None:
        self._prior = {key: os.environ.pop(key, None) for key in self._KEYS}

    def tearDown(self) -> None:
        for key, value in self._prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_defaults_pin_sonnet_5_5(self) -> None:
        client = ClaudeCliJudgmentClient()
        self.assertEqual(client.model, "claude-sonnet-5-5")
        self.assertEqual(client.claude_bin, "claude")

    def test_env_overrides(self) -> None:
        os.environ["ANCHOR_JUDGE_MODEL"] = "env-model"
        os.environ["CLAUDE_BIN"] = "/opt/claude"
        os.environ["ANCHOR_JUDGE_TIMEOUT_SECONDS"] = "99"
        client = ClaudeCliJudgmentClient()
        self.assertEqual(
            (client.model, client.claude_bin, client.timeout),
            ("env-model", "/opt/claude", 99),
        )

    def test_bad_timeout_env_names_the_variable(self) -> None:
        os.environ["ANCHOR_JUDGE_TIMEOUT_SECONDS"] = "three minutes"
        with self.assertRaises(ValueError) as ctx:
            ClaudeCliJudgmentClient()
        self.assertIn("ANCHOR_JUDGE_TIMEOUT_SECONDS", str(ctx.exception))

    def test_explicit_args_win(self) -> None:
        os.environ["ANCHOR_JUDGE_MODEL"] = "env-model"
        client = ClaudeCliJudgmentClient(model="arg-model", timeout=11)
        self.assertEqual((client.model, client.timeout), ("arg-model", 11))


@skipUnless(
    os.getenv("RUN_CLAUDE_JUDGE_TESTS"),
    "live claude CLI call; set RUN_CLAUDE_JUDGE_TESTS=1 to enable",
)
class ClaudeCliJudgmentIntegration(TestCase):
    """One real judgment through a logged-in `claude` CLI."""

    def test_methodology_anchor_round_trip(self) -> None:
        prompt = build_anchor_prompt(
            dataset_id="ds000117",
            dataset_description=(
                "Multi-subject MEG and EEG dataset for a face-perception "
                "experiment, BIDS-formatted."
            ),
            anchor_doi="10.3389/fnins.2013.00267",
            anchor_relation="IsDerivedFrom",
            paper_title="MEG and EEG data analysis with MNE-Python",
            paper_abstract=(
                "MNE-Python is an open-source software package for "
                "processing MEG and EEG data."
            ),
            paper_venue="Frontiers in Neuroscience",
            paper_year=2013,
        )
        result = ClaudeCliJudgmentClient().judge_anchor(prompt)
        self.assertIn(result["classification"], ALLOWED_CLASSIFICATIONS)
        self.assertEqual(result["classification"], "methodology")
