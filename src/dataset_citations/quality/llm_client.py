"""
Claude-backed LLM client for anchor adjudication.

Epic #76 fixes citation inflation by asking an LLM to classify each DOI anchor
in a dataset's metadata as one of five buckets. Since issue #241 the judge is
Claude Sonnet 5.5 driven through the `claude` CLI in headless mode on hallu
(the previous Ollama/Gemma judge on the shared GPU host lost its models and
silently errored on every anchor). opencite still resolves each anchor's
title / abstract / venue / year before the call; the model judges resolved
metadata, never a bare DOI.

This module is the single place where the prompt + classification taxonomy
live, so a prompt revision is a single-file change, not a sweep.

The classification schema is:

  - data_paper:    the paper IS the data paper for this dataset (or the
                   dataset's preprint / curation paper / a deposit of the
                   same data).
  - umbrella:      the paper is a multi-dataset / multi-study initiative
                   (HBN, UK Biobank, ABCD) that contains this dataset but
                   is not its data paper.
  - methodology:   the paper is a software / method / analysis tool or a
                   standard the dataset's protocol uses (MNE-Python,
                   EEG-BIDS spec).
  - related_work:  the paper is topically related but does not describe
                   this dataset specifically.
  - irrelevant:    the paper has no meaningful relationship to this
                   dataset (mis-attached anchor, typo'd DOI, etc.).

Only `data_paper` lets an anchor contribute citations (`core.anchor_gate`).

Copyright (c) 2026 Seyed Yahya Shirazi (neuromechanist)
All rights reserved.

Author: Seyed Yahya Shirazi
GitHub: https://github.com/neuromechanist
Email: shirazi@ieee.org
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)

# The five-class taxonomy is the contract phase 2's sidecar schema and
# phase 3's pipeline filter both depend on. Adding or removing a class is a
# cross-phase change; doc the rationale before edits.
ALLOWED_CLASSIFICATIONS: frozenset[str] = frozenset(
    {"data_paper", "umbrella", "methodology", "related_work", "irrelevant"}
)

_ENV_MODEL = "ANCHOR_JUDGE_MODEL"
_ENV_BIN = "CLAUDE_BIN"
_ENV_TIMEOUT = "ANCHOR_JUDGE_TIMEOUT_SECONDS"

# Single source of truth for the judge model; the cron pins the same value.
# A sidecar judged by any other model is re-judged (cli.judge_anchors).
_DEFAULT_MODEL = "claude-sonnet-5-5"
_DEFAULT_BIN = "claude"
# A judgment takes ~4s on hallu; the ceiling only guards a hung CLI.
_DEFAULT_TIMEOUT = 180

# Structured output: the CLI validates the model's answer against this schema
# and returns it as `structured_output`, so there is no free-text JSON to parse.
_OUTPUT_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "classification": {
                "type": "string",
                "enum": sorted(ALLOWED_CLASSIFICATIONS),
            },
            "reason": {"type": "string"},
        },
        "required": ["classification", "reason"],
        "additionalProperties": False,
    }
)
_SYSTEM_PROMPT = (
    "You are a careful research librarian who decides whether a paper is the "
    "data paper of a specific neuroscience dataset. Answer only through the "
    "structured output."
)

# Truncate long dataset descriptions to keep the prompt small while leaving
# room for the candidate paper.
_DATASET_DESCRIPTION_CHAR_LIMIT = 1500
_ABSTRACT_CHAR_LIMIT = 2000


class LlmJudgmentError(RuntimeError):
    """Raised when the LLM returns malformed JSON or an out-of-taxonomy label.

    The probe and the phase 2 CLI catch this so a single bad anchor doesn't
    abort a batch run. The detail string carries the raw response for
    auditing.
    """

    def __init__(self, message: str, *, raw_response: str | None = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response


def _truncate(text: str | None, limit: int) -> str:
    if not text:
        return "[unavailable]"
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _format_authors(authors: Iterable[Any]) -> str:
    """Render a small authors list for the prompt. Trims at 5 names."""
    names: list[str] = []
    for author in authors:
        name = getattr(author, "name", None) or str(author)
        if name:
            names.append(name)
        if len(names) >= 5:
            names.append("et al.")
            break
    return ", ".join(names) if names else "[unavailable]"


def build_anchor_prompt(
    *,
    dataset_id: str,
    dataset_description: str | None,
    anchor_doi: str,
    anchor_relation: str,
    paper_title: str | None,
    paper_abstract: str | None,
    paper_venue: str | None = None,
    paper_authors: Iterable[Any] | None = None,
    paper_year: int | None = None,
) -> str:
    """Return the full prompt for one (dataset, anchor) judgment.

    Kept as a module-level pure function so every caller builds identical
    prompts. The opening lays out the taxonomy with one-line definitions,
    then hands the model the dataset description + candidate paper + the
    DataCite `source_relation` value, then four few-shot examples (HBN
    umbrella, dataset preprint, MNE-Python methodology, and a journal
    analysis-method paper added for issue #131 so method papers are not
    mistaken for data papers). `data_paper` is the only class that counts
    citations, so the prompt asks for concrete evidence before choosing it.
    """
    description = _truncate(dataset_description, _DATASET_DESCRIPTION_CHAR_LIMIT)
    abstract = _truncate(paper_abstract, _ABSTRACT_CHAR_LIMIT)
    title = paper_title or "[unavailable]"
    venue = paper_venue or "[unavailable]"
    authors = _format_authors(paper_authors or [])
    year = str(paper_year) if paper_year else "[unavailable]"

    return f"""You are classifying the relationship between a neuroscience dataset and a candidate paper that the dataset's metadata cites as a related identifier.

Choose exactly one class from this taxonomy:

- data_paper: the paper IS this dataset's data paper, dataset preprint, or curation paper. It introduces, describes, or releases the data in this specific dataset: a data descriptor, the study paper that reports recording exactly this data, or another deposit of the same data (for example its figshare or Zenodo record). Choose it only on concrete evidence (matching title, task, participants, or authors); a different study by the same lab on a similar task is related_work.
- umbrella: the paper is a multi-dataset / multi-study initiative (e.g. HBN, UK Biobank, ABCD) that this dataset belongs to, but the paper is NOT this specific dataset's data paper.
- methodology: the paper is a software tool, analysis method, algorithm, or technical specification that this dataset's protocol or a downstream analysis uses (e.g. MNE-Python, EEGLAB, FieldTrip, a standard or specification such as BIDS, EEG-BIDS, or HED, a data platform such as OpenNeuro or NEMAR, or an analysis-method paper in a journal such as NeuroImage). A peer-reviewed method/algorithm paper is methodology, NOT data_paper, even when it reads like a normal research article: it describes a technique reused across many studies, it does not introduce THIS dataset.
- related_work: the paper is topically related (same brain region, task, modality) but does not describe this dataset specifically.
- irrelevant: the paper has no meaningful relationship to this dataset (mis-attached anchor, a typo'd DOI that resolves to an unrelated paper, token-collision false positive).

Only data_paper makes the candidate's citations count as citations of this dataset, so when the evidence is ambiguous prefer related_work.

Respond with strict JSON only, no prose, no markdown:
{{"classification": "<one of the five labels>", "reason": "<one sentence, <= 200 chars, citing concrete evidence from the title or abstract>"}}

=== DATASET ===
dataset_id: {dataset_id}
description (truncated):
{description}

=== CANDIDATE PAPER ===
DOI: {anchor_doi}
source_relation (DataCite, may be hint but is NOT ground truth): {anchor_relation}
title: {title}
authors: {authors}
venue: {venue}
year: {year}
abstract:
{abstract}

=== EXAMPLES ===
Example 1 (HBN umbrella):
  dataset_id: ds004186 (one of many HBN sibling releases)
  candidate paper: "The Healthy Brain Network Serial Scanning Initiative: a resource for evaluating inter-individual differences and their reliabilities across scan conditions and sessions"
  Correct output: {{"classification": "umbrella", "reason": "Paper describes the broader Healthy Brain Network initiative containing many sibling datasets, not this specific release."}}

Example 2 (dataset preprint as data paper):
  dataset_id: ds002718
  candidate paper: "An open dataset of EEG recordings from face perception experiments"
  Correct output: {{"classification": "data_paper", "reason": "Title and abstract describe the release of this specific EEG face-perception dataset."}}

Example 3 (methodology tool):
  dataset_id: ds000117 (anchor DOI 10.3389/fnins.2013.00267)
  candidate paper: "MEG and EEG data analysis with MNE-Python"
  Correct output: {{"classification": "methodology", "reason": "Describes the MNE-Python analysis library; tool used in the protocol, not a paper about this dataset."}}

Example 4 (analysis-method paper in a journal -> methodology, NOT data_paper):
  dataset_id: ds004362
  candidate paper: "An automated pipeline for EEG artifact rejection and independent component analysis" (NeuroImage)
  Correct output: {{"classification": "methodology", "reason": "Describes a general EEG analysis method reused across many studies; a journal method paper, not a description of this dataset."}}

Now classify the candidate paper for dataset {dataset_id}."""


class ClaudeCliJudgmentClient:
    """Judge anchors with Claude through the `claude` CLI in headless mode.

    Each judgment is one `claude -p` call with tools disabled, no session
    persistence, only project-level settings (so user hooks and plugins stay
    out), and a JSON schema that makes the CLI return the verdict as
    `structured_output`. The CLI uses whatever login the host already has;
    hallu runs it under the pipeline user's Claude account. The subprocess
    runs from the system temp dir so no repository `CLAUDE.md` is loaded into
    every call.

    Thread-safe: `judge_anchor` only spawns a subprocess, so callers may run
    several judgments concurrently.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        claude_bin: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.model = model or os.environ.get(_ENV_MODEL) or _DEFAULT_MODEL
        self.claude_bin = claude_bin or os.environ.get(_ENV_BIN) or _DEFAULT_BIN
        timeout_env = os.environ.get(_ENV_TIMEOUT)
        if timeout is not None:
            self.timeout = timeout
        elif timeout_env:
            self.timeout = int(timeout_env)
        else:
            self.timeout = _DEFAULT_TIMEOUT

    def health_check(self) -> bool:
        """Return True iff a real judgment round-trips through the CLI.

        A cheap liveness probe is not enough: the retired Ollama judge passed
        its `/api/tags` probe while every judgment 404ed because the model was
        gone (issue #241). This runs one real, tiny judgment instead, so a
        logged-out CLI, a missing binary, or an unknown model all fail here.
        """
        prompt = build_anchor_prompt(
            dataset_id="healthcheck",
            dataset_description="Dataset Name: EEG recordings during a visual task",
            anchor_doi="10.3389/fnins.2013.00267",
            anchor_relation="References",
            paper_title="MEG and EEG data analysis with MNE-Python",
            paper_abstract="Describes the MNE-Python software package.",
            paper_year=2013,
        )
        try:
            self.judge_anchor(prompt)
        except LlmJudgmentError as exc:
            logger.error("anchor judge health check failed: %s", exc)
            return False
        return True

    def judge_anchor(self, prompt: str) -> dict[str, Any]:
        """Run one judgment and validate it.

        Returns a dict with keys:
          - classification (str, one of ALLOWED_CLASSIFICATIONS)
          - reason (str, non-empty)
          - raw_response (str, the CLI's verbatim stdout)
          - model (str)

        Raises LlmJudgmentError on any CLI failure (missing binary, timeout,
        non-zero exit, `is_error` result such as "Not logged in") or when the
        structured output is missing or outside the taxonomy.
        """
        raw = self._run_cli(prompt)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LlmJudgmentError(
                f"claude CLI printed non-JSON output: {exc}", raw_response=raw
            ) from exc
        if not isinstance(payload, dict):
            raise LlmJudgmentError(
                f"claude CLI output root is not an object ({type(payload).__name__})",
                raw_response=raw,
            )
        if payload.get("is_error"):
            raise LlmJudgmentError(
                f"claude CLI reported an error: {payload.get('result')!r}",
                raw_response=raw,
            )

        verdict = payload.get("structured_output")
        if not isinstance(verdict, dict):
            raise LlmJudgmentError(
                "claude CLI returned no structured_output", raw_response=raw
            )
        classification = verdict.get("classification")
        reason = verdict.get("reason")
        if not isinstance(classification, str):
            raise LlmJudgmentError(
                "missing or non-string 'classification' field", raw_response=raw
            )
        if classification not in ALLOWED_CLASSIFICATIONS:
            raise LlmJudgmentError(
                f"classification {classification!r} not in taxonomy "
                f"{sorted(ALLOWED_CLASSIFICATIONS)}",
                raw_response=raw,
            )
        if not isinstance(reason, str) or not reason.strip():
            raise LlmJudgmentError("missing or empty 'reason' field", raw_response=raw)

        return {
            "classification": classification,
            "reason": reason.strip(),
            "raw_response": raw,
            "model": self.model,
        }

    def _command(self) -> list[str]:
        return [
            self.claude_bin,
            "-p",
            "--model",
            self.model,
            "--output-format",
            "json",
            "--json-schema",
            _OUTPUT_SCHEMA,
            "--system-prompt",
            _SYSTEM_PROMPT,
            "--tools",
            "",
            "--setting-sources",
            "project",
            "--strict-mcp-config",
            "--no-session-persistence",
        ]

    def _run_cli(self, prompt: str) -> str:
        """Run the CLI with `prompt` on stdin and return its stdout.

        Split out so tests can subclass the client and return a recorded CLI
        output instead of spawning a process (the no-mocks pattern used across
        this repo). Every process-level failure becomes `LlmJudgmentError`, so
        one bad anchor never aborts a batch.
        """
        try:
            proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
                self._command(),
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=tempfile.gettempdir(),
                check=False,
            )
        except FileNotFoundError as exc:
            raise LlmJudgmentError(
                f"claude CLI not found at {self.claude_bin!r}; set {_ENV_BIN}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise LlmJudgmentError(
                f"claude CLI timed out after {self.timeout}s"
            ) from exc
        if proc.returncode != 0 and not proc.stdout.strip():
            raise LlmJudgmentError(
                f"claude CLI exited {proc.returncode}: {proc.stderr.strip()[:500]}"
            )
        # A non-zero exit that still printed a JSON result (e.g. is_error) is
        # parsed by the caller, which surfaces the CLI's own message.
        return proc.stdout
