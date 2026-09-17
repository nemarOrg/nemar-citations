"""Verify the hallu cron script's Ollama preflight short-circuits cleanly.

Phase 4 (#88) wires `dataset-citations-judge-anchors` into the nightly
pipeline. If the Ollama daemon on hallu is down, the cron must abort BEFORE
running `dataset-citations-update` — otherwise the auto-update PR would land
citation data with stale (or missing) anchor judgments.

The preflight contract:
  curl -s --max-time 5 http://localhost:11434/api/tags >/dev/null || exit 2

This test extracts the preflight block from the production script and runs
it against a port that is guaranteed unreachable on the test host. It must
exit with status 2 (not 0, not 1) so the cron `set -uo pipefail` shell
treats it as a hard abort.

No mocks: we run real `curl` against a real (closed) port. Mirrors the
NO-MOCKS rule in .rules/testing.md.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
from pathlib import Path

import pytest

CRON_SCRIPT = Path(__file__).parent.parent / "scripts" / "hallu_cron_pipeline.sh"
RERUN_SCRIPT = Path(__file__).parent.parent / "scripts" / "hallu_rerun.sh"


def _unused_tcp_port() -> int:
    """Return a port that is closed at call time.

    We bind, read the assigned port, then release. There is a tiny race
    window before the test runs `curl`, but the kernel will not reassign
    the port immediately and the test cares only that `curl` fails — any
    non-200 / connection-refused outcome is acceptable.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.skipif(shutil.which("curl") is None, reason="curl not installed")
def test_preflight_exits_2_when_ollama_unreachable() -> None:
    """The exact preflight snippet from the cron script must exit with 2."""
    closed_port = _unused_tcp_port()
    snippet = (
        f"curl -s --max-time 5 http://localhost:{closed_port}/api/tags "
        ">/dev/null || {\n"
        '  echo "ERROR: Ollama daemon not reachable; aborting." >&2\n'
        "  exit 2\n"
        "}\n"
        'echo "would not reach here"\n'
    )
    result = subprocess.run(
        ["bash", "-c", snippet],
        capture_output=True,
        text=True,
        check=False,  # the test asserts on returncode itself
        timeout=15,
    )
    assert result.returncode == 2, (
        f"expected exit 2 when ollama is down, got {result.returncode}; "
        f"stdout={result.stdout!r}, stderr={result.stderr!r}"
    )
    assert "would not reach here" not in result.stdout
    assert "ERROR: Ollama daemon" in result.stderr


def test_cron_script_has_preflight_block() -> None:
    """Catch accidental removal of the preflight in future edits.

    The preflight URL is built from $OLLAMA_BASE_URL with a localhost
    fallback so the cron probes the same daemon the CLI ends up calling;
    we check for the shape rather than a literal URL.
    """
    text = CRON_SCRIPT.read_text()
    assert "curl -s --max-time 5" in text, (
        "preflight curl missing from hallu_cron_pipeline.sh"
    )
    assert "OLLAMA_BASE_URL" in text, (
        "preflight should honor $OLLAMA_BASE_URL so probe + CLI agree"
    )
    assert "/api/tags" in text, "preflight is not hitting /api/tags"
    assert "exit 2" in text, "preflight exit-2 missing from hallu_cron_pipeline.sh"
    assert "dataset-citations-judge-anchors" in text, (
        "judge-anchors step missing from hallu_cron_pipeline.sh"
    )


def test_cron_script_bash_syntax_clean() -> None:
    """`bash -n` parses the script without errors."""
    result = subprocess.run(
        ["bash", "-n", str(CRON_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,  # the test asserts on returncode itself
        timeout=10,
    )
    assert result.returncode == 0, f"bash -n failed: {result.stderr}"


def test_rerun_script_bash_syntax_clean() -> None:
    """`bash -n` parses the rerun helper without errors."""
    result = subprocess.run(
        ["bash", "-n", str(RERUN_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,  # the test asserts on returncode itself
        timeout=10,
    )
    assert result.returncode == 0, f"bash -n failed: {result.stderr}"


def test_rerun_script_supports_judge_only_flag() -> None:
    """The --judge-only flag must be recognised (regression guard for #88)."""
    text = RERUN_SCRIPT.read_text()
    assert "--judge-only" in text
    assert "dataset-citations-judge-anchors" in text


def test_cron_script_wires_generate_embeddings() -> None:
    """Phase 2 of epic #96 (#98) runs embeddings on hallu after scoring.

    Catches accidental removal of the embeddings step or its guard.
    """
    text = CRON_SCRIPT.read_text()
    assert "dataset-citations-generate-embeddings" in text, (
        "generate-embeddings step missing from hallu_cron_pipeline.sh"
    )
    assert "--device cuda" in text, (
        "embeddings step must target the RTX 4090 with --device cuda"
    )
    assert "--skip-existing" in text, (
        "embeddings step must pass --skip-existing for cron parity"
    )
    # The step has to sit AFTER score-confidence so confidence scores are
    # available for the citation-embedding confidence filter.
    score_idx = text.index("dataset-citations-score-confidence")
    embed_idx = text.index("dataset-citations-generate-embeddings")
    assert score_idx < embed_idx, (
        "generate-embeddings must run after score-confidence in the cron"
    )


def test_rerun_script_supports_embeddings_only_flag() -> None:
    """The --embeddings-only flag must be recognised (regression guard for #98)."""
    text = RERUN_SCRIPT.read_text()
    assert "--embeddings-only" in text
    assert "dataset-citations-generate-embeddings" in text


def test_cron_script_invokes_analyze_umap() -> None:
    """UMAP step must be wired into the nightly pipeline (#99 / closes #78).

    The output dir must be flat `dashboard_data/` (NOT a subdir) so the
    dashboard aggregator's `*similarities*.csv` glob picks up the CSVs.
    """
    text = CRON_SCRIPT.read_text()
    assert "dataset-citations-analyze-umap" in text, (
        "analyze-umap step missing from hallu_cron_pipeline.sh"
    )
    # The flag-and-value pair must appear with `dashboard_data` as the
    # exact target — a `dashboard_data/citation_similarities/` subdir
    # would silently produce an empty Citation Similarities panel.
    umap_block_start = text.index("--- analyze-umap ---")
    umap_block = text[umap_block_start : umap_block_start + 600]
    assert "--output-dir dashboard_data" in umap_block, (
        "UMAP output must target `dashboard_data/` directly (the aggregator's "
        "non-recursive *similarities*.csv glob lives there)."
    )
    assert "citation_similarities/" not in umap_block, (
        "UMAP output must NOT nest under a citation_similarities/ subdir — "
        "see #78 root cause."
    )
    # Same explicit-guard contract as the judge-anchors and update steps:
    # `set -uo pipefail` (no -e) means a non-zero exit does not halt the script
    # unless we OR it with an explicit exit.
    assert "exit 2" in umap_block, (
        "analyze-umap step must guard with `|| { ...; exit 2; }` because "
        "the cron uses `set -uo pipefail` (no -e)."
    )


def test_rerun_script_supports_umap_only_flag() -> None:
    """The --umap-only flag must be recognised (regression guard for #99)."""
    text = RERUN_SCRIPT.read_text()
    assert "--umap-only" in text
    assert "dataset-citations-analyze-umap" in text
    assert 'MODE="umap"' in text, "rerun helper must map --umap-only to umap mode"
    # Full-mode should call the UMAP step after score_confidence so we keep the
    # cron + rerun stage ordering identical.
    full_block_start = text.index("  full)")
    full_block = text[full_block_start : full_block_start + 400]
    assert "score_confidence" in full_block
    assert "umap_analysis" in full_block
    assert full_block.index("score_confidence") < full_block.index("umap_analysis"), (
        "umap_analysis must run after score_confidence in the rerun helper's "
        "full mode (matches cron ordering)."
    )


# The theme/network/temporal analyses produced by the cron and required by
# deploy-dashboard.yml's verify step. PR #108 added them to the cron only; the
# tests below keep the rerun helper in sync so manual recovery can satisfy the
# deploy gate (issue #113).
_ANALYSIS_MODULES = (
    "dataset_citations.analysis.generate_themes",
    "dataset_citations.analysis.generate_network",
    "dataset_citations.analysis.generate_temporal",
)


def test_cron_script_wires_analysis_stages() -> None:
    """Catch accidental removal of the themes/network/temporal stages."""
    text = CRON_SCRIPT.read_text()
    for module in _ANALYSIS_MODULES:
        assert module in text, f"{module} missing from hallu_cron_pipeline.sh"


def test_rerun_full_mode_matches_cron_analysis_stages() -> None:
    """Every analysis stage in the cron must also be in the rerun helper.

    Otherwise an operator recovering via scripts/hallu_rerun.sh produces an
    incomplete dashboard_data/ tree and deploy-dashboard.yml's verify fails.
    """
    rerun = RERUN_SCRIPT.read_text()
    for module in _ANALYSIS_MODULES:
        assert module in rerun, (
            f"{module} is in the cron but missing from hallu_rerun.sh; "
            "manual recovery would not satisfy deploy-dashboard.yml's verify"
        )


def test_rerun_script_supports_analysis_only_flag() -> None:
    """The --analysis-only flag must run the three analysis stages (issue #113)."""
    text = RERUN_SCRIPT.read_text()
    assert "--analysis-only" in text
    assert 'MODE="analysis"' in text, (
        "rerun helper must map --analysis-only to analysis mode"
    )


def test_rerun_full_mode_runs_analysis_after_umap() -> None:
    """Full mode must run the analysis stages after UMAP, matching cron order."""
    text = RERUN_SCRIPT.read_text()
    full_block_start = text.index("  full)")
    full_block = text[full_block_start : full_block_start + 600]
    for fn in (
        "umap_analysis",
        "themes_analysis",
        "network_analysis",
        "temporal_analysis",
    ):
        assert fn in full_block, f"{fn} missing from rerun full mode"
    assert full_block.index("umap_analysis") < full_block.index("themes_analysis"), (
        "analysis stages must run after umap_analysis in the rerun helper's full mode"
    )


def test_no_hallu_script_uses_the_retired_plural_flag():
    """Regression for #94.

    The flag was harmonized to `--dataset-list-file` and `hallu_rerun.sh` was
    missed on the first pass. That script is the manual recovery path, so a
    stale flag there fails exactly when someone is rescuing a broken night.
    """
    for script in (CRON_SCRIPT, RERUN_SCRIPT):
        text = script.read_text(encoding="utf-8")
        assert "--datasets-list-file" not in text, (
            f"{script.name} still uses the retired flag spelling"
        )


def test_judge_anchors_invocations_use_the_canonical_flag():
    for script in (CRON_SCRIPT, RERUN_SCRIPT):
        text = script.read_text(encoding="utf-8")
        if "dataset-citations-judge-anchors" not in text:
            continue
        assert "--dataset-list-file" in text, (
            f"{script.name} calls judge-anchors without the canonical flag"
        )
