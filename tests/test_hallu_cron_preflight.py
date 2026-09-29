"""Static and syntax checks on the hallu cron and rerun scripts.

Phase 4 (#88) wired `dataset-citations-judge-anchors` into the nightly
pipeline. Since #241 the judge is Claude Sonnet 5.5 through the `claude` CLI,
and the judge CLI health-checks it with one real judgment (exit 2 on failure;
covered by tests/test_cli_judge_anchors.py). The scripts must therefore pin the
judge model, abort with exit 2 when the judge step fails, and never reference
the retired Ollama preflight again: that probe passed while every judgment
failed, which is how the September 2026 inflation happened.

No mocks: the syntax checks run the real `bash -n`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

CRON_SCRIPT = Path(__file__).parent.parent / "scripts" / "hallu_cron_pipeline.sh"
RERUN_SCRIPT = Path(__file__).parent.parent / "scripts" / "hallu_rerun.sh"


def test_cron_script_pins_the_claude_judge() -> None:
    text = CRON_SCRIPT.read_text()
    assert 'ANCHOR_JUDGE_MODEL="${ANCHOR_JUDGE_MODEL:-claude-sonnet-5-5}"' in text
    assert "dataset-citations-judge-anchors" in text, (
        "judge-anchors step missing from hallu_cron_pipeline.sh"
    )
    assert "aborting before update." in text, "judge step lost its exit-2 guard"


def test_gate_runs_right_after_update_and_is_fatal() -> None:
    """The anchor gate must run on every file before anything scores or
    publishes them, and a failure must abort rather than publish ungated."""
    text = CRON_SCRIPT.read_text()
    update_idx = text.index("uv run dataset-citations-update")
    gate_idx = text.index("uv run dataset-citations-gate-anchors")
    score_idx = text.index("uv run dataset-citations-score-confidence")
    assert update_idx < gate_idx < score_idx
    assert "dataset-citations-gate-anchors failed; aborting before score." in text
    assert "gate_anchors" in RERUN_SCRIPT.read_text()


def test_scripts_no_longer_probe_ollama() -> None:
    for script in (CRON_SCRIPT, RERUN_SCRIPT):
        text = script.read_text()
        assert "/api/tags" not in text, f"{script.name} still probes Ollama"
        assert "OLLAMA_" not in text, f"{script.name} still reads OLLAMA_* vars"


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
