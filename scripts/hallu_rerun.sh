#!/usr/bin/env bash
# Manual rerun helper for the hallu nightly pipeline. Same steps as
# scripts/hallu_cron_pipeline.sh but without the flock/PR/auto-merge
# scaffolding so an operator can iterate on a single stage.
#
# Usage:
#   scripts/hallu_rerun.sh                   # full pipeline (no lock, no PR)
#   scripts/hallu_rerun.sh --judge-only      # just step 3a (anchor adjudication)
#   scripts/hallu_rerun.sh --update-only     # just step 3b (opencite fetch)
#   scripts/hallu_rerun.sh --score-only      # just step 4 (GPU scoring)
#   scripts/hallu_rerun.sh --embeddings-only # just step 5a (GPU embeddings)
#   scripts/hallu_rerun.sh --umap-only       # just step 5b (UMAP on embeddings)
#   scripts/hallu_rerun.sh --analysis-only   # just step 6 (themes + network + temporal)
#   scripts/hallu_rerun.sh --dry-run         # print the steps that would run
#
# Assumes:
#   - cwd is the repo root (or $REPO_DIR is exported).
#   - `uv` and `gh` are on PATH.
#   - For --judge-only: an Ollama daemon is reachable at
#     $OLLAMA_BASE_URL (default http://localhost:11434). On hallu the
#     daemon is local; from a workstation, set up an ssh tunnel per
#     scripts/probe_anchor_judgment.py's docstring and point
#     OLLAMA_BASE_URL at the forwarded port (NOT 11434, which collides
#     with a workstation-local ollama serve).
set -uo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
DATASETS_LIST="${DATASETS_LIST:-/tmp/hallu_rerun_discovered.txt}"

MODE="full"
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --judge-only) MODE="judge" ;;
    --update-only) MODE="update" ;;
    --score-only) MODE="score" ;;
    --embeddings-only) MODE="embeddings" ;;
    --umap-only) MODE="umap" ;;
    --analysis-only) MODE="analysis" ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *)
      echo "unknown flag: $arg" >&2
      exit 2
      ;;
  esac
done

cd "$REPO_DIR"

# Mirror the cron's GitHub auth path: pull the token from `gh` so the
# downstream CLIs hit the authenticated GitHub API (5000 req/hr) instead
# of falling back to the unauthenticated public API (60 req/hr). An
# operator running this in a fresh login shell where gh is configured
# but $GITHUB_TOKEN isn't exported would otherwise silently rate-limit
# midway through retrieve-metadata.
if [ -z "${GITHUB_TOKEN:-}" ]; then
  if command -v gh >/dev/null 2>&1; then
    export GITHUB_TOKEN="$(gh auth token 2>/dev/null)"
  fi
fi
if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "WARNING: GITHUB_TOKEN not set and gh auth token failed; GitHub API will rate-limit at 60 req/hr." >&2
fi

run() {
  echo "+ $*"
  if [ "$DRY_RUN" -eq 0 ]; then
    "$@"
  fi
}

discover() {
  run uv run dataset-citations-discover \
    --source catalog \
    --output-file "$DATASETS_LIST" \
    --no-catalog-cache
}

retrieve_metadata() {
  # --max-failures mirrors the cron: a few dead/undecodable repos must not
  # abort a manual recovery run (issue #118).
  run uv run dataset-citations-retrieve-metadata \
    --citations-dir citations/json_opencite \
    --output-dir datasets \
    --max-failures 10
}

preflight_ollama() {
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "+ curl -s --max-time 5 ${OLLAMA_BASE_URL:-http://localhost:11434}/api/tags >/dev/null"
    return 0
  fi
  curl -s --max-time 5 "${OLLAMA_BASE_URL:-http://localhost:11434}/api/tags" >/dev/null || {
    echo "ERROR: Ollama daemon at ${OLLAMA_BASE_URL:-http://localhost:11434} not reachable; aborting." >&2
    exit 2
  }
}

judge_anchors() {
  preflight_ollama
  # `set -uo pipefail` does not abort on non-zero exit; explicit guard
  # so a partial judgment run does not let `update_citations` proceed
  # with a half-written sidecar tree.
  run uv run dataset-citations-judge-anchors \
    --dataset-list-file "$DATASETS_LIST" \
    --output-dir citations/anchor_judgments \
    --skip-existing || {
    echo "ERROR: dataset-citations-judge-anchors failed; aborting." >&2
    exit 2
  }
}

update_citations() {
  run env OPENCITE_CONCURRENCY=4 \
    uv run dataset-citations-update \
      --dataset-list-file "$DATASETS_LIST" \
      --output-dir citations/
}

score_confidence() {
  run uv run dataset-citations-score-confidence \
    --citations-dir citations/json_opencite \
    --datasets-dir datasets \
    --device cuda \
    --skip-existing
}

embeddings() {
  # Phase 2 of epic #96 (#98) moved sentence-transformer embedding
  # generation off CI onto hallu's RTX 4090. `--skip-existing` is the
  # explicit form of the CLI's default registry-skip behavior; mirrors
  # the score-confidence convention. The cron guards this step with an
  # explicit `|| exit 2` so a partial run does not feed downstream
  # analysis; the rerun helper leaves error handling to the operator.
  run uv run dataset-citations-generate-embeddings \
    --citations citations/json_opencite \
    --datasets datasets \
    --embeddings-dir embeddings \
    --embedding-type both \
    --device cuda \
    --skip-existing
}

umap_analysis() {
  # Reads `embeddings/` (produced by step 5a above) and writes UMAP outputs
  # FLAT under `dashboard_data/` so the dashboard aggregator's
  # `*similarities*.csv` glob picks them up. Writing under a
  # `citation_similarities/` subdir would render the panel empty (the glob
  # is non-recursive). Explicit guard mirrors the cron's `set -uo pipefail`
  # semantics so a partial UMAP run does not poison a subsequent dashboard
  # build.
  run uv run dataset-citations-analyze-umap \
    --embeddings-dir embeddings \
    --output-dir dashboard_data \
    --embedding-type both || {
    echo "ERROR: dataset-citations-analyze-umap failed; aborting." >&2
    exit 2
  }
  # Export coords + real labels to dashboard_data/umap_points.json for the Astro
  # maps (#140). Non-fatal: maps degrade to empty if absent.
  run uv run dataset-citations-export-umap-points ||
    echo "WARN: export-umap-points failed; maps will be stale (non-fatal)."
}

# Step 6: theme / network / temporal analyses -> dashboard_data/{themes,network,
# temporal}/. PR #108 added these to the cron but not here, so manual recovery
# could not satisfy deploy-dashboard.yml's verify (it hard-requires the three
# dirs be non-empty). Mirror the cron exactly: same CLIs, mkdir, and explicit
# `|| exit 2` guard (the helper runs under `set -uo pipefail`, no -e).
themes_analysis() {
  mkdir -p dashboard_data/themes
  run uv run python -m dataset_citations.analysis.generate_themes \
    --citations-dir citations/json_opencite \
    --output-dir dashboard_data/themes || {
    echo "ERROR: generate-themes failed; aborting." >&2
    exit 2
  }
}

network_analysis() {
  mkdir -p dashboard_data/network
  run uv run python -m dataset_citations.analysis.generate_network \
    --citations-dir citations/json_opencite \
    --output-dir dashboard_data/network || {
    echo "ERROR: generate-network failed; aborting." >&2
    exit 2
  }
}

temporal_analysis() {
  mkdir -p dashboard_data/temporal
  run uv run python -m dataset_citations.analysis.generate_temporal \
    --citations-dir citations/json_opencite \
    --output-dir dashboard_data/temporal || {
    echo "ERROR: generate-temporal failed; aborting." >&2
    exit 2
  }
}

case "$MODE" in
  judge)
    # judge-only needs the dataset list AND fresh dataset descriptions
    # (README + dataset_description.json) under `datasets/` because the
    # judgment prompt feeds them into the LLM context. Discover first if
    # the list is missing, then refresh metadata so an iterating operator
    # gets up-to-date descriptions on every rerun.
    if [ ! -s "$DATASETS_LIST" ]; then
      echo "$DATASETS_LIST missing or empty; running discover first"
      discover
    fi
    retrieve_metadata
    judge_anchors
    ;;
  update)
    if [ ! -s "$DATASETS_LIST" ]; then
      echo "$DATASETS_LIST missing or empty; running discover first"
      discover
    fi
    update_citations
    ;;
  score)
    score_confidence
    ;;
  embeddings)
    # embeddings-only needs the dataset metadata cache (READMEs +
    # dataset_description.json) under `datasets/` because the dataset
    # embedding text is built from those files. Discover first if the
    # list is missing, then refresh metadata so an iterating operator
    # gets up-to-date dataset embeddings on every rerun. Mirrors the
    # `judge` case (discover -> retrieve_metadata -> step).
    if [ ! -s "$DATASETS_LIST" ]; then
      echo "$DATASETS_LIST missing or empty; running discover first"
      discover
    fi
    retrieve_metadata
    embeddings
    ;;
  umap)
    # UMAP only needs the embeddings directory; no GitHub / opencite / Ollama
    # traffic. Assumes step 5a (embeddings) has populated `embeddings/`;
    # the analyze-umap CLI logs a clear error if the directory is missing.
    umap_analysis
    ;;
  analysis)
    # Theme/network/temporal only; reads citations/json_opencite and writes
    # the three dashboard_data/ subdirs deploy-dashboard.yml verifies. No
    # GitHub / opencite / Ollama traffic.
    themes_analysis
    network_analysis
    temporal_analysis
    ;;
  full)
    discover
    retrieve_metadata
    judge_anchors
    update_citations
    score_confidence
    embeddings
    umap_analysis
    themes_analysis
    network_analysis
    temporal_analysis
    ;;
esac
