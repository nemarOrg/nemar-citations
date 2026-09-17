#!/usr/bin/env bash
# Nightly hallu pipeline: pull main, run full opencite pipeline with GPU
# scoring, push results to an auto-update branch, open a PR. Designed to
# run from cron at 03:00 PDT (see crontab installed alongside this script).
#
# Locking: flock prevents concurrent runs. Logs land in $REPO_DIR/.logs/.
# Safety: a `git reset --hard origin/main` runs before any pipeline step,
# so this script assumes the working tree is disposable between runs.
#
# Two stages: stage 1 takes the lock, opens the log and checks out main, then
# re-execs into the script it just checked out. Stage 2 runs the pipeline. See
# the re-exec block below for why that indirection is load-bearing.
set -uo pipefail

REPO_DIR="$HOME/dataset_citations"
LOG_DIR="$REPO_DIR/.logs"
LOCK_FILE="$REPO_DIR/.cron.lock"
# Stage 2 inherits the timestamp so both halves append to one log rather than
# splitting the run across two files.
TS="${HALLU_CRON_TS:-$(date -u +%Y-%m-%dT%H-%M-%SZ)}"
LOG="$LOG_DIR/cron-$TS.log"

mkdir -p "$LOG_DIR"

# Stage 1 only. Both the lock fd and stdout survive the exec, so stage 2
# inherits them: re-running `exec 200>` would close the descriptor, dropping
# the lock, and reopening it leaves a window for a second run to slip in.
if [ -z "${HALLU_CRON_STAGE2:-}" ]; then
  # Single-writer lock; concurrent invocations short-circuit instead of clobbering each other.
  exec 200>"$LOCK_FILE"
  if ! flock -n 200; then
    echo "[hallu-cron $TS] another run in progress, skipping" | tee -a "$LOG_DIR/cron-skips.log"
    exit 0
  fi

  # Tee all subsequent output to the timestamped log file.
  exec > >(tee -a "$LOG") 2>&1
fi

trap 'rc=$?; if [ $rc -ne 0 ]; then echo "FAILED rc=$rc at line $LINENO"; fi' EXIT

echo "=== hallu-cron $TS start (host=$(hostname), pid=$$) ==="
cd "$REPO_DIR"

# Pull the GitHub token from gh CLI. cron jobs run with a minimal env;
# without this, dataset-citations-retrieve-metadata falls back to the
# unauthenticated public API (60 req/hr) and stalls after a handful of
# repos. gh stores the token in ~/.config/gh/hosts.yml from a prior
# `gh auth login`; we never have to write it to disk.
export GITHUB_TOKEN="$(gh auth token 2>/dev/null)"
if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "ERROR: gh auth token returned empty; aborting" >&2
  exit 2
fi

# Reset working tree to fresh main so each cron run starts from a known
# good state, then hand over to the version of this script that was just
# checked out.
#
# WHY THE RE-EXEC: bash streams a script from an open file descriptor as it
# runs, and git replaces files by writing a temp file and renaming it over
# the original. The rename gives the path a new inode; the descriptor bash
# is reading keeps pointing at the old one. So without this handover the
# process executes the PREVIOUS run's copy of this file to completion,
# while every `uv run` below builds the CLIs from the tree that was just
# checked out. The script and the code it calls are then one commit apart.
#
# That is not theoretical. On 2026-09-17 the 03:00 run died 60 seconds in
# with
#     dataset-citations-judge-anchors: error: the following arguments are
#     required: --dataset-list-file
# because 2f2fc985 renamed that flag in the CLI and in this script in one
# atomic commit, and the run still managed to pair the old script with the
# new CLI. No data, no commit, no PR. Any future change that touches both a
# CLI signature and its call site here would break the same way.
#
# `git pull` in place of `git reset` would not help: the hazard is the
# inode swap under a running bash, not how the new content arrives.
# Re-exec is the fix, because it makes bash open the new file and start
# over from the top.
#
# Stage 2 skips this block: the tree is already correct, and fetching again
# could land a newer commit and swap the inode a second time, reintroducing
# exactly the bug this avoids.
if [ -z "${HALLU_CRON_STAGE2:-}" ]; then
  git fetch --quiet origin main
  git checkout --quiet main
  git reset --quiet --hard origin/main
  git clean --quiet -fd citations/.checkpoints/ 2>/dev/null || true

  export HALLU_CRON_STAGE2=1
  export HALLU_CRON_TS="$TS"
  echo "stage 1: checked out $(git rev-parse --short HEAD), re-exec into it"
  exec "$REPO_DIR/scripts/hallu_cron_pipeline.sh" "$@"
fi

echo "main at $(git rev-parse --short HEAD)"

DATASETS_LIST="/tmp/hallu_cron_discovered.txt"

# 1. Discover via catalog (api.nemar.org/datasets, no GitHub for this step).
echo "--- discover ---"
uv run dataset-citations-discover \
  --source catalog \
  --output-file "$DATASETS_LIST" \
  --no-catalog-cache

# 2. Retrieve dataset metadata from GitHub (hallu's own IP = own rate-limit budget).
# Moved ahead of `update` in phase 4 (#88) so anchor adjudication has dataset
# descriptions available, and the subsequent `update` step can consume the
# anchor-judgment sidecars produced below.
# `--skip-existing` keeps steady-state runs cheap and (more importantly) ducks
# the GitHub secondary rate limit that bit us during the post-#76 backfill:
# refetching ~3000 unchanged dataset_description.json + README files every
# weekly run is wasted quota. Trade-off: a dataset's GitHub-side description
# / README will only refresh when the cached file is deleted; #82 (follow-up)
# adds a `--max-age-days` flag mirroring `update.py` so the freshness window
# is configurable without an explicit wipe.
echo "--- retrieve-metadata ---"
# Guard: the cron uses `set -uo pipefail` (no -e), so a non-zero exit from
# any step would otherwise let the script continue. Adding the same guard
# pattern that the downstream steps already use so a GitHub rate-limit or
# transient PyGithub failure aborts cleanly instead of feeding the next
# step a stale `datasets/` tree.
uv run dataset-citations-retrieve-metadata \
  --citations-dir citations/json_opencite \
  --output-dir datasets \
  --skip-existing \
  --max-failures 10 || {
  echo "ERROR: dataset-citations-retrieve-metadata failed; aborting." >&2
  exit 2
}

# Pin the anchor-judgment model. gemma4:e4b (9.6GB) is used instead of the
# larger gemma4:31b (19GB) because hallu is a shared host and 31B OOMs /
# blocks other GPU jobs. Honors an explicit OLLAMA_MODEL override for a
# dedicated host. Keep in sync with llm_client._DEFAULT_MODEL.
export OLLAMA_MODEL="${OLLAMA_MODEL:-gemma4:e4b}"

# 3. Preflight: Ollama must be reachable for anchor adjudication. If the
# daemon is down, abort cleanly instead of producing a citation update
# with stale judgments. Exit 2 mirrors the contract documented for
# `dataset-citations-judge-anchors` (phase 2, #86). Honors
# $OLLAMA_BASE_URL so a non-default daemon URL is probed at the same
# host the CLI ends up calling.
OLLAMA_PROBE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
curl -s --max-time 5 "${OLLAMA_PROBE_URL}/api/tags" >/dev/null || {
  echo "ERROR: Ollama daemon at ${OLLAMA_PROBE_URL} not reachable; aborting." >&2
  exit 2
}

# 3a. Anchor adjudication: classify each anchor DOI as data_paper / umbrella /
# methodology / related_work / irrelevant and write sidecars under
# citations/anchor_judgments/. `--skip-existing` keeps steady-state runs cheap;
# the full ~3000-anchor backfill happens on first run after the epic merges.
# The cron uses `set -uo pipefail` (no -e), so a non-zero exit from the CLI
# does NOT halt the script by default; the explicit `|| { exit; }` guard
# below ensures a partial judgment run does not feed downstream `update`
# with a half-written sidecar tree.
echo "--- judge-anchors (gpu, ollama) ---"
uv run dataset-citations-judge-anchors \
  --dataset-list-file "$DATASETS_LIST" \
  --output-dir citations/anchor_judgments \
  --skip-existing || {
  echo "ERROR: dataset-citations-judge-anchors failed; aborting before update." >&2
  exit 2
}

# 3b. Fetch citations via opencite. Phase 3 (#87) made this CLI consume the
# sidecars from step 3a transparently; the invocation is unchanged from the
# pre-phase-4 script. Skip-existing (7d) keeps the run cheap.
#
# OPERATIONAL NOTE: on the first run after a fresh anchor-judgment backfill,
# `--max-age-days 7` (the `update` CLI default) will keep existing citation
# JSONs "fresh" and skip them, so the new bucketing does not take effect on
# already-cached datasets until the freshness window expires. To apply the
# new judgments immediately, manually re-run `dataset-citations-update`
# with `--max-age-days 0` once, then resume the normal weekly cron.
echo "--- update (skip-existing default 7d) ---"
# --datasets-dir lets ds-* DOI extraction reuse the dataset_description cached
# by retrieve-metadata above instead of refetching it from GitHub, which on a
# cold .fetch_state.json would otherwise exhaust the GitHub rate limit (#174).
OPENCITE_CONCURRENCY=4 \
  uv run dataset-citations-update \
    --dataset-list-file "$DATASETS_LIST" \
    --output-dir citations/ \
    --datasets-dir datasets || {
  echo "ERROR: dataset-citations-update failed; aborting before score." >&2
  exit 2
}

# 3b-prune. ds->on dedup (#126). A dataset imported from OpenNeuro gets an on-*
#     catalog id (its ds-* accession survives only as the on-* row's source_id),
#     so discover --source catalog emits only the on-*. But a ds<N>_citations.json
#     produced before the mirror existed lingers on disk and double-counts. Prune
#     those stale ds-* files now -- AFTER update (so the on-* mirror file exists in
#     citations/json_opencite) and BEFORE find-mentions/score/dashboard glob them.
#     Non-fatal: pruning is cleanup, and the CLI already prunes nothing on a
#     catalog failure.
echo "--- prune-mirrored (ds->on dedup) ---"
uv run dataset-citations-prune-mirrored \
  --citations-dir citations/json_opencite \
  || echo "WARN: dataset-citations-prune-mirrored failed; stale ds-* files may remain and double-count in this run's find-mentions/score/dashboard output. See the error above." >&2

# 3c. Accession-mention discovery (#169). People cite datasets by accession
#     number in text (ds*/on*/nm*, plus the OpenNeuro ds- alias for on-*),
#     not by DOI. Full-text search OpenAlex for those and fold the hits into
#     citation_details tagged discovery_method=accession_mention ("cites
#     dataset" bucket). Runs AFTER update (needs the anchor citation JSONs to
#     merge into) and BEFORE score-confidence (which then ranks the merged
#     citations; merge drops the stale confidence block when it adds new ones
#     so --skip-existing re-scores). OpenAlex-only + idempotent writes, so it
#     is non-churning on the git side (the request cost is a separate
#     constraint, see the budget note below). Guard with `|| exit 2` like the
#     other steps.
#     Budget note (#197, measured 2026-07-03): OpenAlex bills ~10 credits per request against a
#     daily allowance that resets at 00:00 UTC (10,000 credits/day with our
#     API key, so ~1,000 requests). A full pass over the corpus is ~1,979
#     requests (~2.6 per dataset, so it scales with the corpus), i.e. ~2
#     days of budget, and an exhausted budget makes opencite
#     sleep on a 24h Retry-After while still holding this script's flock. The
#     7-day rolling window refreshes ~1/7 of the corpus per night (~280
#     requests) and --max-datasets caps the cold-start pass so the first few
#     nights stagger instead of blowing the budget.
echo "--- find-mentions (openalex accession search) ---"
uv run dataset-citations-find-mentions \
  --citations-dir citations/json_opencite \
  --max-age-days 7 \
  --max-datasets 250 || {
  echo "ERROR: dataset-citations-find-mentions failed; aborting before score." >&2
  exit 2
}

# 3d. Merge duplicate citing works (#216). The fetch path dedupes what it
#     writes, but a file produced before #216 can still carry a paper twice:
#     the same DOI under two punctuations, a concept DOI beside its versioned
#     form (10.82901/nemar.onNNNNNN vs ...v1.0.0), or a preprint beside its
#     version of record. Runs AFTER find-mentions (so newly merged mentions are
#     included) and BEFORE score-confidence, which then scores the final list;
#     the CLI drops the stale confidence block on any file it changes so
#     --skip-existing re-scores exactly those. Idempotent, so a steady-state
#     night writes nothing and adds no git churn. Non-fatal: leftover
#     duplicates inflate counts but do not corrupt anything downstream.
echo "--- dedupe (merge duplicate citing works) ---"
uv run dataset-citations-dedupe \
  --citations-dir citations/json_opencite \
  || echo "WARN: dataset-citations-dedupe failed; duplicate citing works may remain and inflate counts. See the error above." >&2

# 4. Semantic confidence scoring on RTX 4090. --skip-existing is a small speedup
#    for unchanged citation files. Same `|| exit 2` guard as the other GPU
#    steps so a CUDA OOM aborts cleanly instead of feeding empty scores
#    downstream.
#
#    No HuggingFace credential is required or wanted here (#217). Both
#    checkpoints are public and already in this host's hub cache, and
#    `utils.model_loading` loads them with the network disabled before it will
#    consider the Hub. An EXPIRED token used to be worse than none at all: it
#    turned anonymous-readable public models into hard 401s and took down the
#    runs on 2026-09-15 and 2026-09-16.
echo "--- score-confidence (cuda) ---"
uv run dataset-citations-score-confidence \
  --citations-dir citations/json_opencite \
  --datasets-dir datasets \
  --device cuda \
  --skip-existing || {
  echo "ERROR: dataset-citations-score-confidence failed; aborting." >&2
  exit 2
}

# 5a. Sentence-transformer embeddings on the RTX 4090. Phase 2 of epic #96
#     (#98) moved this step off CI because CPU torch in GitHub Actions took
#     ~10x longer than CUDA on hallu. `--skip-existing` is consistent with
#     the rest of the pipeline; the CLI also skips via the embedding
#     registry by default, so the flag is explicit-intent rather than a
#     behavior change. Outputs land under `embeddings/`, ready for the
#     UMAP step phase 3 (#99) wires in right after this block.
#
#     Guard with `|| { exit 2; }` because the cron uses `set -uo pipefail`
#     (no -e); a non-zero exit otherwise would not halt the script and we
#     would publish a citation update without refreshed embeddings.
echo "--- generate-embeddings (cuda) ---"
uv run dataset-citations-generate-embeddings \
  --citations citations/json_opencite \
  --datasets datasets \
  --embeddings-dir embeddings \
  --embedding-type both \
  --device cuda \
  --skip-existing || {
  echo "ERROR: dataset-citations-generate-embeddings failed; aborting before commit." >&2
  exit 2
}

# 5b. UMAP analysis on embeddings (closes #78 / phase 3 of epic #96). Reads
#     from `embeddings/` (produced by step 5a) and writes 2D projections +
#     similarity exports directly into `dashboard_data/`. The dashboard
#     aggregator (`dashboard/data/aggregator.py::_load_citation_similarities`)
#     globs `*similarities*.csv` at the top level of `dashboard_data/`, so
#     the CSVs MUST land flat there — NOT under a `citation_similarities/`
#     subdir, which would render the panel empty.
#
#     UMAP itself is CPU-bound and cheap; we keep it on hallu so the entire
#     data refresh happens on one host instead of bouncing through CI. The
#     CLI has no `--skip-existing` today (output filename is timestamped,
#     see analyze_umap.py); rebuilding every run is acceptable since the
#     compute is small. `set -uo pipefail` does not abort on non-zero
#     exit, so the explicit guard prevents a half-written UMAP output from
#     poisoning the dashboard build.
echo "--- analyze-umap ---"
uv run dataset-citations-analyze-umap \
  --embeddings-dir embeddings \
  --output-dir dashboard_data \
  --embedding-type both || {
  echo "ERROR: dataset-citations-analyze-umap failed; aborting before commit." >&2
  exit 2
}

# 5c. Export UMAP coords + real labels to dashboard_data/umap_points.json for the
#     Astro maps (#140). Non-fatal: the maps degrade to empty if this is missing,
#     so a failure here should not block publishing the citation data.
echo "--- export-umap-points ---"
uv run dataset-citations-export-umap-points ||
  echo "WARN: export-umap-points failed; maps will be stale (non-fatal)."

# 5d. Export per-dataset modality from the api.nemar.org catalog to
#     dashboard_data/dataset_modalities.json for the Trends modality donut
#     (#154). Non-fatal: the chart is omitted if this is missing.
echo "--- export-modalities ---"
uv run dataset-citations-export-modalities ||
  echo "WARN: export-modalities failed; modality chart will be stale (non-fatal)."

# 6. Theme / network / temporal analyses. Epic #96 phase 5 removed these
#    invocations from `.github/workflows/deploy-dashboard.yml` on the
#    expectation that the cron produces them; the initial epic ship only
#    wired generate-embeddings + analyze-umap and missed these three. They
#    are CPU-only and cheap; running here keeps the "hallu is the sole
#    producer" invariant true so the CI verify step at deploy time finds
#    the populated dirs. Each is guarded so a single analysis failure
#    aborts cleanly under `set -uo pipefail` (no -e).
echo "--- generate-themes ---"
mkdir -p dashboard_data/themes
uv run python -m dataset_citations.analysis.generate_themes \
  --citations-dir citations/json_opencite \
  --output-dir dashboard_data/themes || {
  echo "ERROR: generate-themes failed; aborting before commit." >&2
  exit 2
}

echo "--- generate-network ---"
mkdir -p dashboard_data/network
uv run python -m dataset_citations.analysis.generate_network \
  --citations-dir citations/json_opencite \
  --output-dir dashboard_data/network || {
  echo "ERROR: generate-network failed; aborting before commit." >&2
  exit 2
}

echo "--- generate-temporal ---"
mkdir -p dashboard_data/temporal
uv run python -m dataset_citations.analysis.generate_temporal \
  --citations-dir citations/json_opencite \
  --output-dir dashboard_data/temporal || {
  echo "ERROR: generate-temporal failed; aborting before commit." >&2
  exit 2
}

# Bail cleanly if no tracked data changed (typical when nothing is stale).
# `dashboard_data/` and `embeddings/` are included because deploy-dashboard.yml
# verifies their presence on a fresh CI checkout (epic #96 / #101 / #103). The
# previous-pipeline gitignore patterns that excluded the analysis subdirs were
# removed in the same epic so the cron can actually commit them.
if git diff --quiet citations/ datasets/ embeddings/ dashboard_data/; then
  echo "no tracked data changes, nothing to commit"
  exit 0
fi

# Commit + push to a timestamped branch; open a PR (manual merge gates the deploy).
BRANCH="auto-update/$TS"
git checkout -b "$BRANCH"
git add citations/ datasets/ embeddings/ dashboard_data/
DIFFSTAT="$(git diff --cached --stat | tail -5)"
git commit -m "data: hallu nightly pipeline ($TS)

GPU semantic scoring + embeddings on RTX 4090. Pipeline:
  catalog discover -> metadata -> judge-anchors -> opencite fetch
  -> prune-mirrored -> find-mentions -> dedupe -> score-confidence
  -> generate-embeddings

$(echo "$DIFFSTAT")"

git push -u --quiet origin "$BRANCH"

PR_URL=$(gh pr create --base main --head "$BRANCH" \
  --title "Nightly pipeline: $TS" \
  --body "Auto-generated by hallu cron (03:00 PDT). GPU-scored on RTX 4090.

Diffstat:

\`\`\`
$DIFFSTAT
\`\`\`

Auto-merging on green CI. Cloudflare deploy fires via deploy-dashboard.yml's push trigger.")
echo "PR: $PR_URL"

# Auto-merge once CI is green. Data PRs have no human-reviewable
# content; CI is the only gate. --merge preserves commit history
# (no squash), --delete-branch keeps the remote tidy.
gh pr merge --auto --merge --delete-branch "$PR_URL"
echo "auto-merge enabled on $PR_URL"
echo "=== hallu-cron $TS done ==="
