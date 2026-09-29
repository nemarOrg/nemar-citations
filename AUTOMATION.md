# Automated Citation Updates

## Overview

The pipeline runs nightly on **hallu** (the GPU host), not in GitHub Actions.
`scripts/hallu_cron_pipeline.sh` is the single producer of citation data,
embeddings and analysis outputs.
CI only consumes what that script commits.

```
hallu cron (03:00 PDT, nightly)
  -> discover -> retrieve-metadata -> judge-anchors -> update (opencite fetch)
  -> gate-anchors -> prune-mirrored -> find-mentions -> dedupe -> score-confidence
  -> generate-embeddings -> analyze-umap -> themes / network / temporal
  -> commit to auto-update/<timestamp> -> PR -> auto-merge on green CI
  -> close older open nightly PRs (superseded)
       |
       v
  push to main triggers .github/workflows/deploy-dashboard.yml
  -> builds the HTML and deploys to Cloudflare Pages
```

The heavy steps live on hallu because its RTX 4090 runs the semantic scoring and
embeddings roughly ten times faster than CPU torch in GitHub Actions (epic #96),
and because hallu's own IP carries its own GitHub rate-limit budget.

There is no `update_citations.yml` workflow;
it was removed when hallu became the producer.
The two workflows that remain are `test.yml` (lint, types, pytest) and
`deploy-dashboard.yml` (build and deploy from already-produced inputs).

## Host Setup

The crontab entry on hallu:

```
0 3 * * * PATH=/home/yahya/.local/bin:/usr/local/bin:/usr/bin:/bin \
  /home/yahya/dataset_citations/scripts/hallu_cron_pipeline.sh \
  > /home/yahya/dataset_citations/.logs/cron-stdout.log 2>&1
```

Requirements on the host:

- A `gh auth login` session. The script reads `gh auth token` at run time rather
  than storing a token on disk; an empty token aborts the run with exit 2.
- The `claude` CLI, installed in `~/.local/bin` (on the crontab PATH) and
  logged in as the pipeline user, for anchor adjudication. The judge is Claude
  Sonnet 5.5 (`ANCHOR_JUDGE_MODEL`, default `claude-sonnet-5-5`). A judge or
  prompt change re-judges every anchor once, on the order of a thousand calls;
  steady-state nights judge only new, relabeled, or failed anchors.
- **No HuggingFace credential.** Both sentence-transformer checkpoints are
  public and cached locally, and `utils/model_loading.py` loads them with the
  network disabled before it will consider the Hub. Do not set `HF_TOKEN` here
  expecting it to help: an *expired* token is worse than none, because it turns
  an anonymous-readable public repo into a hard 401. That is what took the
  pipeline down on 2026-09-15 and 2026-09-16 (issue #217).

Optional, to raise opencite's limits: `OPENALEX_API_KEY`,
`SEMANTIC_SCHOLAR_API_KEY`, `PUBMED_API_KEY`.
`OPENCITE_DISABLED_SOURCES` turns a source off without a code change.

## Safety Properties

- **Lock.** `flock` on `.cron.lock`; a concurrent invocation logs a skip and
  exits 0 rather than clobbering the run in progress.
- **Disposable tree.** Each run starts with `git reset --hard origin/main`, so
  the working tree is assumed disposable between runs. A run that aborts before
  its commit loses its work entirely and starts over the next night.
- **Guarded steps.** The script uses `set -uo pipefail` without `-e`, so every
  step carries an explicit `|| { exit 2; }`. Cleanup steps (`prune-mirrored`,
  `dedupe`, the export steps) warn instead, since they cannot corrupt anything
  downstream.
- **No-op exit.** If nothing tracked changed, the script exits 0 before
  committing, which is the normal outcome on a quiet night.

## Logs

On hallu:

```bash
ls -lt ~/dataset_citations/.logs/          # one cron-<timestamp>.log per run
grep -n '^--- \|^ERROR\|FAILED' ~/dataset_citations/.logs/cron-<ts>.log
```

Each step announces itself as `--- <step> ---`, so the first `ERROR` after the
last such line identifies where a run died.

**Byte-identical logs across nights mean the pipeline is wedged**, not idle: the
hard reset makes a deterministic failure reproduce exactly, so the same work is
redone and fails the same way every night. Compare sizes with `ls -l` before
reading.

CI runs are at https://github.com/nemarOrg/nemar-citations/actions.

## Monitoring

- Merged `auto-update/<timestamp>` PRs.
  A gap longer than a day or two means the cron is aborting before its commit,
  or its PRs are not merging (see Troubleshooting).
- `dashboard.nemar.org/citations/` for the deployed result.

## Troubleshooting

**No auto-update PR for several days.** Read the most recent log, find the last
`--- <step> ---` before the error. Known wedges:

- `dataset-citations-update` exiting 3 means every processed dataset returned a
  transient API-failure status. If the same small set of datasets is processed
  every night, check whether their statuses are genuinely transient rather than
  permanent properties of their anchor DOIs (issue #217).
- A model-loading failure in `score-confidence` should no longer be possible
  from a credential problem; if one appears, confirm the checkpoints are still
  in `~/.cache/huggingface/hub/`.

**Judge unavailable.** `judge-anchors` runs one real judgment as its health
check and exits 2 if the `claude` CLI is missing, logged out, or rejects the
model. It also exits 2 when 10 judgments fail in a row, a sidecar cannot be
written, or the judge, opencite, or the anchor source fails on more than 10%
(`--max-failure-share`) of its fresh calls (anchors failing again as on their
previous run do not count); the cron then stops before `update`, so yesterday's data stays live.
Fix the login (`claude` on hallu) and wait for the next night, or run the
script by hand. Even if a judgment is missing, nothing inflates: the anchor
gate fails closed, so an unjudged anchor never contributes citations (issue
#241). A dataset that lost citations to a judgment failed below those
thresholds recovers on its own: once the anchor is judged a data paper, the
next `update` refetches that dataset without waiting for its 7-day window.

**Gate sweep aborted.** `gate-anchors` writes nothing and exits 1 when a
citation file or a sidecar is unreadable (a merge conflict, a truncated
write), when `--judgments-dir` is missing or empty, or when more than half of
the datasets that need a judgment have no sidecar from the trusted judge (the
judge did not run, or ran as another model). Repair the file, or confirm the
judge ran, then rerun; `--max-missing-share 1` overrides the last check for a
deliberate run.
**A nightly PR is open but not merging.**
Its CI failed, or the cron could not enable auto-merge
(the log shows `ERROR: could not enable auto-merge`).
Fix the cause on `main`; the PR does not need rescuing.
Each nightly run starts from `main`, so the next night's PR carries the full state of a fresh run,
and once it is open the cron closes every older open nightly PR as superseded.
That cleanup runs only on a night that reaches PR creation,
so an aborted night or one with no data changes leaves older PRs open.
Branches not in the cron's `auto-update/<UTC timestamp>` form are never closed.

**Exit codes.** The script exits 0 on success, when there is nothing to commit, or when another run holds the lock.
It exits 2 when a pipeline step, the push, or PR creation fails,
and also when tonight's PR is open but enabling auto-merge or closing a superseded PR failed;
in that case it finishes the cleanup first, and the last log line says `published ... with errors`.

**Ollama unreachable.** The preflight aborts with exit 2 before any judging, so
no run publishes stale judgments. Restart the daemon and wait for the next
night, or run the script by hand.

**Manual run.** Safe to invoke directly; the lock prevents overlap with cron:

```bash
ssh hallu '~/dataset_citations/scripts/hallu_cron_pipeline.sh'
```

Mind the rate-limit budget: a full pass over the corpus costs roughly two days
of the OpenAlex accession-search allowance, so avoid re-running it casually.

## Schedule Customization

Edit the crontab entry on hallu (`crontab -e`). The script itself takes no
schedule argument.
