# Automated Citation Updates

## Overview

The pipeline runs nightly on **hallu** (the GPU host), not in GitHub Actions.
`scripts/hallu_cron_pipeline.sh` is the single producer of citation data,
embeddings and analysis outputs.
CI only consumes what that script commits.

```
hallu cron (03:00 PDT, nightly)
  -> discover -> retrieve-metadata -> judge-anchors -> update (opencite fetch)
  -> prune-mirrored -> find-mentions -> dedupe -> score-confidence
  -> generate-embeddings -> analyze-umap -> themes / network / temporal
  -> commit to auto-update/<timestamp> -> PR -> auto-merge on green CI
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
- A reachable Ollama daemon for anchor adjudication, probed before the judging
  step. Override the URL with `OLLAMA_BASE_URL` and the checkpoint with
  `OLLAMA_MODEL` (default `gemma4:e4b`; `gemma4:31b` discriminates better but
  OOMs on the shared host).
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

- Merged `auto-update/<timestamp>` PRs. A gap longer than a day or two means the
  cron is aborting before its commit.
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
