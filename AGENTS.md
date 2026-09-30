# NEMAR Citations Instructions

## Project Context
**Purpose:** Automated Brain Imaging Data Structure (BIDS) dataset citation tracking system for the NEMAR.org project. Discovers and tracks citations for ~594 neuroscience datasets (the live `api.nemar.org/datasets` total as of 2026-05-18) from OpenNeuro and NEMAR, scores them with AI confidence, and renders an interactive dashboard hosted at `dashboard.nemar.org/citations/`.
**Tech Stack:** Python 3.13+, UV, Ruff, Ty, pytest, sentence-transformers, opencite (OpenAlex / Semantic Scholar / PubMed aggregator), PyGithub, Cloudflare Pages.
**Architecture:** CLI-first package (`dataset_citations.*`) with discovery → DOI extraction → opencite lookup → scoring → analysis → dashboard pipeline. Automated via GitHub Actions.

## Reading the shared documentation

`docs.nemar.org` is the canonical surface for anything about the NEMAR platform rather than about
this pipeline (nemar-cli ADR 0057). Public pages need nothing; fetch the URL. **Every page also
has a Markdown mirror at the same path plus `.md`**, which is what to fetch if you are a program,
and `https://docs.nemar.org/llms.txt` indexes them.

```bash
curl -s https://docs.nemar.org/platform/data-api.md
```

Pages under `/admin/` are gated: `nemarOrg/docs` is private at source and the gate admits the
`admin` and `owner` roles only. An admin holding a NEMAR CLI key reads one without a browser:

```bash
nemar admin docs admin/operations/systems-inventory
```

## If you cannot read something you need

**Open the issue anyway.** Losing read access must not cost you the ability to report a problem
(nemar-cli ADR 0057).

If you hit an admin-adjacent problem, or need a runbook you cannot open, file the issue on the
relevant repository, say plainly what you could not read and what you were trying to do, and tag
**`@nemarOrg/admins`**. Someone with access will either answer or open the page for you.

Escalation replaces read access. **Silence does not.** A blocked agent that stops without saying
so is the failure mode this instruction exists to prevent.

## Related Repositories
Three sibling repos jointly produce the public NEMAR surface. They live under `/Users/yahya/Documents/git/nemar/` locally and under `github.com/nemarOrg/` remotely.

| Repo | Path (local) | Role |
|---|---|---|
| `nemar-cli` | `../nemar/nemar-cli/` | Bun CLI for BIDS dataset upload/version/DOI. Cloudflare Worker backend serves `api.nemar.org` (D1 catalog) + `data.nemar.org` (S3-backed BIDS view). LLM enrichment writes `.nemar/metadata.json` into each dataset repo. |
| `website` | `../nemar/website/` | Astro 6 SSR frontend for `nemar.org`, deployed to Cloudflare Pages. SSR reads dataset metadata at request time from `api.nemar.org` and `data.nemar.org`. No build-time data bundling. |
| `nemar-citations` (this repo) | `../nemar-citations/` | Citation discovery, scoring, analysis, and dashboard. Reads `.nemar/metadata.json` for DOIs; publishes `citations/json_opencite/` and the dashboard at `dashboard.nemar.org/citations/`. |

**Cross-repo contracts** (details: `.rules/cross_repo.md`). The platform side of each is
documented at [`docs.nemar.org/platform/api/`](https://docs.nemar.org/platform/api/) and
[`docs.nemar.org/platform/data-api/`](https://docs.nemar.org/platform/data-api/); what follows is
how THIS repo consumes them:
- `.nemar/metadata.json` schema (`NemarMetadataV2`) — producer: `nemar-cli/backend/src/services/enrich-dataset.ts`; consumer: `src/dataset_citations/sources/nemar_metadata.py`. Authoritative DOI source via `related_identifiers[]` with DataCite relation types.
- `https://api.nemar.org/datasets` — public, no auth, returns full catalog (nm-* and on-* IDs). Preferred over GitHub API for discovery.
- `https://data.nemar.org/<id>/metadata.json` — per-dataset neuroschema. Live for `nm-*` IDs; legacy `ds-*` still requires GitHub-based discovery.
- Citation data reaches nemar-cli by **pull**, never by push. This repo publishes static manifests under `dashboard.nemar.org/citations/api/`: `index.json` (`nemar-citations/counts@1`, per-dataset citation counts) and `data-papers.json` (`nemar-citations/data-papers@1`, the papers the trusted judge called each dataset's data paper, issue #250). nemar-cli's daily Worker cron pulls the counts into D1 (`citation-counts-sync`) and pulls `data-papers.json` the same way to serve `data_papers` in `metadata.json`. Per-dataset detail stays at `citations/api/dataset/<id>.json`. No citations endpoint or credential on the nemar-cli side is needed. Details: `.rules/cross_repo.md`.

## Architecture Map
```
src/dataset_citations/
├── core/        # citation_utils.py (JSON helpers), opencite_pipeline.py (orchestrator), anchor_gate.py (fail-closed anchor gate)
├── sources/     # DOI extractors: nemar_metadata, bids_metadata, doi helpers
├── backends/    # opencite citation backend (sync facade)
├── quality/     # AI confidence scoring, dataset metadata retrieval, anchor judgment (Claude judge, sidecars), never-anchor list
├── cli/         # command-line entry points (see pyproject [project.scripts])
├── graph/       # Network analysis, Neo4j integration
├── embeddings/  # Semantic vector storage, UMAP, registry
├── dashboard/   # Interactive HTML/JS dashboard generation
├── analysis/    # Theme / network / temporal generators
└── utils/       # Shared helpers
tests/           # Real-data tests (NO MOCKS)
.github/workflows/
├── test.yml             # Lint, ruff format check, ty, pytest, integration
├── update_citations.yml # Weekly cron + workflow_dispatch: opencite fetch + dashboard deploy
└── deploy-dashboard.yml # Manual dashboard rebuild + Cloudflare Pages deploy
```

## Environment Setup
```bash
# Bootstrap (one time)
uv sync --all-extras       # Install deps + dev tools from pyproject.toml
uv run pre-commit install  # Enable ruff hook

# Secrets
echo "GITHUB_TOKEN=..."  > .env
# Optional, raise opencite rate limits:
# echo "SEMANTIC_SCHOLAR_API_KEY=..." >> .env
# echo "OPENALEX_API_KEY=..." >> .env
# Optional: .secrets for integration tests

# Anchor adjudication (epic #76, issue #241): the judge is Claude Sonnet 5.5
# through the `claude` CLI in headless mode, so the CLI must be installed and
# logged in wherever judge-anchors runs (hallu runs it as the pipeline user).
# export ANCHOR_JUDGE_MODEL=claude-sonnet-5-5   # tracks llm_client._DEFAULT_MODEL; only
#                                               # sidecars from this model count
# export CLAUDE_BIN=/path/to/claude             # default: `claude` on PATH
# export ANCHOR_JUDGE_TIMEOUT_SECONDS=180       # per-call ceiling; guards a hung CLI
```

## Development Workflow
1. **Check context:** `.context/plan.md` (current tasks), `.context/current_issues.md` (priorities)
2. **Understand deeply:** `.context/architecture.md`, `.context/ideas.md`
3. **Branch:** `gh issue develop <issue-number>` (creates branch off `main` for each issue)
4. **Code:** Follow patterns in `.rules/`
5. **Test:** Real data only (`.rules/testing.md`)
6. **Document failures:** Log in `.context/scratch_history.md`
7. **Commit:** Atomic, conventional prefix (`feat:`, `fix:`, `chore:`, ...), <50 chars, no emojis, no AI attribution
8. **PR:** Reference the issue; ALL CI must be green before merge
9. **Code review:** Run `/review-pr` after creating PR (`.rules/code_review.md`)

## [CRITICAL] Core Principles - Never Compromise

### [FUNDAMENTAL] NO MOCKS - Test Reality Only
- Use real OpenNeuro datasets (subsets are fine; keep small controlled fixtures)
- No mocked GitHub or opencite responses; use real fixtures or skip the test
- If real testing is impossible, no test is better than a fake passing test
**Details:** `.rules/testing.md`

### UV Exclusively
- `uv sync`, `uv run`, `uv add` — never pip, conda, or virtualenv
- pyproject.toml is the single source of truth (no requirements.txt)
**Details:** `.rules/python.md`

### Ruff + Ty
- Format and lint: `uv run ruff format` / `uv run ruff check --fix`
- Type check: `uv run ty check` (replaces mypy)
**Details:** `.rules/python.md`

### Commits & Git
- Atomic commits, focused changes
- Conventional prefix, <50 chars, no emojis, no AI attribution
**Details:** `.rules/git.md`

### No Technical Debt Carried Forward
- Address ALL PR review findings
- Skip only genuine false positives or intentional design choices, with rationale in the PR
**Details:** `.rules/code_review.md`

## [NEVER DO THIS]
- Never use mocks, stubs, or fake data in tests
- Never use `pip`, `conda`, or `virtualenv`; use UV
- Never use `mypy`; use `ty`
- Never use `black` or `isort`; ruff handles both
- Never commit secrets, .env files, or credentials
- Never leave empty catch blocks or silent failures
- Never add backward-compatibility shims; replace directly
- Never add TODO without a linked issue
- Never use emojis in commits, PRs, or code
- Never merge a PR without all CI green

## Key Development Commands
```bash
# Tests (fast)
uv run pytest tests/ -v
uv run pytest --cov=dataset_citations tests/

# Integration test (live opencite, set the gate var to enable)
RUN_INTEGRATION_TESTS=1 uv run pytest tests/test_backends_opencite.py -v

# Live anchor-judge test (real `claude` CLI + opencite; costs about a cent)
RUN_CLAUDE_JUDGE_TESTS=1 uv run pytest tests/test_quality_llm_client.py tests/test_quality_anchor_judgment.py -v

# Dashboard (web/): lint, types, counting-rule tests
cd web && bun run lint && bun run check && bun test

# Lint, format, types
uv run ruff format src/ tests/
uv run ruff check --fix src/ tests/
uv run ty check src/

# Pipeline CLIs (see pyproject [project.scripts] for the full set)
uv run dataset-citations-discover --output-file discovered_datasets.txt
uv run dataset-citations-update --dataset-list-file discovered_datasets.txt --output-dir citations/
uv run dataset-citations-retrieve-metadata --citations-dir citations/json_opencite --output-dir datasets
uv run dataset-citations-gate-anchors --citations-dir citations/json_opencite  # re-apply the anchor gate offline (#241); --dry-run to preview
uv run dataset-citations-prune-mirrored --citations-dir citations/json_opencite  # ds->on dedup (#126); --dry-run to preview
uv run dataset-citations-score-confidence --citations-dir citations/json_opencite --datasets-dir datasets
```

## Data Flow
1. **Discovery**: Prefer `https://api.nemar.org/datasets` (D1 catalog, no auth, ~40KB for the full list of 594 datasets) for both nm-* and on-* (NEMAR-imported OpenNeuro) IDs. One request gives every dataset's `dataset_id`, `doi`, `concept_doi`, `source`, `source_id`, `github_repo`, and modality/task/author metadata. Legacy ds-* IDs not yet in the catalog still come from GitHub via `cli/discover.py`.
2. **DOI extraction**: For nm-* / on-*, fetch `https://data.nemar.org/<id>/metadata.json` (the same neuroschema doc the worker generates from `.nemar/metadata.json` + D1 enrichment) and parse `related_identifiers[]` via `sources/nemar_metadata.py`. Confirmed shape: `{ identifier, identifier_type, relation_type }` with DataCite relation values. For legacy ds-*, fall back to `dataset_description.json` via `sources/bids_metadata.py`. The dataset's **own** concept DOI (from the catalog `concept_doi` field, `10.82901/nemar.<id>`; it always resolves to the latest version) is also a citation anchor with `relation_type = References`; its citers land in the "cites dataset" bucket. Relation types we consume: `References`, `IsDerivedFrom`, `IsIdenticalTo`, `IsVersionOf`, `IsDescribedBy` (how a data paper is linked, e.g. `10.1038/sdata.2015.1` describes `on000117`), and `IsSupplementTo` (how older enrichments linked some data papers, e.g. `on007615`). The relation type is a hint, not the decision: relation labels are noisy in both directions (data papers labeled `References`, BIDS papers labeled `IsDescribedBy`), so the anchor judgment and the gate decide. The one exception is an identity relation (`IsIdenticalTo` / `IsVersionOf`: another record of the same data, such as a figshare or PhysioNet deposit), which the gate keeps when no judgment exists.
3. **Anchor judgment + citation fetching** (epic #76):
   - **3a. Anchor judgment**: `dataset-citations-judge-anchors` classifies each anchor DOI as `data_paper` / `umbrella` / `methodology` / `related_work` / `irrelevant` with Claude Sonnet 5.5 through the `claude` CLI (`quality/llm_client.ClaudeCliJudgmentClient`, model via `ANCHOR_JUDGE_MODEL`); opencite resolves each anchor's title / abstract / year first. Writes per-dataset sidecars to `citations/anchor_judgments/<id>.json`. A health check runs one real judgment (retried) and exits 2 if the CLI is missing, logged out, or the model is unknown. The run also exits 2 when 10 judge calls fail in a row (it stops there, leaving the remaining sidecars untouched), when a sidecar write fails, or when the judge, opencite lookups, or the anchor source fail on at least 5 fresh attempts and more than `--max-failure-share` (default 10%) of them. Fresh means an anchor that did not fail the same way on its previous run, so a few anchors that always fail cannot stop the cron. The backend reports only an OpenAlex 404 as `not_found`: a first one counts as a failed lookup, and a repeat is retried on a 30-day back-off without counting. A source-lookup failure leaves the existing sidecar untouched; a verdict from the same model and prompt version under the same relation is reused (and kept when a re-judge fails); a sidecar from another model or prompt version is re-judged; anchors are not judged blind when the dataset description is unavailable. `--skip-existing` skips a dataset only when its sidecar covers every recorded anchor under the same relation, so an enrichment relabel is re-judged. (The Ollama/Gemma judge was retired in #241 after hallu lost its models and every judgment silently errored.)
   - **3b. Anchor gate + citation fetching**: `core/anchor_gate.py` decides which anchors contribute citations, and it FAILS CLOSED: only the dataset's own concept DOI, unjudged identity-relation records, and anchors with a successful `data_paper` judgment from the trusted judge model are fetched, and never an anchor on the never-anchor list (`quality/never_anchor_dois.json`: the BIDS family, MNE, EEGLAB, FieldTrip, Brainstorm, PREP, ICLabel, fMRIPrep, FreeSurfer, FSL, Nipype, OpenNeuro, NEMAR, HED, and the Healthy Brain Network (HBN) umbrella paper; plus a BIDS-spec title rule). Other unjudged or errored anchors, and verdicts from any other judge model, are context only. `core/opencite_pipeline.py` applies it at fetch time and records every anchor in `metadata.anchors[]` with `kept` and `kept_reason`; a citing work published more than a year before the anchor it came through is skipped, and a citing work that is itself a NEMAR or OpenNeuro dataset record (the dataset, its OpenNeuro mirror, or a sibling release) is never counted. An unreadable sidecar fetches nothing (`fetch_status="judgment_unreadable"`), and `update` never writes that or an API-failure stub over an existing file. `backends/opencite_backend.py` (sync facade over opencite) then delegates the `cited_by` lookup to opencite's `CitationExplorer` for the surviving anchors (the own concept DOI, judged data papers, and identity records); opencite selects sources (OpenAlex + S2 today, via `config.disabled_sources`) and applies its per-source shared rate limiter. Concurrency throttled by `OPENCITE_MAX_CONCURRENCY` (CI sets to 1, see PR #50).
4. **Processing**: `core/opencite_pipeline.py` deduplicates across anchors and produces schema-v2 citation JSON.
4a. **Gate sweep** (#241): `dataset-citations-gate-anchors` re-applies the anchor gate offline to every citation file right after `update`, so a judgment that removes an anchor lands the same night rather than after the 7-day refetch window. It only removes: an anchor that newly qualifies is recorded `kept_reason="awaiting_fetch"`, and `update` refetches any dataset with such an anchor inside its freshness window. Files written before schema 2.1 have their anchors rebuilt from `source_doi` and are gated too. Accession mentions (and citations that also name the accession) always stay. It writes nothing and exits 1 when a citation file or sidecar is unreadable, the judgments directory is missing or empty, or more than half of the datasets that need a judgment have no sidecar from the trusted model. Fatal in the cron.
4b. **ds->on dedup** (epic #180 phase 4, #126): `dataset-citations-prune-mirrored` removes stale `ds<N>_citations.json` files whose dataset has been mirrored into NEMAR as an `on<N>` (the on-* catalog row's `source_id` is the `ds-*` accession). It only deletes a `ds-*` file when its `on-*` mirror file is also present, so no coverage is lost. Runs in the hallu cron after `update` and before `find-mentions`; `core/ds_on_dedup.py` holds the pure logic. New duplicates are not generated (the catalog replaces the ds-* row with the on-* row, so `discover --source catalog` emits only the on-*); this sweeps leftovers produced before a dataset was mirrored.
5. **Quality scoring**: sentence-transformer similarity between dataset metadata and citation abstract. Runs on hallu's RTX 4090 via `scripts/hallu_cron_pipeline.sh` (epic #76).
6. **Embeddings**: `dataset-citations-generate-embeddings` writes per-dataset and per-citation sentence-transformer embeddings to `embeddings/` (registry-backed, deduped by content hash). Produced on hallu's RTX 4090 next to step 5 (phase 2 of epic #96 / #98); CI no longer runs the embedding step because CPU torch was ~10x slower than CUDA.
7. **Analysis**: network, temporal, theme, and UMAP analyses run on hallu next to the embedding step (epic #96 / #99 wired UMAP via `dataset-citations-analyze-umap`; CI no longer runs the heavy analysis). Outputs land in `dashboard_data/` (themes / network / temporal subdirs + `*similarities*.csv` flat for UMAP) so the deploy workflow consumes them on a fresh checkout.
8. **Dashboard**: interactive HTML + D3 built by `deploy-dashboard.yml` against the hallu-produced inputs and deployed to Cloudflare Pages at `dashboard.nemar.org/citations/`. CI verifies the expected `dashboard_data/` + `embeddings/` shape before deploy (epic #96 / #101); a missing input fails the run cleanly so the previous deploy stays live. The build counts a citation only when its source anchor is kept (`web/src/lib/gate.ts` honors each anchor's `kept` / `kept_reason`, and the never-anchor list always excludes); a file the sweep has not processed falls back to heuristics, and the build warns.

## Fetch Strategy & Rate Limits
Both upstream sources we depend on have throttled us in production. Treat external APIs as scarce; cache, retry, and prefer the NEMAR backend.

**Order of preference for dataset / DOI discovery:**
1. `api.nemar.org/datasets` (D1, public, generous limits — primary).
2. `data.nemar.org/<id>/metadata.json` (S3-backed, nm-* IDs only today — secondary).
3. GitHub API on `nemarDatasets/` and `OpenNeuroDatasets/` (`GITHUB_TOKEN`, 5000/hr, hits ceiling on full reindex — fallback only).
4. Local checkout under `~/Documents/git/nemar/` for development (offline).

**Citation backends inside opencite — delegated to opencite (>= v0.5.4):**
`backends/opencite_backend.py` opens one `CitationExplorer` per batch and lets opencite select sources and rate-limit them. No per-anchor routing lives here.
1. **OpenAlex** — free, no key required but `OPENALEX_API_KEY` raises the politeness pool. Most reliable for `cited_by` against DOIs; opencite's primary.
2. **Semantic Scholar (S2)** — kept for its ~9.71% unique coverage on mainstream journals. Its only issue is throughput (1 req/s); opencite's process-wide shared rate limiter paces it so it no longer 429-storms. `SEMANTIC_SCHOLAR_API_KEY` lifts it off the shared pool.
3. **PubMed** — `PubMedClient.citing_papers` exists but `CitationExplorer` does not wire it yet, so it is not in the production fetch path. The per-source probe (`.context/research/source_coverage_2026-06-19.json`) shows material coverage; wiring it in is tracked upstream (neuromechanist/opencite#48), not locally.
- **Disable a source** with `OPENCITE_DISABLED_SOURCES` (comma-separated); opencite honors it everywhere with no code change.

**Guardrails to keep in code (not aspirational — already partially in place):**
- Respect `OPENCITE_MAX_CONCURRENCY` and `OPENCITE_RATE_LIMIT_*` env vars in the backend facade.
- Persist partial progress per anchor; resume on next invocation rather than refetch.
- Shard discovery and fetch into chunks that fit GitHub Actions' 6h job limit (the May-18 full-catalog run was cancelled at 6h with concurrency=1).
- Avoid push-on-every-dataset; batch commits to the auto-update branch.

## Key Data Formats
- **Citation JSON** (`citations/json_opencite/`), the canonical output. Schema v2.1 (epic #180): top-level `dataset_id`, `num_citations`, `date_last_updated`, `citation_details[]`, `metadata.{schema_version="2.1", discovery_backend="opencite", fetch_status, anchor_count, anchor_errors, anchor_judgment_model, anchors[], searched_dois[], keywords[], methods_description, funding[], ...}`; per-citation `source_doi` + `source_relation` (one of `References`, `IsDerivedFrom`, `IsIdenticalTo`, `IsVersionOf`, `IsDescribedBy`, `IsSupplementTo`). `metadata.anchors[]` (replaces the v2.0 `context_anchors[]`) is the self-describing list of EVERY anchor, kept or not; each entry: `identifier`, `identifier_type`, `source_relation`, `classification` (the judge's verdict; null if unjudged), `kept` (bool: fetched for citations vs context-only), `kept_reason` (`own_doi` / `dataset_record` / `judged_data_paper` / `judged_not_data_paper` / `never_anchor` / `unjudged` / `awaiting_fetch`), `paper_title`, `paper_year`, `paper_venue`, `reason`, `judgment_model`. Everything except the own DOI, identity records, and judged data papers is `kept=false`; `anchor_count` is the number of kept anchors. `metadata.searched_dois[]` is the flat list of DOI anchors actually fetched (the kept DOIs); it is the citation analog of `metadata.searched_accessions[]` (written later by `cli/find_mentions.py` for the accession full-text search). The two are disjoint provenance keys; DOI anchors and accession-mention search never mix. `metadata.keywords[]` / `methods_description` / `funding[]` are pulled from the GitHub `.nemar/metadata.json` (schema 2.0; the freshest copy, since data.nemar.org ships stale tag-versioned data and omits `methods_description`); empty/null for legacy ds-* datasets.
- **Anchor judgments** (`citations/anchor_judgments/<id>.json`): phase 2 judge sidecar with `dataset_id`, `judged_at`, `judgment_model`, `prompt_version`, `judgments[]` (each: `anchor_identifier`, `anchor_identifier_type`, `source_relation`, `classification`, `reason`, `paper_title`, `paper_year`, `paper_venue`, `judged_at`, `error`). Read by `quality/anchor_judgment_io.load_judgment_sidecar`.
- **Dataset metadata** (`datasets/`): GitHub-sourced descriptions.
- **Embeddings** (`embeddings/`): semantic vectors + registry.

## [REFERENCE] Rules Directory
### Core Standards
- `.rules/testing.md` — NO MOCKS policy
- `.rules/self_improve.md` — Learning from projects
- `.rules/documentation.md` — Docs standards
- `.rules/code_review.md` — PR review toolkit and checklist

### Language & Tools
- `.rules/python.md` — UV, ruff, ty
- `.rules/ci_cd.md` — GitHub Actions setup
- `.rules/git.md` — Branch + commit conventions

### Cross-Repo
- `.rules/cross_repo.md` — Contracts with `nemar-cli` and `website`; fetch strategy; rate-limit posture

### MCP Tools
- `.rules/serena_mcp.md` — Code intelligence with Serena MCP

## Context Files
- `.context/plan.md` — Current tasks and phases (gitignored copy lives in plan.md at root for personal notes)
- `.context/research.md` — Technical explorations
- `.context/ideas.md` — Design concepts
- `.context/scratch_history.md` — Failed attempts and lessons
- `.context/current_issues.md` — Open GitHub issues and priorities
- `.context/architecture.md` — System architecture
- `.context/testing_strategy.md` — Test approach with controlled datasets
- `.context/development_plan.md` — Phased development roadmap

## CI/CD
- `.github/workflows/test.yml`: lint (ruff format/check, ty), pytest 3.13, integration tests, and the dashboard's biome lint, astro type check, and `bun test`
- `.github/workflows/update_citations.yml`: weekly cron (Sunday 06:00 UTC) plus `workflow_dispatch`. Fetches citations via opencite, regenerates the dashboard, deploys to Cloudflare Pages, opens PR.
- `.github/workflows/deploy-dashboard.yml` — Builds the HTML from already-produced `dashboard_data/` + `embeddings/` and deploys to Cloudflare Pages. The heavy analysis steps (themes, network, temporal, embeddings, UMAP) now run on the hallu GPU host via `scripts/hallu_cron_pipeline.sh` (epic #96, phases 2 + 3); CI just consumes the artifacts the hallu auto-update PR commits. The workflow fires on `push` to main when `citations/json_opencite/**`, `datasets/**`, `dashboard_data/**`, or `embeddings/**` change, and has a fail-fast guard that refuses to deploy if any required input directory is missing or empty (the previous Cloudflare Pages deploy stays live in that case).
- Required secrets: `GITHUB_TOKEN`, `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`. Optional: `OPENALEX_API_KEY` (politeness pool), `SEMANTIC_SCHOLAR_API_KEY` (lifts S2 off the shared rate-limit pool), `PUBMED_API_KEY` (raises NCBI E-utilities to 10 req/s).
- Sole citation source (epic #180 phase 5): `citations/json_opencite/` is now populated (the opencite backfill produced data) and is the ONLY citation store. The legacy `citations/json/` (scholarly-format) and `citations/pickle/` directories, the `dataset-citations-migrate` CLI, and the pre-opencite `save_citation_json` / `create_citation_json_structure` write path were removed; the dashboard aggregator reads one source and `DiscoveryBackend` is `opencite`-only. Six legacy datasets that no longer exist in the live NEMAR catalog (no opencite coverage by any path) were dropped with the legacy store; their records remain in git history.

## Project-Specific Guidelines
- **Domain:** NEMAR / OpenNeuro / BIDS / Hierarchical Event Descriptors (HED)
- **Writing style:** No em-dashes (use commas or semicolons); spell out acronyms on first use
- **Performance:** Dashboard payload was reduced from 18MB to 106KB — keep it optimized
- **External:** `dashboard.nemar.org/citations/` is the public-facing URL (Cloudflare Pages project `nemar-dashboard`)

---
Maintainable systems, not just code. Check `.rules/` for detailed guidance on any topic.
