# Cross-Repo Contracts (NEMAR triangle)

This repo (`dataset_citations`) ships citation data and a dashboard. Two sibling repos jointly own the surrounding system; they live under `github.com/nemarOrg/` (locally: `/Users/yahya/Documents/git/nemar/`).

| Repo | Surface | Role |
|---|---|---|
| `nemar-cli` | `api.nemar.org`, `data.nemar.org`, `.nemar/metadata.json` | Catalog + manifest + LLM enrichment |
| `website` | `nemar.org` | Astro 6 SSR frontend |
| `dataset_citations` (here) | `dashboard.nemar.org/citations/`, `citations/json_opencite/` | Citation discovery, scoring, dashboard |

## The `.nemar/metadata.json` contract
Each NEMAR-managed dataset repo (`github.com/nemarDatasets/<id>/`) carries `.nemar/metadata.json` at the root. This file is the authoritative DOI source for citation work.

- **Schema:** `NemarMetadataV2` — defined in `nemar-cli/shared/datacite-constants.ts:160-185`.
- **Producer:** `nemar-cli/backend/src/services/enrich-dataset.ts` (LLM enrichment job, committed via PR to the dataset repo).
- **Consumer:** `src/dataset_citations/sources/nemar_metadata.py:83-125` (`parse_nemar_metadata()`).

**DOI-bearing fields we consume:**
- `related_identifiers[]` — array of `{ identifier, identifier_type, relation_type }`.
  - `identifier_type` accepted: `"DOI"` only (PMID / arXiv / URL / handle are skipped by the parser today; widen here if you start needing them).
  - `relation_type` accepted: `References`, `IsDerivedFrom`, `IsIdenticalTo`, `IsVersionOf`, `IsDescribedBy`. These are the DataCite values we surface in citation JSON as `source_relation`. `IsDescribedBy` is how a data paper is linked, but the producer (`nemar-cli`) also uses it for landing pages; those are URL-typed and the parser skips every URL-typed identifier, so only DOI-typed entries become anchors. `IsSupplementedBy` is not read.
- The specific relation-type mix varies per dataset. Reference dataset `nm000104` carries `IsVersionOf`, `IsIdenticalTo`, and `IsDescribedBy` in its live payload; other datasets (e.g. `nm000103`) carry `References` and `IsDerivedFrom`. Don't assume all four accepted types are present on any single dataset.
- OpenNeuro dataset DOIs are deduplicated; do not double-count when a dataset is mirrored under multiple IDs.

**When producer and consumer drift:** treat as a contract break. Open issues in both repos; do not silently widen the parser to accept fields the producer doesn't yet emit.

## api.nemar.org endpoints we depend on
Public, no token required for any of these. Routes are mounted by `nemar-cli/backend/src/index.ts`; see `routes/datasets.ts` and `routes/data.ts` for handlers.

| URL | Returns | Use for |
|---|---|---|
| `GET https://api.nemar.org/datasets` | Full catalog as `{datasets: [...], count, total_count, limit, offset}` (~40KB at default limit, nm-* and on-* IDs). Per-row fields: `dataset_id`, `doi`, `concept_doi`, `source`, `source_id`, `modalities`, `participants`, `tasks`, `github_repo` | **Primary discovery source.** Replaces GitHub-API pagination of `OpenNeuroDatasets/` + `nemarDatasets/`. Use `offset` + `limit` to paginate. |
| `GET https://api.nemar.org/datasets/:id` | Single dataset wrapped as `{"dataset": {...}}` (note the wrap; not the bare row shape returned by the list endpoint), including the version list. | Per-dataset enrichment without cloning the repo. |
| `GET https://api.nemar.org/datasets/resolve/:sourceId` | nm-ID for a given OpenNeuro `ds*` ID | Mapping legacy ds-* to NEMAR-managed nm-*. |
| `GET https://data.nemar.org/<id>/metadata.json` | Per-dataset neuroschema doc — includes `related_identifiers[]` with `{identifier, identifier_type, relation_type}` (DataCite values: `References`, `IsDerivedFrom`, `IsIdenticalTo`, `IsVersionOf`, `IsDescribedBy`, ...) | Per-dataset metadata + citation anchors for nm-* / on-* IDs. Legacy ds-* returns 404 here, fall back to GitHub. |
| `GET https://data.nemar.org/<id>/<version>/manifest.json` | File-level BIDS manifest with presigned S3 URLs | Generally not needed for citations; reference for context. |

There is **no citations endpoint** on the backend today. We do not publish citation JSON back into the D1 catalog. If/when we do, the natural shape is `GET /datasets/:id/citations.json` returning the schema-v2 payload — coordinate with `nemar-cli/backend/src/routes/`.

## Citation JSON contract (we produce)
Path: `citations/json_opencite/<id>_citations.json`. Schema v2 (see `AGENTS.md` § Key Data Formats).

Downstream consumers of this artifact should treat:
- `metadata.schema_version` as the version gate. Bump it on any breaking shape change.
- `metadata.discovery_backend == "opencite"` as confirmation of provenance. Legacy `"scholarly"` files (under `citations/json/`) are read-only history.
- `citation_details[].source_doi` + `source_relation` as required fields; they tell the website which DOI anchor surfaced the citation. `source_relation` is the DataCite label of that anchor and only a hint; whether an anchor contributed citations is `metadata.anchors[].kept` / `kept_reason` (the fail-closed anchor gate, #241).

## Deploy targets
- `dashboard.nemar.org/citations/` — Cloudflare Pages project `nemar-dashboard`. Canonical citation surface. Deployed by `.github/workflows/deploy-dashboard.yml` when the hallu cron's auto-update PR (or a `web/` change) lands on `main`.
- `nemar.org` — Cloudflare Pages project owned by `nemar/website`. Does not embed citation data today.
- `api.nemar.org` / `data.nemar.org` — Cloudflare Worker owned by `nemar/nemar-cli/backend/`.

## Fetch strategy & rate-limit posture
Two of our upstreams have throttled us in production: GitHub (on full-catalog discovery) and Semantic Scholar (on `cited_by` lookups). Default to the NEMAR backend; treat third-party APIs as scarce.

### Dataset / DOI discovery — order of preference
1. **`api.nemar.org/datasets`** — primary. One request gets the full catalog with DOIs.
2. **`data.nemar.org/<id>/metadata.json`** — secondary, when you need richer per-dataset metadata than the catalog row.
3. **Local checkout** at `/Users/yahya/Documents/git/nemar/<repo>/` — development / offline.
4. **GitHub REST** — fallback only, for legacy `ds-*` IDs not yet ingested into the D1 catalog. Use a token; respect `X-RateLimit-Remaining`.

### Citation backends inside opencite — delegated, not hand-routed
Source selection and per-source rate limiting are **delegated to opencite** (>= v0.5.4), not implemented in this repo. `backends/opencite_backend.py` opens one `CitationExplorer` per batch and lets opencite fan the `cited_by(DOI)` lookup across its enabled sources. There is no local per-anchor routing (the old `S2_SKIP_PREFIXES` filter was removed in epic #180, phase 2).

- **OpenAlex** — free, no key required. Set `OPENALEX_API_KEY` to enter the politeness pool. Most reliable for `cited_by(DOI)`; opencite's primary source.
- **Semantic Scholar (S2)** — kept. The May 2026 probe (`.context/research/s2_vs_openalex_2026-05-19.json`) showed S2 adds ~9.71% unique coverage on mainstream journals; per-source numbers refreshed in `.context/research/source_coverage_2026-06-19.json`. S2's only problem is throughput (1 req/s); opencite's **process-wide shared rate limiter** (`shared_limiter_key="s2"`) paces it globally so it no longer 429-storms, which is what the prefix filter used to work around. Set `SEMANTIC_SCHOLAR_API_KEY` to lift it off the unreliable shared pool.
- **PubMed** — opencite has a `PubMedClient.citing_papers` (NCBI `elink`), but its `CitationExplorer` does **not** wire it yet, so it is not in our production fetch path. The per-source probe shows it adds material coverage; wiring it in is tracked **upstream in opencite** (neuromechanist/opencite#48) rather than via a local multi-client merge.

**Disabling a source:** set `OPENCITE_DISABLED_SOURCES` (comma-separated, e.g. `s2`) — opencite skips it everywhere; no code change. Disabling both `openalex` and `s2` makes `CitationExplorer` raise.

### Guardrails (already partly in code; keep enforced)
- `OPENCITE_MAX_CONCURRENCY` — env var, CI sets to 1 (PR #50). Don't raise in CI without sharding.
- `OPENCITE_MAX_RETRIES` — env var, CI sets to 1 (down from opencite's default 3). The retry budget was the main contributor to the May 2026 6h timeout; opencite's shared per-source rate limiter (>= v0.5.4) now caps the worst 429 sources, so one attempt is enough in CI. Local dev keeps the default 3 for single-dataset debugging.
- Per-anchor checkpointing — persist partial progress; resume rather than refetch on retry.
- Sharded backfill — split discovery+fetch into chunks small enough to finish inside a single GitHub Actions job (≤6h). Full catalog at concurrency=1 does **not** fit.
- Batch git operations — don't commit per dataset; one commit per shard, one PR per run.
- Cache the catalog response locally during a CI run; don't refetch `api.nemar.org/datasets` between steps.

## Where the website is *today* vs. where this might go
Today: `website/src/lib/data-api.ts` SSR-fetches from `data.nemar.org` and `api.nemar.org` per request; nothing pulls from `dataset_citations`. The dashboard is a standalone link.

If/when citations are surfaced inside `nemar.org` per-dataset pages, the integration path is:
1. Publish `citations/json_opencite/<id>.json` to a known CDN URL (likely behind `data.nemar.org/<id>/citations.json` via the worker, or directly to a Pages route).
2. Add `getCitations(id)` to `website/src/lib/data-api.ts` mirroring `getMetadata()` / `getManifest()`.
3. Render a per-dataset panel that splits citations by `source_relation` (uses vs related).
4. Schema gate via `metadata.schema_version`.

The reverse direction — the website **producing** anything we consume — is not currently in scope.
