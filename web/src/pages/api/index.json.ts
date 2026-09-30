/**
 * Citation counts manifest, emitted as a static file at build time:
 *   dashboard.nemar.org/citations/api/index.json
 *
 * One row per dataset with citations, plus an explicit zero row for every
 * catalog-served dataset that has a citation file but nothing left to count
 * (rules in lib/counts-manifest.ts: nemar-cli's sync only updates datasets it
 * finds here, so a dataset missing from this file would keep its old count).
 * nemar-cli ingests this into D1 (citation count columns) so the catalog can
 * order datasets by citation count without loading any per-paper detail
 * (Phase 3, issue #170). The detail lives in the per-dataset endpoints; this
 * manifest stays small (counts only).
 */
import type { APIRoute } from "astro";
import { COUNTS_SCHEMA, buildCountsRows } from "../../lib/counts-manifest";
import { loadAll } from "../../lib/data";

export const GET: APIRoute = () => {
  const { datasets, datasetIds, overview } = loadAll();
  const body = {
    schema: COUNTS_SCHEMA,
    last_updated: overview.lastUpdated,
    datasets: buildCountsRows(datasets, datasetIds),
  };
  return new Response(JSON.stringify(body), {
    headers: { "Content-Type": "application/json" },
  });
};
