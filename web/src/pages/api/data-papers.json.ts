/**
 * Data-paper manifest, emitted as a static file at build time:
 *   dashboard.nemar.org/citations/api/data-papers.json
 *
 * One row per dataset whose anchors the pipeline's gate has finished with: the
 * papers the trusted judge called the dataset's data paper (an empty list when
 * there is none). nemar-cli pulls this daily, as it does the counts manifest,
 * and serves it as `data_papers` in data.nemar.org/<id>/metadata.json (issue
 * #250). A dataset the gate has not finished with is omitted, so the consumer
 * keeps "not judged". The rules live in lib/data-papers.ts.
 */
import type { APIRoute } from "astro";
import { loadAll, loadDataPapers } from "../../lib/data";
import { DATA_PAPERS_SCHEMA } from "../../lib/data-papers";

export const GET: APIRoute = () => {
  const { overview } = loadAll();
  const body = {
    schema: DATA_PAPERS_SCHEMA,
    last_updated: overview.lastUpdated,
    datasets: loadDataPapers(),
  };
  return new Response(JSON.stringify(body), {
    headers: { "Content-Type": "application/json" },
  });
};
