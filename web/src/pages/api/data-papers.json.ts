/**
 * Data-paper manifest, emitted as a static file at build time:
 *   dashboard.nemar.org/citations/api/data-papers.json
 *
 * Per catalog-served dataset the gate can back a statement for, the papers the
 * pipeline's trusted judge called its data paper (an empty list when it judged
 * none). nemar-cli will pull this daily, as it does the counts manifest, and
 * serve it as `data_papers` in data.nemar.org/<id>/metadata.json (issue #250).
 * A row replaces the consumer's stored value; a dataset with no row means no
 * statement, so the consumer leaves its stored value untouched. The rules and
 * the description embedded in the file live in lib/data-papers.ts.
 */
import type { APIRoute } from "astro";
import { loadDataPapers } from "../../lib/data";

export const GET: APIRoute = () => {
  return new Response(JSON.stringify(loadDataPapers(new Date())), {
    headers: { "Content-Type": "application/json" },
  });
};
