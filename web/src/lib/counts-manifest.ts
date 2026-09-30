/**
 * Rows of the citation-count manifest (api/index.json), kept pure (no file
 * I/O) so the rule is unit-testable.
 *
 * nemar-cli's daily sync only UPDATEs the datasets it finds in this manifest;
 * it never resets one that is missing. The dashboard builds a row only for a
 * dataset that still has a counted or low-confidence citation, so a dataset
 * whose citations all dropped away (an anchor judged not its data paper, the
 * gate tightened) simply vanished from the file and kept its old count in the
 * catalog, served by api.nemar.org and shown on the nemar.org front page. So
 * every catalog-served dataset that has a citation file gets a row here, with
 * explicit zeros when nothing is left to count.
 *
 * Only catalog-served ids (nm and on) get a zero row. Legacy ds* ids are never
 * served, and nemar-cli matches a `ds*` row to its `on*` mirror through
 * `source_id`, so a zero row for a ds* file could overwrite the mirror's real
 * count.
 */
import { SERVED_ID } from "./data-papers";

export const COUNTS_SCHEMA = "nemar-citations/counts@1";

export interface CountsRow {
  dataset_id: string;
  num_citations: number;
  num_dataset_citations: number;
  num_datapaper_citations: number;
}

/** The fields of a dataset's detail the manifest needs. */
export interface CountedDataset {
  id: string;
  numCitations: number;
  numDatasetCitations: number;
  numDataPaperCitations: number;
}

/** A dataset with no counted citation: present in the catalog, nothing to count. */
function zeroRow(id: string): CountsRow {
  return {
    dataset_id: id,
    num_citations: 0,
    num_dataset_citations: 0,
    num_datapaper_citations: 0,
  };
}

function compareText(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

/** One row per dataset in `datasets` (in the order given), then an explicit
 * zero row, by id, for every served id in `fileIds` that has none. */
export function buildCountsRows(
  datasets: readonly CountedDataset[],
  fileIds: readonly string[],
): CountsRow[] {
  const rows: CountsRow[] = datasets.map((d) => ({
    dataset_id: d.id,
    num_citations: d.numCitations,
    num_dataset_citations: d.numDatasetCitations,
    num_datapaper_citations: d.numDataPaperCitations,
  }));
  const listed = new Set(rows.map((r) => r.dataset_id));
  const missing = [...new Set(fileIds)]
    .filter((id) => SERVED_ID.test(id) && !listed.has(id))
    .sort(compareText);
  return [...rows, ...missing.map(zeroRow)];
}
