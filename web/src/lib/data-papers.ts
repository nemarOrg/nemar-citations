/**
 * Data-paper manifest (issue #250): which papers the pipeline treats as a
 * dataset's data paper, for nemar-cli to serve as `data_papers` in
 * data.nemar.org/<id>/metadata.json.
 *
 * Pure (no file I/O), so the rules are unit-testable. The verdict is the
 * pipeline's anchor gate's, read from each file's `metadata.anchors[]`
 * (`kept` plus `kept_reason`, see gate.ts): the same verdict that decides which
 * citations count, so the list and the counts cannot disagree. Only anchors the
 * trusted judge called `data_paper` and the gate kept are listed. The dataset's
 * own DOI (`own_doi`), identity records (`dataset_record`), and every dropped
 * anchor are not. An entry may be a deposit of the same data (a figshare or
 * Zenodo record): the judge's `data_paper` class includes those, so there is no
 * type field and the list is judge-confirmed only.
 *
 * Consumer semantics. A row always replaces the consumer's stored value; a
 * dataset with no row means "no statement", and the consumer leaves its stored
 * value untouched. A row's list holds the judge-confirmed papers known so far,
 * and an empty list means "judged, no data paper". So a dataset gets a row only
 * when the gate can back it:
 *  - at least one kept judged data paper exists: list them, even if other
 *    anchors are still unjudged (true but possibly incomplete beats silence);
 *  - none exists and the gate decided every anchor (own DOI, identity record,
 *    never-anchor, or judged not a data paper): an empty list;
 *  - otherwise (no anchors, an anchor unjudged, awaiting its fetch, or without
 *    a `kept_reason` because the gate sweep has not run): no row.
 * Only catalog-served ids (nm and on) get a row; legacy ds* ids are never
 * served, so a row for one would be pointless.
 */
import { type RawAnchor, normalizeDoi } from "./gate";

export const DATA_PAPERS_SCHEMA = "nemar-citations/data-papers@1";

/** `kept_reason` of an anchor the trusted judge called the dataset's data paper. */
export const KEPT_DATA_PAPER = "judged_data_paper";

/** `kept_reason` of an anchor that newly qualifies but whose citers are not
 * fetched yet (core.anchor_gate.AWAITING_FETCH). */
export const AWAITING_FETCH = "awaiting_fetch";

/** `kept_reason` values by which the gate decided an anchor is not a data paper
 * to list: the dataset's own DOI, an identity record, a never-anchor paper, or
 * a paper the judge called something else. Anything else (`unjudged`,
 * `awaiting_fetch`, a missing reason) means the gate has not decided. */
const DECIDED_NOT_LISTED = new Set([
  "own_doi",
  "dataset_record",
  "never_anchor",
  "judged_not_data_paper",
]);

/** Catalog-served dataset ids: NEMAR-managed (nm) and NEMAR-imported OpenNeuro
 * (on). Legacy ds* ids are never served, only redirected when mirrored. */
export const SERVED_ID = /^(?:nm|on)\d{6}$/;

/** Trailing version marker (".v1", ".v1.0.0", "/v2"). Mirrors
 * core.citation_identity._VERSION_SUFFIX. */
const VERSION_SUFFIX = /(?:\.v\d+(?:\.\d+)*|\/v\d+(?:\.\d+)*)$/i;

/** Shown in the manifest itself, so a reader of the file sees the contract. */
export const DATA_PAPERS_DESCRIPTION =
  "Papers the pipeline's trusted judge classified as each dataset's data paper, after the fail-closed anchor gate. " +
  "A row always replaces the consumer's stored value: data_papers holds the judge-confirmed papers known so far, and an empty list means judged with no data paper. " +
  "A dataset with no row means no statement, so the consumer leaves its stored value untouched. " +
  "Entries may be deposits of the same data (for example a figshare or Zenodo record), so there is no type field. " +
  "DOIs are the pipeline's lowercase canonical form; compare them case-insensitively.";

export interface DataPaper {
  doi: string;
  title: string | null;
  year: number | null;
  venue: string | null;
  judge_model: string | null;
}

export interface DataPapersRow {
  dataset_id: string;
  data_papers: DataPaper[];
}

export interface DataPapersManifest {
  schema: string;
  /** When this manifest was built (ISO 8601), not when the data last changed. */
  last_updated: string;
  description: string;
  datasets: DataPapersRow[];
}

export interface AnchorEntry {
  id: string;
  anchors: RawAnchor[];
}

/** A DOI for display: resolver prefix and trailing punctuation removed. The
 * case is left as the pipeline wrote it, which is its lowercase canonical form,
 * so consumers must compare DOIs case-insensitively. */
export function cleanDoi(doi: string): string {
  return doi
    .trim()
    .replace(/^doi:\s*/i, "")
    .replace(/^https?:\/\/(?:dx\.)?doi\.org\//i, "")
    .replace(/[.,;:]+$/, "");
}

/** Normalized DOI with trailing version suffixes collapsed to the concept DOI,
 * applied repeatedly (`....v1.0.0` reduces fully). Mirrors
 * core.citation_identity.base_doi. */
export function baseDoi(doi: string): string {
  let text = normalizeDoi(doi);
  let previous = "";
  while (previous !== text) {
    previous = text;
    text = text.replace(VERSION_SUFFIX, "");
  }
  return text;
}

/** The version a DOI ends in as numbers (`.v5` is [5], `.v1.0.0` is [1, 0, 0]),
 * or null for a version-less DOI. */
function versionOf(doi: string): number[] | null {
  const match = VERSION_SUFFIX.exec(normalizeDoi(doi));
  if (!match) {
    return null;
  }
  return match[0]
    .replace(/^(?:\.v|\/v)/i, "")
    .split(".")
    .map(Number);
}

function compareText(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

function compareVersions(a: number[], b: number[]): number {
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    const diff = (a[i] ?? 0) - (b[i] ?? 0);
    if (diff !== 0) {
      return diff;
    }
  }
  return 0;
}

/** Which of two DOIs of one work to keep: the version-less one, else the
 * highest version (v10 over v9), else the smaller text, whatever the input
 * order. Negative when `a` is the better one. */
function compareVariants(a: string, b: string): number {
  const va = versionOf(a);
  const vb = versionOf(b);
  if (va === null && vb !== null) {
    return -1;
  }
  if (va !== null && vb === null) {
    return 1;
  }
  if (va !== null && vb !== null) {
    const diff = compareVersions(vb, va);
    if (diff !== 0) {
      return diff;
    }
  }
  return compareText(normalizeDoi(a), normalizeDoi(b)) || compareText(a, b);
}

const ENTITIES: Array<[RegExp, string]> = [
  [/&lt;/g, "<"],
  [/&gt;/g, ">"],
  [/&quot;/g, '"'],
  [/&#39;/g, "'"],
  // Last, so "&amp;lt;" decodes once to "&lt;" and not twice to "<".
  [/&amp;/g, "&"],
];

/** Title and venue text as the registries leave it: basic HTML entities
 * decoded, trimmed, null when nothing is left. */
function text(value: string | null | undefined): string | null {
  if (typeof value !== "string") {
    return null;
  }
  let out = value;
  for (const [pattern, replacement] of ENTITIES) {
    out = out.replace(pattern, replacement);
  }
  out = out.trim();
  return out === "" ? null : out;
}

function isDoiAnchor(anchor: RawAnchor): boolean {
  return (anchor.identifier_type ?? "doi").toLowerCase() === "doi";
}

function merge(a: DataPaper, b: DataPaper): DataPaper {
  const [keep, other] = compareVariants(a.doi, b.doi) <= 0 ? [a, b] : [b, a];
  return {
    doi: keep.doi,
    title: keep.title ?? other.title,
    year: keep.year ?? other.year,
    venue: keep.venue ?? other.venue,
    judge_model: keep.judge_model ?? other.judge_model,
  };
}

/** The data papers of one dataset's anchors, or null when the gate cannot back
 * a statement (see the file header). `neverAnchors` is a backstop: a
 * never-anchor DOI is dropped, and counts as decided, even if a stale file
 * still says `judged_data_paper`. */
export function dataPapersOf(
  anchors: RawAnchor[],
  neverAnchors: ReadonlySet<string>,
): DataPaper[] | null {
  const byBase = new Map<string, DataPaper>();
  let undecided = false;
  for (const anchor of anchors) {
    const reason = typeof anchor.kept_reason === "string" ? anchor.kept_reason : "";
    if (reason !== KEPT_DATA_PAPER) {
      if (!DECIDED_NOT_LISTED.has(reason)) {
        undecided = true;
      }
      continue;
    }
    if (!anchor.identifier || !isDoiAnchor(anchor)) {
      // A judged data paper that is not a DOI cannot be listed, so any list
      // would be wrong in a way the manifest cannot express.
      return null;
    }
    if (neverAnchors.has(normalizeDoi(anchor.identifier))) {
      continue;
    }
    if (anchor.kept !== true) {
      undecided = true;
      continue;
    }
    const doi = cleanDoi(anchor.identifier);
    if (!doi) {
      undecided = true;
      continue;
    }
    const paper: DataPaper = {
      doi,
      title: text(anchor.paper_title),
      year:
        typeof anchor.paper_year === "number" &&
        Number.isInteger(anchor.paper_year) &&
        anchor.paper_year > 0
          ? anchor.paper_year
          : null,
      venue: text(anchor.paper_venue),
      judge_model: anchor.judgment_model?.trim() || null,
    };
    const key = baseDoi(doi);
    const existing = byBase.get(key);
    byBase.set(key, existing ? merge(existing, paper) : paper);
  }
  if (byBase.size > 0) {
    return [...byBase.values()].sort((a, b) =>
      compareText(normalizeDoi(a.doi), normalizeDoi(b.doi)),
    );
  }
  return anchors.length === 0 || undecided ? null : [];
}

/** Manifest rows: catalog-served datasets by id, omitting those the gate cannot
 * back a statement for. */
export function buildDataPapersRows(
  entries: AnchorEntry[],
  neverAnchors: ReadonlySet<string>,
): DataPapersRow[] {
  const rows: DataPapersRow[] = [];
  for (const { id, anchors } of entries) {
    if (!SERVED_ID.test(id)) {
      continue;
    }
    const dataPapers = dataPapersOf(anchors, neverAnchors);
    if (dataPapers !== null) {
      rows.push({ dataset_id: id, data_papers: dataPapers });
    }
  }
  return rows.sort((a, b) => compareText(a.dataset_id, b.dataset_id));
}

/** The manifest. `builtAt` is passed in so tests stay deterministic. */
export function buildDataPapersManifest(
  entries: AnchorEntry[],
  neverAnchors: ReadonlySet<string>,
  builtAt: Date,
): DataPapersManifest {
  return {
    schema: DATA_PAPERS_SCHEMA,
    last_updated: builtAt.toISOString(),
    description: DATA_PAPERS_DESCRIPTION,
    datasets: buildDataPapersRows(entries, neverAnchors),
  };
}
