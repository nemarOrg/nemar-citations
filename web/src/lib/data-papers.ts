/**
 * Data-paper manifest rows (issue #250): which papers the pipeline treats as a
 * dataset's data paper, for nemar-cli to serve as `data_papers` in
 * data.nemar.org/<id>/metadata.json.
 *
 * Pure (no file I/O), so the rules are unit-testable. The verdict is the
 * pipeline's anchor gate's, read from each file's `metadata.anchors[]`
 * (`kept` plus `kept_reason`, see gate.ts): the same verdict that decides which
 * citations count, so the list and the counts cannot disagree. Only anchors the
 * trusted judge called `data_paper` and the gate kept are listed. The dataset's
 * own DOI (`own_doi`), identity records (`dataset_record`), and every dropped
 * anchor are not.
 *
 * Absent vs empty is part of the contract. A dataset is omitted when the gate
 * has not finished with its file, so the consumer keeps NULL, meaning "not
 * judged"; a dataset the gate has finished with and that has no data paper gets
 * an empty list.
 */
import { type RawAnchor, normalizeDoi } from "./gate";

export const DATA_PAPERS_SCHEMA = "nemar-citations/data-papers@1";

/** `kept_reason` of an anchor the trusted judge called the dataset's data paper. */
export const KEPT_DATA_PAPER = "judged_data_paper";

/** `kept_reason` of an anchor that newly qualifies but whose citers are not
 * fetched yet (core.anchor_gate.AWAITING_FETCH). */
export const AWAITING_FETCH = "awaiting_fetch";

/** Trailing version marker (".v1", ".v1.0.0", "/v2"). Mirrors
 * core.citation_identity._VERSION_SUFFIX. */
const VERSION_SUFFIX = /(?:\.v\d+(?:\.\d+)*|\/v\d+(?:\.\d+)*)$/i;

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

export interface AnchorEntry {
  id: string;
  anchors: RawAnchor[];
}

/** A DOI for display: resolver prefix and trailing punctuation removed, the
 * registrant's capitalization kept (e.g. 10.7554/eLife.85012). */
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

function text(value: string | null | undefined): string | null {
  const trimmed = value?.trim();
  return trimmed ? trimmed : null;
}

function isDoiAnchor(anchor: RawAnchor): boolean {
  return (anchor.identifier_type ?? "doi").toLowerCase() === "doi";
}

/** True when the gate has finished with this file: it has anchors, every one
 * carries a `kept_reason`, and none is waiting for its citers to be fetched. A
 * file with no anchors is not judged (it may be a failure stub), and a file
 * written before the gate sweep has no `kept_reason` at all. */
export function isGated(anchors: RawAnchor[]): boolean {
  return (
    anchors.length > 0 &&
    anchors.every((a) => text(a.kept_reason) !== null && a.kept_reason !== AWAITING_FETCH)
  );
}

/** The data papers of one dataset's anchors, or null when the gate has not
 * finished with the file. `neverAnchors` is a backstop: a never-anchor DOI is
 * dropped even if a stale file still says `judged_data_paper`. */
export function dataPapersOf(
  anchors: RawAnchor[],
  neverAnchors: ReadonlySet<string>,
): DataPaper[] | null {
  if (!isGated(anchors)) {
    return null;
  }
  const byBase = new Map<string, DataPaper>();
  for (const anchor of anchors) {
    if (
      !anchor.identifier ||
      !isDoiAnchor(anchor) ||
      anchor.kept !== true ||
      anchor.kept_reason !== KEPT_DATA_PAPER ||
      neverAnchors.has(normalizeDoi(anchor.identifier))
    ) {
      continue;
    }
    const doi = cleanDoi(anchor.identifier);
    if (!doi) {
      continue;
    }
    const year = Number.isInteger(anchor.paper_year) ? (anchor.paper_year as number) : null;
    const paper: DataPaper = {
      doi,
      title: text(anchor.paper_title),
      year,
      venue: text(anchor.paper_venue),
      judge_model: text(anchor.judgment_model),
    };
    const key = baseDoi(doi);
    const existing = byBase.get(key);
    if (!existing) {
      byBase.set(key, paper);
      continue;
    }
    // Two DOIs of the same work (a trailing period, or versions of a deposit):
    // keep one, preferring the version-less and then the shorter form, and
    // fill its gaps from the other.
    const [keep, other] =
      compareDois(paper.doi, existing.doi) < 0 ? [paper, existing] : [existing, paper];
    byBase.set(key, {
      doi: keep.doi,
      title: keep.title ?? other.title,
      year: keep.year ?? other.year,
      venue: keep.venue ?? other.venue,
      judge_model: keep.judge_model ?? other.judge_model,
    });
  }
  return [...byBase.values()].sort((a, b) => compareText(normalizeDoi(a.doi), normalizeDoi(b.doi)));
}

function compareText(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

/** Which of two DOIs of the same work to keep: the shorter one first (so the
 * version-less form of a deposit beats its versions), then by normalized text. */
function compareDois(a: string, b: string): number {
  return a.length !== b.length
    ? a.length - b.length
    : compareText(normalizeDoi(a), normalizeDoi(b));
}

/** Manifest rows: datasets by id, omitting those the gate has not finished with. */
export function buildDataPapersRows(
  entries: AnchorEntry[],
  neverAnchors: ReadonlySet<string>,
): DataPapersRow[] {
  const rows: DataPapersRow[] = [];
  for (const { id, anchors } of entries) {
    const dataPapers = dataPapersOf(anchors, neverAnchors);
    if (dataPapers !== null) {
      rows.push({ dataset_id: id, data_papers: dataPapers });
    }
  }
  return rows.sort((a, b) => compareText(a.dataset_id, b.dataset_id));
}
