/**
 * Pure counting rules shared by the data layer and its tests (no file I/O).
 *
 * The pipeline's anchor gate (src/dataset_citations/core/anchor_gate.py, issue
 * #241) decides which anchors contribute citations and records the verdict in
 * each file's `metadata.anchors[]` as `kept` plus `kept_reason`. This layer
 * honors that verdict. Two backstops remain: the never-anchor list shared with
 * the pipeline always excludes, and an anchor the gate has not processed yet
 * (no `kept_reason`: a file written before the sweep ran) falls back to the
 * `kept` flag and the over-spread heuristic.
 */

/** Dataset DOIs, the dataset's own record rather than a publication: NEMAR
 * (10.82901/, e.g. 10.82901/nemar.nm000275) and OpenNeuro (10.18112/openneuro.*;
 * the trailing dot avoids other registrants under that prefix). Mirrors
 * core.accession_mentions.cites_dataset. */
export const DATASET_DOI_RE = /^(?:10\.82901\/|10\.18112\/openneuro\.)/;

/** source_relation values whose anchor IS a record of the dataset (vs a paper).
 * Mirrors core.anchor_gate.DATASET_RECORD_RELATIONS. */
export const DATASET_RELATIONS = new Set(["IsVersionOf", "IsIdenticalTo"]);

/** A source anchor attributed across more datasets than this, with no gate
 * verdict, is treated as a methods/umbrella paper. */
export const METHODS_SPREAD = 5;

export interface GateCitation {
  source_doi?: string | null;
  source_relation?: string | null;
  discovery_method?: string | null;
  mentions_accession?: boolean | null;
}

export interface RawAnchor {
  identifier?: string | null;
  kept?: boolean | null;
  kept_reason?: string | null;
  classification?: string | null;
  paper_title?: string | null;
}

/** An anchor's verdict as the file records it. `gated` is false for a file the
 * gate has not processed (no kept_reason), whose `kept` is only the fetch-time
 * flag of an older pipeline. */
export interface AnchorVerdict {
  kept: boolean | null;
  gated: boolean;
  classification: string | null;
  title: string | null;
}

export function normalizeDoi(doi: string): string {
  return doi
    .trim()
    .toLowerCase()
    .replace(/^doi:/, "")
    .replace(/^https?:\/\/(?:dx\.)?doi\.org\//, "")
    .replace(/[.,;:]+$/, "");
}

/** Anchor DOI (normalized) -> verdict, from a file's metadata.anchors[]. */
export function anchorVerdicts(
  anchors: RawAnchor[] | null | undefined,
): Map<string, AnchorVerdict> {
  const out = new Map<string, AnchorVerdict>();
  for (const anchor of anchors ?? []) {
    if (!anchor?.identifier) {
      continue;
    }
    out.set(normalizeDoi(anchor.identifier), {
      kept: typeof anchor.kept === "boolean" ? anchor.kept : null,
      gated: typeof anchor.kept_reason === "string" && anchor.kept_reason.length > 0,
      classification: anchor.classification ?? null,
      title: anchor.paper_title?.trim() || null,
    });
  }
  return out;
}

/** Parse the never-anchor list file. Throws when it lists nothing: building
 * without it would publish standards-paper citers. */
export function parseNeverAnchors(text: string): Set<string> {
  const data = JSON.parse(text) as { dois?: Array<{ doi?: string }> };
  const dois = (data.dois ?? []).flatMap((entry) =>
    entry.doi?.trim() ? [normalizeDoi(entry.doi)] : [],
  );
  if (dois.length === 0) {
    throw new Error("never-anchor list lists no DOIs; refusing to build.");
  }
  return new Set(dois);
}

/** Source DOIs attributed across more than METHODS_SPREAD datasets. */
export function overSpreadAnchors(
  entries: Array<{ id: string; details: GateCitation[] }>,
): Set<string> {
  const spread = new Map<string, Set<string>>();
  for (const { id, details } of entries) {
    for (const c of details) {
      if (!c.source_doi) {
        continue;
      }
      const key = normalizeDoi(c.source_doi);
      const set = spread.get(key) ?? new Set<string>();
      set.add(id);
      spread.set(key, set);
    }
  }
  const out = new Set<string>();
  for (const [anchor, datasets] of spread) {
    if (datasets.size > METHODS_SPREAD) {
      out.add(anchor);
    }
  }
  return out;
}

/** True when a citation must not count toward its dataset: it came in only
 * through an anchor that is not kept. A paper that names the dataset accession
 * cites the dataset whatever anchor surfaced it, matching the pipeline's gate. */
export function isExcludedCitation(
  c: GateCitation,
  neverAnchors: Set<string>,
  overSpread: Set<string>,
  verdicts: Map<string, AnchorVerdict>,
): boolean {
  if (c.discovery_method === "accession_mention" || c.mentions_accession === true) {
    return false;
  }
  if (!c.source_doi) {
    return false;
  }
  const source = normalizeDoi(c.source_doi);
  if (neverAnchors.has(source)) {
    return true;
  }
  const verdict = verdicts.get(source);
  if (verdict?.gated) {
    return verdict.kept !== true;
  }
  return verdict?.kept === false || overSpread.has(source);
}

/** True when the citation cites the dataset record itself rather than a paper.
 * Keep in sync with core.accession_mentions.cites_dataset. */
export function citesDataset(c: GateCitation): boolean {
  if (c.discovery_method === "accession_mention" || c.mentions_accession === true) {
    return true;
  }
  if (c.source_relation != null && DATASET_RELATIONS.has(c.source_relation)) {
    return true;
  }
  return c.source_doi != null && DATASET_DOI_RE.test(normalizeDoi(c.source_doi));
}
