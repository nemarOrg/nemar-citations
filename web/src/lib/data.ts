/**
 * Build-time data layer for the citations dashboard.
 *
 * Reads the schema-v2 citation JSON committed to this repo
 * (citations/json_opencite/) plus dataset names from datasets/, and exposes a
 * typed contract the pages render. Runs in node during `astro build`; nothing
 * here ships to the client. Epic #127.
 *
 * Counting policy (issue #138): the HEADLINE counts only HIGH-CONFIDENCE
 * citations (confidence_score >= HIGH_CONF) and EXCLUDES BIDS / methods /
 * umbrella anchor papers (a citation pulled in via a method/standards paper is
 * not a citation OF the dataset). Methods anchors are detected as a source DOI
 * that is over-spread across many datasets, plus a curated denylist of canonical
 * BIDS/methods papers. Low-confidence citations are not counted but are kept so
 * the per-dataset view can surface them ("also N low-confidence").
 */
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

import { findRepoPath } from "./repo";

const ALLOW_EMPTY = process.env.CITATIONS_ALLOW_EMPTY === "1";
const EMPTY_HINT =
  "Run the hallu pipeline (or check out a tree with citations/json_opencite/), " +
  "or set CITATIONS_ALLOW_EMPTY=1 to intentionally build an empty dashboard.";

const HIGH_CONF = 0.4;
// A source anchor attributed across more datasets than this is treated as a
// methods/umbrella paper (its citers are not citations of any one dataset).
const METHODS_SPREAD = 5;

// Curated canonical BIDS / methods / standards / umbrella anchor DOIs (normalized:
// lowercase, no "doi:" prefix). Excluded even if not over-spread. The over-spread
// heuristic catches the rest. The principled long-term fix is anchor judgment on
// the un-judged ds-* datasets (#131); this is the build-local stand-in.
const METHODS_DENYLIST = new Set<string>([
  "10.21105/joss.01896", // MNE-BIDS
  "10.1038/s41597-019-0104-8", // EEG-BIDS
  "10.1038/s41597-019-0105-7", // iEEG-BIDS
  "10.1038/sdata.2018.110", // MEG-BIDS
  "10.1038/sdata.2016.44", // BIDS (original)
  "10.1038/s41592-018-0235-4", // fMRIPrep
  "10.3389/fnins.2013.00267", // MNE-Python
  "10.1016/j.jneumeth.2003.10.009", // EEGLAB
  "10.1155/2011/156869", // FieldTrip
  "10.1038/sdata.2017.181", // HBN (umbrella)
  "10.1038/sdata.2017.40", // HBN resource (umbrella)
]);

const citationsDir = findRepoPath(join("citations", "json_opencite"));
const datasetsDir = findRepoPath("datasets");
const anchorJudgmentsDir = findRepoPath(join("citations", "anchor_judgments"));

// Dataset DOIs, the dataset's own record rather than a publication: NEMAR
// (10.82901/, e.g. 10.82901/nemar.nm000275) and OpenNeuro (10.18112/openneuro.*;
// the trailing dot avoids other registrants under that prefix). Mirrors
// core.accession_mentions.cites_dataset. Citations whose source anchor matches
// this are "cites dataset"; everything else is a publication.
const DATASET_DOI_RE = /^(?:10\.82901\/|10\.18112\/openneuro\.)/;

/** Where a citation was found: did the citing paper cite the dataset's own DOI,
 * or a publication (data paper) describing it? Derived by joining the citation's
 * source anchor DOI against the anchor-judgment sidecars. */
export interface CitationProvenance {
  /** "dataset" = source anchor is the dataset's own OpenNeuro/NEMAR DOI;
   * "paper" = source anchor is a publication. */
  kind: "dataset" | "paper";
  /** Badge text, e.g. "Cites dataset" / "Cites data paper" / "Cites paper". */
  label: string;
  /** Title of the cited anchor paper (provenance), when known from the sidecar. */
  anchorTitle: string | null;
}

export interface Citation {
  title: string;
  authors: string;
  year: number | null;
  venue: string;
  url: string | null;
  doi: string | null;
  citedBy: number;
  confidence: number | null;
  provenance: CitationProvenance;
}

export interface DatasetDetail {
  id: string;
  /** High-confidence, non-methods citations — the counted set. */
  numCitations: number;
  /** Of numCitations, those that cite the dataset itself (accession mention /
   * version / own DOI). The leaderboard ranks on this (issue #169). */
  numDatasetCitations: number;
  /** Of numCitations, those that cite a data paper / publication. */
  numDataPaperCitations: number;
  name: string;
  citations: Citation[];
  /** Low-confidence citations (kept for the detail view, not counted). */
  lowConfCitations: Citation[];
  /** Count of method/standards references excluded from the dataset's citations. */
  methodsExcluded: number;
}

export interface Overview {
  datasetCount: number;
  datasetsWithCitations: number;
  /** Summed high-confidence, non-methods citations across datasets. */
  totalCitations: number;
  /** Unique high-confidence, non-methods citing papers (the headline). */
  uniqueCitations: number;
  /** Summed low-confidence citations (surfaced as a secondary figure). */
  lowConfidenceTotal: number;
  /** Most recent date_last_updated across the corpus (ISO), or null. */
  lastUpdated: string | null;
}

/** One year of the citation-growth series (unique high-confidence papers). */
export interface YearPoint {
  year: number;
  /** Unique high-confidence papers first cited in this year. */
  newPapers: number;
  /** Running total of unique high-confidence papers through this year. */
  cumulative: number;
}

/** Chart-ready aggregates derived from the same filtered/deduped pass as the
 * overview, so the trends page reconciles exactly to the headline figures. */
export interface ChartData {
  /** Unique high-confidence citing papers by publication year (papers with a
   * known year; the cumulative total approaches Overview.uniqueCitations). */
  temporal: YearPoint[];
  /** Confidence split of citation-dataset attributions (methods excluded):
   * high = confidence >= HIGH_CONF, low = below it. */
  confidence: { high: number; low: number };
}

export interface LoadedData {
  overview: Overview;
  datasets: DatasetDetail[];
  charts: ChartData;
}

interface RawCitation {
  title?: string | null;
  author?: string | null;
  venue?: string | null;
  year?: number | null;
  url?: string | null;
  doi?: string | null;
  pmid?: string | null;
  openalex_id?: string | null;
  source_doi?: string | null;
  source_relation?: string | null;
  /** "accession_mention" = found by full-text search for the dataset accession
   * (issue #169); absent/"anchor" = the DOI-anchored opencite path. */
  discovery_method?: string | null;
  /** True when an anchor citation also names the dataset accession in text. */
  mentions_accession?: boolean | null;
  cited_by?: number | null;
  confidence_scoring?: { confidence_score?: number | null } | null;
}

/** source_relation values whose anchor IS the dataset (vs a data paper). Mirrors
 * core.accession_mentions._DATASET_RELATIONS. */
const DATASET_RELATIONS = new Set(["IsVersionOf", "IsIdenticalTo"]);

/** anchor DOI (normalized) -> its judged classification + paper title. */
type AnchorMap = Map<string, { classification: string; title: string | null }>;

interface RawEntry {
  id: string;
  details: RawCitation[];
  /** Top-level date_last_updated (ISO) from the citation JSON, or null. */
  lastUpdated: string | null;
}

function normalizeDoi(doi: string): string {
  return doi
    .trim()
    .toLowerCase()
    .replace(/^doi:/, "")
    .replace(/[.,;:]+$/, "");
}

/** DOI-first identity for a citing paper (matches the attribution audit's dedup). */
function citingKey(c: RawCitation): string {
  if (c.doi) {
    return `doi:${normalizeDoi(c.doi)}`;
  }
  if (c.openalex_id) {
    return `openalex:${c.openalex_id.trim().toLowerCase()}`;
  }
  if (c.pmid) {
    return `pmid:${String(c.pmid).trim()}`;
  }
  return `title:${(c.title ?? "").trim().toLowerCase().replace(/\s+/g, " ")}`;
}

function isHighConf(c: RawCitation): boolean {
  const conf = c.confidence_scoring?.confidence_score;
  return typeof conf === "number" && conf >= HIGH_CONF;
}

/** Classify where a citation was found from its source anchor DOI + the
 * dataset's anchor-judgment sidecar. */
function provenanceOf(c: RawCitation, anchors: AnchorMap): CitationProvenance {
  // Accession mentions (the paper names the dataset accession in text) and
  // version/identical anchors ARE dataset citations even though they carry no
  // OpenNeuro source DOI. Keep in sync with core.accession_mentions.cites_dataset.
  if (
    c.discovery_method === "accession_mention" ||
    c.mentions_accession === true ||
    (c.source_relation != null && DATASET_RELATIONS.has(c.source_relation))
  ) {
    return { kind: "dataset", label: "Cites dataset", anchorTitle: null };
  }
  const sd = c.source_doi ? normalizeDoi(c.source_doi) : null;
  if (sd && DATASET_DOI_RE.test(sd)) {
    return { kind: "dataset", label: "Cites dataset", anchorTitle: null };
  }
  const info = sd ? anchors.get(sd) : undefined;
  const label = info?.classification === "data_paper" ? "Cites data paper" : "Cites paper";
  return { kind: "paper", label, anchorTitle: info?.title ?? null };
}

function toCitation(c: RawCitation, anchors: AnchorMap): Citation {
  return {
    title: c.title?.trim() || "Untitled",
    authors: c.author?.trim() || "",
    year: typeof c.year === "number" && c.year > 0 ? c.year : null,
    venue: c.venue?.trim() || "",
    url: c.url?.trim() || null,
    doi: c.doi?.trim() || null,
    citedBy: typeof c.cited_by === "number" ? c.cited_by : 0,
    confidence: c.confidence_scoring?.confidence_score ?? null,
    provenance: provenanceOf(c, anchors),
  };
}

/** Read a dataset's anchor-judgment sidecar into a DOI -> {classification,title}
 * map. Returns an empty map when the sidecar is missing or unreadable. */
function readAnchorMap(id: string): AnchorMap {
  const map: AnchorMap = new Map();
  if (!anchorJudgmentsDir) {
    return map;
  }
  const path = join(anchorJudgmentsDir, `${id}.json`);
  if (!existsSync(path)) {
    return map;
  }
  try {
    const data = JSON.parse(readFileSync(path, "utf-8")) as {
      judgments?: Array<{
        anchor_identifier?: string | null;
        classification?: string | null;
        paper_title?: string | null;
      }>;
    };
    for (const j of data.judgments ?? []) {
      if (j.anchor_identifier) {
        map.set(normalizeDoi(j.anchor_identifier), {
          classification: j.classification ?? "",
          title: j.paper_title?.trim() || null,
        });
      }
    }
  } catch (err) {
    console.warn(
      `[data] skipping anchor sidecar ${id}: ${err instanceof Error ? err.message : err}`,
    );
  }
  return map;
}

function readDatasetName(id: string): string {
  if (!datasetsDir) {
    return id;
  }
  const path = join(datasetsDir, `${id}_datasets.json`);
  if (!existsSync(path)) {
    return id;
  }
  try {
    const meta = JSON.parse(readFileSync(path, "utf-8")) as {
      dataset_description?: { Name?: string } | null;
    };
    return meta.dataset_description?.Name?.trim() || id;
  } catch {
    return id;
  }
}

function readEntries(): RawEntry[] | null {
  if (!citationsDir) {
    return null;
  }
  const files = readdirSync(citationsDir).filter((f) => f.endsWith("_citations.json"));
  if (files.length === 0) {
    return null;
  }
  const entries: RawEntry[] = [];
  for (const fileName of files) {
    try {
      const raw = JSON.parse(readFileSync(join(citationsDir, fileName), "utf-8")) as {
        dataset_id?: string;
        citation_details?: RawCitation[];
        date_last_updated?: string | null;
      };
      entries.push({
        id: raw.dataset_id || fileName.replace("_citations.json", ""),
        details: raw.citation_details ?? [],
        lastUpdated: raw.date_last_updated ?? null,
      });
    } catch (err) {
      console.warn(`[data] skipping ${fileName}: ${err instanceof Error ? err.message : err}`);
    }
  }
  return entries;
}

/** Source DOIs attributed across more than METHODS_SPREAD datasets, unioned with
 * the curated denylist — the methods/umbrella anchors whose citers we exclude. */
function methodsAnchors(entries: RawEntry[]): Set<string> {
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
  const methods = new Set<string>(METHODS_DENYLIST);
  for (const [anchor, datasets] of spread) {
    if (datasets.size > METHODS_SPREAD) {
      methods.add(anchor);
    }
  }
  return methods;
}

function isMethodsCitation(c: RawCitation, methods: Set<string>): boolean {
  return c.source_doi ? methods.has(normalizeDoi(c.source_doi)) : false;
}

let cache: LoadedData | null = null;

/** Load + classify the full corpus once (memoized across page builds). */
export function loadAll(): LoadedData {
  if (cache) {
    return cache;
  }
  const empty: LoadedData = {
    overview: {
      datasetCount: 0,
      datasetsWithCitations: 0,
      totalCitations: 0,
      uniqueCitations: 0,
      lowConfidenceTotal: 0,
      lastUpdated: null,
    },
    datasets: [],
    charts: { temporal: [], confidence: { high: 0, low: 0 } },
  };

  const entries = readEntries();
  if (!entries) {
    if (ALLOW_EMPTY) {
      console.warn(`[data] no citation data found; building empty. ${EMPTY_HINT}`);
      cache = empty;
      return cache;
    }
    throw new Error(`[data] Cannot load citations/json_opencite/. ${EMPTY_HINT}`);
  }

  const methods = methodsAnchors(entries);
  const uniqueHighConf = new Set<string>();
  // First-seen publication year per unique high-confidence paper (null = unknown).
  const firstYearByKey = new Map<string, number | null>();
  let totalCitations = 0;
  let lowConfidenceTotal = 0;
  let datasetsWithCitations = 0;
  const datasets: DatasetDetail[] = [];

  for (const { id, details } of entries) {
    const counted: Citation[] = [];
    const lowConf: Citation[] = [];
    let methodsExcluded = 0;
    const anchors = readAnchorMap(id);

    for (const c of details) {
      if (isMethodsCitation(c, methods)) {
        methodsExcluded += 1;
        continue;
      }
      if (isHighConf(c)) {
        const cit = toCitation(c, anchors);
        counted.push(cit);
        const key = citingKey(c);
        if (!uniqueHighConf.has(key)) {
          uniqueHighConf.add(key);
          firstYearByKey.set(key, cit.year);
        }
      } else {
        lowConf.push(toCitation(c, anchors));
      }
    }

    totalCitations += counted.length;
    lowConfidenceTotal += lowConf.length;
    if (counted.length > 0) {
      datasetsWithCitations += 1;
    }
    // Generate a page for any dataset with citations (high- OR low-confidence),
    // so every low-confidence citation is actually surfaced per dataset.
    if (counted.length > 0 || lowConf.length > 0) {
      counted.sort((a, b) => b.citedBy - a.citedBy);
      lowConf.sort((a, b) => b.citedBy - a.citedBy);
      const numDatasetCitations = counted.filter((c) => c.provenance.kind === "dataset").length;
      datasets.push({
        id,
        name: readDatasetName(id),
        numCitations: counted.length,
        numDatasetCitations,
        numDataPaperCitations: counted.length - numDatasetCitations,
        citations: counted,
        lowConfCitations: lowConf,
        methodsExcluded,
      });
    }
  }

  datasets.sort((a, b) => b.numCitations - a.numCitations);

  // Build the citation-growth series from the unique high-confidence papers with
  // a known year, so the cumulative total reconciles to uniqueCitations.
  const byYear = new Map<number, number>();
  for (const year of firstYearByKey.values()) {
    if (year !== null) {
      byYear.set(year, (byYear.get(year) ?? 0) + 1);
    }
  }
  let cumulative = 0;
  const temporal: YearPoint[] = [...byYear.keys()]
    .sort((a, b) => a - b)
    .map((year) => {
      const newPapers = byYear.get(year) ?? 0;
      cumulative += newPapers;
      return { year, newPapers, cumulative };
    });

  // Most recent per-dataset date_last_updated (ISO strings sort lexicographically).
  let lastUpdated: string | null = null;
  for (const { lastUpdated: d } of entries) {
    if (d && (lastUpdated === null || d > lastUpdated)) {
      lastUpdated = d;
    }
  }

  cache = {
    overview: {
      datasetCount: entries.length,
      datasetsWithCitations,
      totalCitations,
      uniqueCitations: uniqueHighConf.size,
      lowConfidenceTotal,
      lastUpdated,
    },
    datasets,
    charts: {
      temporal,
      confidence: { high: totalCitations, low: lowConfidenceTotal },
    },
  };
  return cache;
}
