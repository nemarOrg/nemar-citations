/**
 * Build-time data layer for the citations dashboard.
 *
 * Reads the schema-v2 citation JSON committed to this repo
 * (citations/json_opencite/) plus dataset names from datasets/, and exposes a
 * typed contract the pages render. Runs in node during `astro build`; nothing
 * here ships to the client. Epic #127.
 *
 * Counting policy (issues #138, #241): the HEADLINE counts only HIGH-CONFIDENCE
 * citations (confidence_score >= HIGH_CONF) and EXCLUDES citations surfaced
 * through anchors that are not kept. The pipeline's anchor gate decides and
 * records that per anchor (`kept` / `kept_reason`), and this layer honors it
 * (`gate.ts`). The never-anchor list shared with the pipeline (BIDS / software /
 * platform / umbrella papers) always excludes; a file the gate has not
 * processed yet also falls back to its `kept: false` flags and the over-spread
 * heuristic, and the build warns about it. Accession mentions always count.
 * Low-confidence citations are not counted but are kept so the per-dataset view
 * can surface them ("also N low-confidence").
 */
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

import {
  type AnchorVerdict,
  type RawAnchor,
  anchorVerdicts,
  citesDataset,
  isExcludedCitation,
  normalizeDoi,
  overSpreadAnchors,
  parseNeverAnchors,
} from "./gate";
import { findRepoPath } from "./repo";

const ALLOW_EMPTY = process.env.CITATIONS_ALLOW_EMPTY === "1";
const EMPTY_HINT =
  "Run the hallu pipeline (or check out a tree with citations/json_opencite/), " +
  "or set CITATIONS_ALLOW_EMPTY=1 to intentionally build an empty dashboard.";

const HIGH_CONF = 0.4;

// Standards / software / platform / umbrella anchor DOIs whose citers are never
// citations of a dataset. One list, shared with the pipeline's anchor gate
// (issue #241).
export const NEVER_ANCHOR_FILE = join(
  "src",
  "dataset_citations",
  "quality",
  "never_anchor_dois.json",
);

const citationsDir = findRepoPath(join("citations", "json_opencite"));
const datasetsDir = findRepoPath("datasets");

/** Where a citation was found: did the citing paper cite the dataset's own DOI,
 * or a publication (data paper) describing it? Derived from the citation's
 * source anchor and that anchor's record in the file's metadata.anchors[]. */
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
  /** High-confidence citations through kept anchors: the counted set. */
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
  /** Citations excluded because their source anchor is not kept (not the
   * dataset's data paper or record: methods, standards, related work, unjudged). */
  excludedByAnchor: number;
}

export interface Overview {
  datasetCount: number;
  datasetsWithCitations: number;
  /** Summed high-confidence citations through kept anchors, across datasets. */
  totalCitations: number;
  /** Unique high-confidence citing papers through kept anchors (the headline). */
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
  /** Confidence split of citation-dataset attributions (excluded anchors dropped):
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

interface RawEntry {
  id: string;
  details: RawCitation[];
  /** Normalized anchor DOI -> the gate's verdict as the file records it. */
  verdicts: Map<string, AnchorVerdict>;
  /** Top-level date_last_updated (ISO) from the citation JSON, or null. */
  lastUpdated: string | null;
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

/** Classify where a citation was found from its source anchor DOI + that
 * anchor's record in the file. */
function provenanceOf(c: RawCitation, verdicts: Map<string, AnchorVerdict>): CitationProvenance {
  // Accession mentions, identity-relation anchors, and dataset DOIs ARE dataset
  // citations. Keep in sync with core.accession_mentions.cites_dataset.
  if (citesDataset(c)) {
    return { kind: "dataset", label: "Cites dataset", anchorTitle: null };
  }
  const info = c.source_doi ? verdicts.get(normalizeDoi(c.source_doi)) : undefined;
  const label = info?.classification === "data_paper" ? "Cites data paper" : "Cites paper";
  return { kind: "paper", label, anchorTitle: info?.title ?? null };
}

function toCitation(c: RawCitation, verdicts: Map<string, AnchorVerdict>): Citation {
  return {
    title: c.title?.trim() || "Untitled",
    authors: c.author?.trim() || "",
    year: typeof c.year === "number" && c.year > 0 ? c.year : null,
    venue: c.venue?.trim() || "",
    url: c.url?.trim() || null,
    doi: c.doi?.trim() || null,
    citedBy: typeof c.cited_by === "number" ? c.cited_by : 0,
    confidence: c.confidence_scoring?.confidence_score ?? null,
    provenance: provenanceOf(c, verdicts),
  };
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
        metadata?: { anchors?: RawAnchor[] };
      };
      entries.push({
        id: raw.dataset_id || fileName.replace("_citations.json", ""),
        details: raw.citation_details ?? [],
        verdicts: anchorVerdicts(raw.metadata?.anchors),
        lastUpdated: raw.date_last_updated ?? null,
      });
    } catch (err) {
      console.warn(`[data] skipping ${fileName}: ${err instanceof Error ? err.message : err}`);
    }
  }
  return entries;
}

/** The never-anchor DOI list shared with the pipeline. Throws when the file is
 * missing or empty: building without it would publish standards-paper citers. */
function readNeverAnchors(): Set<string> {
  const path = findRepoPath(NEVER_ANCHOR_FILE);
  if (!path) {
    throw new Error(`[data] ${NEVER_ANCHOR_FILE} not found; refusing to build without it.`);
  }
  return parseNeverAnchors(readFileSync(path, "utf-8"));
}

/** Datasets with an anchor-sourced citation whose anchor carries no gate
 * verdict: files the gate sweep has not processed yet. */
function ungatedDatasets(entries: RawEntry[]): string[] {
  return entries
    .filter(({ details, verdicts }) =>
      details.some(
        (c) =>
          c.source_doi &&
          c.discovery_method !== "accession_mention" &&
          !verdicts.get(normalizeDoi(c.source_doi))?.gated,
      ),
    )
    .map(({ id }) => id);
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

  const neverAnchors = readNeverAnchors();
  const overSpread = overSpreadAnchors(entries);
  const ungated = ungatedDatasets(entries);
  if (ungated.length > 0) {
    const shown = ungated.slice(0, 10).join(", ") + (ungated.length > 10 ? ", ..." : "");
    console.warn(
      `[data] ${ungated.length} dataset(s) have citations through anchors the gate has not processed (run dataset-citations-gate-anchors); falling back to heuristics for them: ${shown}`,
    );
  }
  const uniqueHighConf = new Set<string>();
  // First-seen publication year per unique high-confidence paper (null = unknown).
  const firstYearByKey = new Map<string, number | null>();
  let totalCitations = 0;
  let lowConfidenceTotal = 0;
  let datasetsWithCitations = 0;
  const datasets: DatasetDetail[] = [];

  for (const { id, details, verdicts } of entries) {
    const counted: Citation[] = [];
    const lowConf: Citation[] = [];
    let excludedByAnchor = 0;

    for (const c of details) {
      if (isExcludedCitation(c, neverAnchors, overSpread, verdicts)) {
        excludedByAnchor += 1;
        continue;
      }
      if (isHighConf(c)) {
        const cit = toCitation(c, verdicts);
        counted.push(cit);
        const key = citingKey(c);
        if (!uniqueHighConf.has(key)) {
          uniqueHighConf.add(key);
          firstYearByKey.set(key, cit.year);
        }
      } else {
        lowConf.push(toCitation(c, verdicts));
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
        excludedByAnchor,
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
