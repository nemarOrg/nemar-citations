/// <reference types="bun" />
import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import {
  AWAITING_FETCH,
  type AnchorEntry,
  KEPT_DATA_PAPER,
  baseDoi,
  buildDataPapersRows,
  cleanDoi,
  dataPapersOf,
  isGated,
} from "../src/lib/data-papers";
import type { RawAnchor } from "../src/lib/gate";

const REPO = join(import.meta.dir, "..", "..");
const NONE = new Set<string>();

function fixtureAnchors(name: string): RawAnchor[] {
  const payload = JSON.parse(readFileSync(join(REPO, "tests", "test_data", name), "utf-8")) as {
    metadata: { anchors: RawAnchor[] };
  };
  return payload.metadata.anchors;
}

/** A gated anchor as the pipeline records it. */
function anchor(
  identifier: string,
  kept_reason: string,
  extra: Partial<RawAnchor> = {},
): RawAnchor {
  return {
    identifier,
    identifier_type: "doi",
    kept:
      kept_reason === KEPT_DATA_PAPER ||
      kept_reason === "own_doi" ||
      kept_reason === "dataset_record",
    kept_reason,
    paper_title: `Paper ${identifier}`,
    paper_year: 2020,
    paper_venue: "A Journal",
    judgment_model: "claude-sonnet-5-5",
    ...extra,
  };
}

describe("real gate output (nm000275)", () => {
  // tests/test_data/gate_nm000275_gated_citations.json is the real pre-gate
  // nm000275 fixture run through the repo's real anchor gate
  // (cli.gate_anchors.gate_citation_file) with a judge sidecar whose verdicts
  // were written by hand, since no Claude judgment exists yet. Titles, venues,
  // and years are the real registry records.
  const gated = fixtureAnchors("gate_nm000275_gated_citations.json");

  test("lists exactly the judge-confirmed data paper and same-data deposit", () => {
    expect(dataPapersOf(gated, NONE)).toEqual([
      {
        doi: "10.1038/s41597-019-0027-4",
        title: "Multi-channel EEG recordings during a sustained-attention driving task",
        year: 2019,
        venue: "Scientific Data",
        judge_model: "claude-sonnet-5-5",
      },
      {
        doi: "10.6084/m9.figshare.6427334.v5",
        title:
          "Multi-channel EEG recordings during a sustained-attention driving task (raw dataset)",
        year: 2019,
        venue: "figshare",
        judge_model: "claude-sonnet-5-5",
      },
    ]);
  });

  test("never lists the dataset's own DOI, related work, or unjudged anchors", () => {
    const dois = (dataPapersOf(gated, NONE) ?? []).map((p) => p.doi);
    expect(dois).not.toContain("10.82901/nemar.nm000275");
    expect(dois).not.toContain("10.1016/j.neuroimage.2014.01.015");
    expect(dois).not.toContain("10.1038/srep21353");
  });

  test("the file from before the gate sweep is not judged, so it is omitted", () => {
    // The real pre-gate fixture: kept true, no kept_reason, Gemma judge.
    const preGate = fixtureAnchors("gate_nm000275_citations.json");
    expect(isGated(preGate)).toBe(false);
    expect(dataPapersOf(preGate, NONE)).toBeNull();
    expect(buildDataPapersRows([{ id: "nm000275", anchors: preGate }], NONE)).toEqual([]);
  });
});

describe("which anchors are listed", () => {
  test("only kept judged data papers, never another kept_reason", () => {
    const anchors = [
      anchor("10.1/data-paper", KEPT_DATA_PAPER),
      anchor("10.82901/nemar.nm000001", "own_doi"),
      anchor("10.18112/openneuro.ds000001.v1.0.0", "dataset_record"),
      anchor("10.21105/joss.01896", "never_anchor", { kept: false }),
      anchor("10.1/unjudged", "unjudged", { kept: false }),
      anchor("10.1/related", "judged_not_data_paper", { kept: false }),
    ];
    expect(dataPapersOf(anchors, NONE)?.map((p) => p.doi)).toEqual(["10.1/data-paper"]);
  });

  test("a kept_reason alone is not enough: the anchor must also be kept", () => {
    const anchors = [anchor("10.1/odd", KEPT_DATA_PAPER, { kept: false })];
    expect(dataPapersOf(anchors, NONE)).toEqual([]);
  });

  test("a stale judged_data_paper on a never-anchor DOI is still dropped", () => {
    const anchors = [
      anchor("10.21105/joss.01896", KEPT_DATA_PAPER),
      anchor("10.1/real", KEPT_DATA_PAPER),
    ];
    const never = new Set(["10.21105/joss.01896"]);
    expect(dataPapersOf(anchors, never)?.map((p) => p.doi)).toEqual(["10.1/real"]);
  });

  test("a non-DOI anchor is ignored", () => {
    const anchors = [
      anchor("https://example.org/paper", KEPT_DATA_PAPER, { identifier_type: "url" }),
      anchor("10.1/real", KEPT_DATA_PAPER),
    ];
    expect(dataPapersOf(anchors, NONE)?.map((p) => p.doi)).toEqual(["10.1/real"]);
  });
});

describe("absent versus empty", () => {
  test("a gated dataset with no data paper gets an empty list", () => {
    const anchors = [
      anchor("10.82901/nemar.nm000002", "own_doi"),
      anchor("10.1/related", "judged_not_data_paper", { kept: false }),
    ];
    expect(dataPapersOf(anchors, NONE)).toEqual([]);
    expect(buildDataPapersRows([{ id: "nm000002", anchors }], NONE)).toEqual([
      { dataset_id: "nm000002", data_papers: [] },
    ]);
  });

  test("a file with no anchors is omitted: it may be a failure stub", () => {
    expect(isGated([])).toBe(false);
    expect(buildDataPapersRows([{ id: "nm000003", anchors: [] }], NONE)).toEqual([]);
  });

  test("a partly gated file is omitted", () => {
    const anchors = [
      anchor("10.1/data-paper", KEPT_DATA_PAPER),
      { identifier: "10.1/older", identifier_type: "doi", kept: true },
    ];
    expect(dataPapersOf(anchors, NONE)).toBeNull();
  });

  test("a dataset with an anchor awaiting its fetch is omitted, not listed short", () => {
    // The anchor newly qualifies but its citers are not fetched yet; a list
    // without it (or an empty list) would claim it is not a data paper.
    const anchors = [
      anchor("10.1/already", KEPT_DATA_PAPER),
      anchor("10.1/new", AWAITING_FETCH, { kept: false }),
    ];
    expect(isGated(anchors)).toBe(false);
    expect(dataPapersOf(anchors, NONE)).toBeNull();
  });
});

describe("one entry per work", () => {
  test("a trailing period and letter case do not make a second paper", () => {
    const anchors = [
      anchor("10.7554/eLife.85012.", KEPT_DATA_PAPER),
      anchor("10.7554/elife.85012", KEPT_DATA_PAPER),
    ];
    const papers = dataPapersOf(anchors, NONE) ?? [];
    expect(papers).toHaveLength(1);
    expect(papers[0]?.doi).not.toMatch(/\.$/);
    expect(papers[0]?.doi.toLowerCase()).toBe("10.7554/elife.85012");
  });

  test("versions of one deposit collapse to the version-less DOI when present", () => {
    const anchors = [
      anchor("10.6084/m9.figshare.6427334.v5", KEPT_DATA_PAPER),
      anchor("10.6084/m9.figshare.6427334", KEPT_DATA_PAPER, { paper_title: null }),
      anchor("10.6084/m9.figshare.6427334.v4", KEPT_DATA_PAPER),
    ];
    const papers = dataPapersOf(anchors, NONE) ?? [];
    expect(papers.map((p) => p.doi)).toEqual(["10.6084/m9.figshare.6427334"]);
    // The kept entry had no title; it is filled from a sibling.
    expect(papers[0]?.title).not.toBeNull();
  });

  test("empty and non-integer metadata become null", () => {
    const anchors = [
      anchor("10.1/x", KEPT_DATA_PAPER, {
        paper_title: "  ",
        paper_venue: "",
        paper_year: 2019.5,
        judgment_model: null,
      }),
    ];
    expect(dataPapersOf(anchors, NONE)).toEqual([
      { doi: "10.1/x", title: null, year: null, venue: null, judge_model: null },
    ]);
  });
});

describe("deterministic output", () => {
  const entry = (id: string, dois: string[]): AnchorEntry => ({
    id,
    anchors: dois.map((d) => anchor(d, KEPT_DATA_PAPER)),
  });

  test("datasets sort by id and papers by DOI, whatever the input order", () => {
    const entries = [
      entry("on000002", ["10.9/z", "10.1/a"]),
      entry("nm000010", ["10.5/m"]),
      entry("nm000002", ["10.2/b", "10.1/a", "10.10/c"]),
    ];
    const rows = buildDataPapersRows(entries, NONE);
    expect(rows.map((r) => r.dataset_id)).toEqual(["nm000002", "nm000010", "on000002"]);
    expect(rows[0]?.data_papers.map((p) => p.doi)).toEqual(["10.1/a", "10.10/c", "10.2/b"]);
    expect(buildDataPapersRows([...entries].reverse(), NONE)).toEqual(rows);
  });
});

describe("DOI helpers", () => {
  test("cleanDoi strips the resolver and trailing punctuation but keeps case", () => {
    expect(cleanDoi(" https://doi.org/10.7554/eLife.85012. ")).toBe("10.7554/eLife.85012");
    expect(cleanDoi("doi:10.1038/S41597")).toBe("10.1038/S41597");
  });

  test("baseDoi collapses version suffixes, repeatedly", () => {
    expect(baseDoi("10.6084/m9.figshare.6427334.v5")).toBe("10.6084/m9.figshare.6427334");
    expect(baseDoi("10.5281/zenodo.123.v1.0.0")).toBe("10.5281/zenodo.123");
    expect(baseDoi("10.18112/openneuro.ds007763.v1.1.1")).toBe("10.18112/openneuro.ds007763");
    expect(baseDoi("10.1038/s41597-019-0027-4")).toBe("10.1038/s41597-019-0027-4");
  });
});
