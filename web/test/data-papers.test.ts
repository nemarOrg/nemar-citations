/// <reference types="bun" />
import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import {
  AWAITING_FETCH,
  type AnchorEntry,
  KEPT_DATA_PAPER,
  baseDoi,
  buildDataPapersManifest,
  buildDataPapersRows,
  cleanDoi,
  dataPapersOf,
} from "../src/lib/data-papers";
import type { RawAnchor } from "../src/lib/gate";

const REPO = join(import.meta.dir, "..", "..");
const NONE = new Set<string>();
const DECIDED_KEPT = new Set([KEPT_DATA_PAPER, "own_doi", "dataset_record"]);

function fixtureAnchors(name: string): RawAnchor[] {
  const payload = JSON.parse(readFileSync(join(REPO, "tests", "test_data", name), "utf-8")) as {
    metadata: { anchors: RawAnchor[] };
  };
  return payload.metadata.anchors;
}

/** An anchor as the gate records it. */
function anchor(
  identifier: string,
  kept_reason: string,
  extra: Partial<RawAnchor> = {},
): RawAnchor {
  return {
    identifier,
    identifier_type: "doi",
    kept: DECIDED_KEPT.has(kept_reason),
    kept_reason,
    paper_title: `Paper ${identifier}`,
    paper_year: 2020,
    paper_venue: "A Journal",
    judgment_model: "claude-sonnet-5-5",
    ...extra,
  };
}

const paper = (doi: string) => anchor(doi, KEPT_DATA_PAPER);
const dois = (anchors: RawAnchor[]) => (dataPapersOf(anchors, NONE) ?? []).map((p) => p.doi);

describe("real gate output (nm000275)", () => {
  // tests/test_data/gate_nm000275_gated_citations.json is the real pre-gate
  // nm000275 fixture run through the repo's real anchor gate
  // (cli.gate_anchors.gate_citation_file) with a judge sidecar whose verdicts
  // were written by hand, since no Claude judgment exists yet. Titles, venues,
  // and years are the real registry records. tests/test_gate_fixture.py
  // regenerates the file and asserts it matches.
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
    const listed = dois(gated);
    expect(listed).not.toContain("10.82901/nemar.nm000275");
    expect(listed).not.toContain("10.1016/j.neuroimage.2014.01.015");
    expect(listed).not.toContain("10.1038/srep21353");
  });

  test("the file from before the gate sweep gets no row", () => {
    // The real pre-gate fixture: kept true, no kept_reason, Gemma judge.
    const preGate = fixtureAnchors("gate_nm000275_citations.json");
    expect(dataPapersOf(preGate, NONE)).toBeNull();
    expect(buildDataPapersRows([{ id: "nm000275", anchors: preGate }], NONE)).toEqual([]);
  });
});

describe("which anchors are listed", () => {
  test("only kept judged data papers, never another kept_reason", () => {
    const anchors = [
      paper("10.1/data-paper"),
      anchor("10.82901/nemar.nm000001", "own_doi"),
      anchor("10.18112/openneuro.ds000001.v1.0.0", "dataset_record"),
      anchor("10.21105/joss.01896", "never_anchor", { kept: false }),
      anchor("10.1/unjudged", "unjudged", { kept: false }),
      anchor("10.1/related", "judged_not_data_paper", { kept: false }),
    ];
    expect(dois(anchors)).toEqual(["10.1/data-paper"]);
  });

  test("a kept_reason alone is not enough: the anchor must also be kept", () => {
    const anchors = [paper("10.1/odd"), anchor("10.82901/nemar.nm000001", "own_doi")];
    anchors[0] = { ...anchors[0], kept: false } as RawAnchor;
    // Inconsistent record: not listed, and not decided either, so no row.
    expect(dataPapersOf(anchors, NONE)).toBeNull();
  });

  test("a stale judged_data_paper on a never-anchor DOI is dropped", () => {
    const never = new Set(["10.21105/joss.01896"]);
    const anchors = [paper("10.21105/joss.01896"), paper("10.1/real")];
    expect(dataPapersOf(anchors, never)?.map((p) => p.doi)).toEqual(["10.1/real"]);
    // With nothing else listed it counts as decided, so the row is empty.
    expect(dataPapersOf([paper("10.21105/joss.01896")], never)).toEqual([]);
  });

  test("a judged data paper that is not a DOI means no row, even beside a DOI paper", () => {
    const url = anchor("https://example.org/paper", KEPT_DATA_PAPER, { identifier_type: "url" });
    expect(dataPapersOf([url], NONE)).toBeNull();
    expect(dataPapersOf([url, paper("10.1/real")], NONE)).toBeNull();
  });

  test("a non-DOI anchor the gate dropped does not matter", () => {
    const url = anchor("https://example.org/page", "judged_not_data_paper", {
      identifier_type: "url",
      kept: false,
    });
    expect(dois([url, paper("10.1/real")])).toEqual(["10.1/real"]);
  });
});

describe("absent versus empty", () => {
  test("a judged paper is listed even when other anchors are still unjudged", () => {
    const anchors = [paper("10.1/data-paper"), anchor("10.1/other", "unjudged", { kept: false })];
    expect(dois(anchors)).toEqual(["10.1/data-paper"]);
  });

  test("a judged paper is listed even when another anchor awaits its fetch", () => {
    const anchors = [paper("10.1/already"), anchor("10.1/new", AWAITING_FETCH, { kept: false })];
    expect(dois(anchors)).toEqual(["10.1/already"]);
  });

  test("no judged paper and an unjudged anchor gets no row", () => {
    const anchors = [
      anchor("10.82901/nemar.nm000001", "own_doi"),
      anchor("10.1/other", "unjudged", { kept: false }),
    ];
    expect(dataPapersOf(anchors, NONE)).toBeNull();
  });

  test("only unjudged anchors get no row", () => {
    const anchors = [
      anchor("10.1/a", "unjudged", { kept: false }),
      anchor("10.1/b", "unjudged", { kept: false }),
    ];
    expect(dataPapersOf(anchors, NONE)).toBeNull();
  });

  test("no judged paper and an anchor awaiting its fetch gets no row", () => {
    const anchors = [
      anchor("10.82901/nemar.nm000001", "own_doi"),
      anchor("10.1/new", AWAITING_FETCH, { kept: false }),
    ];
    expect(dataPapersOf(anchors, NONE)).toBeNull();
  });

  test("every anchor decided and none a data paper gives an empty list", () => {
    const anchors = [
      anchor("10.82901/nemar.nm000002", "own_doi"),
      anchor("10.18112/openneuro.ds000002.v1.0.0", "dataset_record"),
      anchor("10.21105/joss.01896", "never_anchor", { kept: false }),
      anchor("10.1/related", "judged_not_data_paper", { kept: false }),
    ];
    expect(dataPapersOf(anchors, NONE)).toEqual([]);
    expect(buildDataPapersRows([{ id: "nm000002", anchors }], NONE)).toEqual([
      { dataset_id: "nm000002", data_papers: [] },
    ]);
  });

  test("a file with no anchors gets no row: it may be a failure stub", () => {
    expect(dataPapersOf([], NONE)).toBeNull();
    expect(buildDataPapersRows([{ id: "nm000003", anchors: [] }], NONE)).toEqual([]);
  });

  test("an anchor without a kept_reason (file from before the sweep) gets no row", () => {
    const anchors = [
      anchor("10.82901/nemar.nm000001", "own_doi"),
      { identifier: "10.1/older", identifier_type: "doi", kept: true },
    ];
    expect(dataPapersOf(anchors, NONE)).toBeNull();
  });
});

describe("only catalog-served ids get a row", () => {
  // A fully gated file, as the sweep stamps on legacy ds004944 and ds005234.
  const gated = fixtureAnchors("gate_nm000275_gated_citations.json");

  test("legacy ds ids are excluded even when fully gated", () => {
    const entries: AnchorEntry[] = [
      { id: "ds004944", anchors: gated },
      { id: "ds005234", anchors: gated },
      { id: "nm000275", anchors: gated },
      { id: "on007763", anchors: gated },
    ];
    expect(buildDataPapersRows(entries, NONE).map((r) => r.dataset_id)).toEqual([
      "nm000275",
      "on007763",
    ]);
  });

  test("other id shapes are excluded too", () => {
    const ids = ["xx099901", "nm00027", "nm0002750", "NM000275", "ds000001", "nm000275-copy"];
    expect(
      buildDataPapersRows(
        ids.map((id) => ({ id, anchors: gated })),
        NONE,
      ),
    ).toEqual([]);
  });
});

describe("one entry per work", () => {
  test("a trailing period and letter case do not make a second paper", () => {
    const papers = dataPapersOf(
      [paper("10.7554/eLife.85012."), paper("10.7554/elife.85012")],
      NONE,
    );
    expect(papers).toHaveLength(1);
    expect(papers?.[0]?.doi).not.toMatch(/\.$/);
    expect(papers?.[0]?.doi.toLowerCase()).toBe("10.7554/elife.85012");
  });

  test("versions of one deposit collapse to the version-less DOI when present", () => {
    const anchors = [
      paper("10.6084/m9.figshare.6427334.v5"),
      anchor("10.6084/m9.figshare.6427334", KEPT_DATA_PAPER, { paper_title: null }),
      paper("10.6084/m9.figshare.6427334.v4"),
    ];
    const papers = dataPapersOf(anchors, NONE) ?? [];
    expect(papers.map((p) => p.doi)).toEqual(["10.6084/m9.figshare.6427334"]);
    // The kept entry had no title; it is filled from a sibling.
    expect(papers[0]?.title).not.toBeNull();
  });

  test("with no version-less DOI the highest version wins, numerically", () => {
    const versions = [
      "10.6084/m9.figshare.1.v9",
      "10.6084/m9.figshare.1.v10",
      "10.6084/m9.figshare.1.v2",
    ];
    for (const order of [
      versions,
      [...versions].reverse(),
      [versions[1], versions[2], versions[0]],
    ]) {
      expect(dois(order.map((d) => paper(d as string)))).toEqual(["10.6084/m9.figshare.1.v10"]);
    }
    const dotted = ["10.5281/zenodo.7.v1.2.0", "10.5281/zenodo.7.v1.10.0"];
    expect(dois(dotted.map(paper))).toEqual(["10.5281/zenodo.7.v1.10.0"]);
  });

  test("a year of zero, a negative year, a fraction, or a string becomes null", () => {
    for (const year of [0, -1, 2019.5, "2019" as unknown as number, null]) {
      const [entry] =
        dataPapersOf([anchor("10.1/x", KEPT_DATA_PAPER, { paper_year: year })], NONE) ?? [];
      expect(entry?.year).toBeNull();
    }
    const [ok] =
      dataPapersOf([anchor("10.1/x", KEPT_DATA_PAPER, { paper_year: 1999 })], NONE) ?? [];
    expect(ok?.year).toBe(1999);
  });

  test("empty title, venue, and model become null", () => {
    const anchors = [
      anchor("10.1/x", KEPT_DATA_PAPER, {
        paper_title: "  ",
        paper_venue: "",
        judgment_model: null,
      }),
    ];
    expect(dataPapersOf(anchors, NONE)).toEqual([
      { doi: "10.1/x", title: null, year: 2020, venue: null, judge_model: null },
    ]);
  });

  test("basic HTML entities in title and venue are decoded and trimmed", () => {
    const anchors = [
      anchor("10.1/x", KEPT_DATA_PAPER, {
        paper_title: "  Sleep &amp; memory: a &lt;b&gt;&quot;dataset&quot;&#39;s&lt;/b&gt; paper ",
        paper_venue: " Brain &amp;amp; Behavior ",
      }),
    ];
    const [entry] = dataPapersOf(anchors, NONE) ?? [];
    expect(entry?.title).toBe('Sleep & memory: a <b>"dataset"\'s</b> paper');
    // "&amp;amp;" decodes once, not twice.
    expect(entry?.venue).toBe("Brain &amp; Behavior");
  });
});

describe("deterministic output", () => {
  const entry = (id: string, list: string[]): AnchorEntry => ({ id, anchors: list.map(paper) });

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

describe("the manifest", () => {
  const built = new Date("2026-09-30T12:34:56.789Z");
  const gated = fixtureAnchors("gate_nm000275_gated_citations.json");
  const manifest = buildDataPapersManifest([{ id: "nm000275", anchors: gated }], NONE, built);

  test("carries the literal schema id and the build time it was given", () => {
    expect(manifest.schema).toBe("nemar-citations/data-papers@1");
    expect(manifest.last_updated).toBe("2026-09-30T12:34:56.789Z");
  });

  test("its description states the consumer contract", () => {
    expect(manifest.description).toContain("replaces the consumer's stored value");
    expect(manifest.description).toContain("no row means no statement");
    expect(manifest.description).toContain("deposits of the same data");
    expect(manifest.description).toContain("case-insensitively");
    // The project writes no em-dashes (code 0x2014).
    expect(manifest.description).not.toContain(String.fromCharCode(0x2014));
  });

  test("holds the rows", () => {
    expect(manifest.datasets.map((r) => r.dataset_id)).toEqual(["nm000275"]);
  });
});

describe("DOI helpers", () => {
  test("cleanDoi strips the resolver and trailing punctuation and leaves case alone", () => {
    expect(cleanDoi(" https://doi.org/10.7554/eLife.85012. ")).toBe("10.7554/eLife.85012");
    expect(cleanDoi("doi:10.1038/S41597")).toBe("10.1038/S41597");
    expect(cleanDoi("10.1109/tbcas.2014.2316224")).toBe("10.1109/tbcas.2014.2316224");
  });

  test("baseDoi collapses version suffixes, repeatedly", () => {
    expect(baseDoi("10.6084/m9.figshare.6427334.v5")).toBe("10.6084/m9.figshare.6427334");
    expect(baseDoi("10.5281/zenodo.123.v1.0.0")).toBe("10.5281/zenodo.123");
    expect(baseDoi("10.18112/openneuro.ds007763.v1.1.1")).toBe("10.18112/openneuro.ds007763");
    expect(baseDoi("10.1038/s41597-019-0027-4")).toBe("10.1038/s41597-019-0027-4");
  });
});
