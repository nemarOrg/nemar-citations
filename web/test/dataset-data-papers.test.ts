/// <reference types="bun" />
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { NEVER_ANCHOR_FILE } from "../src/lib/data";
import type { DataPaper, DataPapersManifest } from "../src/lib/data-papers";
import type { RawAnchor } from "../src/lib/gate";

/**
 * The data papers box at the top of a dataset page (issue #256), through the
 * real loader and file reading.
 *
 * data.ts resolves its directories from the working directory at import time,
 * so the case runs in a bun subprocess whose cwd is a temp repo root; nothing
 * here changes this process's cwd. Every citation file is the real nm000275
 * fixture or a small named edit of it; the never-anchor list is the real one.
 */
const REPO = join(import.meta.dir, "..", "..");
const DATA_MODULE = join(import.meta.dir, "..", "src", "lib", "data.ts");
const PAGE = join(import.meta.dir, "..", "src", "pages", "dataset", "[id].astro");

type Payload = { dataset_id: string; metadata: { anchors: RawAnchor[] } } & Record<string, unknown>;

function fixture(name: string): Payload {
  return JSON.parse(readFileSync(join(REPO, "tests", "test_data", name), "utf-8")) as Payload;
}

let root: string;

function write(id: string, payload: Payload): void {
  writeFileSync(
    join(root, "citations", "json_opencite", `${id}_citations.json`),
    JSON.stringify({ ...payload, dataset_id: id }),
  );
}

interface Loaded {
  pages: Array<{ id: string; dataPapers: DataPaper[] }>;
  manifest: DataPapersManifest;
}

function load(cwd: string = root): Loaded {
  const script = `import { loadAll, loadDataPapers } from ${JSON.stringify(DATA_MODULE)};
console.log(JSON.stringify({
  pages: loadAll().datasets.map((d) => ({ id: d.id, dataPapers: d.dataPapers })),
  manifest: loadDataPapers(new Date("2026-09-30T12:00:00.000Z")),
}));`;
  const result = Bun.spawnSync([process.execPath, "-e", script], {
    cwd,
    env: { ...process.env, CITATIONS_ALLOW_EMPTY: "" },
  });
  if (result.exitCode !== 0) {
    throw new Error(`loader failed: ${result.stderr.toString()}`);
  }
  return JSON.parse(result.stdout.toString()) as Loaded;
}

beforeAll(() => {
  root = mkdtempSync(join(tmpdir(), "dataset-data-papers-"));
  mkdirSync(join(root, "citations", "json_opencite"), { recursive: true });
  mkdirSync(join(root, NEVER_ANCHOR_FILE, ".."), { recursive: true });
  copyFileSync(join(REPO, NEVER_ANCHOR_FILE), join(root, NEVER_ANCHOR_FILE));

  const gated = fixture("gate_nm000275_gated_citations.json");
  // nm000275: the real gated fixture, unedited: two confirmed data papers.
  write("nm000275", gated);
  // nm000300: every anchor but the dataset's own DOI relabeled judged_not_data_paper
  // (edited), so the gate has decided and none is a data paper.
  const none = structuredClone(gated);
  for (const a of none.metadata.anchors) {
    if (a.kept_reason !== "own_doi") {
      a.kept = false;
      a.kept_reason = "judged_not_data_paper";
    }
  }
  // One citation is pointed at the dataset's own DOI (edited), so the dataset
  // still has a page: it is low-confidence (0.35), not excluded by its anchor.
  (none.citation_details as Array<{ source_doi: string }>)[0].source_doi =
    "10.82901/nemar.nm000275";
  write("nm000300", none);
  // ds000778: the gated fixture under a legacy id (id edited): it has a page,
  // but NEMAR does not serve ds* ids and the manifest gives them no row.
  write("ds000778", gated);
  // nm000301: the real pre-gate fixture under another id (id edited): the gate
  // has not decided its anchors, so there is nothing to list.
  write("nm000301", fixture("gate_nm000275_citations.json"));
  // nm000302: the gated fixture plus a BIDS-paper anchor (added) that a stale
  // file marks judged_data_paper; the never-anchor list must keep it out.
  const stale = structuredClone(gated);
  stale.metadata.anchors.push({
    identifier: "10.1038/sdata.2016.44",
    identifier_type: "doi",
    kept: true,
    kept_reason: "judged_data_paper",
    paper_title: "The brain imaging data structure",
    paper_year: 2016,
    paper_venue: "Scientific Data",
    judgment_model: "claude-sonnet-5-5",
  });
  write("nm000302", stale);
});

afterAll(() => {
  rmSync(root, { recursive: true, force: true });
});

describe("the data papers of a dataset page, through the real loader", () => {
  test("lists the judge-confirmed data papers the paper citations came through", () => {
    const page = load().pages.find((p) => p.id === "nm000275");
    expect(page?.dataPapers.map((p) => p.doi)).toEqual([
      "10.1038/s41597-019-0027-4",
      "10.6084/m9.figshare.6427334.v5",
    ]);
    const first = page?.dataPapers[0];
    expect(first?.title).toBe(
      "Multi-channel EEG recordings during a sustained-attention driving task",
    );
    expect(first?.venue).toBe("Scientific Data");
    expect(first?.year).toBe(2019);
  });

  test("lists nothing when the gate decided there is no data paper, or has not decided", () => {
    const pages = new Map(load().pages.map((p) => [p.id, p.dataPapers]));
    // Both have a page, so these assert the list is empty, not that it is missing.
    expect(pages.has("nm000300")).toBe(true);
    expect(pages.get("nm000300")).toEqual([]);
    expect(pages.has("nm000301")).toBe(true);
    expect(pages.get("nm000301")).toEqual([]);
  });

  test("lists nothing for an id NEMAR does not serve, as the manifest has no row for it", () => {
    const { pages, manifest } = load();
    const page = pages.find((p) => p.id === "ds000778");
    expect(page).toBeDefined();
    expect(page?.dataPapers).toEqual([]);
    expect(manifest.datasets.map((r) => r.dataset_id)).not.toContain("ds000778");
  });

  test("never lists a standards paper, even one a stale file marks as a data paper", () => {
    const page = load().pages.find((p) => p.id === "nm000302");
    const dois = page?.dataPapers.map((p) => p.doi) ?? [];
    expect(dois).not.toContain("10.1038/sdata.2016.44");
    expect(dois).toContain("10.1038/s41597-019-0027-4");
  });

  test("agrees with data-papers.json, so the box, the counts and the metadata cannot disagree", () => {
    const { pages, manifest } = load();
    const byId = new Map(pages.map((p) => [p.id, p.dataPapers]));
    expect(manifest.datasets.length).toBeGreaterThan(0);
    for (const row of manifest.datasets) {
      // A dataset with no citations has no page, and so no box to compare.
      if (byId.has(row.dataset_id)) {
        expect(byId.get(row.dataset_id)).toEqual(row.data_papers);
      }
    }
  });
});

describe("the committed corpus", () => {
  test("every page's box equals its data-papers.json row", () => {
    const { pages, manifest } = load(REPO);
    const rows = new Map(manifest.datasets.map((r) => [r.dataset_id, r.data_papers]));
    expect(pages.length).toBeGreaterThan(0);
    const differing = pages
      .filter((p) => rows.has(p.id))
      .filter((p) => JSON.stringify(p.dataPapers) !== JSON.stringify(rows.get(p.id)))
      .map((p) => p.id);
    expect(differing).toEqual([]);
    // A page without a row (a legacy id, or a dataset the gate has not decided) shows no box.
    for (const page of pages.filter((p) => !rows.has(p.id))) {
      expect(page.dataPapers).toEqual([]);
    }
  });
});

describe("the dataset page template", () => {
  const source = readFileSync(PAGE, "utf-8");
  const markup = source.slice(0, source.indexOf("<style>"));

  test("puts the box above the filter tags and the list", () => {
    const box = markup.indexOf('class="papers"');
    expect(box).toBeGreaterThan(-1);
    expect(box).toBeLessThan(markup.indexOf('class="filter"'));
    expect(box).toBeLessThan(markup.indexOf('id="cites-main"'));
  });

  test("shows it only when the dataset has a confirmed data paper", () => {
    expect(markup).toMatch(/dataset\.dataPapers\.length\s*>\s*0\s*&&/);
  });

  test("links each paper through the tested view, in a new tab, safely", () => {
    expect(markup).toContain("dataPaperView(paper)");
    expect(markup).toMatch(/href=\{view\.href\}/);
    expect(markup).toContain('target="_blank"');
    expect(markup).toContain('rel="noopener noreferrer"');
  });

  test("says what the tags mean, only when the tags are on the page", () => {
    expect(markup).toContain('count under "Cites a paper"');
    expect(markup).toContain('count under "Cites dataset"');
    expect(markup).toMatch(/dataset\.citations\.length\s*>\s*0\s*&&\s*['"`]/);
  });

  test("keeps the small meta text at AA contrast in the light theme", () => {
    // `--color-fg-subtle` on the box's background is 4.2:1 (AA needs 4.5:1).
    const style = source.slice(source.indexOf("<style>"));
    const meta = /\.papers__meta\s*\{([^}]*)\}/.exec(style)?.[1] ?? "";
    expect(meta).toMatch(/color:\s*var\(--color-fg-muted\)/);
  });

  test("is a labelled section with a heading from the tested helper", () => {
    expect(markup).toMatch(/<section class="papers" aria-labelledby="data-papers-title">/);
    expect(markup).toContain("dataPapersHeading(dataset.dataPapers.length)");
  });
});
