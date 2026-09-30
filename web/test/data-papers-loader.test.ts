/// <reference types="bun" />
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { NEVER_ANCHOR_FILE } from "../src/lib/data";
import type { DataPapersManifest } from "../src/lib/data-papers";
import type { RawAnchor } from "../src/lib/gate";

/**
 * loadDataPapers against a temporary repo root, through the real file reading.
 *
 * data.ts resolves its directories from the working directory at import time,
 * so each case runs in a bun subprocess whose cwd is the temp root; nothing
 * here changes this process's cwd. Every citation file is the real nm000275
 * fixture or a small edit of it (the edits are named in each case); the
 * never-anchor list is the real one.
 */
const REPO = join(import.meta.dir, "..", "..");
const DATA_MODULE = join(import.meta.dir, "..", "src", "lib", "data.ts");
const BUILT = "2026-09-30T12:00:00.000Z";

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

function load(): DataPapersManifest {
  const script = `import { loadDataPapers } from ${JSON.stringify(DATA_MODULE)};
console.log(JSON.stringify(loadDataPapers(new Date(${JSON.stringify(BUILT)}))));`;
  const result = Bun.spawnSync([process.execPath, "-e", script], {
    cwd: root,
    env: { ...process.env, CITATIONS_ALLOW_EMPTY: "" },
  });
  if (result.exitCode !== 0) {
    throw new Error(`loader failed: ${result.stderr.toString()}`);
  }
  return JSON.parse(result.stdout.toString()) as DataPapersManifest;
}

beforeAll(() => {
  root = mkdtempSync(join(tmpdir(), "data-papers-loader-"));
  mkdirSync(join(root, "citations", "json_opencite"), { recursive: true });
  const neverDir = join(root, NEVER_ANCHOR_FILE, "..");
  mkdirSync(neverDir, { recursive: true });
  copyFileSync(join(REPO, NEVER_ANCHOR_FILE), join(root, NEVER_ANCHOR_FILE));

  const gated = fixture("gate_nm000275_gated_citations.json");
  const preGate = fixture("gate_nm000275_citations.json");

  // nm000275: the real gated fixture, unedited.
  write("nm000275", gated);
  // ds004944: the gated fixture under a legacy id (id edited): never served.
  write("ds004944", gated);
  // on000001: the real pre-gate fixture under another id (id edited): no row.
  write("on000001", preGate);
  // on000002: the gated fixture plus a BIDS-paper anchor (added) that a stale
  // file marks judged_data_paper; the never-anchor list must drop it.
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
  write("on000002", stale);
  // nm000300: the gated fixture with every anchor but the dataset's own DOI
  // relabeled judged_not_data_paper (edited): all decided, none listed. The
  // unjudged anchors are relabeled too, since one would withhold the row.
  const none = structuredClone(gated);
  for (const a of none.metadata.anchors) {
    if (a.kept_reason !== "own_doi") {
      a.kept = false;
      a.kept_reason = "judged_not_data_paper";
    }
  }
  write("nm000300", none);
});

afterAll(() => {
  rmSync(root, { recursive: true, force: true });
});

describe("loadDataPapers against a temp repo root", () => {
  test("reads the files, maps their anchors, and applies the never-anchor list", () => {
    const manifest = load();
    expect(manifest.schema).toBe("nemar-citations/data-papers@1");
    expect(manifest.last_updated).toBe(BUILT);
    expect(manifest.datasets.map((r) => r.dataset_id)).toEqual([
      "nm000275",
      "nm000300",
      "on000002",
    ]);
    const byId = new Map(manifest.datasets.map((r) => [r.dataset_id, r.data_papers]));
    const expected = ["10.1038/s41597-019-0027-4", "10.6084/m9.figshare.6427334.v5"];
    expect(byId.get("nm000275")?.map((p) => p.doi)).toEqual(expected);
    // The BIDS paper is on the never-anchor list, read from the real file.
    expect(byId.get("on000002")?.map((p) => p.doi)).toEqual(expected);
    expect(byId.get("nm000300")).toEqual([]);
  });

  test("a legacy ds id and a pre-gate file get no row", () => {
    const ids = load().datasets.map((r) => r.dataset_id);
    expect(ids).not.toContain("ds004944");
    expect(ids).not.toContain("on000001");
  });
});
