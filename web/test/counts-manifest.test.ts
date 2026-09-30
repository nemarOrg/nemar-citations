/// <reference types="bun" />
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { type CountedDataset, type CountsRow, buildCountsRows } from "../src/lib/counts-manifest";
import { NEVER_ANCHOR_FILE } from "../src/lib/data";
import type { RawAnchor } from "../src/lib/gate";

function dataset(id: string, n: number, own = 0): CountedDataset {
  return { id, numCitations: n, numDatasetCitations: own, numDataPaperCitations: n - own };
}

describe("buildCountsRows", () => {
  test("keeps the dataset rows, in order and unchanged", () => {
    const rows = buildCountsRows([dataset("nm000275", 67), dataset("on000002", 5, 2)], []);
    expect(rows).toEqual([
      {
        dataset_id: "nm000275",
        num_citations: 67,
        num_dataset_citations: 0,
        num_datapaper_citations: 67,
      },
      {
        dataset_id: "on000002",
        num_citations: 5,
        num_dataset_citations: 2,
        num_datapaper_citations: 3,
      },
    ]);
  });

  test("adds an explicit zero row for a served dataset with a file but no row", () => {
    // The 0.10.11 incident shape: on007763 had 335 citations in the catalog,
    // none left after the gate, and no row in the manifest, so the sync never
    // reset it.
    const rows = buildCountsRows([dataset("nm000275", 67)], ["nm000275", "on007763"]);
    expect(rows.at(-1)).toEqual({
      dataset_id: "on007763",
      num_citations: 0,
      num_dataset_citations: 0,
      num_datapaper_citations: 0,
    });
    expect(rows).toHaveLength(2);
  });

  test("never lists a dataset twice", () => {
    // Already listed, and a file id repeated (two files naming one dataset).
    const rows = buildCountsRows(
      [dataset("nm000275", 67)],
      ["nm000275", "on000009", "on000009", "on000009"],
    );
    expect(rows.map((r) => r.dataset_id)).toEqual(["nm000275", "on000009"]);
  });

  test("does not add zero rows for legacy ds* ids or malformed ids", () => {
    // nemar-cli matches a ds* row to its on* mirror through source_id, so a
    // zero row for a ds* file could overwrite the mirror's real count.
    const rows = buildCountsRows([], ["ds004944", "xx099900", "nm00027", "nm0002750", "on000009"]);
    expect(rows.map((r) => r.dataset_id)).toEqual(["on000009"]);
  });

  test("orders the zero rows by id whatever the input order", () => {
    const rows = buildCountsRows([dataset("nm000275", 1)], ["on000009", "nm000100", "on000001"]);
    expect(rows.map((r) => r.dataset_id)).toEqual(["nm000275", "nm000100", "on000001", "on000009"]);
  });

  test("keeps a legacy id that already has a counted row", () => {
    const rows = buildCountsRows([dataset("ds004944", 3)], ["ds004944"]);
    expect(rows.map((r) => r.dataset_id)).toEqual(["ds004944"]);
  });
});

/**
 * The same rule through the real loader and file reading.
 *
 * data.ts resolves its directories from the working directory at import time,
 * so the case runs in a bun subprocess whose cwd is a temp repo root; nothing
 * here changes this process's cwd. Every citation file is the real nm000275
 * fixture or a small named edit of it; the never-anchor list is the real one.
 */
const REPO = join(import.meta.dir, "..", "..");
const DATA_MODULE = join(import.meta.dir, "..", "src", "lib", "data.ts");
const COUNTS_MODULE = join(import.meta.dir, "..", "src", "lib", "counts-manifest.ts");

type Payload = { dataset_id: string; metadata: { anchors: RawAnchor[] } } & Record<string, unknown>;

let root: string;

function write(id: string, payload: Payload): void {
  writeFileSync(
    join(root, "citations", "json_opencite", `${id}_citations.json`),
    JSON.stringify({ ...payload, dataset_id: id }),
  );
}

function rowsFromLoader(): CountsRow[] {
  const script = `import { loadAll } from ${JSON.stringify(DATA_MODULE)};
import { buildCountsRows } from ${JSON.stringify(COUNTS_MODULE)};
const { datasets, datasetIds } = loadAll();
console.log(JSON.stringify(buildCountsRows(datasets, datasetIds)));`;
  const result = Bun.spawnSync([process.execPath, "-e", script], {
    cwd: root,
    env: { ...process.env, CITATIONS_ALLOW_EMPTY: "" },
  });
  if (result.exitCode !== 0) {
    throw new Error(`loader failed: ${result.stderr.toString()}`);
  }
  return JSON.parse(result.stdout.toString()) as CountsRow[];
}

beforeAll(() => {
  root = mkdtempSync(join(tmpdir(), "counts-manifest-loader-"));
  mkdirSync(join(root, "citations", "json_opencite"), { recursive: true });
  mkdirSync(join(root, NEVER_ANCHOR_FILE, ".."), { recursive: true });
  copyFileSync(join(REPO, NEVER_ANCHOR_FILE), join(root, NEVER_ANCHOR_FILE));

  const gated = JSON.parse(
    readFileSync(join(REPO, "tests", "test_data", "gate_nm000275_gated_citations.json"), "utf-8"),
  ) as Payload;

  // nm000275: the real gated fixture, unedited: citations are counted.
  write("nm000275", gated);
  // nm000300 and ds000777: the fixture with every anchor but the dataset's own
  // DOI relabeled judged_not_data_paper (edited), so nothing is left to count.
  // The first is served, the second is a legacy id.
  const dropped = structuredClone(gated);
  for (const a of dropped.metadata.anchors) {
    if (a.kept_reason !== "own_doi") {
      a.kept = false;
      a.kept_reason = "judged_not_data_paper";
    }
  }
  write("nm000300", dropped);
  write("ds000777", dropped);
  // nm000301: the same fixture with no citations at all (edited), the shape of
  // 61 of the 65 stale datasets on 2026-09-30: a file the gate emptied.
  const empty = structuredClone(dropped);
  empty.citation_details = [];
  empty.num_citations = 0;
  write("nm000301", empty);
});

afterAll(() => {
  rmSync(root, { recursive: true, force: true });
});

describe("the counts manifest through the real loader", () => {
  test("lists a served dataset whose citations all dropped away as an explicit zero", () => {
    const rows = rowsFromLoader();
    const byId = new Map(rows.map((r) => [r.dataset_id, r]));
    expect(byId.get("nm000275")?.num_citations).toBeGreaterThan(0);
    expect(byId.get("nm000300")).toEqual({
      dataset_id: "nm000300",
      num_citations: 0,
      num_dataset_citations: 0,
      num_datapaper_citations: 0,
    });
  });

  test("lists a served dataset whose file holds no citations at all", () => {
    const row = rowsFromLoader().find((r) => r.dataset_id === "nm000301");
    expect(row).toEqual({
      dataset_id: "nm000301",
      num_citations: 0,
      num_dataset_citations: 0,
      num_datapaper_citations: 0,
    });
  });

  test("does not list a legacy dataset that has nothing to count", () => {
    const ids = rowsFromLoader().map((r) => r.dataset_id);
    expect(ids).not.toContain("ds000777");
    expect(ids.filter((id) => id === "nm000275")).toHaveLength(1);
  });
});
