/// <reference types="bun" />
import { describe, expect, test } from "bun:test";
import { readdirSync } from "node:fs";
import { join } from "node:path";

const CITATIONS_DIR = join(import.meta.dir, "..", "..", "citations", "json_opencite");

type Row = {
  dataset_id: string;
  num_citations: number;
  num_dataset_citations: number;
  num_datapaper_citations: number;
};

describe("the emitted counts manifest (real corpus)", () => {
  test("is well formed and lists every served dataset that has a citation file", async () => {
    const { GET } = await import("../src/pages/api/index.json");
    const response = await (GET as () => Response | Promise<Response>)();
    expect(response.headers.get("Content-Type")).toBe("application/json");
    const body = (await response.json()) as {
      schema: string;
      last_updated: unknown;
      datasets: Row[];
    };
    // The literal value, not the constant the module exports: nemar-cli's sync
    // and the docs name this string.
    expect(body.schema).toBe("nemar-citations/counts@1");
    expect(Array.isArray(body.datasets)).toBe(true);

    const ids = body.datasets.map((r) => r.dataset_id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const row of body.datasets) {
      // The four fields nemar-cli's `isCitationCountRow` requires.
      for (const field of [
        row.num_citations,
        row.num_dataset_citations,
        row.num_datapaper_citations,
      ]) {
        expect(Number.isInteger(field)).toBe(true);
        expect(field).toBeGreaterThanOrEqual(0);
      }
      expect(row.num_citations).toBe(row.num_dataset_citations + row.num_datapaper_citations);
    }

    // The 0.10.11 incident: a dataset with a file but nothing left to count
    // must be listed (as zeros), or nemar-cli's sync never resets its old count.
    const served = readdirSync(CITATIONS_DIR)
      .map((f) => f.replace(/_citations\.json$/, ""))
      .filter((id) => /^(nm|on)\d{6}$/.test(id));
    expect(served.length).toBeGreaterThan(0);
    const listed = new Set(ids);
    expect(served.filter((id) => !listed.has(id))).toEqual([]);
  });
});
