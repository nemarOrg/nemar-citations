/// <reference types="bun" />
import { describe, expect, test } from "bun:test";

import { DATA_PAPERS_SCHEMA } from "../src/lib/data-papers";

describe("the emitted manifest (real corpus)", () => {
  test("is well formed for whatever the committed corpus holds", async () => {
    const { GET } = await import("../src/pages/api/data-papers.json");
    const response = await (GET as () => Response | Promise<Response>)();
    const body = (await response.json()) as {
      schema: string;
      last_updated: string | null;
      datasets: Array<{ dataset_id: string; data_papers: Array<{ doi: string }> }>;
    };
    expect(body.schema).toBe(DATA_PAPERS_SCHEMA);
    expect(Array.isArray(body.datasets)).toBe(true);
    const ids = body.datasets.map((d) => d.dataset_id);
    expect(ids).toEqual([...ids].sort());
    expect(new Set(ids).size).toBe(ids.length);
    for (const row of body.datasets) {
      expect(row.dataset_id).toMatch(/^(nm|on)\d{6}$/);
      for (const paper of row.data_papers) {
        expect(paper.doi).toMatch(/^10\.\d{4,}\//);
      }
    }
  });
});
