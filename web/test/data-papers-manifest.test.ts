/// <reference types="bun" />
import { describe, expect, test } from "bun:test";

describe("the emitted manifest (real corpus)", () => {
  test("is well formed for whatever the committed corpus holds", async () => {
    const { GET } = await import("../src/pages/api/data-papers.json");
    const response = await (GET as () => Response | Promise<Response>)();
    expect(response.headers.get("Content-Type")).toBe("application/json");
    const body = (await response.json()) as {
      schema: string;
      last_updated: unknown;
      description: unknown;
      datasets: Array<{ dataset_id: string; data_papers: Array<{ doi: string }> }>;
    };
    // The literal values, not the constants the module exports: a consumer
    // matches on these strings.
    expect(body.schema).toBe("nemar-citations/data-papers@1");
    expect(typeof body.last_updated).toBe("string");
    expect(Number.isNaN(Date.parse(body.last_updated as string))).toBe(false);
    expect(typeof body.description).toBe("string");
    expect(Array.isArray(body.datasets)).toBe(true);
    const ids = body.datasets.map((d) => d.dataset_id);
    expect(ids).toEqual([...ids].sort());
    expect(new Set(ids).size).toBe(ids.length);
    for (const row of body.datasets) {
      // Only catalog-served ids: a legacy ds* id is never served.
      expect(row.dataset_id).toMatch(/^(nm|on)\d{6}$/);
      for (const paper of row.data_papers) {
        expect(paper.doi).toMatch(/^10\.\d{4,}\//);
      }
    }
  });
});
