/// <reference types="bun" />
import { describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { NEVER_ANCHOR_FILE } from "../src/lib/data";
import {
  type GateCitation,
  anchorVerdicts,
  citesDataset,
  isExcludedCitation,
  overSpreadAnchors,
  parseNeverAnchors,
} from "../src/lib/gate";

const REPO = join(import.meta.dir, "..", "..");
const NEVER = new Set(["10.21105/joss.01896"]);
const NONE = new Set<string>();

function cite(source: string | null, extra: Partial<GateCitation> = {}): GateCitation {
  return { source_doi: source, ...extra };
}

describe("isExcludedCitation", () => {
  const verdicts = anchorVerdicts([
    { identifier: "10.1/kept", kept: true, kept_reason: "judged_data_paper" },
    { identifier: "10.1/related", kept: false, kept_reason: "judged_not_data_paper" },
    { identifier: "10.1/waiting", kept: false, kept_reason: "awaiting_fetch" },
    { identifier: "10.1/legacy-kept", kept: true },
    { identifier: "10.1/legacy-context", kept: false },
  ]);
  const spread = new Set(["10.1/kept", "10.1/legacy-kept"]);

  test("a gate verdict is honored, even over the spread heuristic", () => {
    // The HBN-EEG data paper is kept for 10+ release datasets.
    expect(isExcludedCitation(cite("10.1/kept"), NEVER, spread, verdicts)).toBe(false);
    expect(isExcludedCitation(cite("10.1/related"), NEVER, NONE, verdicts)).toBe(true);
    expect(isExcludedCitation(cite("10.1/waiting"), NEVER, NONE, verdicts)).toBe(true);
  });

  test("files the gate has not processed fall back to kept flags and spread", () => {
    expect(isExcludedCitation(cite("10.1/legacy-context"), NEVER, NONE, verdicts)).toBe(true);
    expect(isExcludedCitation(cite("10.1/legacy-kept"), NEVER, spread, verdicts)).toBe(true);
    expect(isExcludedCitation(cite("10.1/legacy-kept"), NEVER, NONE, verdicts)).toBe(false);
    expect(isExcludedCitation(cite("10.1/unknown"), NEVER, NONE, verdicts)).toBe(false);
  });

  test("the never-anchor list always excludes", () => {
    const kept = anchorVerdicts([
      { identifier: "10.21105/joss.01896", kept: true, kept_reason: "judged_data_paper" },
    ]);
    expect(isExcludedCitation(cite("10.21105/JOSS.01896"), NEVER, NONE, kept)).toBe(true);
  });

  test("mentions and records without a source anchor always count", () => {
    const mention = cite("10.1/related", { mentions_accession: true });
    const accession = cite(null, { discovery_method: "accession_mention" });
    expect(isExcludedCitation(mention, NEVER, NONE, verdicts)).toBe(false);
    expect(isExcludedCitation(accession, NEVER, NONE, verdicts)).toBe(false);
    expect(isExcludedCitation(cite(null), NEVER, NONE, verdicts)).toBe(false);
  });
});

describe("overSpreadAnchors", () => {
  test("flags a source DOI shared by more than five datasets", () => {
    const entries = Array.from({ length: 6 }, (_, i) => ({
      id: `nm00000${i}`,
      details: [cite("10.1/shared"), cite(`10.1/own-${i}`)],
    }));
    expect([...overSpreadAnchors(entries)]).toEqual(["10.1/shared"]);
    expect(overSpreadAnchors(entries.slice(0, 5)).size).toBe(0);
  });
});

describe("parseNeverAnchors", () => {
  test("normalizes the listed DOIs", () => {
    const parsed = parseNeverAnchors('{"dois": [{"doi": "https://doi.org/10.1/ABC."}]}');
    expect([...parsed]).toEqual(["10.1/abc"]);
  });

  test("refuses an empty or broken list", () => {
    expect(() => parseNeverAnchors('{"dois": []}')).toThrow();
    expect(() => parseNeverAnchors('{"dois": [{"doi": "  "}]}')).toThrow();
    expect(() => parseNeverAnchors("not json")).toThrow();
  });

  test("the data layer's path resolves to the pipeline's list", () => {
    const path = join(REPO, NEVER_ANCHOR_FILE);
    expect(existsSync(path)).toBe(true);
    expect(parseNeverAnchors(readFileSync(path, "utf-8")).size).toBeGreaterThan(0);
  });
});

describe("citesDataset", () => {
  test("matches core.accession_mentions.cites_dataset on the shared cases", () => {
    const fixture = JSON.parse(
      readFileSync(join(REPO, "tests", "test_data", "cites_dataset_cases.json"), "utf-8"),
    ) as { cases: Array<{ citation: GateCitation; cites_dataset: boolean }> };
    expect(fixture.cases.length).toBeGreaterThan(0);
    for (const { citation, cites_dataset } of fixture.cases) {
      expect([citation, citesDataset(citation)]).toEqual([citation, cites_dataset]);
    }
  });
});
