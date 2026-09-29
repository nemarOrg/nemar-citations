/// <reference types="bun" />
import { describe, expect, test } from "bun:test";

import { nemarDatasetUrl } from "../src/lib/nemar";

describe("nemarDatasetUrl", () => {
  test("links the datasets NEMAR serves", () => {
    expect(nemarDatasetUrl("nm000275")).toBe("https://nemar.org/dataset/nm000275");
    expect(nemarDatasetUrl("on002778")).toBe("https://nemar.org/dataset/on002778");
  });

  test("gives a legacy OpenNeuro id no link", () => {
    expect(nemarDatasetUrl("ds004944")).toBeNull();
  });

  test("fails closed on anything else", () => {
    // nemar.org 404s on an uppercase id, so it is not normalized into one.
    for (const id of [
      "NM000275",
      " nm000275",
      "nm000275 ",
      "nm000275\n",
      "nm",
      "",
      "nm000275/../x",
    ]) {
      expect(nemarDatasetUrl(id)).toBeNull();
    }
  });
});
