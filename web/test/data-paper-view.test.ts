import { describe, expect, test } from "bun:test";

import { dataPaperView, dataPapersHeading, doiUrl } from "../src/lib/data-paper-view";
import type { DataPaper } from "../src/lib/data-papers";

// The first two are shaped like the real entries of nm000275's box.
const paper: DataPaper = {
  doi: "10.1038/s41597-019-0027-4",
  title: "Multi-channel EEG recordings during a sustained-attention driving task",
  year: 2019,
  venue: "Scientific Data",
  judge_model: "claude-sonnet-5-5",
};
const deposit: DataPaper = {
  doi: "10.6084/m9.figshare.6427334.v5",
  title: "Multi-channel EEG recordings during a sustained-attention driving task (raw dataset)",
  year: 2019,
  venue: "Figshare",
  judge_model: "claude-sonnet-5-5",
};

describe("doiUrl", () => {
  test("links a plain DOI through doi.org", () => {
    expect(doiUrl("10.1038/s41597-019-0027-4")).toBe("https://doi.org/10.1038/s41597-019-0027-4");
    expect(doiUrl("10.6084/m9.figshare.6427334.v5")).toBe(
      "https://doi.org/10.6084/m9.figshare.6427334.v5",
    );
  });

  test("keeps a # or ? inside the DOI instead of starting a fragment or a query", () => {
    expect(doiUrl("10.1000/a#b")).toBe("https://doi.org/10.1000/a%23b");
    expect(doiUrl("10.1000/a?b=c")).toBe("https://doi.org/10.1000/a%3Fb%3Dc");
  });

  test("encodes characters that could break out of an attribute or a URL", () => {
    expect(doiUrl('10.1000/<x>"y z')).toBe("https://doi.org/10.1000/%3Cx%3E%22y%20z");
  });

  test("keeps the slashes of a DOI with several path segments", () => {
    expect(doiUrl("10.1000/a/b c/d")).toBe("https://doi.org/10.1000/a/b%20c/d");
  });
});

describe("dataPaperView", () => {
  test("shows the title as the link, with venue, year and DOI under it", () => {
    expect(dataPaperView(paper)).toEqual({
      href: "https://doi.org/10.1038/s41597-019-0027-4",
      text: "Multi-channel EEG recordings during a sustained-attention driving task",
      meta: "Scientific Data · 2019 · doi:10.1038/s41597-019-0027-4",
    });
  });

  test("tells a deposit of the same data apart by its venue", () => {
    expect(dataPaperView(deposit).meta).toBe(
      "Figshare · 2019 · doi:10.6084/m9.figshare.6427334.v5",
    );
  });

  test("leaves out the venue and the year when they are unknown", () => {
    expect(dataPaperView({ ...paper, venue: null, year: null }).meta).toBe(
      "doi:10.1038/s41597-019-0027-4",
    );
    expect(dataPaperView({ ...paper, venue: null }).meta).toBe(
      "2019 · doi:10.1038/s41597-019-0027-4",
    );
  });

  test("uses the DOI as the link text, once, when there is no title", () => {
    expect(dataPaperView({ ...paper, title: null })).toEqual({
      href: "https://doi.org/10.1038/s41597-019-0027-4",
      text: "10.1038/s41597-019-0027-4",
      meta: "Scientific Data · 2019",
    });
  });
});

describe("dataPapersHeading", () => {
  test("is singular for one paper and plural otherwise", () => {
    expect(dataPapersHeading(1)).toBe("Data paper");
    expect(dataPapersHeading(2)).toBe("Data papers");
    expect(dataPapersHeading(3)).toBe("Data papers");
  });
});
