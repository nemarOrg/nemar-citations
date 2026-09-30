/**
 * How a dataset page shows its data papers (the box above the filter tags).
 * Pure, so the link, the text and the heading are unit-tested rather than
 * string-matched in the template.
 */
import type { DataPaper } from "./data-papers";

export interface DataPaperView {
  href: string;
  /** The link text: the title, or the DOI when the paper has none. */
  text: string;
  /** The line under it: venue, year and the DOI (the DOI is the link text when there is no title). */
  meta: string;
}

/** A doi.org URL for a DOI. Each path segment is encoded, so a `#` or `?` in
 * a DOI stays part of the DOI instead of starting a fragment or a query. */
export function doiUrl(doi: string): string {
  return `https://doi.org/${doi.split("/").map(encodeURIComponent).join("/")}`;
}

export function dataPaperView(paper: DataPaper): DataPaperView {
  const parts: Array<string | null> = [
    paper.venue,
    paper.year === null ? null : String(paper.year),
  ];
  if (paper.title) {
    parts.push(`doi:${paper.doi}`);
  }
  return {
    href: doiUrl(paper.doi),
    text: paper.title ?? paper.doi,
    meta: parts.filter((part): part is string => Boolean(part)).join(" · "),
  };
}

/** The box's heading. */
export function dataPapersHeading(count: number): string {
  return count === 1 ? "Data paper" : "Data papers";
}
