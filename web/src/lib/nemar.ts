/** Links from the dashboard to the NEMAR catalog (nemar.org). */

/** A dataset's landing page on NEMAR, the same URL nemar-cli records as the
 * dataset's `IsDescribedBy` landing link. Legacy `ds*` ids redirect there to
 * their `on*` mirror. */
export function nemarDatasetUrl(id: string): string {
  return `https://nemar.org/dataset/${encodeURIComponent(id)}`;
}
