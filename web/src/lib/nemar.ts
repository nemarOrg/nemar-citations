/** Links from the dashboard to the NEMAR catalog (nemar.org). */

/** A dataset's landing page on NEMAR, the same URL nemar-cli records as the
 * dataset's `IsDescribedBy` landing link, or null when NEMAR has no page for
 * it. NEMAR serves only its own `nm*` and imported `on*` datasets; a legacy
 * OpenNeuro `ds*` id is not served (nemar.org only redirects one that has an
 * `on*` mirror, and those are pruned from the dashboard in favor of the
 * mirror), so it gets no link. */
export function nemarDatasetUrl(id: string): string | null {
  return /^(?:nm|on)\d+$/.test(id) ? `https://nemar.org/dataset/${id}` : null;
}
