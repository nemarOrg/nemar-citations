"""Anchors that are never a dataset's data paper, whatever the judge says.

Standards (the Brain Imaging Data Structure (BIDS) family), analysis software
(MNE, EEGLAB, FieldTrip, ...), data platforms (OpenNeuro, NEMAR), and umbrella
initiatives (the Healthy Brain Network) are cited by thousands of papers that
never touch a given dataset. Counting their citers as citations of the dataset
is the worst failure this pipeline can have, so the anchor gate refuses them
deterministically instead of trusting an LLM verdict:
in the 2026-09-17 snapshot the judge called EEG-BIDS a data paper for three
datasets and iEEG-BIDS for one (issue #241).

Only papers that are never any dataset's data paper belong on the list. A data
descriptor that is merely related to many datasets (e.g. MIPDB,
`10.1038/sdata.2017.40`, which is nm000153's own data paper) stays off it and
is left to the judge, dataset by dataset.

Two checks, either one is enough:

* the DOI is on the curated list in `never_anchor_dois.json` (next to this
  module, so the dashboard build in `web/src/lib/data.ts` reads the same file);
* the resolved title reads like a BIDS specification paper, which catches
  future `<Modality>-BIDS` extensions before anyone adds them to the list. The
  pattern is deliberately narrow: the spec wording ("an extension to the brain
  imaging data structure", "... extended to ...") must sit next to the phrase,
  so a data paper that merely says its data follow BIDS, or calls itself an
  "extended dataset ... in the Brain Imaging Data Structure", does not match.
"""

from __future__ import annotations

import json
import re
from functools import cache
from importlib.resources import files

from dataset_citations.sources.doi import normalize_doi

_LIST_FILE = "never_anchor_dois.json"

# "EEG-BIDS, an extension ...", "Motion-BIDS: an extension ...",
# "MNE-BIDS: Organizing ...", "BIDS apps: Improving ...". A title that only
# starts with "BIDS-formatted ..." does not match: the name must be followed
# by the comma or colon that spec and tool papers use.
_BIDS_NAME_TITLE = re.compile(r"^(?:[\w]+-)?bids(?:\s+apps)?\s*[,:]", re.IGNORECASE)
_BIDS_SPEC_PHRASE = re.compile(
    r"\b(?:extension to|extending) the brain imaging data structure"
    r"|\bbrain imaging data structure extended to\b"
    r"|^the brain imaging data structure, a format for organizing",
    re.IGNORECASE,
)


def parse_never_anchor_list(raw: str) -> frozenset[str]:
    """Parse the list file's text into normalized DOIs; raise if it is broken.

    A missing or malformed list must fail loudly: silently treating it as
    empty would let every standards paper back into the counts.
    """
    entries = json.loads(raw)["dois"]
    dois = frozenset(normalize_doi(entry["doi"]) for entry in entries)
    if not dois or "" in dois:
        raise ValueError(f"{_LIST_FILE} lists no DOIs, or an empty one")
    return dois


@cache
def never_anchor_dois() -> frozenset[str]:
    """The packaged curated DOI list, normalized (`parse_never_anchor_list`)."""
    raw = files("dataset_citations.quality").joinpath(_LIST_FILE).read_text("utf-8")
    return parse_never_anchor_list(raw)


def is_spec_title(title: str | None) -> bool:
    """True when `title` reads like a BIDS specification or BIDS tool paper."""
    if not title:
        return False
    text = title.strip()
    return bool(_BIDS_NAME_TITLE.match(text) or _BIDS_SPEC_PHRASE.search(text))


def is_never_anchor(identifier: str, title: str | None = None) -> bool:
    """True when this anchor must never be counted as the dataset's data paper.

    `identifier` is a DOI in any common spelling (bare, `doi:`, or a doi.org
    URL). `title` is the resolved paper title when one is known.
    """
    return normalize_doi(identifier) in never_anchor_dois() or is_spec_title(title)
