"""Anchors that are never a dataset's data paper, whatever the judge says.

Standards (the Brain Imaging Data Structure (BIDS) family), analysis software
(MNE, EEGLAB, FieldTrip, ...), and data platforms (OpenNeuro, NEMAR) are cited
by thousands of papers that never touch a given dataset. Counting their citers
as citations of the dataset is the worst failure this pipeline can have, so the
anchor gate refuses them deterministically instead of trusting an LLM verdict:
in the 2026-09-18 snapshot the judge called EEG-BIDS a data paper for three
datasets and iEEG-BIDS for one (issue #241).

Two checks, either one is enough:

* the DOI is on the curated list in `never_anchor_dois.json` (next to this
  module, so the dashboard build in `web/src/lib/data.ts` reads the same file);
* the resolved title reads like a BIDS specification paper, which catches
  future `<Modality>-BIDS` extensions before anyone adds them to the list. The
  pattern is deliberately narrow: a data paper that merely says its data follow
  BIDS ("... in BIDS format") does not match.
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
_BIDS_PHRASE = re.compile(r"brain imaging data structure", re.IGNORECASE)
_SPEC_WORDS = re.compile(
    r"\b(?:extension|extending|extended|a format for organizing)\b", re.IGNORECASE
)


@cache
def never_anchor_dois() -> frozenset[str]:
    """The curated DOI list, normalized. Raises if the packaged file is broken.

    A missing or malformed list must fail loudly: silently treating it as
    empty would let every standards paper back into the counts.
    """
    raw = files("dataset_citations.quality").joinpath(_LIST_FILE).read_text("utf-8")
    entries = json.loads(raw)["dois"]
    dois = frozenset(normalize_doi(entry["doi"]) for entry in entries)
    if not dois:
        raise ValueError(f"{_LIST_FILE} lists no DOIs")
    return dois


def is_spec_title(title: str | None) -> bool:
    """True when `title` reads like a BIDS specification or BIDS tool paper."""
    if not title:
        return False
    text = title.strip()
    if _BIDS_NAME_TITLE.match(text):
        return True
    return bool(_BIDS_PHRASE.search(text) and _SPEC_WORDS.search(text))


def is_never_anchor(identifier: str, title: str | None = None) -> bool:
    """True when this anchor must never be counted as the dataset's data paper.

    `identifier` is a DOI in any common spelling (bare, `doi:`, or a doi.org
    URL). `title` is the resolved paper title when one is known.
    """
    return normalize_doi(identifier) in never_anchor_dois() or is_spec_title(title)
