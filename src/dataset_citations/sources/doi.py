"""DOI / PMID / arXiv identifier helpers.

`normalize_doi` strips the common prefixes (`doi:`, `https://doi.org/`, ...),
lowercases the result, and trims trailing punctuation that often hitchhikes
from prose (`.,;)`). `validate_identifier` rejects strings with control
characters, newlines, or shapes that obviously aren't valid identifiers.
`extract_identifiers` walks free text and returns the DOIs, PMIDs, and
arXiv IDs it can confidently find.
"""

from __future__ import annotations

import re

# DOIs always start with "10." and a registrant code, then a `/`, then a suffix.
# We allow the common punctuation classes inside the suffix but balance any
# trailing parens via post-processing in `_balance_parens`.
_DOI_BARE = re.compile(r"10\.\d{4,9}/[-._;()/:\w]+", re.IGNORECASE)

_PMID_URL = re.compile(
    r"(?:pubmed\.ncbi\.nlm\.nih\.gov/|pubmed/)(\d{1,9})", re.IGNORECASE
)
_PMID_TAG = re.compile(r"\bpmid[:\s]+(\d{1,9})\b", re.IGNORECASE)

_ARXIV_TAG = re.compile(r"\barxiv[:\s]+(\d{4}\.\d{4,5}(?:v\d+)?)\b", re.IGNORECASE)
_ARXIV_DOI = re.compile(r"10\.48550/arxiv\.(\S+)", re.IGNORECASE)

_OPENNEURO_DOI = re.compile(r"^10\.18112/openneuro\.", re.IGNORECASE)

# DataCite prefix NEMAR mints dataset DOIs under (`10.82901/nemar.<id>`).
NEMAR_DOI_PREFIX = "10.82901/"

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def normalize_doi(s: str) -> str:
    """Return a lowercase, prefix-stripped DOI string.

    Does not validate; pair with `validate_identifier` before persisting.
    """
    s = s.strip()
    for prefix in (
        "doi:",
        "DOI:",
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
    ):
        s = s.removeprefix(prefix)
    s = _balance_parens(s).rstrip(".,;:")
    return s.strip().lower()


def _balance_parens(s: str) -> str:
    """Trim trailing `)` characters that don't have a matching `(` in the
    string. Helps with DOIs captured from prose like ``see (10.x/y)``."""
    opens = s.count("(")
    closes = s.count(")")
    while closes > opens and s.endswith(")"):
        s = s[:-1]
        closes -= 1
    return s


def validate_identifier(identifier: str) -> bool:
    """Reject identifiers with control characters, newlines, or obvious junk.

    A `True` result does not guarantee the identifier resolves to a record;
    it just guarantees the string is safe to write to CSVs / pass to APIs.
    """
    if not identifier or len(identifier) > 256:
        return False
    if _CONTROL_CHARS.search(identifier):
        return False
    if identifier.startswith("pmid:"):
        return identifier[5:].isdigit()
    if identifier.startswith("arxiv:"):
        # tolerate version suffix like 2106.15928v2
        suffix = identifier[6:]
        return bool(re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", suffix))
    # Otherwise treat as DOI shape.
    return bool(re.fullmatch(r"10\.\d{4,9}/\S+", identifier))


def is_own_dataset_doi(identifier: str, dataset_id: str) -> bool:
    """True when `identifier` is `dataset_id`'s own NEMAR concept DOI.

    Only the concept DOI (`10.82901/nemar.<id>`) is recognized: it always
    resolves to the latest version on NEMAR, and it is the DOI the catalog
    publishes, so versioned DOIs never reach the pipeline as anchors.
    """
    return normalize_doi(identifier) == f"{NEMAR_DOI_PREFIX}nemar.{dataset_id.lower()}"


def is_openneuro_dataset_doi(identifier: str) -> bool:
    """True for OpenNeuro `DatasetDOI` strings that are not indexed in OpenAlex.

    These are useful for record linkage but should not be sent to opencite;
    OpenAlex / Semantic Scholar return zero results for them.
    """
    return bool(_OPENNEURO_DOI.match(identifier))


def extract_identifiers(text: str | None) -> list[tuple[str, str]]:
    """Walk free text and return `(identifier, identifier_type)` pairs.

    `identifier_type` is one of `"doi"`, `"pmid"`, `"arxiv"`. Order of
    detection matters: arXiv DOI shorthand (10.48550/arxiv.X) is rewritten
    to the canonical `arxiv:X` form before the bare DOI regex runs, so we
    don't double-extract.
    """
    if not text:
        return []
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _add(value: str, kind: str) -> None:
        norm = value.lower() if kind != "doi" else normalize_doi(value)
        ident = norm if kind == "doi" else f"{kind}:{norm}"
        if not validate_identifier(ident):
            return
        key = (ident, kind)
        if key in seen:
            return
        seen.add(key)
        out.append((ident, kind))

    for m in _ARXIV_DOI.finditer(text):
        _add(m.group(1), "arxiv")
    for m in _ARXIV_TAG.finditer(text):
        _add(m.group(1), "arxiv")

    for m in _PMID_URL.finditer(text):
        _add(m.group(1), "pmid")
    for m in _PMID_TAG.finditer(text):
        _add(m.group(1), "pmid")

    masked = _ARXIV_DOI.sub("", text)
    for m in _DOI_BARE.finditer(masked):
        _add(m.group(0), "doi")

    return out
