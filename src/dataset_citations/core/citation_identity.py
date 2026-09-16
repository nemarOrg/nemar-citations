"""Canonical identity for a citing work, and the dedup that follows from it.

A single paper reaches us by several routes -- two different anchor DOIs, an
anchor plus an accession-mention hit, OpenAlex plus Semantic Scholar -- and the
copies rarely agree byte for byte. Three kinds of disagreement produced real
double-counting in `citations/json_opencite/`:

1. Same DOI, different title text. OpenAlex and S2 punctuate differently.
   For `10.1016/j.cortex.2024.12.017` the two copies differ by a trailing
   period AND by the dash before "A multi-lab replication": OpenAlex uses an
   ASCII hyphen (U+002D), S2 an en dash (U+2013). This source file is
   deliberately ASCII, so the codepoints are named rather than shown. Keying
   on (doi, title) as a tuple meant these never collapsed.
2. Concept DOI vs version DOI. `10.82901/nemar.on004842` and
   `10.82901/nemar.on004842.v1.0.0` are the same record, as are figshare's
   `....v1` / `....v2`. Both carry a `.vN` suffix, so `base_doi` collapses
   them outright.
   Zenodo is NOT one of these, despite looking like it: it mints a wholly
   distinct integer per deposit (`zenodo.19051613` / `zenodo.19051614`), so
   no suffix stripping can relate them. Those merge only through the title
   pass below, which means Zenodo deposits whose titles drift between
   versions ("... (Version 1)" vs "... (Version 2)") stay separate. That is
   a known limit, not an oversight.
3. Preprint and version of record. `10.31219/osf.io/pu5vb` and
   `10.1016/j.neuroimage.2022.119623` are one work; counting both inflates a
   dataset's citation total.

The rule this module implements: a DOI identifies a work, after stripping
version suffixes. Where two DOIs disagree but the normalized titles match and
one side is a preprint, the version of record wins and the preprint DOI is
retained on the survivor as `superseded_dois` so nothing becomes untraceable.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# Repository / preprint-server DOI prefixes. A DOI under one of these is a
# preprint or a deposit record, so it loses to a journal DOI for the same work.
# 10.64898 is bioRxiv/medRxiv's prefix for all submissions since December
# 2025, alongside the legacy 10.1101.
_PREPRINT_PREFIXES = (
    "10.1101/",  # bioRxiv / medRxiv (legacy)
    "10.64898/",  # bioRxiv / medRxiv (current since December 2025)
    "10.48550/",  # arXiv
    "10.21203/",  # Research Square
    "10.31234/",  # PsyArXiv
    "10.31219/",  # OSF Preprints
    "10.17605/",  # OSF projects
    "10.31222/",  # MetaArXiv
    "10.31235/",  # SocArXiv
    "10.26434/",  # ChemRxiv
    "10.20944/",  # Preprints.org
    "10.2139/",  # SSRN
    "10.36227/",  # TechRxiv (IEEE)
    "10.22541/",  # Authorea
    "10.12688/",  # F1000Research / Wellcome Open (versioned records)
    "10.5281/",  # Zenodo
    "10.6084/",  # figshare
    "10.82901/",  # NEMAR dataset DOIs
)

# bioRxiv/medRxiv style suffix: YYYY.MM.DD.NNNNNN. Catches new preprint-server
# prefixes before this module learns about them.
_PREPRINT_SUFFIX = re.compile(r"/\d{4}\.\d{2}\.\d{2}\.\d{4,}$")

# Trailing version marker: ".v1", ".v1.0.0", "/v2". Zenodo and figshare mint a
# fresh DOI per version alongside a version-less concept DOI.
_VERSION_SUFFIX = re.compile(r"(?:\.v\d+(?:\.\d+)*|/v\d+(?:\.\d+)*)$", re.IGNORECASE)

_DOI_URL_PREFIX = re.compile(r"^(?:https?://)?(?:dx\.)?doi\.org/", re.IGNORECASE)

# Unicode dashes (hyphen through horizontal bar, plus minus sign) that sources
# swap for a plain hyphen at random.
_DASHES = re.compile("[\\u2010-\\u2015\\u2212]")


def normalize_doi(doi: Any) -> str:
    """Lowercase a DOI and strip any resolver prefix. Empty string if absent."""
    if not doi:
        return ""
    text = _DOI_URL_PREFIX.sub("", str(doi).strip())
    return text.casefold().rstrip("/")


def base_doi(doi: Any) -> str:
    """Normalize `doi` and collapse trailing version suffixes to the concept DOI.

    Applied repeatedly so a doubly-versioned DOI (`....v1.0.0`) reduces fully.
    """
    text = normalize_doi(doi)
    previous = None
    while previous != text:
        previous = text
        text = _VERSION_SUFFIX.sub("", text)
    return text


def is_preprint_doi(doi: Any) -> bool:
    """True when `doi` belongs to a preprint server or a deposit repository."""
    text = normalize_doi(doi)
    if not text:
        return False
    return text.startswith(_PREPRINT_PREFIXES) or bool(_PREPRINT_SUFFIX.search(text))


def normalize_title(title: Any) -> str:
    """Fold a title to a comparison key: case, unicode, punctuation, spacing.

    Deliberately lossy. The inputs differ only in presentation (trailing period,
    en dash vs hyphen, HTML-escaped entities), so everything but the letters and
    digits is discarded.
    """
    if not title:
        return ""
    text = unicodedata.normalize("NFKD", str(title))
    text = text.casefold()
    text = _DASHES.sub("-", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _prefer(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Pick the better of two records for the same work.

    In order: version of record beats preprint; then having a DOI at all
    beats having none (this dominates, so a one-field record with a DOI still
    beats a ten-field record without one); then the richer record (more
    populated fields); then the higher `cited_by`; and finally the lexically
    smaller DOI, purely so the outcome never depends on input order.
    """
    left_pre = is_preprint_doi(left.get("doi"))
    right_pre = is_preprint_doi(right.get("doi"))
    if left_pre != right_pre:
        return right if left_pre else left

    def rank(record: dict[str, Any]) -> tuple[int, int, int, str]:
        return (
            1 if normalize_doi(record.get("doi")) else 0,
            sum(1 for value in record.values() if value not in (None, "", 0, "n/a")),
            int(record.get("cited_by") or 0),
            # Stored as-is. The first three fields are compared greatest-wins
            # as a slice; this one is NOT part of that comparison and is
            # broken out separately below, smallest-wins, so no ordering
            # trick is applied to it here.
            normalize_doi(record.get("doi")),
        )

    left_rank, right_rank = rank(left), rank(right)
    if left_rank[:3] != right_rank[:3]:
        return left if left_rank[:3] > right_rank[:3] else right
    return left if left_rank[3] <= right_rank[3] else right


def _absorb(winner: dict[str, Any], loser: dict[str, Any]) -> None:
    """Fold a dropped duplicate's provenance into the surviving record.

    The loser's DOI is kept in `superseded_dois` so a preprint or version DOI
    that used to appear as its own citation stays traceable, and the
    accession-mention flags survive a merge in either direction.
    """
    winner_doi = normalize_doi(winner.get("doi"))
    # The loser's own DOI, plus anything it had already absorbed. A three-way
    # merge (version DOI -> concept DOI -> version of record) would otherwise
    # lose the first DOI when the pass-1 winner loses again in pass 2.
    incoming = [normalize_doi(loser.get("doi"))]
    incoming.extend(normalize_doi(d) for d in loser.get("superseded_dois") or [])
    new_dois = {d for d in incoming if d and d != winner_doi}
    if new_dois:
        superseded = winner.setdefault("superseded_dois", [])
        superseded[:] = sorted(set(superseded) | new_dois)
    if loser.get("mentions_accession") and not winner.get("mentions_accession"):
        winner["mentions_accession"] = True
        winner.setdefault("matched_accession", loser.get("matched_accession"))


def dedupe_citations(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Collapse duplicate citing works. Returns (kept records, dropped count).

    Two passes, because DOI identity is stronger evidence than title identity
    and must settle first:

    1. Group by version-collapsed DOI. Records without a DOI sit out this pass.
    2. Group what survives by normalized title, and merge only where the title
       match is corroborated -- one side is a preprint, or one side has no DOI
       at all. Two distinct journal DOIs sharing a title are left alone; that
       pattern is a correction or a reprint, not a double count.

    Input order is preserved for the survivors.
    """
    by_doi: dict[str, dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    dropped = 0

    for record in records:
        key = base_doi(record.get("doi"))
        if not key:
            ordered.append(record)
            continue
        existing = by_doi.get(key)
        if existing is None:
            by_doi[key] = record
            ordered.append(record)
            continue
        winner = _prefer(existing, record)
        loser = record if winner is existing else existing
        _absorb(winner, loser)
        if winner is not existing:
            ordered[ordered.index(existing)] = winner
            by_doi[key] = winner
        dropped += 1

    by_title: dict[str, dict[str, Any]] = {}
    survivors: list[dict[str, Any]] = []
    for record in ordered:
        key = normalize_title(record.get("title"))
        existing = by_title.get(key) if key else None
        if existing is None:
            if key:
                by_title[key] = record
            survivors.append(record)
            continue
        has_doi = bool(normalize_doi(record.get("doi")))
        existing_has_doi = bool(normalize_doi(existing.get("doi")))
        corroborated = (
            not has_doi
            or not existing_has_doi
            or is_preprint_doi(record.get("doi"))
            or is_preprint_doi(existing.get("doi"))
        )
        if not corroborated:
            survivors.append(record)
            continue
        winner = _prefer(existing, record)
        loser = record if winner is existing else existing
        _absorb(winner, loser)
        if winner is not existing:
            survivors[survivors.index(existing)] = winner
            by_title[key] = winner
        dropped += 1

    return survivors, dropped
