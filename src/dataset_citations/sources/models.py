"""Typed records and result wrappers shared by source extractors and backends.

`FetchResult` is a discriminated union: either `FetchSuccess(value)` or
`FetchError(reason, detail)`. Callers must check `result.ok` (or use
isinstance) before accessing the value. This is the project's answer to the
Phase 1 review finding that broad except blocks conflated transient failures
with legitimately empty results.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Never, TypeVar

IdentifierType = Literal["doi", "pmid", "arxiv"]
SourceKind = Literal["nemar_metadata", "openneuro_description", "nemar_catalog"]
# DataCite relation types we read anchors from. IsDescribedBy is how a data
# paper is linked from .nemar/metadata.json (e.g. 10.1038/sdata.2015.1 describes
# on000117); older enrichments linked some data papers as IsSupplementTo (the
# dataset supplements the paper, e.g. on007615). The relation alone never makes
# an anchor count: the anchor judgment and gate (`core.anchor_gate`) then decide
# which anchors count.
RelationType = Literal[
    "References",
    "IsDerivedFrom",
    "IsIdenticalTo",
    "IsVersionOf",
    "IsDescribedBy",
    "IsSupplementTo",
]
FetchErrorReason = Literal[
    "not_found", "auth", "network", "parse", "rate_limit", "other"
]

_IDENTIFIER_PREFIX: dict[IdentifierType, str] = {
    "pmid": "pmid:",
    "arxiv": "arxiv:",
    "doi": "10.",
}


@dataclass(frozen=True, slots=True)
class Author:
    """Structured author record.

    `name` is the display string (typically family + given). `family` and
    `given` are optional; downstream consumers may show them when present.
    """

    name: str
    family: str | None = None
    given: str | None = None
    orcid: str | None = None


@dataclass(frozen=True, slots=True)
class DoiReference:
    """One DOI/PMID/arXiv anchor extracted from a dataset's metadata.

    Construction-time invariants:
      - identifier is non-empty
      - identifier matches the prefix expected for identifier_type
        ("pmid:" / "arxiv:" / "10."). Producers must normalize first.
    """

    identifier: str
    identifier_type: IdentifierType
    relation_type: RelationType
    source: SourceKind
    source_field: str | None = None

    def __post_init__(self) -> None:
        if not self.identifier:
            raise ValueError("DoiReference.identifier may not be empty")
        expected_prefix = _IDENTIFIER_PREFIX[self.identifier_type]
        if not self.identifier.startswith(expected_prefix):
            raise ValueError(
                f"identifier {self.identifier!r} does not match "
                f"identifier_type={self.identifier_type!r} "
                f"(expected prefix {expected_prefix!r})"
            )


@dataclass(frozen=True, slots=True)
class CitingWork:
    """One citing paper, normalized from an opencite Paper and tagged with the
    source DOI / relation it was discovered through."""

    title: str
    doi: str | None
    pmid: str | None
    openalex_id: str | None
    year: int | None
    authors: tuple[Author, ...]
    venue: str | None
    abstract: str | None
    citation_count: int | None
    source_doi: str
    source_relation: RelationType

    def __post_init__(self) -> None:
        if not self.title:
            raise ValueError("CitingWork.title may not be empty")
        if not self.source_doi:
            raise ValueError("CitingWork.source_doi may not be empty")


T = TypeVar("T")
U = TypeVar("U")


@dataclass(frozen=True, slots=True)
class FetchSuccess[T]:
    value: T

    @property
    def ok(self) -> Literal[True]:
        return True

    def map(self, fn: Callable[[T], U]) -> FetchSuccess[U]:
        return FetchSuccess(fn(self.value))

    def unwrap(self) -> T:
        return self.value

    def unwrap_or(self, default: T) -> T:
        return self.value


@dataclass(frozen=True, slots=True)
class FetchError:
    reason: FetchErrorReason
    detail: str

    @property
    def ok(self) -> Literal[False]:
        return False

    def map(self, fn: Callable[[U], U]) -> FetchError:
        return self

    def unwrap(self) -> Never:
        raise RuntimeError(f"unwrap on FetchError({self.reason}): {self.detail}")

    def unwrap_or(self, default: U) -> U:
        return default


FetchResult = FetchSuccess[T] | FetchError
