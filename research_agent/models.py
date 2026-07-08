from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


MIN_PUBLICATION_YEAR = 1800
MAX_PUBLICATION_YEAR = datetime.now().year + 1
MAX_QUOTE_COUNT = 20


@dataclass(frozen=True)
class SearchQuery:
    query: str
    sources: list[str]
    filters: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlanResult:
    iteration: int
    queries: list[SearchQuery]
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "queries": [query.to_dict() for query in self.queries],
            "reasoning": self.reasoning,
        }


@dataclass(frozen=True)
class Article:
    title: str
    source: str
    url: str
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    journal: str | None = None
    year: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    doi: str | None = None
    mesh_terms: list[str] = field(default_factory=list)
    publication_types: list[str] = field(default_factory=list)
    citation_count: int | None = None
    influential_citation_count: int | None = None
    tldr: str | None = None
    full_text: str | None = None
    oa_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        if self.pmid:
            return f"pmid:{self.pmid}"
        if self.doi:
            return f"doi:{self.doi.lower()}"
        return f"{self.source}:{self.title.lower()}"

    def evidence_text(self) -> str:
        parts = [self.title, self.tldr or "", self.abstract or "", self.full_text or ""]
        return " ".join(part for part in parts if part).strip()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchSettings:
    year_start: int | None = None
    year_end: int | None = None
    text_mode: str = "any"
    quote_count: int = 8

    def __post_init__(self) -> None:
        if self.text_mode not in {"any", "abstract", "full-text"}:
            raise ValueError("text_mode must be any, abstract, or full-text.")
        self._validate_year("year_start", self.year_start)
        self._validate_year("year_end", self.year_end)
        if not isinstance(self.quote_count, int) or isinstance(self.quote_count, bool):
            raise ValueError("quote_count must be a whole number.")
        if self.quote_count < 1 or self.quote_count > MAX_QUOTE_COUNT:
            raise ValueError(f"quote_count must be between 1 and {MAX_QUOTE_COUNT}.")
        if self.year_start is not None and self.year_end is not None and self.year_start > self.year_end:
            raise ValueError("year_start cannot be later than year_end.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def _validate_year(name: str, value: int | None) -> None:
        if value is None:
            return
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be a whole year.")
        if value < MIN_PUBLICATION_YEAR or value > MAX_PUBLICATION_YEAR:
            raise ValueError(f"{name} must be between {MIN_PUBLICATION_YEAR} and {MAX_PUBLICATION_YEAR}.")


@dataclass(frozen=True)
class ObserveResult:
    sufficient: bool
    reason: str
    relevant_count: int
    higher_quality_evidence: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    suggested_refined_query: str | None = None
    relevant_article_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerifiedClaim:
    claim: str
    supported: bool
    article_keys: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerifyResult:
    passed: bool
    claims: list[VerifiedClaim]
    removed_claims: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "claims": [claim.to_dict() for claim in self.claims],
            "removed_claims": self.removed_claims,
        }


@dataclass(frozen=True)
class EvidenceQuote:
    rank: int
    quote: str
    article: Article
    source_section: str
    relevance_score: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "quote": self.quote,
            "article": self.article.to_dict(),
            "source_section": self.source_section,
            "relevance_score": self.relevance_score,
            "reason": self.reason,
            "url": self.article.oa_url or self.article.url,
            "pmid": self.article.pmid,
            "pmcid": self.article.pmcid,
            "doi": self.article.doi,
        }


@dataclass(frozen=True)
class LoopTraceEntry:
    step: str
    iteration: int
    status: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchAnswer:
    summary: str
    ranked_evidence: list[EvidenceQuote]
    citations: list[Article]
    evidence_quality_note: str
    sufficient: bool
    iterations: int
    trace: list[LoopTraceEntry]
    verification: VerifyResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "ranked_evidence": [quote.to_dict() for quote in self.ranked_evidence],
            "citations": [citation.to_dict() for citation in self.citations],
            "evidence_quality_note": self.evidence_quality_note,
            "sufficient": self.sufficient,
            "iterations": self.iterations,
            "trace": [entry.to_dict() for entry in self.trace],
            "verification": self.verification.to_dict(),
        }


LiteratureAnswer = ResearchAnswer
