from __future__ import annotations

import re

from research_agent.agent.llm import LLMClient, NoopLLMClient
from research_agent.models import Article, ObserveResult


HIGH_QUALITY_TERMS = {
    "meta-analysis",
    "systematic review",
    "randomized controlled trial",
    "clinical trial",
    "practice guideline",
    "guideline",
    "review",
    "cohort",
    "case-control",
    "consensus",
}


def _tokens(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "article",
        "articles",
        "be",
        "can",
        "does",
        "evidence",
        "for",
        "from",
        "how",
        "in",
        "is",
        "of",
        "or",
        "paper",
        "papers",
        "research",
        "show",
        "study",
        "studies",
        "the",
        "to",
        "what",
        "whether",
        "why",
        "with",
    }
    return {token for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if token not in stopwords and len(token) > 1}


class Observer:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or NoopLLMClient()

    def observe(self, *, question: str, articles: list[Article]) -> ObserveResult:
        payload = self.llm.complete_json(
            system=(
                "Judge whether retrieved scholarly articles directly answer the research question. "
                "Do not rely on result count alone. Consider topical relevance, quotable evidence, "
                "source quality, and conflict."
            ),
            user={
                "question": question,
                "articles": [self._article_brief(article) for article in articles[:20]],
                "output_schema": {
                    "sufficient": "boolean",
                    "reason": "string",
                    "relevant_count": "integer",
                    "higher_quality_evidence": ["string"],
                    "conflicts": ["string"],
                    "suggested_refined_query": "string or null",
                    "relevant_article_keys": ["string"],
                },
            },
        )
        if payload and "sufficient" in payload:
            return ObserveResult(
                sufficient=bool(payload.get("sufficient")),
                reason=str(payload.get("reason") or ""),
                relevant_count=int(payload.get("relevant_count") or 0),
                higher_quality_evidence=list(payload.get("higher_quality_evidence") or []),
                conflicts=list(payload.get("conflicts") or []),
                suggested_refined_query=payload.get("suggested_refined_query"),
                relevant_article_keys=list(payload.get("relevant_article_keys") or []),
            )
        return self._fallback_observe(question, articles)

    def _fallback_observe(self, question: str, articles: list[Article]) -> ObserveResult:
        query_terms = _tokens(question)
        relevant: list[Article] = []
        high_quality: list[str] = []
        for article in articles:
            text = article.evidence_text().lower()
            overlap = len(query_terms & _tokens(text)) / max(len(query_terms), 1)
            direct = self._directly_addresses_question(question, text)
            has_quotable_text = len(_tokens(article.abstract or article.full_text or article.tldr or "")) >= 8
            if overlap >= 0.22 and direct and has_quotable_text:
                relevant.append(article)
                publication_text = " ".join(article.publication_types).lower()
                if any(term in publication_text or term in text[:800] for term in HIGH_QUALITY_TERMS):
                    high_quality.append(article.key)

        sufficient = len(relevant) >= 2
        if not relevant:
            reason = "No retrieved articles directly addressed the key terms in the question."
        elif not sufficient:
            reason = "Some relevant evidence was retrieved, but it is too sparse for a confident researcher-facing synthesis."
        else:
            reason = "Retrieved evidence contains multiple relevant articles with quotable abstracts or full text."

        refined = None
        if not sufficient:
            refined = self._refine_query(question)
        return ObserveResult(
            sufficient=sufficient,
            reason=reason,
            relevant_count=len(relevant),
            higher_quality_evidence=high_quality,
            conflicts=[],
            suggested_refined_query=refined,
            relevant_article_keys=[article.key for article in relevant],
        )

    @staticmethod
    def _refine_query(question: str) -> str:
        terms = [term for term in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9/-]*", question) if term.lower() in _tokens(question)]
        core = " ".join(terms[:10]) or question
        return f"{core} review evidence"

    @staticmethod
    def _directly_addresses_question(question: str, text: str) -> bool:
        question_terms = _tokens(question)
        if not question_terms:
            return False
        matched = question_terms & _tokens(text)
        coverage = len(matched) / max(len(question_terms), 1)
        distinctive_terms = {term for term in question_terms if len(term) >= 6}
        distinctive_match = bool(distinctive_terms & matched)
        return coverage >= 0.34 or (coverage >= 0.24 and distinctive_match)

    @staticmethod
    def _article_brief(article: Article) -> dict:
        return {
            "key": article.key,
            "title": article.title,
            "abstract": article.abstract[:1200],
            "publication_types": article.publication_types,
            "year": article.year,
            "pmid": article.pmid,
            "doi": article.doi,
            "citation_count": article.citation_count,
            "tldr": article.tldr,
        }
