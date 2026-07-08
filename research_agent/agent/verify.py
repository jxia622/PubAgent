from __future__ import annotations

import re

from research_agent.agent.llm import LLMClient, NoopLLMClient
from research_agent.models import Article, EvidenceQuote, VerifiedClaim, VerifyResult


class Verifier:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or NoopLLMClient()

    def verify(self, *, answer: str, articles: list[Article], evidence_quotes: list[EvidenceQuote] | None = None) -> VerifyResult:
        payload = self.llm.complete_json(
            system=(
                "Verify that each specific claim in the research summary is supported by retrieved passages "
                "and that each quoted excerpt appears in its source article. Return unsupported claims or quote problems."
            ),
            user={
                "answer": answer,
                "articles": [{"key": article.key, "text": article.evidence_text()[:1800]} for article in articles[:12]],
                "evidence_quotes": [
                    {"article_key": quote.article.key, "quote": quote.quote}
                    for quote in (evidence_quotes or [])
                ],
                "output_schema": {
                    "passed": "boolean",
                    "claims": [{"claim": "string", "supported": "boolean", "article_keys": ["string"], "reason": "string"}],
                    "removed_claims": ["string"],
                },
            },
        )
        if payload and "passed" in payload:
            claims = [
                VerifiedClaim(
                    claim=str(item.get("claim") or ""),
                    supported=bool(item.get("supported")),
                    article_keys=list(item.get("article_keys") or []),
                    reason=str(item.get("reason") or ""),
                )
                for item in payload.get("claims", [])
            ]
            return VerifyResult(passed=bool(payload.get("passed")), claims=claims, removed_claims=list(payload.get("removed_claims") or []))
        return self._fallback_verify(answer, articles, evidence_quotes or [])

    def _fallback_verify(self, answer: str, articles: list[Article], evidence_quotes: list[EvidenceQuote]) -> VerifyResult:
        evidence = " ".join(article.evidence_text().lower() for article in articles)
        claims: list[VerifiedClaim] = []
        removed: list[str] = []
        article_text_by_key = {article.key: " ".join(article.evidence_text().lower().split()) for article in articles}
        for quote in evidence_quotes:
            normalized_quote = " ".join(quote.quote.lower().replace("...", "").split())
            source_text = article_text_by_key.get(quote.article.key, "")
            supported = bool(normalized_quote) and normalized_quote in source_text
            if not supported:
                removed.append(quote.quote)
            claims.append(
                VerifiedClaim(
                    claim=f"quote:{quote.article.key}",
                    supported=supported,
                    article_keys=[quote.article.key],
                    reason="Rule-based exact excerpt grounding check.",
                )
            )
        for bullet in re.findall(r"^- (.+)$", answer, flags=re.M):
            claim_text = re.sub(r"\([^)]*(PMID|DOI|pubmed|semantic_scholar|europe_pmc)[^)]*\)", "", bullet, flags=re.I).strip()
            terms = [term for term in re.findall(r"[a-zA-Z0-9]+", claim_text.lower()) if len(term) > 3]
            matched = sum(1 for term in set(terms) if term in evidence)
            supported = matched >= max(2, min(5, len(set(terms)) // 3))
            if not supported:
                removed.append(bullet)
            claims.append(VerifiedClaim(claim=bullet, supported=supported, article_keys=[article.key for article in articles[:3]], reason="Rule-based lexical grounding check."))
        return VerifyResult(passed=not removed, claims=claims, removed_claims=removed)
