from __future__ import annotations

import json
from dataclasses import replace

from research_agent.cache.store import JsonCacheStore
from research_agent.models import Article, PlanResult
from research_agent.sources.europe_pmc import EuropePMCClient
from research_agent.sources.pmc import PMCClient
from research_agent.sources.pubmed import PubMedClient
from research_agent.sources.semantic_scholar import SemanticScholarClient
from research_agent.sources.unpaywall import UnpaywallClient


class RetrievalActor:
    def __init__(
        self,
        *,
        cache: JsonCacheStore | None = None,
        pubmed: PubMedClient | None = None,
        pmc: PMCClient | None = None,
        semantic_scholar: SemanticScholarClient | None = None,
        europe_pmc: EuropePMCClient | None = None,
        unpaywall: UnpaywallClient | None = None,
    ) -> None:
        self.cache = cache or JsonCacheStore()
        self.pubmed = pubmed or PubMedClient()
        self.pmc = pmc or PMCClient()
        self.semantic_scholar = semantic_scholar or SemanticScholarClient()
        self.europe_pmc = europe_pmc or EuropePMCClient()
        self.unpaywall = unpaywall or UnpaywallClient()

    def act(self, plan: PlanResult, *, per_source_limit: int = 20) -> list[Article]:
        all_articles: list[Article] = []
        for query in plan.queries:
            for source in query.sources:
                if source == "pubmed":
                    all_articles.extend(self._pubmed(query.query, per_source_limit))
                elif source == "semantic_scholar":
                    all_articles.extend(self._semantic_scholar(query.query, min(per_source_limit, 10)))
                elif source == "europe_pmc":
                    all_articles.extend(self._europe_pmc(query.query, min(per_source_limit, 10)))

        merged = self._dedupe(all_articles)
        if any("pmc" in query.sources for query in plan.queries):
            merged = self._attach_pmc_full_text(merged)
        merged = self._attach_unpaywall_links(merged)
        return merged

    def _pubmed(self, query: str, limit: int) -> list[Article]:
        cached = self.cache.get("pubmed_search_fetch", json.dumps({"query": query, "limit": limit}, sort_keys=True))
        if cached is not None:
            return [Article(**item) for item in cached]
        try:
            pmids = self.pubmed.search(query, retmax=limit)
            articles = self.pubmed.fetch(pmids)
        except Exception:
            return []
        self.cache.set("pubmed_search_fetch", json.dumps({"query": query, "limit": limit}, sort_keys=True), [article.to_dict() for article in articles])
        return articles

    def _semantic_scholar(self, query: str, limit: int) -> list[Article]:
        cached = self.cache.get("semantic_scholar_search", json.dumps({"query": query, "limit": limit}, sort_keys=True))
        if cached is not None:
            return [Article(**item) for item in cached]
        try:
            articles = self.semantic_scholar.search(query, limit=limit)
        except Exception:
            return []
        self.cache.set("semantic_scholar_search", json.dumps({"query": query, "limit": limit}, sort_keys=True), [article.to_dict() for article in articles])
        return articles

    def _europe_pmc(self, query: str, limit: int) -> list[Article]:
        cached = self.cache.get("europe_pmc_search", json.dumps({"query": query, "limit": limit}, sort_keys=True))
        if cached is not None:
            return [Article(**item) for item in cached]
        try:
            articles = self.europe_pmc.search(query, page_size=limit)
        except Exception:
            return []
        self.cache.set("europe_pmc_search", json.dumps({"query": query, "limit": limit}, sort_keys=True), [article.to_dict() for article in articles])
        return articles

    def _attach_pmc_full_text(self, articles: list[Article]) -> list[Article]:
        updated: list[Article] = []
        for article in articles:
            if not article.pmcid or article.full_text:
                updated.append(article)
                continue
            cached = self.cache.get("pmc_full_text", article.pmcid)
            try:
                full_text_article = Article(**cached) if cached is not None else self.pmc.fetch_full_text(article.pmcid)
            except Exception:
                full_text_article = None
            if full_text_article and cached is None:
                self.cache.set("pmc_full_text", article.pmcid, full_text_article.to_dict())
            updated.append(replace(article, full_text=full_text_article.full_text if full_text_article else article.full_text))
        return updated

    def _attach_unpaywall_links(self, articles: list[Article]) -> list[Article]:
        updated: list[Article] = []
        for article in articles:
            if not article.doi or article.oa_url:
                updated.append(article)
                continue
            cached = self.cache.get("unpaywall", article.doi)
            if cached is None:
                try:
                    cached = self.unpaywall.find_oa_url(article.doi)
                except Exception:
                    cached = None
                self.cache.set("unpaywall", article.doi, cached)
            updated.append(replace(article, oa_url=cached))
        return updated

    @staticmethod
    def _dedupe(articles: list[Article]) -> list[Article]:
        merged: dict[str, Article] = {}
        for article in articles:
            existing = merged.get(article.key)
            if existing is None:
                merged[article.key] = article
                continue
            merged[article.key] = replace(
                existing,
                abstract=existing.abstract or article.abstract,
                full_text=existing.full_text or article.full_text,
                tldr=existing.tldr or article.tldr,
                citation_count=existing.citation_count if existing.citation_count is not None else article.citation_count,
                influential_citation_count=existing.influential_citation_count
                if existing.influential_citation_count is not None
                else article.influential_citation_count,
                pmcid=existing.pmcid or article.pmcid,
                doi=existing.doi or article.doi,
                oa_url=existing.oa_url or article.oa_url,
            )
        return list(merged.values())
