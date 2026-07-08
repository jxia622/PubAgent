from __future__ import annotations

import os

from research_agent.models import Article
from research_agent.sources.http import RetryHttpClient


SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


class SemanticScholarClient:
    def __init__(self, *, http: RetryHttpClient | None = None, api_key: str | None = None) -> None:
        self.http = http or RetryHttpClient()
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    def search(self, query: str, *, limit: int = 10) -> list[Article]:
        headers = {"x-api-key": self.api_key} if self.api_key else None
        payload = self.http.get_json(
            SEMANTIC_SCHOLAR_SEARCH_URL,
            params={
                "query": query,
                "limit": str(limit),
                "fields": "title,abstract,year,authors,citationCount,influentialCitationCount,tldr,externalIds,url,journal",
            },
            headers=headers,
        )
        articles: list[Article] = []
        for item in payload.get("data", []):
            external_ids = item.get("externalIds") or {}
            authors = [author.get("name", "") for author in item.get("authors", []) if author.get("name")]
            tldr = item.get("tldr") or {}
            doi = external_ids.get("DOI")
            pmid = external_ids.get("PubMed")
            pmcid = external_ids.get("PubMedCentral")
            if pmcid and not str(pmcid).upper().startswith("PMC"):
                pmcid = f"PMC{pmcid}"
            articles.append(
                Article(
                    title=item.get("title") or "Untitled Semantic Scholar result",
                    source="semantic_scholar",
                    url=item.get("url") or "",
                    abstract=item.get("abstract") or "",
                    authors=authors,
                    journal=(item.get("journal") or {}).get("name") if isinstance(item.get("journal"), dict) else None,
                    year=str(item.get("year")) if item.get("year") else None,
                    pmid=pmid,
                    pmcid=pmcid,
                    doi=doi,
                    citation_count=item.get("citationCount"),
                    influential_citation_count=item.get("influentialCitationCount"),
                    tldr=tldr.get("text") if isinstance(tldr, dict) else None,
                    metadata={"paper_id": item.get("paperId")},
                )
            )
        return articles

