from __future__ import annotations

import re

from research_agent.models import Article
from research_agent.sources.http import RetryHttpClient


EUROPE_PMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


class EuropePMCClient:
    def __init__(self, *, http: RetryHttpClient | None = None) -> None:
        self.http = http or RetryHttpClient()

    def search(self, query: str, *, page_size: int = 10) -> list[Article]:
        payload = self.http.get_json(
            EUROPE_PMC_SEARCH_URL,
            params={
                "query": query,
                "format": "json",
                "pageSize": str(page_size),
                "resultType": "core",
            },
        )
        results = payload.get("resultList", {}).get("result", [])
        articles: list[Article] = []
        for item in results:
            pmcid = item.get("pmcid")
            abstract = re.sub(r"<[^>]+>", " ", item.get("abstractText") or "")
            abstract = " ".join(abstract.split())
            articles.append(
                Article(
                    title=item.get("title") or "Untitled Europe PMC result",
                    source="europe_pmc",
                    url=item.get("fullTextUrlList", {}).get("fullTextUrl", [{}])[0].get("url")
                    if isinstance(item.get("fullTextUrlList"), dict)
                    else item.get("doiUrl")
                    or f"https://europepmc.org/article/{item.get('source', 'MED')}/{item.get('id', '')}",
                    abstract=abstract,
                    authors=[name.strip() for name in (item.get("authorString") or "").split(",") if name.strip()],
                    journal=item.get("journalTitle"),
                    year=item.get("pubYear"),
                    pmid=item.get("pmid"),
                    pmcid=pmcid if not pmcid or pmcid.upper().startswith("PMC") else f"PMC{pmcid}",
                    doi=item.get("doi"),
                    metadata={"europe_pmc_id": item.get("id"), "source": item.get("source")},
                )
            )
        return articles
