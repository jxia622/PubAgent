from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET

from research_agent.models import Article
from research_agent.sources.http import RetryHttpClient


NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_URL = "https://pubmed.ncbi.nlm.nih.gov"


def build_pubmed_query(question: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9/-]*", question)
    stopwords = {
        "a",
        "an",
        "are",
        "article",
        "articles",
        "be",
        "can",
        "does",
        "evidence",
        "for",
        "how",
        "is",
        "me",
        "might",
        "of",
        "show",
        "supports",
        "the",
        "to",
        "what",
        "whether",
        "why",
        "with",
    }
    terms = [token for token in tokens if token.lower() not in stopwords]
    return " ".join(terms[:12]) or question


def _text(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    value = " ".join("".join(element.itertext()).split())
    return value or None


def _first(parent: ET.Element, path: str) -> str | None:
    return _text(parent.find(path))


def _abstract(article: ET.Element) -> str:
    sections: list[str] = []
    for abstract_text in article.findall(".//Article/Abstract/AbstractText"):
        label = abstract_text.attrib.get("Label")
        text = _text(abstract_text)
        if not text:
            continue
        sections.append(f"{label}: {text}" if label else text)
    return " ".join(sections)


def _authors(article: ET.Element, limit: int = 8) -> list[str]:
    authors: list[str] = []
    for author in article.findall(".//Article/AuthorList/Author"):
        collective = _first(author, "CollectiveName")
        if collective:
            authors.append(collective)
            continue
        last = _first(author, "LastName")
        initials = _first(author, "Initials")
        if last:
            authors.append(f"{last} {initials or ''}".strip())
    if len(authors) > limit:
        return authors[:limit] + ["et al."]
    return authors


def _year(article: ET.Element) -> str | None:
    for path in [
        ".//Article/Journal/JournalIssue/PubDate/Year",
        ".//Article/ArticleDate/Year",
        ".//DateCompleted/Year",
        ".//DateRevised/Year",
    ]:
        year = _first(article, path)
        if year:
            return year
    medline_date = _first(article, ".//Article/Journal/JournalIssue/PubDate/MedlineDate")
    if medline_date:
        match = re.search(r"\d{4}", medline_date)
        return match.group(0) if match else None
    return None


def _article_ids(article: ET.Element) -> tuple[str | None, str | None]:
    doi = None
    pmcid = None
    for article_id in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
        id_type = article_id.attrib.get("IdType")
        value = _text(article_id)
        if id_type == "doi":
            doi = value
        elif id_type in {"pmc", "pmcid"}:
            pmcid = value if value and value.upper().startswith("PMC") else f"PMC{value}" if value else None
    return doi, pmcid


def parse_pubmed_articles(xml_text: str) -> list[Article]:
    root = ET.fromstring(xml_text)
    articles: list[Article] = []
    for item in root.findall(".//PubmedArticle"):
        pmid = _first(item, ".//MedlineCitation/PMID")
        title = _first(item, ".//Article/ArticleTitle") or "Untitled PubMed article"
        doi, pmcid = _article_ids(item)
        mesh_terms = [_text(node) for node in item.findall(".//MeshHeading/DescriptorName")]
        publication_types = [_text(node) for node in item.findall(".//PublicationTypeList/PublicationType")]
        articles.append(
            Article(
                title=title,
                source="pubmed",
                url=f"{PUBMED_URL}/{pmid}/" if pmid else PUBMED_URL,
                abstract=_abstract(item),
                authors=_authors(item),
                journal=_first(item, ".//Article/Journal/Title"),
                year=_year(item),
                pmid=pmid,
                pmcid=pmcid,
                doi=doi,
                mesh_terms=[term for term in mesh_terms if term],
                publication_types=[kind for kind in publication_types if kind],
            )
        )
    return articles


class PubMedClient:
    def __init__(self, *, http: RetryHttpClient | None = None, email: str | None = None, api_key: str | None = None) -> None:
        self.http = http or RetryHttpClient()
        self.email = email or os.getenv("NCBI_EMAIL")
        self.api_key = api_key or os.getenv("NCBI_API_KEY")

    def _base_params(self) -> dict[str, str]:
        params = {"tool": "research_agent"}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def search(self, query: str, *, retmax: int = 20) -> list[str]:
        params = {
            **self._base_params(),
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": str(retmax),
            "sort": "relevance",
        }
        payload = self.http.get_json(f"{NCBI_EUTILS}/esearch.fcgi", params=params)
        return list(payload.get("esearchresult", {}).get("idlist", []))

    def fetch(self, pmids: list[str]) -> list[Article]:
        if not pmids:
            return []
        params = {
            **self._base_params(),
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        xml_text = self.http.get_text(f"{NCBI_EUTILS}/efetch.fcgi", params=params)
        return parse_pubmed_articles(xml_text)
