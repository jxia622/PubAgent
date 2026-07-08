from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from research_agent.models import Article
from research_agent.sources.http import RetryHttpClient


NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def normalize_pmcid(pmcid: str) -> str:
    value = pmcid.strip()
    return value if value.upper().startswith("PMC") else f"PMC{value}"


def _plain_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def parse_pmc_article(xml_text: str, pmcid: str) -> Article | None:
    root = ET.fromstring(xml_text)
    title = _plain_text(root.find(".//front//article-title")) or f"PMC article {pmcid}"
    abstract = _plain_text(root.find(".//abstract"))
    body = _plain_text(root.find(".//body"))
    if not abstract and not body:
        return None
    if len(body) > 7000:
        body = body[:6997].rstrip() + "..."
    return Article(
        title=title,
        source="pmc_open_access",
        url=f"https://pmc.ncbi.nlm.nih.gov/articles/{normalize_pmcid(pmcid)}/",
        abstract=abstract,
        full_text=body,
        pmcid=normalize_pmcid(pmcid),
    )


class PMCClient:
    def __init__(self, *, http: RetryHttpClient | None = None) -> None:
        self.http = http or RetryHttpClient()

    def fetch_full_text(self, pmcid: str) -> Article | None:
        normalized = normalize_pmcid(pmcid)
        numeric = re.sub(r"^PMC", "", normalized, flags=re.I)
        xml_text = self.http.get_text(
            f"{NCBI_EUTILS}/efetch.fcgi",
            params={"db": "pmc", "id": numeric, "retmode": "xml", "tool": "research_agent"},
        )
        return parse_pmc_article(xml_text, normalized)

