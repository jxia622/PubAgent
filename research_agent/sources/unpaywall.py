from __future__ import annotations

import os

from research_agent.sources.http import RetryHttpClient


UNPAYWALL_URL = "https://api.unpaywall.org/v2"


class UnpaywallClient:
    def __init__(self, *, http: RetryHttpClient | None = None, email: str | None = None) -> None:
        self.http = http or RetryHttpClient()
        self.email = email or os.getenv("UNPAYWALL_EMAIL") or os.getenv("NCBI_EMAIL")

    def find_oa_url(self, doi: str) -> str | None:
        if not self.email:
            return None
        payload = self.http.get_json(f"{UNPAYWALL_URL}/{doi}", params={"email": self.email})
        best = payload.get("best_oa_location") or {}
        return best.get("url_for_pdf") or best.get("url")

