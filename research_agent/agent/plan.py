from __future__ import annotations

from research_agent.agent.llm import LLMClient, NoopLLMClient
from research_agent.models import ObserveResult, PlanResult, SearchQuery
from research_agent.sources.pubmed import build_pubmed_query


SOURCE_NAMES = {"pubmed", "pmc", "semantic_scholar", "europe_pmc", "unpaywall"}


class Planner:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or NoopLLMClient()

    def plan(
        self,
        *,
        question: str,
        iteration: int,
        previous_observation: ObserveResult | None = None,
    ) -> PlanResult:
        payload = self.llm.complete_json(
            system=(
                "Plan live scholarly literature searches for a researcher. Choose only from "
                "semantic_scholar, pubmed, europe_pmc, pmc, unpaywall. Prefer Semantic Scholar "
                "for broad academic coverage and PubMed/PMC/Europe PMC for biomedical or life-science questions. "
                "Prefer queries likely to retrieve citable primary articles, reviews, or guidelines."
            ),
            user={
                "question": question,
                "iteration": iteration,
                "previous_observation": previous_observation.to_dict() if previous_observation else None,
                "output_schema": {
                    "queries": [{"query": "string", "sources": ["pubmed"], "filters": {}, "reasoning": "string"}],
                    "reasoning": "string",
                },
            },
        )
        if payload:
            parsed = self._parse_llm_payload(payload, iteration)
            if parsed.queries:
                return parsed
        return self._fallback_plan(question, iteration, previous_observation)

    def _parse_llm_payload(self, payload: dict, iteration: int) -> PlanResult:
        queries: list[SearchQuery] = []
        for raw in payload.get("queries", []):
            query = str(raw.get("query") or "").strip()
            sources = [source for source in raw.get("sources", []) if source in SOURCE_NAMES]
            if query and sources:
                queries.append(
                    SearchQuery(
                        query=query,
                        sources=sources,
                        filters=raw.get("filters") if isinstance(raw.get("filters"), dict) else {},
                        reasoning=str(raw.get("reasoning") or ""),
                    )
                )
        return PlanResult(iteration=iteration, queries=queries, reasoning=str(payload.get("reasoning") or "llm plan"))

    def _fallback_plan(
        self,
        question: str,
        iteration: int,
        previous_observation: ObserveResult | None,
    ) -> PlanResult:
        base_query = previous_observation.suggested_refined_query if previous_observation and previous_observation.suggested_refined_query else build_pubmed_query(question)
        sources = ["semantic_scholar", "pubmed"]
        if iteration >= 2:
            sources.append("europe_pmc")
        if iteration >= 3:
            sources.append("pmc")
        return PlanResult(
            iteration=iteration,
            queries=[
                SearchQuery(
                    query=base_query,
                    sources=sources,
                    filters={"english_only": True},
                    reasoning="Deterministic fallback: broad scholarly search first, add full-text/open biomedical sources in later iterations if needed.",
                )
            ],
            reasoning="deterministic fallback plan",
        )
