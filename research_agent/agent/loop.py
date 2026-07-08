from __future__ import annotations

from dataclasses import replace

from research_agent.agent.act import RetrievalActor
from research_agent.agent.observe import Observer
from research_agent.agent.plan import Planner
from research_agent.agent.synthesize import Synthesizer, select_evidence_quotes
from research_agent.agent.verify import Verifier
from research_agent.models import Article, LoopTraceEntry, ObserveResult, PlanResult, ResearchAnswer, SearchQuery, SearchSettings


class ResearchAgent:
    def __init__(
        self,
        *,
        planner: Planner | None = None,
        actor: RetrievalActor | None = None,
        observer: Observer | None = None,
        synthesizer: Synthesizer | None = None,
        verifier: Verifier | None = None,
        max_iterations: int = 3,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")
        self.planner = planner or Planner()
        self.actor = actor or RetrievalActor()
        self.observer = observer or Observer()
        self.synthesizer = synthesizer or Synthesizer()
        self.verifier = verifier or Verifier()
        self.max_iterations = max_iterations

    def run(self, question: str, *, per_source_limit: int = 20, settings: SearchSettings | None = None) -> ResearchAnswer:
        if per_source_limit < 1:
            raise ValueError("per_source_limit must be at least 1.")
        settings = settings or SearchSettings()
        trace: list[LoopTraceEntry] = []
        articles: list[Article] = []
        observation: ObserveResult | None = None
        final_iteration = 0

        for iteration in range(1, self.max_iterations + 1):
            final_iteration = iteration
            plan = self.planner.plan(question=question, iteration=iteration, previous_observation=observation)
            plan = self._apply_settings_to_plan(plan, settings)
            trace.append(LoopTraceEntry(step="plan", iteration=iteration, status="ok", details=plan.to_dict()))

            new_articles = self.actor.act(plan, per_source_limit=per_source_limit)
            articles = self._dedupe([*articles, *new_articles])
            articles = self._filter_articles(articles, settings)
            trace.append(
                LoopTraceEntry(
                    step="act",
                    iteration=iteration,
                    status="ok",
                    details={
                        "new_articles": len(new_articles),
                        "filtered_total": len(articles),
                        "settings": settings.to_dict(),
                    },
                )
            )

            observation = self.observer.observe(question=question, articles=articles)
            trace.append(LoopTraceEntry(step="observe", iteration=iteration, status="sufficient" if observation.sufficient else "insufficient", details=observation.to_dict()))

            if observation.sufficient:
                trace.append(LoopTraceEntry(step="decide", iteration=iteration, status="stop", details={"reason": observation.reason}))
                break
            trace.append(LoopTraceEntry(step="decide", iteration=iteration, status="refine" if iteration < self.max_iterations else "cap_reached", details={"reason": observation.reason}))

        assert observation is not None
        ranked_articles = self._rank(question, articles)
        if observation.relevant_article_keys:
            relevant_keys = set(observation.relevant_article_keys)
            ranked_articles = [article for article in ranked_articles if article.key in relevant_keys] + [
                article for article in ranked_articles if article.key not in relevant_keys
            ]
        if not observation.sufficient:
            synthesis_articles = [article for article in ranked_articles if article.key in set(observation.relevant_article_keys)]
        else:
            synthesis_articles = ranked_articles
        evidence_quotes = select_evidence_quotes(
            question,
            synthesis_articles,
            max_quotes=settings.quote_count,
            text_mode=settings.text_mode,
        )
        summary = self.synthesizer.synthesize(
            question=question,
            articles=synthesis_articles,
            observation=observation,
            evidence_quotes=evidence_quotes,
        )
        trace.append(
            LoopTraceEntry(
                step="synthesize",
                iteration=final_iteration,
                status="ok",
                details={"article_count": len(synthesis_articles), "quote_count": len(evidence_quotes), "settings": settings.to_dict()},
            )
        )
        verification = self.verifier.verify(answer=summary, articles=synthesis_articles, evidence_quotes=evidence_quotes)
        trace.append(LoopTraceEntry(step="verify", iteration=final_iteration, status="passed" if verification.passed else "issues_found", details=verification.to_dict()))
        evidence_note = self._evidence_quality_note(observation)
        if not observation.sufficient:
            evidence_note = f"Low confidence: iteration cap reached or evidence remained insufficient. {observation.reason}"

        return ResearchAnswer(
            summary=summary,
            ranked_evidence=evidence_quotes,
            citations=synthesis_articles[:10],
            evidence_quality_note=evidence_note,
            sufficient=observation.sufficient,
            iterations=final_iteration,
            trace=trace,
            verification=verification,
        )

    @staticmethod
    def _dedupe(articles: list[Article]) -> list[Article]:
        merged: dict[str, Article] = {}
        for article in articles:
            merged.setdefault(article.key, article)
        return list(merged.values())

    @staticmethod
    def _apply_settings_to_plan(plan: PlanResult, settings: SearchSettings) -> PlanResult:
        queries: list[SearchQuery] = []
        for query in plan.queries:
            sources = list(query.sources)
            if settings.text_mode == "full-text" and "pmc" not in sources:
                sources.append("pmc")
            filters = {
                **query.filters,
                **{key: value for key, value in settings.to_dict().items() if value is not None},
            }
            queries.append(replace(query, sources=sources, filters=filters))
        return replace(plan, queries=queries)

    @staticmethod
    def _filter_articles(articles: list[Article], settings: SearchSettings) -> list[Article]:
        filtered: list[Article] = []
        for article in articles:
            year = ResearchAgent._article_year(article)
            if settings.year_start is not None and (year is None or year < settings.year_start):
                continue
            if settings.year_end is not None and (year is None or year > settings.year_end):
                continue
            if settings.text_mode == "abstract" and not article.abstract:
                continue
            if settings.text_mode == "full-text" and not article.full_text:
                continue
            filtered.append(article)
        return filtered

    @staticmethod
    def _article_year(article: Article) -> int | None:
        if not article.year:
            return None
        raw_year = str(article.year).strip()[:4]
        return int(raw_year) if raw_year.isdigit() else None

    @staticmethod
    def _rank(question: str, articles: list[Article]) -> list[Article]:
        terms = {term.strip(".,:;!?()[]{}").lower() for term in question.replace("/", " ").split() if len(term) > 2}

        def score(article: Article) -> tuple[int, int, int]:
            text = article.evidence_text().lower()
            overlap = sum(1 for term in terms if term in text)
            quality = sum(1 for kind in article.publication_types if "review" in kind.lower() or "trial" in kind.lower() or "guideline" in kind.lower())
            citations = article.citation_count or 0
            full_text = 1 if article.full_text else 0
            recency = int(article.year) if article.year and article.year.isdigit() else 0
            return (overlap + quality + full_text, citations, recency)

        return sorted(articles, key=score, reverse=True)

    @staticmethod
    def _evidence_quality_note(observation: ObserveResult) -> str:
        if observation.higher_quality_evidence:
            return f"Moderate confidence: {observation.relevant_count} relevant articles found, including higher-quality evidence signals."
        return f"Limited confidence: {observation.relevant_count} relevant articles found, with limited evidence-quality signals."


LiteratureSearchAgent = ResearchAgent
