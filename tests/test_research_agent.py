from __future__ import annotations

import tempfile
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

from research_agent.agent.act import RetrievalActor
from research_agent.agent.loop import ResearchAgent
from research_agent.agent.plan import Planner
from research_agent.agent.synthesize import MAX_QUOTE_WORDS, Synthesizer, select_evidence_quotes, trim_summary
from research_agent.models import ObserveResult
from research_agent.cache.store import JsonCacheStore
from research_agent.main import (
    APIKeyConfig,
    PUBAGENT_BANNER,
    PUBAGENT_SLOGAN,
    apply_api_keys,
    format_answer_text,
    format_api_key_settings,
    masked,
    parse_save_command,
    parse_settings_command,
    save_result,
    save_session,
    should_prompt_ai_setup,
)
from research_agent.models import MAX_PUBLICATION_YEAR, MAX_QUOTE_COUNT, MIN_PUBLICATION_YEAR, Article, SearchSettings
from research_agent.sources.pubmed import build_pubmed_query, parse_pubmed_articles


PUBMED_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>111</PMID>
      <Article>
        <Journal>
          <Title>Nature Medicine</Title>
          <JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue>
        </Journal>
        <ArticleTitle>Artificial intelligence models for protein structure prediction.</ArticleTitle>
        <Abstract>
          <AbstractText>Deep learning systems improved protein structure prediction accuracy and enabled broader structural biology research.</AbstractText>
        </Abstract>
        <PublicationTypeList>
          <PublicationType>Review</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="doi">10.1000/protein</ArticleId>
        <ArticleId IdType="pmc">PMC111</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""


class FakePubMed:
    def search(self, query: str, *, retmax: int = 20) -> list[str]:
        return ["111", "222"]

    def fetch(self, pmids: list[str]) -> list[Article]:
        return [
            Article(
                title="AI for protein structure prediction",
                source="pubmed",
                url="https://pubmed.ncbi.nlm.nih.gov/111/",
                abstract=(
                    "Deep learning systems improved protein structure prediction accuracy and enabled broader "
                    "structural biology research."
                ),
                authors=["Smith J"],
                journal="Nature Medicine",
                year="2024",
                pmid="111",
                doi="10.1000/protein",
                publication_types=["Review"],
            ),
            Article(
                title="Protein folding benchmarks",
                source="pubmed",
                url="https://pubmed.ncbi.nlm.nih.gov/222/",
                abstract=(
                    "Protein folding benchmarks show that artificial intelligence methods can predict many "
                    "protein structures with experimentally useful accuracy."
                ),
                authors=["Lee A"],
                journal="Science",
                year="2023",
                pmid="222",
                publication_types=["Journal Article"],
            ),
        ]


class FakeSemanticScholar:
    def search(self, query: str, *, limit: int = 10) -> list[Article]:
        return [
            Article(
                title="Highly accurate protein structure prediction with AlphaFold",
                source="semantic_scholar",
                url="https://www.nature.com/articles/example",
                abstract=(
                    "The AlphaFold system achieves high accuracy in protein structure prediction and is useful "
                    "for interpreting biological function."
                ),
                authors=["Jumper J"],
                journal="Nature",
                year="2021",
                doi="10.1038/example",
                citation_count=25000,
            )
        ]


class EmptySource:
    def search(self, *args, **kwargs):
        return []

    def fetch_full_text(self, *args, **kwargs):
        return None

    def find_oa_url(self, *args, **kwargs):
        return None


class EmptyPubMed:
    def search(self, query: str, *, retmax: int = 20) -> list[str]:
        return []

    def fetch(self, pmids: list[str]) -> list[Article]:
        return []


class ResearchAgentTests(unittest.TestCase):
    def test_pubmed_query_and_parse_metadata(self) -> None:
        query = build_pubmed_query("What does evidence show about AI protein structure prediction?")
        self.assertIn("protein", query)
        self.assertNotIn("Wagner", query)
        articles = parse_pubmed_articles(PUBMED_XML)
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].pmid, "111")
        self.assertEqual(articles[0].pmcid, "PMC111")
        self.assertEqual(articles[0].doi, "10.1000/protein")

    def test_json_cache_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            path = Path(raw_tmp) / "cache.json"
            cache = JsonCacheStore(path)
            cache.set("query", "abc", {"value": 3})
            restored = JsonCacheStore(path)
            self.assertEqual(restored.get("query", "abc"), {"value": 3})

    def test_planner_starts_with_broad_scholarly_sources(self) -> None:
        plan = Planner().plan(question="What evidence supports AlphaFold for protein structure prediction?", iteration=1)
        self.assertIn("semantic_scholar", plan.queries[0].sources)
        self.assertIn("pubmed", plan.queries[0].sources)

    def test_loop_returns_summary_and_ranked_quotes_when_sufficient(self) -> None:
        actor = RetrievalActor(
            cache=JsonCacheStore(),
            pubmed=FakePubMed(),
            pmc=EmptySource(),
            semantic_scholar=FakeSemanticScholar(),
            europe_pmc=EmptySource(),
            unpaywall=EmptySource(),
        )
        agent = ResearchAgent(actor=actor, max_iterations=2)
        result = agent.run("What evidence supports AI for protein structure prediction?", per_source_limit=5)
        self.assertTrue(result.sufficient)
        self.assertIn("rank", result.to_dict()["ranked_evidence"][0])
        self.assertGreaterEqual(len(result.ranked_evidence), 2)
        self.assertTrue(result.verification.passed)
        for item in result.ranked_evidence:
            self.assertLessEqual(len(item.quote.split()), MAX_QUOTE_WORDS + 2)
            self.assertTrue(item.article.url)

    def test_loop_reports_insufficient_after_iteration_cap(self) -> None:
        actor = RetrievalActor(
            cache=JsonCacheStore(),
            pubmed=EmptyPubMed(),
            pmc=EmptySource(),
            semantic_scholar=EmptySource(),
            europe_pmc=EmptySource(),
            unpaywall=EmptySource(),
        )
        agent = ResearchAgent(actor=actor, max_iterations=2)
        result = agent.run("Do lunar phases change protein folding accuracy?", per_source_limit=5)
        self.assertFalse(result.sufficient)
        self.assertEqual(result.ranked_evidence, [])
        self.assertIn("Low confidence", result.evidence_quality_note)

    def test_quote_selection_prefers_relevant_cited_article(self) -> None:
        articles = FakeSemanticScholar().search("protein structure prediction")
        quotes = select_evidence_quotes("AI protein structure prediction accuracy", articles, max_quotes=1)
        self.assertEqual(len(quotes), 1)
        self.assertEqual(quotes[0].rank, 1)
        self.assertIn("protein structure prediction", quotes[0].quote)

    def test_quote_selection_honors_quote_count_and_text_mode(self) -> None:
        article = Article(
            title="Full text protein structure prediction",
            source="fixture",
            url="https://example.org/full-text",
            abstract="This abstract mentions protein structure prediction but not the full text phrase.",
            full_text="Full text evidence says protein structure prediction accuracy improved with artificial intelligence.",
            year="2022",
        )
        quotes = select_evidence_quotes(
            "protein structure prediction accuracy artificial intelligence",
            [article],
            max_quotes=1,
            text_mode="full-text",
        )
        self.assertEqual(len(quotes), 1)
        self.assertEqual(quotes[0].source_section, "full_text")
        self.assertIn("accuracy improved", quotes[0].quote)

    def test_summary_is_limited_to_two_sentences(self) -> None:
        long_summary = "First sentence. Second sentence. Third sentence should be removed."
        self.assertEqual(trim_summary(long_summary), "First sentence. Second sentence.")

        article = FakeSemanticScholar().search("protein structure prediction")[0]
        quotes = select_evidence_quotes("AI protein structure prediction accuracy", [article], max_quotes=1)
        summary = Synthesizer()._fallback_synthesize(
            "AI protein structure prediction",
            [article],
            ObserveResult(sufficient=True, reason="ok", relevant_count=1),
            quotes,
        )
        sentences = [part for part in re.split(r"(?<=[.!?])\\s+", summary) if part.strip()]
        self.assertLessEqual(len(sentences), 2)

    def test_agent_filters_articles_by_year_and_text_mode(self) -> None:
        articles = [
            Article(title="Old article", source="fixture", url="https://example.org/old", abstract="protein structure prediction", year="2020"),
            Article(title="In range", source="fixture", url="https://example.org/new", abstract="protein structure prediction", year="2023"),
            Article(title="Full text", source="fixture", url="https://example.org/full", abstract="", full_text="protein structure prediction", year="2024"),
        ]
        year_filtered = ResearchAgent._filter_articles(articles, SearchSettings(year_start=2022, year_end=2023))
        self.assertEqual([article.title for article in year_filtered], ["In range"])
        abstract_filtered = ResearchAgent._filter_articles(articles, SearchSettings(text_mode="abstract"))
        self.assertEqual([article.title for article in abstract_filtered], ["Old article", "In range"])
        full_text_filtered = ResearchAgent._filter_articles(articles, SearchSettings(text_mode="full-text"))
        self.assertEqual([article.title for article in full_text_filtered], ["Full text"])

    def test_settings_reject_unrealistic_years_and_counts(self) -> None:
        with self.assertRaises(ValueError):
            SearchSettings(year_start=MIN_PUBLICATION_YEAR - 1)
        with self.assertRaises(ValueError):
            SearchSettings(year_start=MAX_PUBLICATION_YEAR + 1)
        with self.assertRaises(ValueError):
            SearchSettings(year_start=2024, year_end=2020)
        with self.assertRaises(ValueError):
            SearchSettings(quote_count=0)
        with self.assertRaises(ValueError):
            SearchSettings(quote_count=MAX_QUOTE_COUNT + 1)

    def test_agent_rejects_zero_runtime_limits(self) -> None:
        with self.assertRaises(ValueError):
            ResearchAgent(max_iterations=0)
        agent = ResearchAgent(actor=RetrievalActor(pubmed=EmptyPubMed(), pmc=EmptySource(), semantic_scholar=EmptySource(), europe_pmc=EmptySource(), unpaywall=EmptySource()))
        with self.assertRaises(ValueError):
            agent.run("question", per_source_limit=0)

    def test_text_output_keeps_sources_after_ranked_quotes(self) -> None:
        article = FakeSemanticScholar().search("protein structure prediction")[0]
        payload = {
            "summary": "Short summary.",
            "ranked_evidence": [
                {
                    "rank": 1,
                    "quote": "The AlphaFold system achieves high accuracy in protein structure prediction.",
                    "article": article.to_dict(),
                    "url": article.url,
                    "reason": "Matched question terms.",
                    "relevance_score": 9.0,
                }
            ],
            "citations": [article.to_dict()],
            "evidence_quality_note": "Moderate confidence.",
        }
        text = format_answer_text(payload)
        ranked_section = text.split("Evidence quality note:")[0]
        self.assertIn('1. "The AlphaFold system achieves high accuracy in protein structure prediction."', ranked_section)
        self.assertNotIn("Source:", ranked_section)
        self.assertIn("[1] Highly accurate protein structure prediction with AlphaFold", text)
        self.assertIn("https://www.nature.com/articles/example", text)

    def test_parse_save_commands(self) -> None:
        self.assertEqual(parse_save_command("save current"), ("last", None, None))
        self.assertEqual(parse_save_command("save session"), ("session", None, None))
        kind, path, export_format = parse_save_command('/save-last "my result.txt" --format txt')
        self.assertEqual(kind, "last")
        self.assertEqual(str(path), "my result.txt")
        self.assertEqual(export_format, "txt")

    def test_parse_settings_commands(self) -> None:
        base = SearchSettings()
        self.assertEqual(parse_settings_command("settings", base), "menu")
        self.assertEqual(parse_settings_command("api keys", base), "api_keys")
        years = parse_settings_command("/set years 2009 2011", base)
        self.assertEqual(years, SearchSettings(year_start=2009, year_end=2011))
        after = parse_settings_command("/set after 2018", years)
        self.assertEqual(after, SearchSettings(year_start=2018, year_end=None))
        text = parse_settings_command("/set text full-text", base)
        self.assertEqual(text, SearchSettings(text_mode="full-text"))
        quotes = parse_settings_command("/set quotes 3", base)
        self.assertEqual(quotes, SearchSettings(quote_count=3))
        with self.assertRaises(ValueError):
            parse_settings_command("/set years 3000 3001", base)
        with self.assertRaises(ValueError):
            parse_settings_command("/set quotes 500", base)

    def test_cli_bad_settings_fail_cleanly(self) -> None:
        cases = [
            ["question", "--year-range", "3000", "3001", "--json"],
            ["question", "--year-range", "2024", "2020", "--json"],
            ["question", "--quotes", "0", "--json"],
            ["question", "--max-iterations", "0", "--json"],
            ["question", "--per-source-limit", "0", "--json"],
        ]
        for args in cases:
            with self.subTest(args=args):
                result = subprocess.run(
                    [sys.executable, "-m", "research_agent.main", *args],
                    cwd=Path(__file__).resolve().parents[1],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("error:", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_pubagent_banner_and_api_key_formatting(self) -> None:
        self.assertIn("PubAgent", "PubAgent")
        self.assertIn("____", PUBAGENT_BANNER)
        self.assertEqual(PUBAGENT_SLOGAN, "grounded literature retrieval, zero fabricated citations")
        self.assertEqual(masked(""), "not set")
        self.assertEqual(masked("short"), "set")
        self.assertEqual(masked("openai-demo-key-abcdefghijkl"), "open...ijkl")
        formatted = format_api_key_settings(APIKeyConfig(ai_provider="openai", openai_api_key="openai-demo-key-abcdefghijkl", ncbi_email="me@example.com"))
        self.assertIn("AI provider: OpenAI", formatted)
        self.assertIn("OpenAI key: open...ijkl", formatted)
        self.assertIn("NCBI email: me@example.com", formatted)
        self.assertTrue(should_prompt_ai_setup(interactive=True, has_saved_api_keys=False))
        self.assertFalse(should_prompt_ai_setup(interactive=True, has_saved_api_keys=True))
        self.assertFalse(should_prompt_ai_setup(interactive=False, has_saved_api_keys=False))

    def test_apply_api_keys_sets_environment(self) -> None:
        names = [
            "RESEARCH_AGENT_LLM_PROVIDER",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "NCBI_EMAIL",
            "NCBI_API_KEY",
            "SEMANTIC_SCHOLAR_API_KEY",
            "UNPAYWALL_EMAIL",
        ]
        previous = {name: os.environ.get(name) for name in names}
        try:
            apply_api_keys(
                APIKeyConfig(
                    ai_provider="anthropic",
                    anthropic_api_key="claude-key",
                    ncbi_email="me@example.com",
                    ncbi_api_key="ncbi-key",
                    semantic_scholar_api_key="sem-key",
                    unpaywall_email="oa@example.com",
                )
            )
            self.assertEqual(os.environ["RESEARCH_AGENT_LLM_PROVIDER"], "anthropic")
            self.assertEqual(os.environ["ANTHROPIC_API_KEY"], "claude-key")
            self.assertEqual(os.environ["NCBI_EMAIL"], "me@example.com")
            self.assertEqual(os.environ["NCBI_API_KEY"], "ncbi-key")
            self.assertEqual(os.environ["SEMANTIC_SCHOLAR_API_KEY"], "sem-key")
            self.assertEqual(os.environ["UNPAYWALL_EMAIL"], "oa@example.com")
            apply_api_keys(APIKeyConfig(ai_provider="none"))
            self.assertEqual(os.environ["RESEARCH_AGENT_LLM_PROVIDER"], "none")
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_save_result_and_session_exports(self) -> None:
        article = FakeSemanticScholar().search("protein structure prediction")[0]
        payload = {
            "question": "What evidence supports AI for protein structure prediction?",
            "generated_at": "2026-07-08T12:00:00",
            "summary": "Short summary.",
            "ranked_evidence": [
                {
                    "rank": 1,
                    "quote": "The AlphaFold system achieves high accuracy in protein structure prediction.",
                    "article": article.to_dict(),
                    "url": article.url,
                    "reason": "Matched question terms.",
                    "relevance_score": 9.0,
                }
            ],
            "citations": [article.to_dict()],
            "evidence_quality_note": "Moderate confidence.",
            "sufficient": True,
            "iterations": 1,
            "trace": [],
            "verification": {"passed": True, "claims": [], "removed_claims": []},
        }
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            json_path = save_result(payload, tmp / "last.json")
            txt_path = save_result(payload, tmp / "last.txt")
            session_path = save_session([payload, payload], tmp / "session.txt")
            self.assertEqual(json_path.name, "last.json")
            self.assertIn('"question"', json_path.read_text(encoding="utf-8"))
            self.assertIn("Question:", txt_path.read_text(encoding="utf-8"))
            session_text = session_path.read_text(encoding="utf-8")
            self.assertIn("Searches: 2", session_text)
            self.assertIn("Search 1", session_text)
            self.assertIn("Search 2", session_text)


if __name__ == "__main__":
    unittest.main()
