from __future__ import annotations

import re
from dataclasses import replace

from research_agent.agent.llm import LLMClient, NoopLLMClient
from research_agent.models import Article, EvidenceQuote, ObserveResult


MAX_QUOTE_WORDS = 25
MAX_SUMMARY_SENTENCES = 2


def citation_label(article: Article) -> str:
    author = article.authors[0].split()[0] if article.authors else "Unknown"
    year = article.year or "n.d."
    identifier = f"PMID {article.pmid}" if article.pmid else f"DOI {article.doi}" if article.doi else article.source
    return f"{author} et al., {year}, {identifier}"


def _tokens(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "article",
        "articles",
        "be",
        "can",
        "does",
        "evidence",
        "for",
        "from",
        "how",
        "in",
        "is",
        "of",
        "or",
        "paper",
        "papers",
        "research",
        "show",
        "study",
        "studies",
        "the",
        "to",
        "what",
        "whether",
        "why",
        "with",
    }
    return {token for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if token not in stopwords and len(token) > 1}


def _sentences(text: str) -> list[str]:
    text = re.sub(r"<[^>]+>", " ", text)
    text = " ".join(text.split())
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _short_quote(sentence: str, question_terms: set[str]) -> str:
    words = sentence.split()
    if len(words) <= MAX_QUOTE_WORDS:
        return sentence

    best_start = 0
    best_score = -1
    for start in range(0, len(words) - MAX_QUOTE_WORDS + 1):
        window = words[start : start + MAX_QUOTE_WORDS]
        score = len(question_terms & _tokens(" ".join(window)))
        if score > best_score:
            best_score = score
            best_start = start
    quote = " ".join(words[best_start : best_start + MAX_QUOTE_WORDS])
    prefix = "... " if best_start > 0 else ""
    suffix = " ..." if best_start + MAX_QUOTE_WORDS < len(words) else ""
    return f"{prefix}{quote}{suffix}"


def trim_summary(summary: str, *, max_sentences: int = MAX_SUMMARY_SENTENCES) -> str:
    summary = " ".join(summary.split())
    if not summary:
        return summary
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", summary) if part.strip()]
    if len(sentences) <= max_sentences:
        return summary
    return " ".join(sentences[:max_sentences])


def _candidate_passages(article: Article, *, text_mode: str = "any") -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    if text_mode == "abstract":
        section_texts = [("abstract", article.abstract or "")]
    elif text_mode == "full-text":
        section_texts = [("full_text", article.full_text or "")]
    else:
        section_texts = [
            ("tldr", article.tldr or ""),
            ("abstract", article.abstract or ""),
            ("full_text", article.full_text or ""),
        ]
    for section, text in section_texts:
        for sentence in _sentences(text):
            if len(_tokens(sentence)) >= 4:
                candidates.append((section, sentence))
    if not candidates and text_mode == "any":
        for sentence in _sentences(article.title):
            if len(_tokens(sentence)) >= 4:
                candidates.append(("title", sentence))
    return candidates


def select_evidence_quotes(question: str, articles: list[Article], *, max_quotes: int = 6, text_mode: str = "any") -> list[EvidenceQuote]:
    question_terms = _tokens(question)
    scored: list[tuple[float, EvidenceQuote]] = []
    seen_articles: set[str] = set()

    for article in articles:
        passages = _candidate_passages(article, text_mode=text_mode)
        if not passages:
            continue

        def passage_score(item: tuple[str, str]) -> tuple[float, int]:
            section, sentence = item
            sentence_terms = _tokens(sentence)
            overlap = len(question_terms & sentence_terms)
            coverage = overlap / max(len(question_terms), 1)
            section_bonus = {"title": 0.3, "tldr": 0.6, "abstract": 0.8, "full_text": 1.0}.get(section, 0.0)
            return (overlap + coverage + section_bonus, -len(sentence))

        section, sentence = max(passages, key=passage_score)
        article_terms = _tokens(article.evidence_text())
        overlap = len(question_terms & article_terms)
        coverage = overlap / max(len(question_terms), 1)
        citation_signal = min((article.citation_count or 0) / 100.0, 1.5)
        quality_signal = 0.5 if any("review" in kind.lower() or "trial" in kind.lower() for kind in article.publication_types) else 0.0
        full_text_signal = 0.4 if article.full_text else 0.0
        score = round((overlap * 1.5) + (coverage * 4.0) + citation_signal + quality_signal + full_text_signal, 3)
        if score <= 0:
            continue
        quote = _short_quote(sentence, question_terms)
        scored.append(
            (
                score,
                EvidenceQuote(
                    rank=0,
                    quote=quote,
                    article=article,
                    source_section=section,
                    relevance_score=score,
                    reason=f"Matched {overlap} question terms; selected from {section}.",
                ),
            )
        )

    ranked: list[EvidenceQuote] = []
    for _, evidence in sorted(scored, key=lambda item: item[0], reverse=True):
        if evidence.article.key in seen_articles:
            continue
        seen_articles.add(evidence.article.key)
        ranked.append(replace(evidence, rank=len(ranked) + 1))
        if len(ranked) >= max_quotes:
            break
    return ranked


class Synthesizer:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or NoopLLMClient()

    def synthesize(
        self,
        *,
        question: str,
        articles: list[Article],
        observation: ObserveResult,
        evidence_quotes: list[EvidenceQuote],
    ) -> str:
        payload = self.llm.complete_json(
            system=(
                "Write a concise researcher-facing answer summary using only the provided quoted evidence. "
                "Limit the summary to 1-2 sentences. Do not invent facts. Mention uncertainty when evidence is sparse or indirect."
            ),
            user={
                "question": question,
                "observation": observation.to_dict(),
                "evidence": [
                    {
                        "rank": item.rank,
                        "quote": item.quote,
                        "citation": citation_label(item.article),
                        "title": item.article.title,
                    }
                    for item in evidence_quotes
                ],
                "output_schema": {"summary": "string, 1-2 sentences"},
            },
        )
        if payload and payload.get("summary"):
            return trim_summary(str(payload["summary"]))
        return self._fallback_synthesize(question, articles, observation, evidence_quotes)

    def _fallback_synthesize(
        self,
        question: str,
        articles: list[Article],
        observation: ObserveResult,
        evidence_quotes: list[EvidenceQuote],
    ) -> str:
        if not evidence_quotes:
            return (
                "The search did not retrieve enough directly quotable evidence to answer this question. "
                f"Retrieval note: {observation.reason}"
            )
        if not observation.sufficient:
            return (
                "The retrieved literature is not sufficient for a confident synthesis, but the closest quoted "
                "evidence below can guide manual follow-up. Treat this as a retrieval lead, not a settled answer."
            )

        source_count = len({quote.article.key for quote in evidence_quotes})
        year_values = [int(article.year) for article in articles if article.year and article.year.isdigit()]
        year_note = f" spanning {min(year_values)}-{max(year_values)}" if year_values else ""
        return (
            f"PubAgent found {source_count} directly relevant citable source{'s' if source_count != 1 else ''}{year_note}. "
            "Use the ranked quotations below as the evidence trail and verify the full articles before citing."
        )
