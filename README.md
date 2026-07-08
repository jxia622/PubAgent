# PubAgent

PubAgent is a standalone, live literature-search assistant for researchers who need citable evidence. It runs an agent loop:

```text
PLAN -> ACT -> OBSERVE -> DECIDE -> SYNTHESIZE -> VERIFY -> ANSWER
```

It is not a persistent local RAG system and does not build a vector database. It searches public scholarly APIs, ranks retrieved articles, extracts short exact quotations, and returns a concise summary plus quote-backed sources.

```text
grounded literature retrieval, zero fabricated citations
```

## Demo

![PubAgent demo](docs/pubagent_demo.gif)

The video version is available at [`docs/pubagent_demo.webm`](docs/pubagent_demo.webm).

## What It Returns

- `summary`: a cautious answer synthesized only from retrieved evidence.
- `ranked_evidence`: ranked short exact excerpts from articles, each with article metadata and a full article or landing-page link.
- `citations`: source records with title, authors, journal, year, PMID/PMCID/DOI, and URL when available.
- `evidence_quality_note`: confidence and sufficiency note.
- `trace`: plan/act/observe/decide/synthesize/verify steps for debugging.
- `verification`: grounding checks, including whether selected quotes appear in retrieved source text.

Quotes are intentionally short excerpts so the output is useful for source triage without reproducing long copyrighted passages. Researchers should open and verify the full article before citing it in a manuscript.

The default terminal view shows a progress bar while retrieval is running. Ranked quoted evidence is quote-only for readability; source records are listed afterward as `[1]`, `[2]`, etc., matching the evidence rank numbers. Most modern terminals make the raw source URLs clickable. Named hidden hyperlinks are not consistently supported across Terminal, iTerm2, VS Code, and web consoles, so the CLI prints plain URLs for reliability.

## Sources

- Semantic Scholar for broad scholarly search and citation signals.
- PubMed/MEDLINE through NCBI E-utilities.
- Europe PMC as a biomedical/life-science fallback.
- PubMed Central full text when a PMCID is available.
- Unpaywall links when `UNPAYWALL_EMAIL` is configured and a DOI is available.

The current public-source mix is strongest for biomedical and life-science questions, with broader academic coverage coming from Semantic Scholar.

## Install

```bash
cd "/Users/jackxia/Desktop/Python/research_agent"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 -m research_agent.main "What evidence supports AI for protein structure prediction?"
```

Return the full JSON trace:

```bash
python3 -m research_agent.main "What evidence supports AI for protein structure prediction?" --json
```

Disable the file cache for one run:

```bash
python3 -m research_agent.main "What evidence supports AI for protein structure prediction?" --cache none
```

## Save Results

Each one-shot search automatically stores the most recent result in `storage/last_result.json`. Save that result after the command finishes:

```bash
python3 -m research_agent.main --save-last
python3 -m research_agent.main --save-last exports/protein_search.txt --format txt
```

For a multi-question workflow, use interactive mode:

```bash
python3 -m research_agent.main --interactive
```

Interactive launch prints the PubAgent banner and asks whether you want to configure an AI provider. You can choose OpenAI, Claude, or skip and use deterministic fallback. Public literature API keys are optional at startup.

## Search Settings

In interactive mode, type:

```text
settings
```

That opens a small settings menu. The settings are saved to `storage/settings.json` and automatically apply to later searches, so you do not need to repeat them in every question.

Settings available:

- year range, such as `2009-2011`
- after-year filter, such as `2018 or later`
- text mode: `any`, `abstract`, or `full-text`
- number of ranked quotes to show

Year settings are validated against a practical publication range of 1800 through the next calendar year. Quote count is limited to 1-20 so terminal output stays readable.

API keys are also available from the settings menu:

```text
api keys
```

You can configure:

- OpenAI API key
- Claude API key
- NCBI email and optional NCBI API key
- Semantic Scholar API key
- Unpaywall email

Keys are saved locally in `storage/api_keys.json` and applied to future searches. PubMed, Semantic Scholar, and Unpaywall keys are optional; PubAgent runs without them.

Shortcut commands also work:

```text
/set years 2009 2011
/set after 2018
/set text abstract
/set text full-text
/set quotes 5
/reset-settings
```

For one-shot runs, the same settings can be passed as flags:

```bash
python3 -m research_agent.main "your question" --year-range 2009 2011 --text-mode abstract --quotes 5
python3 -m research_agent.main "your question" --after-year 2018 --text-mode full-text
```

Inside interactive mode:

```text
/save-last [path] [--format json|txt]
/save-session [path] [--format json|txt]
```

Friendly aliases also work:

```text
save current
save session
```

## Optional Keys

No paid keys are required for tests. Optional keys can improve live runs:

```bash
export NCBI_EMAIL="you@example.com"
export NCBI_API_KEY="optional_free_ncbi_key"
export SEMANTIC_SCHOLAR_API_KEY="optional_semantic_scholar_key"
export UNPAYWALL_EMAIL="you@example.com"
```

Optional LLM planner/observer/synthesizer/verifier:

```bash
export OPENAI_API_KEY="optional_openai_key"
export OPENAI_MODEL="gpt-4.1-mini"
```

or:

```bash
export ANTHROPIC_API_KEY="optional_anthropic_key"
export ANTHROPIC_MODEL="claude-3-5-sonnet-latest"
export RESEARCH_AGENT_LLM_PROVIDER="anthropic"
```

If no LLM key is configured, deterministic fallback logic is used.

## Test

```bash
python3 -m unittest discover -s tests
python3 -m compileall research_agent tests
```
