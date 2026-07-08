from __future__ import annotations

import argparse
import getpass
import json
import os
import shlex
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, TypeVar

from research_agent.agent.llm import default_llm_client
from research_agent.agent.loop import ResearchAgent
from research_agent.agent.act import RetrievalActor
from research_agent.agent.observe import Observer
from research_agent.agent.plan import Planner
from research_agent.agent.synthesize import Synthesizer
from research_agent.agent.verify import Verifier
from research_agent.cache.store import JsonCacheStore
from research_agent.models import MAX_PUBLICATION_YEAR, MAX_QUOTE_COUNT, MIN_PUBLICATION_YEAR, SearchSettings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_PATH = PROJECT_ROOT / "storage" / "research_agent_cache.json"
LAST_RESULT_PATH = PROJECT_ROOT / "storage" / "last_result.json"
SETTINGS_PATH = PROJECT_ROOT / "storage" / "settings.json"
API_KEYS_PATH = PROJECT_ROOT / "storage" / "api_keys.json"
EXPORT_DIR = PROJECT_ROOT / "exports"
T = TypeVar("T")


PUBAGENT_BANNER = r"""
 ____        _        _                    _
|  _ \ _   _| |__    / \   __ _  ___ _ __ | |_
| |_) | | | | '_ \  / _ \ / _` |/ _ \ '_ \| __|
|  __/| |_| | |_) |/ ___ \ (_| |  __/ | | | |_
|_|    \__,_|_.__//_/   \_\__, |\___|_| |_|\__|
                          |___/
"""
PUBAGENT_SLOGAN = "grounded literature retrieval, zero fabricated citations"


@dataclass(frozen=True)
class APIKeyConfig:
    ai_provider: str = "none"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    ncbi_email: str = ""
    ncbi_api_key: str = ""
    semantic_scholar_api_key: str = ""
    unpaywall_email: str = ""

    def __post_init__(self) -> None:
        if self.ai_provider not in {"none", "openai", "anthropic"}:
            raise ValueError("ai_provider must be none, openai, or anthropic.")

    def to_dict(self) -> dict:
        return asdict(self)


def load_api_keys() -> APIKeyConfig:
    if not API_KEYS_PATH.exists():
        return APIKeyConfig()
    try:
        payload = json.loads(API_KEYS_PATH.read_text(encoding="utf-8"))
        return APIKeyConfig(
            ai_provider=payload.get("ai_provider", "none"),
            openai_api_key=payload.get("openai_api_key", ""),
            anthropic_api_key=payload.get("anthropic_api_key", ""),
            ncbi_email=payload.get("ncbi_email", ""),
            ncbi_api_key=payload.get("ncbi_api_key", ""),
            semantic_scholar_api_key=payload.get("semantic_scholar_api_key", ""),
            unpaywall_email=payload.get("unpaywall_email", ""),
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return APIKeyConfig()


def save_api_keys(config: APIKeyConfig) -> None:
    API_KEYS_PATH.parent.mkdir(parents=True, exist_ok=True)
    API_KEYS_PATH.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")


def apply_api_keys(config: APIKeyConfig) -> None:
    if config.ai_provider == "openai":
        os.environ["RESEARCH_AGENT_LLM_PROVIDER"] = "openai"
    elif config.ai_provider == "anthropic":
        os.environ["RESEARCH_AGENT_LLM_PROVIDER"] = "anthropic"
    else:
        os.environ["RESEARCH_AGENT_LLM_PROVIDER"] = "none"

    values = {
        "OPENAI_API_KEY": config.openai_api_key,
        "ANTHROPIC_API_KEY": config.anthropic_api_key,
        "NCBI_EMAIL": config.ncbi_email,
        "NCBI_API_KEY": config.ncbi_api_key,
        "SEMANTIC_SCHOLAR_API_KEY": config.semantic_scholar_api_key,
        "UNPAYWALL_EMAIL": config.unpaywall_email,
    }
    for name, value in values.items():
        if value:
            os.environ[name] = value
        else:
            os.environ.pop(name, None)


def build_agent(*, cache_path: Path | None = DEFAULT_CACHE_PATH, max_iterations: int = 3, api_keys: APIKeyConfig | None = None) -> ResearchAgent:
    if api_keys:
        apply_api_keys(api_keys)
    cache = JsonCacheStore(cache_path) if cache_path else JsonCacheStore()
    llm = default_llm_client()
    return ResearchAgent(
        planner=Planner(llm),
        actor=RetrievalActor(cache=cache),
        observer=Observer(llm),
        synthesizer=Synthesizer(llm),
        verifier=Verifier(llm),
        max_iterations=max_iterations,
    )


def format_answer_text(payload: dict) -> str:
    lines = ["Summary:", "", payload["summary"], "", "Ranked quoted evidence:", ""]
    if payload["ranked_evidence"]:
        for item in payload["ranked_evidence"]:
            lines.append(f"{item['rank']}. \"{item['quote']}\"")
            lines.append("")
    else:
        lines.append("No directly quotable evidence was retrieved.")
        lines.append("")

    source_lines = []
    for index, item in enumerate(payload["ranked_evidence"], start=1):
        article = item["article"]
        fields = [article["title"]]
        if article.get("authors"):
            fields.append(", ".join(article["authors"][:4]))
        if article.get("journal"):
            fields.append(article["journal"])
        if article.get("year"):
            fields.append(str(article["year"]))
        if article.get("pmid"):
            fields.append(f"PMID {article['pmid']}")
        if article.get("doi"):
            fields.append(f"DOI {article['doi']}")
        fields.append(item.get("url") or article["url"])
        source_lines.append(f"[{index}] " + "; ".join(fields))

    lines.extend(["Evidence quality note:", "", payload["evidence_quality_note"], "", "Sources:", ""])
    lines.append("\n\n".join(source_lines) if source_lines else "No sources retrieved.")
    return "\n".join(lines)


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def enrich_payload(payload: dict, question: str, settings: SearchSettings) -> dict:
    enriched = dict(payload)
    enriched["question"] = question
    enriched["generated_at"] = datetime.now().isoformat(timespec="seconds")
    enriched["settings"] = settings.to_dict()
    return enriched


def load_settings() -> SearchSettings:
    if not SETTINGS_PATH.exists():
        return SearchSettings()
    try:
        payload = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return SearchSettings(
            year_start=payload.get("year_start"),
            year_end=payload.get("year_end"),
            text_mode=payload.get("text_mode", "any"),
            quote_count=int(payload.get("quote_count", 8)),
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return SearchSettings()


def save_settings(settings: SearchSettings) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")


def settings_from_args(args: argparse.Namespace, base: SearchSettings) -> SearchSettings:
    year_start = base.year_start
    year_end = base.year_end
    if args.year_range:
        year_start, year_end = args.year_range
    if args.after_year is not None:
        year_start = args.after_year
        year_end = None
    if args.year_start is not None:
        year_start = args.year_start
    if args.year_end is not None:
        year_end = args.year_end
    return SearchSettings(
        year_start=year_start,
        year_end=year_end,
        text_mode=args.text_mode or base.text_mode,
        quote_count=args.quotes if args.quotes is not None else base.quote_count,
    )


def format_settings(settings: SearchSettings) -> str:
    if settings.year_start is None and settings.year_end is None:
        year_text = "any"
    elif settings.year_start is not None and settings.year_end is not None:
        year_text = f"{settings.year_start}-{settings.year_end}"
    elif settings.year_start is not None:
        year_text = f"{settings.year_start} or later"
    else:
        year_text = f"{settings.year_end} or earlier"
    return (
        "Current settings:\n"
        f"  Year filter: {year_text}\n"
        f"  Text mode: {settings.text_mode}\n"
        f"  Quotes shown: {settings.quote_count}"
    )


def masked(value: str) -> str:
    if not value:
        return "not set"
    if len(value) <= 8:
        return "set"
    return f"{value[:4]}...{value[-4:]}"


def format_api_key_settings(config: APIKeyConfig) -> str:
    provider = {"none": "deterministic fallback", "openai": "OpenAI", "anthropic": "Claude"}.get(config.ai_provider, config.ai_provider)
    return (
        "API key settings:\n"
        f"  AI provider: {provider}\n"
        f"  OpenAI key: {masked(config.openai_api_key)}\n"
        f"  Claude key: {masked(config.anthropic_api_key)}\n"
        f"  NCBI email: {config.ncbi_email or 'not set'}\n"
        f"  NCBI API key: {masked(config.ncbi_api_key)}\n"
        f"  Semantic Scholar key: {masked(config.semantic_scholar_api_key)}\n"
        f"  Unpaywall email: {config.unpaywall_email or 'not set'}"
    )


def prompt_secret(prompt: str, current: str = "") -> str:
    suffix = " (leave blank to keep current): " if current else " (leave blank to skip): "
    value = getpass.getpass(prompt + suffix).strip()
    return value or current


def prompt_plain(prompt: str, current: str = "") -> str:
    suffix = " (leave blank to keep current): " if current else " (leave blank to skip): "
    value = input(prompt + suffix).strip()
    return value or current


def prompt_ai_key_on_launch(config: APIKeyConfig) -> APIKeyConfig:
    if not sys.stdin.isatty():
        return config
    print("First-time AI setup")
    print("Choose an AI provider for planning/summarizing. You can skip and use deterministic fallback.")
    print("  1. OpenAI")
    print("  2. Claude")
    print("  3. Skip")
    current_note = f" Current: {config.ai_provider}." if config.ai_provider != "none" else ""
    choice = input(f"Provider [1-3]{current_note}: ").strip().lower()
    if choice in {"1", "openai"}:
        key = prompt_secret("OpenAI API key", config.openai_api_key)
        config = APIKeyConfig(
            ai_provider="openai" if key else "none",
            openai_api_key=key,
            anthropic_api_key=config.anthropic_api_key,
            ncbi_email=config.ncbi_email,
            ncbi_api_key=config.ncbi_api_key,
            semantic_scholar_api_key=config.semantic_scholar_api_key,
            unpaywall_email=config.unpaywall_email,
        )
    elif choice in {"2", "claude", "anthropic"}:
        key = prompt_secret("Claude API key", config.anthropic_api_key)
        config = APIKeyConfig(
            ai_provider="anthropic" if key else "none",
            openai_api_key=config.openai_api_key,
            anthropic_api_key=key,
            ncbi_email=config.ncbi_email,
            ncbi_api_key=config.ncbi_api_key,
            semantic_scholar_api_key=config.semantic_scholar_api_key,
            unpaywall_email=config.unpaywall_email,
        )
    elif choice in {"", "3", "skip", "none"}:
        pass
    else:
        print("Unknown provider choice. Keeping existing AI setup.")
    save_api_keys(config)
    apply_api_keys(config)
    return config


def should_prompt_ai_setup(*, interactive: bool, has_saved_api_keys: bool) -> bool:
    return interactive and not has_saved_api_keys


def api_keys_menu(config: APIKeyConfig) -> APIKeyConfig:
    current = config
    while True:
        print("\n" + format_api_key_settings(current))
        print(
            "\nAPI keys:\n"
            "  1. Set OpenAI key and use OpenAI\n"
            "  2. Set Claude key and use Claude\n"
            "  3. Use deterministic fallback AI\n"
            "  4. Set NCBI email\n"
            "  5. Set NCBI API key\n"
            "  6. Set Semantic Scholar API key\n"
            "  7. Set Unpaywall email\n"
            "  8. Clear all API keys\n"
            "  9. Done\n"
        )
        choice = input("Choose 1-9: ").strip().lower()
        if choice == "1":
            key = prompt_secret("OpenAI API key", current.openai_api_key)
            current = APIKeyConfig(
                ai_provider="openai" if key else "none",
                openai_api_key=key,
                anthropic_api_key=current.anthropic_api_key,
                ncbi_email=current.ncbi_email,
                ncbi_api_key=current.ncbi_api_key,
                semantic_scholar_api_key=current.semantic_scholar_api_key,
                unpaywall_email=current.unpaywall_email,
            )
        elif choice == "2":
            key = prompt_secret("Claude API key", current.anthropic_api_key)
            current = APIKeyConfig(
                ai_provider="anthropic" if key else "none",
                openai_api_key=current.openai_api_key,
                anthropic_api_key=key,
                ncbi_email=current.ncbi_email,
                ncbi_api_key=current.ncbi_api_key,
                semantic_scholar_api_key=current.semantic_scholar_api_key,
                unpaywall_email=current.unpaywall_email,
            )
        elif choice == "3":
            current = APIKeyConfig(
                ai_provider="none",
                openai_api_key=current.openai_api_key,
                anthropic_api_key=current.anthropic_api_key,
                ncbi_email=current.ncbi_email,
                ncbi_api_key=current.ncbi_api_key,
                semantic_scholar_api_key=current.semantic_scholar_api_key,
                unpaywall_email=current.unpaywall_email,
            )
        elif choice == "4":
            current = APIKeyConfig(**{**current.to_dict(), "ncbi_email": prompt_plain("NCBI email", current.ncbi_email)})
        elif choice == "5":
            current = APIKeyConfig(**{**current.to_dict(), "ncbi_api_key": prompt_secret("NCBI API key", current.ncbi_api_key)})
        elif choice == "6":
            current = APIKeyConfig(**{**current.to_dict(), "semantic_scholar_api_key": prompt_secret("Semantic Scholar API key", current.semantic_scholar_api_key)})
        elif choice == "7":
            current = APIKeyConfig(**{**current.to_dict(), "unpaywall_email": prompt_plain("Unpaywall email", current.unpaywall_email)})
        elif choice == "8":
            current = APIKeyConfig()
        elif choice in {"9", "done", "q", "quit", "exit"}:
            save_api_keys(current)
            apply_api_keys(current)
            print("API keys saved.")
            return current
        else:
            print("Choose one of the listed options.")


def default_export_path(kind: str, export_format: str) -> Path:
    return EXPORT_DIR / f"{kind}_{timestamp_slug()}.{export_format}"


def infer_export_format(path: Path | None, requested_format: str | None) -> str:
    if requested_format:
        return requested_format
    if path and path.suffix.lower() == ".txt":
        return "txt"
    return "json"


def save_last_result_state(payload: dict) -> None:
    LAST_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAST_RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_last_result_state() -> dict:
    if not LAST_RESULT_PATH.exists():
        raise FileNotFoundError("No previous search result is available to save.")
    return json.loads(LAST_RESULT_PATH.read_text(encoding="utf-8"))


def format_saved_result_text(payload: dict) -> str:
    lines = []
    if payload.get("question"):
        lines.extend(["Question:", "", payload["question"], ""])
    if payload.get("generated_at"):
        lines.extend(["Generated at:", "", payload["generated_at"], ""])
    lines.append(format_answer_text(payload))
    return "\n".join(lines)


def format_saved_session_text(session_payload: dict) -> str:
    lines = ["PubAgent Session", ""]
    if session_payload.get("saved_at"):
        lines.extend(["Saved at:", "", session_payload["saved_at"], ""])
    searches = session_payload.get("searches", [])
    lines.extend([f"Searches: {len(searches)}", ""])
    for index, result in enumerate(searches, start=1):
        lines.extend([f"Search {index}", "=" * (8 + len(str(index))), ""])
        lines.append(format_saved_result_text(result))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def save_result(payload: dict, path: Path | None = None, *, export_format: str | None = None) -> Path:
    resolved_format = infer_export_format(path, export_format)
    output_path = path or default_export_path("research_result", resolved_format)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if resolved_format == "txt":
        output_path.write_text(format_saved_result_text(payload), encoding="utf-8")
    else:
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def save_session(searches: list[dict], path: Path | None = None, *, export_format: str | None = None) -> Path:
    resolved_format = infer_export_format(path, export_format)
    output_path = path or default_export_path("research_session", resolved_format)
    session_payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "search_count": len(searches),
        "searches": searches,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if resolved_format == "txt":
        output_path.write_text(format_saved_session_text(session_payload), encoding="utf-8")
    else:
        output_path.write_text(json.dumps(session_payload, indent=2), encoding="utf-8")
    return output_path


def run_with_progress(label: str, work: Callable[[], T]) -> T:
    if not sys.stderr.isatty():
        sys.stderr.write(f"{label}\n")
        sys.stderr.flush()
        result = work()
        sys.stderr.write("Done.\n")
        sys.stderr.flush()
        return result

    frames = [
        "[>         ]",
        "[=>        ]",
        "[==>       ]",
        "[===>      ]",
        "[====>     ]",
        "[=====>    ]",
        "[======>   ]",
        "[=======>  ]",
        "[========> ]",
        "[=========>]",
    ]
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(work)
        index = 0
        while not future.done():
            sys.stderr.write(f"\r{frames[index % len(frames)]} {label}")
            sys.stderr.flush()
            index += 1
            time.sleep(0.12)
        if future.exception():
            sys.stderr.write("\r[     failed] Retrieval failed.             \n")
            sys.stderr.flush()
            return future.result()
        sys.stderr.write("\r[==========] Done.                         \n")
        sys.stderr.flush()
        return future.result()


def print_interactive_help() -> None:
    print(
        "\nCommands:\n"
        "  Type any research question to run a search.\n"
        "  settings                                  Open the settings menu.\n"
        "  api keys                                  Open AI and literature API key settings.\n"
        "  /set years 2009 2011                      Set an inclusive year range.\n"
        "  /set after 2018                           Set a start year with no end year.\n"
        "  /set text any|abstract|full-text          Choose quote/evidence text mode.\n"
        "  /set quotes 5                             Choose how many quotes to show.\n"
        "  /reset-settings                           Reset settings to defaults.\n"
        "  /save-last [path] [--format json|txt]     Save the most recent search.\n"
        "  /save-session [path] [--format json|txt]  Save all searches from this session.\n"
        "  save current                              Alias for /save-last.\n"
        "  save session                              Alias for /save-session.\n"
        "  /help                                     Show this help.\n"
        "  /exit                                     Quit.\n"
    )


def prompt_int(prompt: str, *, allow_blank: bool = False) -> int | None:
    while True:
        value = input(prompt).strip()
        if allow_blank and not value:
            return None
        try:
            return int(value)
        except ValueError:
            print("Enter a whole number.")


def prompt_setting_int(prompt: str, *, field_name: str) -> int:
    while True:
        value = prompt_int(prompt)
        try:
            if field_name == "quote_count":
                SearchSettings(quote_count=value or 0)
            elif field_name == "year_start":
                SearchSettings(year_start=value)
            elif field_name == "year_end":
                SearchSettings(year_end=value)
            return value or 0
        except ValueError as exc:
            print(f"Invalid value: {exc}")


def print_pubagent_banner() -> None:
    print(PUBAGENT_BANNER)
    print(PUBAGENT_SLOGAN)
    print()


def settings_menu(settings: SearchSettings, api_keys: APIKeyConfig) -> tuple[SearchSettings, APIKeyConfig]:
    current = settings
    current_api_keys = api_keys
    while True:
        print("\n" + format_settings(current))
        print(
            "\nSettings:\n"
            "  1. Set year range\n"
            "  2. Set after-year filter\n"
            "  3. Set text mode\n"
            "  4. Set number of quotes\n"
            "  5. Clear year filter\n"
            "  6. Reset all settings\n"
            "  7. API keys\n"
            "  8. Done\n"
        )
        choice = input("Choose 1-8: ").strip().lower()
        try:
            if choice == "1":
                start = prompt_setting_int("Start year: ", field_name="year_start")
                end = prompt_setting_int("End year: ", field_name="year_end")
                current = SearchSettings(year_start=start, year_end=end, text_mode=current.text_mode, quote_count=current.quote_count)
            elif choice == "2":
                start = prompt_setting_int("Only include articles from year: ", field_name="year_start")
                current = SearchSettings(year_start=start, year_end=None, text_mode=current.text_mode, quote_count=current.quote_count)
            elif choice == "3":
                mode = input("Text mode (any, abstract, full-text): ").strip().lower()
                current = SearchSettings(year_start=current.year_start, year_end=current.year_end, text_mode=mode, quote_count=current.quote_count)
            elif choice == "4":
                count = prompt_setting_int("Number of quotes: ", field_name="quote_count")
                current = SearchSettings(year_start=current.year_start, year_end=current.year_end, text_mode=current.text_mode, quote_count=count or current.quote_count)
            elif choice == "5":
                current = SearchSettings(text_mode=current.text_mode, quote_count=current.quote_count)
            elif choice == "6":
                current = SearchSettings()
            elif choice == "7":
                current_api_keys = api_keys_menu(current_api_keys)
            elif choice in {"8", "done", "q", "quit", "exit"}:
                save_settings(current)
                print("Settings saved.")
                return current, current_api_keys
            else:
                print("Choose one of the listed options.")
                continue
        except ValueError as exc:
            print(f"Invalid setting: {exc}")


def parse_settings_command(raw: str, current: SearchSettings) -> SearchSettings | str | None:
    stripped = raw.strip()
    lowered = stripped.lower()
    if lowered in {"settings", "/settings", "setting", "/setting"}:
        return "menu"
    if lowered in {"api keys", "api key", "/api-keys", "/api-key", "keys", "/keys"}:
        return "api_keys"
    if lowered == "/reset-settings":
        return SearchSettings()
    if not lowered.startswith("/set "):
        return None
    try:
        tokens = shlex.split(stripped)
    except ValueError as exc:
        raise ValueError(f"Could not parse command: {exc}") from exc
    if len(tokens) < 3:
        raise ValueError("Use /set years START END, /set after YEAR, /set text MODE, or /set quotes N.")
    target = tokens[1].lower()
    if target in {"years", "year-range", "range"}:
        if len(tokens) != 4:
            raise ValueError("Use /set years START END.")
        try:
            start = int(tokens[2])
            end = int(tokens[3])
        except ValueError as exc:
            raise ValueError("Years must be whole numbers.") from exc
        return SearchSettings(year_start=start, year_end=end, text_mode=current.text_mode, quote_count=current.quote_count)
    if target in {"after", "from", "since"}:
        if len(tokens) != 3:
            raise ValueError("Use /set after YEAR.")
        try:
            start = int(tokens[2])
        except ValueError as exc:
            raise ValueError("Year must be a whole number.") from exc
        return SearchSettings(year_start=start, year_end=None, text_mode=current.text_mode, quote_count=current.quote_count)
    if target in {"text", "text-mode", "mode"}:
        if len(tokens) != 3:
            raise ValueError("Use /set text any|abstract|full-text.")
        return SearchSettings(year_start=current.year_start, year_end=current.year_end, text_mode=tokens[2].lower(), quote_count=current.quote_count)
    if target in {"quotes", "quote-count"}:
        if len(tokens) != 3:
            raise ValueError("Use /set quotes N.")
        try:
            quote_count = int(tokens[2])
        except ValueError as exc:
            raise ValueError("Quote count must be a whole number.") from exc
        return SearchSettings(year_start=current.year_start, year_end=current.year_end, text_mode=current.text_mode, quote_count=quote_count)
    raise ValueError("Unknown setting. Use years, after, text, or quotes.")


def parse_save_command(raw: str) -> tuple[str, Path | None, str | None] | None:
    stripped = raw.strip()
    lowered = stripped.lower()
    if lowered in {"save current", "save last"}:
        return ("last", None, None)
    if lowered == "save session":
        return ("session", None, None)
    if not stripped.startswith("/"):
        return None

    try:
        tokens = shlex.split(stripped)
    except ValueError as exc:
        raise ValueError(f"Could not parse command: {exc}") from exc
    if not tokens:
        return None
    command = tokens[0].lower()
    if command not in {"/save-last", "/save-current", "/save-session"}:
        return None

    kind = "session" if command == "/save-session" else "last"
    export_format: str | None = None
    path_parts: list[str] = []
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token in {"--format", "-f"}:
            if index + 1 >= len(tokens):
                raise ValueError("Missing value after --format.")
            export_format = tokens[index + 1].lower()
            index += 2
            continue
        if token.lower() in {"json", "txt"} and export_format is None:
            export_format = token.lower()
        else:
            path_parts.append(token)
        index += 1
    if export_format not in {None, "json", "txt"}:
        raise ValueError("Format must be json or txt.")
    path = Path(" ".join(path_parts)).expanduser() if path_parts else None
    return (kind, path, export_format)


def run_interactive(
    agent: ResearchAgent,
    *,
    per_source_limit: int,
    settings: SearchSettings,
    api_keys: APIKeyConfig,
    cache_path: Path | None,
    max_iterations: int,
) -> int:
    session_results: list[dict] = []
    last_result: dict | None = None
    print("PubAgent interactive mode. Type /help for commands, /exit to quit.")
    print(format_settings(settings))
    while True:
        try:
            raw = input("\nresearch> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return 0
        if not raw:
            continue
        if raw.lower() in {"/exit", "/quit", "exit", "quit"}:
            return 0
        if raw.lower() == "/help":
            print_interactive_help()
            continue

        try:
            settings_command = parse_settings_command(raw, settings)
        except ValueError as exc:
            print(f"Error: {exc}")
            continue
        if settings_command == "menu":
            settings, new_api_keys = settings_menu(settings, api_keys)
            if new_api_keys != api_keys:
                api_keys = new_api_keys
                agent = build_agent(cache_path=cache_path, max_iterations=max_iterations, api_keys=api_keys)
                print("API key changes are active for future searches.")
            continue
        if settings_command == "api_keys":
            new_api_keys = api_keys_menu(api_keys)
            if new_api_keys != api_keys:
                api_keys = new_api_keys
                agent = build_agent(cache_path=cache_path, max_iterations=max_iterations, api_keys=api_keys)
                print("API key changes are active for future searches.")
            continue
        if isinstance(settings_command, SearchSettings):
            settings = settings_command
            save_settings(settings)
            print("Settings updated.")
            print(format_settings(settings))
            continue

        try:
            save_command = parse_save_command(raw)
        except ValueError as exc:
            print(f"Error: {exc}")
            continue
        if save_command:
            kind, path, export_format = save_command
            if kind == "last":
                if last_result is None:
                    print("No search has been run in this session yet.")
                    continue
                output_path = save_result(last_result, path, export_format=export_format)
                print(f"Saved last search to: {output_path}")
            else:
                if not session_results:
                    print("No searches have been run in this session yet.")
                    continue
                output_path = save_session(session_results, path, export_format=export_format)
                print(f"Saved session to: {output_path}")
            continue

        result = run_with_progress(
            "Retrieving and ranking evidence...",
            lambda: agent.run(raw, per_source_limit=per_source_limit, settings=settings),
        )
        payload = enrich_payload(result.to_dict(), raw, settings)
        save_last_result_state(payload)
        last_result = payload
        session_results.append(payload)
        print(format_answer_text(payload))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PubAgent: grounded literature retrieval, zero fabricated citations")
    parser.add_argument("question", nargs="?", help="Research question to answer")
    parser.add_argument("--max-iterations", type=int, default=3, help="Plan/act/observe iteration cap")
    parser.add_argument("--per-source-limit", type=int, default=12, help="Maximum records per source per iteration")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE_PATH), help="JSON cache path; use 'none' to disable file cache")
    parser.add_argument("--json", action="store_true", help="Print full JSON payload")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start an interactive session with /save-last and /save-session commands")
    parser.add_argument("--save-last", nargs="?", const="", metavar="PATH", help="Save the most recent completed search from storage")
    parser.add_argument("--format", choices=["json", "txt"], help="Export format for --save-last or interactive save commands")
    parser.add_argument("--year-range", nargs=2, type=int, metavar=("START", "END"), help=f"Only include articles published from START through END ({MIN_PUBLICATION_YEAR}-{MAX_PUBLICATION_YEAR})")
    parser.add_argument("--after-year", type=int, help=f"Only include articles from this year or later ({MIN_PUBLICATION_YEAR}-{MAX_PUBLICATION_YEAR})")
    parser.add_argument("--year-start", type=int, help=f"Only include articles from this year or later ({MIN_PUBLICATION_YEAR}-{MAX_PUBLICATION_YEAR})")
    parser.add_argument("--year-end", type=int, help=f"Only include articles from this year or earlier ({MIN_PUBLICATION_YEAR}-{MAX_PUBLICATION_YEAR})")
    parser.add_argument("--text-mode", choices=["any", "abstract", "full-text"], help="Use any text, abstract-only evidence, or full-text-only evidence")
    parser.add_argument("--quotes", type=int, help=f"Number of ranked quotes to show (1-{MAX_QUOTE_COUNT})")
    args = parser.parse_args(argv)

    if args.save_last is not None:
        try:
            payload = load_last_result_state()
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        output_path = save_result(payload, Path(args.save_last).expanduser() if args.save_last else None, export_format=args.format)
        print(f"Saved last search to: {output_path}")
        return 0

    if args.max_iterations < 1:
        parser.error("--max-iterations must be at least 1")
    if args.per_source_limit < 1:
        parser.error("--per-source-limit must be at least 1")

    has_saved_api_keys = API_KEYS_PATH.exists()
    api_keys = load_api_keys()
    if should_prompt_ai_setup(interactive=args.interactive, has_saved_api_keys=has_saved_api_keys):
        print_pubagent_banner()
        api_keys = prompt_ai_key_on_launch(api_keys)
    elif args.interactive:
        print_pubagent_banner()

    cache_path = None if args.cache == "none" else Path(args.cache)
    agent = build_agent(
        cache_path=cache_path,
        max_iterations=args.max_iterations,
        api_keys=api_keys if args.interactive or has_saved_api_keys else None,
    )
    try:
        settings = settings_from_args(args, load_settings())
    except ValueError as exc:
        parser.error(str(exc))
    if args.interactive:
        return run_interactive(
            agent,
            per_source_limit=args.per_source_limit,
            settings=settings,
            api_keys=api_keys,
            cache_path=cache_path,
            max_iterations=args.max_iterations,
        )
    if not args.question:
        parser.error("question is required unless --interactive or --save-last is used")

    if args.json:
        result = agent.run(args.question, per_source_limit=args.per_source_limit, settings=settings)
    else:
        result = run_with_progress("Retrieving and ranking evidence...", lambda: agent.run(args.question, per_source_limit=args.per_source_limit, settings=settings))
    payload = enrich_payload(result.to_dict(), args.question, settings)
    save_last_result_state(payload)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(format_answer_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
