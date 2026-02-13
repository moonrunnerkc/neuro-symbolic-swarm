# Author: Bradley R. Kinnard
"""Web search grounding via Tavily API.

Provides fact-checking context for agent responses by searching
the web for claims that need verification. Designed to be called
selectively (not on every query) to conserve API credits."""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# load .env once at import time
load_dotenv()

_TAVILY_KEY: str = os.getenv("TAVILY_API_KEY", "")
_client = None


def _get_client():
    """lazy-init the Tavily client. returns None if no key configured."""
    global _client
    if _client is not None:
        return _client
    if not _TAVILY_KEY:
        logger.warning("TAVILY_API_KEY not set, web search disabled")
        return None
    try:
        from tavily import TavilyClient
        _client = TavilyClient(api_key=_TAVILY_KEY)
        logger.info("tavily client initialized")
        return _client
    except Exception as exc:
        logger.error("tavily init failed: %s", exc)
        return None


def search(query: str, max_results: int = 3) -> list[dict]:
    """search the web for grounding context.

    Returns a list of dicts with 'title', 'url', 'content' keys.
    Returns empty list on failure or missing key (never raises).
    """
    client = _get_client()
    if client is None:
        return []

    try:
        result = client.search(query, max_results=max_results)
        hits = []
        for r in result.get("results", []):
            hits.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:500],
            })
        logger.info("tavily search: %d results for '%s'", len(hits), query[:60])
        return hits
    except Exception as exc:
        logger.warning("tavily search failed: %s", exc)
        return []


def build_grounding_block(hits: list[dict]) -> str:
    """format search results into a constraint block for injection into synthesis."""
    if not hits:
        return ""

    lines = [
        "WEB SEARCH RESULTS (use these to verify factual claims):",
        "If a draft contradicts these sources, correct it.",
        "",
    ]
    for i, hit in enumerate(hits, 1):
        lines.append(f"  [{i}] {hit['title']}")
        lines.append(f"      {hit['url']}")
        lines.append(f"      {hit['content'][:300]}")
        lines.append("")
    return "\n".join(lines)


# -- query classification --

# patterns that suggest the query needs factual grounding
_FACTUAL_PATTERNS = re.compile(
    r"\b(explain|how does|what is|what are|what was|who is|who was|"
    r"why does|why did|when did|when was|where is|where was|"
    r"define|describe|compare|difference between|"
    r"is it true|actually|correct me|verify|fact.check|"
    r"percentage|how many|how much|what percent|estimated|"
    r"according to|evidence|research|study|scientist|"
    r"history of|origin of|cause of)\b",
    re.IGNORECASE,
)

# patterns suggesting creative/fiction context where search is wasteful
_FICTION_PATTERNS = re.compile(
    r"\b(write the scene|write a scene|continue the story|"
    r"in character|roleplay|next chapter|the protagonist|"
    r"the wizard|the knight|the captain|once upon)\b",
    re.IGNORECASE,
)


def needs_grounding(query: str, genre: str = "") -> bool:
    """decide if a query should trigger web search.

    Conservative: only fires on factual/knowledge queries to
    conserve Tavily credits. Skips fiction/creative prompts.
    """
    # never search for creative writing
    if _FICTION_PATTERNS.search(query):
        return False

    # genre hints: fiction genres skip search
    fiction_genres = {"fantasy", "sci-fi", "cyberpunk", "horror", "romance"}
    if genre.lower() in fiction_genres:
        return False

    # trigger on factual patterns
    if _FACTUAL_PATTERNS.search(query):
        return True

    # trigger on contradiction attempts ("actually, X is Y")
    if query.strip().lower().startswith("actually"):
        return True

    return False


def extract_search_query(user_query: str) -> str:
    """distill the user's message into a clean search query.

    Strips conversational fluff, keeps the factual core.
    """
    # remove common prefixes
    cleaned = re.sub(
        r"^(can you |could you |please |hey |ok so |)"
        r"(explain|tell me|describe|what is|what are|how does)",
        r"\2",
        user_query.strip(),
        flags=re.IGNORECASE,
    )
    # truncate to a reasonable search length
    if len(cleaned) > 200:
        cleaned = cleaned[:200].rsplit(" ", 1)[0]
    return cleaned.strip()
