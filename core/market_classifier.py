"""Market category classification from question text and tags.

Simple keyword-based classifier. No ML dependencies.
"""

from __future__ import annotations

import re

POLITICS_KEYWORDS = [
    "president", "election", "trump", "biden", "congress", "senate",
    "governor", "vote", "democrat", "republican", "party", "cabinet",
    "impeach", "resign", "speaker", "veto", "legislation", "poll",
    "primary", "inaugurat", "minister", "parliament", "brexit",
    "sanction", "nato", "ceasefire", "war", "military", "invasion",
    "regime", "diplomat", "treaty",
]

SPORTS_KEYWORDS = [
    "nba", "nfl", "mlb", "nhl", "ufc", "mma", "boxing", "tennis",
    "fifa", "world cup", "premier league", "champions league",
    "super bowl", "playoff", "mvp", "championship", "grand prix",
    "formula 1", "f1", "olympics", "vs.", "v.", "spread:",
    "lakers", "celtics", "warriors", "chiefs", "eagles",
    "match", "game ", "bout", "fight", "race ", "win the",
]

CRYPTO_KEYWORDS = [
    "bitcoin", "btc", "ethereum", "eth", "solana", "sol",
    "crypto", "token", "defi", "nft", "blockchain", "altcoin",
    "binance", "coinbase", "stablecoin", "halving",
]

MACRO_KEYWORDS = [
    "fed ", "federal reserve", "interest rate", "inflation", "cpi",
    "gdp", "unemployment", "jobs report", "recession", "tariff",
    "treasury", "yield", "bond", "s&p", "nasdaq", "dow jones",
    "stock market", "ipo", "earnings",
]

ENTERTAINMENT_KEYWORDS = [
    "oscar", "grammy", "emmy", "tony award", "album", "movie",
    "box office", "netflix", "spotify", "youtube", "tiktok",
    "reality tv", "bachelor", "survivor", "rihanna", "taylor swift",
    "kanye", "drake", "gta vi", "game release", "steam",
]

SCIENCE_KEYWORDS = [
    "ai ", "artificial intelligence", "gpt", "openai", "google ai",
    "space", "nasa", "mars", "moon", "climate", "vaccine",
    "fda", "breakthrough", "discovery", "quantum",
]


def classify_market(
    question: str,
    description: str = "",
    tags: list[str] | None = None,
) -> str:
    """Classify a market into a category.

    Args:
        question: Market question text.
        description: Optional description text.
        tags: Optional list of tags from the API.

    Returns:
        Category string: "politics", "sports", "crypto", "macro",
        "entertainment", "science", or "other".
    """
    text = f"{question} {description}".lower()
    tag_text = " ".join(tags).lower() if tags else ""
    combined = f"{text} {tag_text}"

    scores = {
        "politics": _score(combined, POLITICS_KEYWORDS),
        "sports": _score(combined, SPORTS_KEYWORDS),
        "crypto": _score(combined, CRYPTO_KEYWORDS),
        "macro": _score(combined, MACRO_KEYWORDS),
        "entertainment": _score(combined, ENTERTAINMENT_KEYWORDS),
        "science": _score(combined, SCIENCE_KEYWORDS),
    }

    best = max(scores, key=scores.get)  # type: ignore
    if scores[best] == 0:
        return "other"
    return best


def _score(text: str, keywords: list[str]) -> int:
    return sum(1 for kw in keywords if kw in text)
