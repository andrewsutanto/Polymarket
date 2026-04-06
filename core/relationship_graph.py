"""Cross-market relationship detection.

Identifies logically related markets for the cross-market consistency
strategy. Detects subset/superset relationships, contradictions,
and correlated outcomes.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from core.gamma_feed import GammaMarket

logger = logging.getLogger(__name__)


@dataclass
class MarketRelationship:
    """A detected relationship between two markets."""

    market_a_id: str
    market_b_id: str
    relationship_type: str  # "subset", "contradiction", "correlated", "temporal"
    confidence: float  # 0-1
    description: str


def find_relationships(
    markets: dict[str, GammaMarket],
    min_confidence: float = 0.5,
) -> list[MarketRelationship]:
    """Find logically related market pairs.

    Args:
        markets: {condition_id: GammaMarket} universe.
        min_confidence: Minimum confidence to include a relationship.

    Returns:
        List of MarketRelationship objects.
    """
    relationships: list[MarketRelationship] = []
    market_list = list(markets.values())

    for i in range(len(market_list)):
        for j in range(i + 1, len(market_list)):
            a = market_list[i]
            b = market_list[j]

            # Skip if different categories (unlikely to be related)
            if a.category != b.category:
                continue

            rels = _detect_relationships(a, b)
            for rel in rels:
                if rel.confidence >= min_confidence:
                    relationships.append(rel)

    logger.info("Found %d cross-market relationships", len(relationships))
    return relationships


def _detect_relationships(a: GammaMarket, b: GammaMarket) -> list[MarketRelationship]:
    """Check if two markets are related."""
    results: list[MarketRelationship] = []

    # Text similarity
    sim = _text_similarity(a.question, b.question)
    if sim < 0.25:
        return results

    # Tag overlap
    tag_overlap = len(set(a.tags) & set(b.tags))

    # Temporal overlap (same resolution date)
    same_date = False
    if a.end_date and b.end_date:
        same_date = abs((a.end_date - b.end_date).total_seconds()) < 86400

    # Subset detection: "X wins Y" is subset of "X's party wins Y"
    # E.g., "Trump wins 2028" implies "Republican wins 2028"
    a_words = set(_extract_key_words(a.question))
    b_words = set(_extract_key_words(b.question))

    if a_words and b_words:
        overlap = len(a_words & b_words)
        union = len(a_words | b_words)
        jaccard = overlap / union if union > 0 else 0

        # High text overlap + same date = likely related
        if jaccard > 0.4 and same_date:
            # Check for subset relationship
            if a_words < b_words or a_words > b_words:
                subset_id = a.condition_id if len(a_words) < len(b_words) else b.condition_id
                superset_id = b.condition_id if subset_id == a.condition_id else a.condition_id
                results.append(MarketRelationship(
                    market_a_id=subset_id,
                    market_b_id=superset_id,
                    relationship_type="subset",
                    confidence=min(jaccard + 0.2, 1.0),
                    description=f"'{a.question[:40]}' is subset of '{b.question[:40]}'",
                ))
            else:
                results.append(MarketRelationship(
                    market_a_id=a.condition_id,
                    market_b_id=b.condition_id,
                    relationship_type="correlated",
                    confidence=jaccard,
                    description=f"Correlated: '{a.question[:40]}' ~ '{b.question[:40]}'",
                ))

    # "by date X" vs "by date Y" for same event — temporal chain
    date_a = _extract_deadline(a.question)
    date_b = _extract_deadline(b.question)
    if date_a and date_b and sim > 0.5:
        base_a = re.sub(r"by\s+\w+\s+\d+", "", a.question.lower()).strip()
        base_b = re.sub(r"by\s+\w+\s+\d+", "", b.question.lower()).strip()
        if _text_similarity(base_a, base_b) > 0.7:
            results.append(MarketRelationship(
                market_a_id=a.condition_id,
                market_b_id=b.condition_id,
                relationship_type="temporal",
                confidence=0.8,
                description=f"Same event, different deadlines",
            ))

    return results


def detect_contradiction(
    rel: MarketRelationship,
    price_a: float,
    price_b: float,
) -> tuple[bool, float]:
    """Check if a relationship has a price contradiction.

    Args:
        rel: The relationship between two markets.
        price_a: YES price of market A.
        price_b: YES price of market B.

    Returns:
        (is_contradiction, gap_size)
    """
    if rel.relationship_type == "subset":
        # Subset should always be <= superset probability
        # If subset_price > superset_price, that's a contradiction
        if price_a > price_b + 0.02:  # A is subset, priced higher than B
            return True, price_a - price_b
        return False, 0.0

    if rel.relationship_type == "temporal":
        # "By April 30" should be >= "By April 15" (later deadline = more likely)
        # We'd need to know which is later to check direction
        gap = abs(price_a - price_b)
        if gap > 0.10:  # Suspicious if similar events with big price gap
            return True, gap
        return False, 0.0

    return False, 0.0


def _text_similarity(a: str, b: str) -> float:
    """Simple word-overlap similarity."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    overlap = len(wa & wb)
    union = len(wa | wb)
    return overlap / union if union > 0 else 0.0


def _extract_key_words(text: str) -> list[str]:
    """Extract meaningful words (skip stop words)."""
    stop = {"will", "the", "a", "an", "in", "on", "by", "be", "to", "of", "or", "and", "is", "at", "for"}
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [w for w in words if w not in stop and len(w) > 2]


def _extract_deadline(text: str) -> str | None:
    """Extract 'by <date>' from question text."""
    match = re.search(r"by\s+(\w+\s+\d{1,2}(?:,?\s*\d{4})?)", text, re.I)
    return match.group(1) if match else None
