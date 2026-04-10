"""Combinatorial arbitrage engine — multi-market constraint exploitation.

Implements the core finding from arxiv:2508.03474: $39.6M extracted from
Polymarket via probability constraint violations. $28.8M came from NegRisk
market rebalancing alone.

Detection hierarchy (most to least rigorous):
1. STRUCTURAL: Same event_slug / negRiskMarketID — mathematically proven link
   P(outcome_1) + P(outcome_2) + ... = 1.0 by contract design
2. TAG-BASED: Same groupItemTitle or shared tags — high confidence
3. SEMANTIC: TF-IDF cosine similarity fallback — lowest confidence

Key insight: NegRisk markets (multi-outcome events) are the richest source.
The Gamma API already groups outcomes via event_slug and negRiskMarketID.
No NLP or LLM needed for the highest-value opportunities.
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Polymarket fee structure
TAKER_FEE_RATE = 0.02  # 2% taker fee on most markets
CRYPTO_FEE_RATE = 0.05  # 5% coefficient on crypto markets


@dataclass
class MarketOutcome:
    """A single tradeable outcome within a market cluster."""

    condition_id: str
    token_id: str
    question: str
    outcome: str  # "Yes", "No", or specific outcome name
    slug: str
    category: str
    yes_price: float
    no_price: float
    best_bid: float = 0.0
    best_ask: float = 0.0
    liquidity: float = 0.0
    volume_24h: float = 0.0


@dataclass
class MarketCluster:
    """A group of semantically related markets forming a constraint set.

    The core invariant: for mutually exclusive, collectively exhaustive
    outcomes, sum of true probabilities = 1.0 at settlement.
    """

    cluster_id: str
    description: str
    outcomes: list[MarketOutcome]
    constraint_type: str  # "mutex" (mutually exclusive) or "conditional"
    link_confidence: float  # how confident we are these are truly linked


@dataclass
class ArbOpportunity:
    """A detected arbitrage opportunity across a market cluster."""

    cluster_id: str
    description: str
    arb_type: str  # "rebalancing" (single market) or "combinatorial" (multi)
    guaranteed_profit: float  # profit per $1 deployed, after fees
    gross_profit: float  # profit before fees
    total_fees: float
    legs: list[dict[str, Any]]  # individual trades to execute
    total_cost: float  # total capital required
    roi_pct: float  # guaranteed_profit / total_cost * 100
    confidence: float  # confidence in the link/constraint


class CombinatorialArbEngine:
    """Detects and sizes arbitrage across semantically linked markets.

    Detection hierarchy (most to least rigorous):
    1. STRUCTURAL: event_slug / negRiskMarketID grouping (confidence: 0.99)
    2. TAG-BASED: shared groupItemTitle / tags (confidence: 0.85)
    3. SEMANTIC: TF-IDF cosine similarity fallback (confidence: 0.60)
    """

    def __init__(
        self,
        min_profit_pct: float = 0.5,   # min 0.5% guaranteed profit
        min_link_confidence: float = 0.6,
        max_legs: int = 6,
        fee_buffer: float = 1.2,  # 20% safety margin on fees
    ):
        self._min_profit_pct = min_profit_pct
        self._min_link_confidence = min_link_confidence
        self._max_legs = max_legs
        self._fee_buffer = fee_buffer
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._clusters: list[MarketCluster] = []

    def build_clusters(self, markets: list[dict[str, Any]]) -> list[MarketCluster]:
        """Cluster related markets using a 3-tier detection hierarchy.

        Tier 1 (STRUCTURAL): Markets sharing event_slug or negRiskMarketID
               are mathematically linked by contract design. Confidence: 0.99
        Tier 2 (TAG-BASED): Markets sharing groupItemTitle or significant
               tag overlap. Confidence: 0.85
        Tier 3 (SEMANTIC): TF-IDF cosine similarity fallback for markets
               not caught by structural/tag detection. Confidence: 0.60

        Args:
            markets: List of market dicts with at minimum:
                condition_id, question, slug, category, outcomes,
                outcome_prices, tokens, tags
                Optional (for structural detection):
                event_slug, neg_risk_market_id, group_item_title

        Returns:
            List of MarketCluster objects.
        """
        if not markets:
            return []

        n = len(markets)
        clusters_map: dict[int, list[int]] = {}  # cluster_id -> market indices
        cluster_confidence: dict[int, float] = {}  # cluster_id -> confidence
        cluster_type: dict[int, str] = {}  # cluster_id -> detection method
        assigned: set[int] = set()
        cluster_counter = 0

        # ── Tier 1: STRUCTURAL (event_slug / negRiskMarketID) ────────
        # These are mathematically guaranteed links — the contract enforces
        # that outcomes within the same event sum to 1 at settlement.
        event_groups: dict[str, list[int]] = defaultdict(list)
        negrisk_groups: dict[str, list[int]] = defaultdict(list)

        for i, m in enumerate(markets):
            event_slug = m.get("event_slug", "") or m.get("eventSlug", "")
            neg_risk_id = m.get("neg_risk_market_id", "") or m.get("negRiskMarketID", "")

            if neg_risk_id:
                negrisk_groups[neg_risk_id].append(i)
            elif event_slug:
                event_groups[event_slug].append(i)

        # NegRisk markets first (highest value — $28.8M of $39.6M in paper)
        for nrid, indices in negrisk_groups.items():
            if len(indices) >= 2:
                clusters_map[cluster_counter] = indices
                cluster_confidence[cluster_counter] = 0.99
                cluster_type[cluster_counter] = "negrisk_structural"
                for i in indices:
                    assigned.add(i)
                cluster_counter += 1

        # Event-slug groups
        for slug, indices in event_groups.items():
            unassigned = [i for i in indices if i not in assigned]
            if len(unassigned) >= 2:
                clusters_map[cluster_counter] = unassigned
                cluster_confidence[cluster_counter] = 0.95
                cluster_type[cluster_counter] = "event_structural"
                for i in unassigned:
                    assigned.add(i)
                cluster_counter += 1

        # ── Tier 2: TAG-BASED (groupItemTitle / tag overlap) ─────────
        group_title_map: dict[str, list[int]] = defaultdict(list)
        for i, m in enumerate(markets):
            if i in assigned:
                continue
            title = m.get("group_item_title", "") or m.get("groupItemTitle", "")
            if title:
                group_title_map[title].append(i)

        for title, indices in group_title_map.items():
            if len(indices) >= 2:
                clusters_map[cluster_counter] = indices
                cluster_confidence[cluster_counter] = 0.85
                cluster_type[cluster_counter] = "tag_based"
                for i in indices:
                    assigned.add(i)
                cluster_counter += 1

        # ── Tier 3: SEMANTIC (TF-IDF fallback) ───────────────────────
        docs = []
        for m in markets:
            text = self._normalize_text(m.get("question", ""))
            tokens = self._tokenize(text)
            docs.append(tokens)

        self._build_idf(docs)
        tfidf_vectors = [self._tfidf_vector(d) for d in docs]

        for i in range(n):
            if i in assigned:
                continue

            group = [i]
            assigned.add(i)

            for j in range(i + 1, n):
                if j in assigned:
                    continue
                if markets[i].get("category", "") != markets[j].get("category", ""):
                    continue

                sim = self._cosine_similarity(tfidf_vectors[i], tfidf_vectors[j])
                if sim >= 0.45:
                    group.append(j)
                    assigned.add(j)

            if len(group) >= 2:
                clusters_map[cluster_counter] = group
                cluster_confidence[cluster_counter] = 0.60
                cluster_type[cluster_counter] = "semantic_tfidf"
                cluster_counter += 1

        # ── Also detect single-market rebalancing (YES+NO != 1) ──────
        for i in range(n):
            if i not in assigned:
                m = markets[i]
                prices = m.get("outcome_prices", [])
                if len(prices) >= 2:
                    total = sum(prices)
                    if abs(total - 1.0) > 0.02:
                        clusters_map[cluster_counter] = [i]
                        cluster_confidence[cluster_counter] = 0.95
                        cluster_type[cluster_counter] = "rebalancing"
                        cluster_counter += 1

        # ── Build MarketCluster objects ──────────────────────────────
        self._clusters = []
        for cid, indices in clusters_map.items():
            outcomes = []
            questions = []

            for idx in indices:
                m = markets[idx]
                prices = m.get("outcome_prices", [])
                tokens = m.get("tokens", [])
                outcome_names = m.get("outcomes", ["Yes", "No"])

                yes_price = prices[0] if len(prices) > 0 else 0.5
                no_price = prices[1] if len(prices) > 1 else 1.0 - yes_price

                token_id = tokens[0].get("token_id", "") if tokens else ""
                if isinstance(token_id, dict):
                    token_id = token_id.get("token_id", "")
                outcomes.append(MarketOutcome(
                    condition_id=m.get("condition_id", ""),
                    token_id=token_id,
                    question=m.get("question", ""),
                    outcome=outcome_names[0] if outcome_names else "Yes",
                    slug=m.get("slug", ""),
                    category=m.get("category", "other"),
                    yes_price=yes_price,
                    no_price=no_price,
                    liquidity=m.get("liquidity", 0),
                    volume_24h=m.get("volume_24h", 0),
                ))
                questions.append(m.get("question", ""))

            # Use pre-computed confidence from detection tier
            confidence = cluster_confidence.get(cid, 0.60)
            detection = cluster_type.get(cid, "unknown")
            ctype = "rebalancing" if len(indices) == 1 else "mutex"

            desc_prefix = {
                "negrisk_structural": "NegRisk",
                "event_structural": "Event",
                "tag_based": "Tag",
                "semantic_tfidf": "Semantic",
                "rebalancing": "Rebalance",
            }.get(detection, "Unknown")

            cluster = MarketCluster(
                cluster_id=f"cluster_{cid}",
                description=f"[{desc_prefix}] {'; '.join(q[:50] for q in questions[:3])}",
                outcomes=outcomes,
                constraint_type=ctype,
                link_confidence=confidence,
            )
            self._clusters.append(cluster)

        logger.info("Built %d market clusters from %d markets", len(self._clusters), n)
        return self._clusters

    def detect_arbitrage(
        self,
        clusters: list[MarketCluster] | None = None,
    ) -> list[ArbOpportunity]:
        """Scan clusters for arbitrage opportunities.

        For each cluster, check if the constraint (probabilities sum to 1)
        is violated by more than fees allow.
        """
        if clusters is None:
            clusters = self._clusters

        opportunities: list[ArbOpportunity] = []

        for cluster in clusters:
            if cluster.link_confidence < self._min_link_confidence:
                continue

            if cluster.constraint_type == "rebalancing":
                opp = self._check_rebalancing_arb(cluster)
            else:
                opp = self._check_combinatorial_arb(cluster)

            if opp and opp.roi_pct >= self._min_profit_pct:
                opportunities.append(opp)

        # Sort by ROI descending
        opportunities.sort(key=lambda x: x.roi_pct, reverse=True)
        logger.info("Found %d arbitrage opportunities", len(opportunities))
        return opportunities

    def _check_rebalancing_arb(self, cluster: MarketCluster) -> ArbOpportunity | None:
        """Check single-market YES+NO < 1 arbitrage.

        If we can buy YES at p_yes and NO at p_no where p_yes + p_no < 1,
        we're guaranteed $1 at settlement for < $1 cost.
        """
        if not cluster.outcomes:
            return None

        outcome = cluster.outcomes[0]
        yes_p = outcome.yes_price
        no_p = outcome.no_price
        total_cost = yes_p + no_p

        if total_cost >= 1.0:
            return None  # no arb if sum >= 1

        # Fee calculation
        is_crypto = outcome.category.lower() in ("crypto", "cryptocurrency")
        fee_rate = CRYPTO_FEE_RATE if is_crypto else TAKER_FEE_RATE
        total_fees = (yes_p + no_p) * fee_rate * self._fee_buffer

        gross_profit = 1.0 - total_cost
        net_profit = gross_profit - total_fees

        if net_profit <= 0:
            return None

        legs = [
            {
                "action": "BUY",
                "outcome": "YES",
                "price": yes_p,
                "token_id": outcome.token_id,
                "condition_id": outcome.condition_id,
                "slug": outcome.slug,
            },
            {
                "action": "BUY",
                "outcome": "NO",
                "price": no_p,
                "token_id": "",  # would need NO token_id
                "condition_id": outcome.condition_id,
                "slug": outcome.slug,
            },
        ]

        return ArbOpportunity(
            cluster_id=cluster.cluster_id,
            description=f"Rebalancing: {outcome.question[:60]}",
            arb_type="rebalancing",
            guaranteed_profit=net_profit,
            gross_profit=gross_profit,
            total_fees=total_fees,
            legs=legs,
            total_cost=total_cost,
            roi_pct=(net_profit / total_cost) * 100 if total_cost > 0 else 0,
            confidence=0.95,  # rebalancing is high confidence
        )

    def _check_combinatorial_arb(self, cluster: MarketCluster) -> ArbOpportunity | None:
        """Check multi-market constraint violations.

        For mutually exclusive outcomes across related markets:
        If sum of YES prices < 1.0, buy all YES sides => guaranteed 1 payout.
        If sum of YES prices > 1.0, sell all (buy NO on each) => guaranteed profit.
        """
        if len(cluster.outcomes) < 2 or len(cluster.outcomes) > self._max_legs:
            return None

        outcomes = cluster.outcomes
        yes_prices = [o.yes_price for o in outcomes]
        no_prices = [o.no_price for o in outcomes]

        # Strategy A: Buy all YES sides (cost = sum of yes_prices, payout = $1)
        sum_yes = sum(yes_prices)

        # Strategy B: Buy all NO sides (cost = sum of no_prices, payout = $(n-1))
        # because exactly one outcome resolves YES, so (n-1) NO tokens pay $1 each
        sum_no = sum(no_prices)
        n = len(outcomes)

        # Determine fee rate (use worst-case)
        is_crypto = any(o.category.lower() in ("crypto", "cryptocurrency") for o in outcomes)
        fee_rate = CRYPTO_FEE_RATE if is_crypto else TAKER_FEE_RATE

        best_opp = None

        # Check Strategy A: sum_yes < 1.0 means buying all YES is profitable
        if sum_yes < 1.0:
            total_fees_a = sum_yes * fee_rate * self._fee_buffer
            gross_a = 1.0 - sum_yes
            net_a = gross_a - total_fees_a

            if net_a > 0:
                legs = [
                    {
                        "action": "BUY",
                        "outcome": f"YES ({o.outcome})",
                        "price": o.yes_price,
                        "token_id": o.token_id,
                        "condition_id": o.condition_id,
                        "slug": o.slug,
                        "question": o.question[:50],
                    }
                    for o in outcomes
                ]
                best_opp = ArbOpportunity(
                    cluster_id=cluster.cluster_id,
                    description=f"Combinatorial YES: {cluster.description[:80]}",
                    arb_type="combinatorial",
                    guaranteed_profit=net_a,
                    gross_profit=gross_a,
                    total_fees=total_fees_a,
                    legs=legs,
                    total_cost=sum_yes,
                    roi_pct=(net_a / sum_yes) * 100 if sum_yes > 0 else 0,
                    confidence=cluster.link_confidence,
                )

        # Check Strategy B: sum_no < (n-1) means buying all NO is profitable
        expected_payout_b = n - 1  # (n-1) NO tokens pay $1
        if sum_no < expected_payout_b:
            total_fees_b = sum_no * fee_rate * self._fee_buffer
            gross_b = expected_payout_b - sum_no
            net_b = gross_b - total_fees_b

            if net_b > 0:
                legs = [
                    {
                        "action": "BUY",
                        "outcome": f"NO ({o.outcome})",
                        "price": o.no_price,
                        "token_id": "",  # need NO token_id
                        "condition_id": o.condition_id,
                        "slug": o.slug,
                        "question": o.question[:50],
                    }
                    for o in outcomes
                ]
                opp_b = ArbOpportunity(
                    cluster_id=cluster.cluster_id,
                    description=f"Combinatorial NO: {cluster.description[:80]}",
                    arb_type="combinatorial",
                    guaranteed_profit=net_b,
                    gross_profit=gross_b,
                    total_fees=total_fees_b,
                    legs=legs,
                    total_cost=sum_no,
                    roi_pct=(net_b / sum_no) * 100 if sum_no > 0 else 0,
                    confidence=cluster.link_confidence,
                )

                if best_opp is None or opp_b.roi_pct > best_opp.roi_pct:
                    best_opp = opp_b

        return best_opp

    # --- TF-IDF utilities (no external dependencies) ---

    def _normalize_text(self, text: str) -> str:
        """Normalize market question text for comparison."""
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize and remove stop words."""
        stop = {
            "will", "the", "a", "an", "in", "on", "by", "be", "to", "of",
            "or", "and", "is", "at", "for", "it", "this", "that", "with",
            "as", "are", "was", "has", "have", "do", "does", "did", "been",
            "being", "from", "its", "more", "than", "not", "but", "what",
            "which", "who", "whom", "how", "when", "where", "there", "here",
        }
        words = text.split()
        return [w for w in words if w not in stop and len(w) > 1]

    def _build_idf(self, docs: list[list[str]]) -> None:
        """Build inverse document frequency from corpus."""
        n = len(docs)
        if n == 0:
            return

        df: dict[str, int] = defaultdict(int)
        for doc in docs:
            for word in set(doc):
                df[word] += 1

        self._idf = {}
        for word, count in df.items():
            self._idf[word] = math.log(n / (1 + count))

    def _tfidf_vector(self, tokens: list[str]) -> dict[str, float]:
        """Compute TF-IDF vector for a document."""
        if not tokens:
            return {}

        tf: dict[str, int] = defaultdict(int)
        for t in tokens:
            tf[t] += 1

        vec: dict[str, float] = {}
        max_tf = max(tf.values()) if tf else 1
        for word, count in tf.items():
            normalized_tf = count / max_tf
            idf = self._idf.get(word, 0)
            vec[word] = normalized_tf * idf

        return vec

    def _cosine_similarity(
        self, a: dict[str, float], b: dict[str, float]
    ) -> float:
        """Cosine similarity between two sparse TF-IDF vectors."""
        if not a or not b:
            return 0.0

        common = set(a.keys()) & set(b.keys())
        if not common:
            return 0.0

        dot = sum(a[w] * b[w] for w in common)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def get_stats(self) -> dict[str, Any]:
        """Return engine statistics with detection tier breakdown."""
        tier_counts: dict[str, int] = defaultdict(int)
        for c in self._clusters:
            # Extract tier from description prefix
            if c.description.startswith("[NegRisk]"):
                tier_counts["structural_negrisk"] += 1
            elif c.description.startswith("[Event]"):
                tier_counts["structural_event"] += 1
            elif c.description.startswith("[Tag]"):
                tier_counts["tag_based"] += 1
            elif c.description.startswith("[Semantic]"):
                tier_counts["semantic_tfidf"] += 1
            elif c.description.startswith("[Rebalance]"):
                tier_counts["rebalancing"] += 1
            else:
                tier_counts["unknown"] += 1

        return {
            "total_clusters": len(self._clusters),
            "mutex_clusters": sum(
                1 for c in self._clusters if c.constraint_type == "mutex"
            ),
            "rebalancing_clusters": sum(
                1 for c in self._clusters if c.constraint_type == "rebalancing"
            ),
            "avg_cluster_size": (
                sum(len(c.outcomes) for c in self._clusters) / len(self._clusters)
                if self._clusters
                else 0
            ),
            "detection_tiers": dict(tier_counts),
            "avg_confidence": (
                sum(c.link_confidence for c in self._clusters) / len(self._clusters)
                if self._clusters
                else 0
            ),
            "vocab_size": len(self._idf),
        }
