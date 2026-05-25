"""
Web researcher – the brain searches the internet to learn new trading concepts.

Periodically searches for articles about:
  - The current asset's technical analysis
  - Market conditions and patterns
  - Trading strategies related to current indicators

Converts what it finds into new KnowledgeNodes (new branches in the root system).
"""
import logging
import re
import time
from typing import List, Optional
from urllib.parse import quote

import requests

from trading_ai.brain.knowledge_node import KnowledgeNode, NodeType

logger = logging.getLogger(__name__)

# Google News RSS – free, no API key
GOOGLE_NEWS_TEMPLATE = (
    "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
)

SEARCH_QUERIES = [
    "{asset} technical analysis today",
    "{asset} forex forecast",
    "{asset} trading signal",
    "forex {asset} support resistance",
    "{asset} market outlook",
]


class WebResearcher:
    """
    Searches the internet for trading knowledge and returns new KnowledgeNodes
    to be planted in the brain graph.

    Runs in the background; the brain calls this periodically.
    """

    REQUEST_TIMEOUT = 8
    SEARCH_COOLDOWN = 1800   # seconds between research sessions (30 min)
    MAX_NODES_PER_SESSION = 5

    def __init__(self, asset: str = "EURUSD"):
        self.asset = asset
        self._last_research: float = 0.0
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; TradingAI-Researcher/1.0)"
        })

    def should_research(self) -> bool:
        return (time.time() - self._last_research) > self.SEARCH_COOLDOWN

    def research(self) -> List[KnowledgeNode]:
        """
        Search the web and return new knowledge nodes.
        Gracefully returns [] if offline or no useful results found.
        """
        if not self.should_research():
            return []

        self._last_research = time.time()
        new_nodes: List[KnowledgeNode] = []

        for query_template in SEARCH_QUERIES:
            query = query_template.format(asset=self.asset)
            try:
                articles = self._search_google_news(query)
                for title, summary in articles[:3]:
                    node = self._article_to_node(title, summary, query)
                    if node:
                        new_nodes.append(node)
                        if len(new_nodes) >= self.MAX_NODES_PER_SESSION:
                            break
            except Exception as exc:
                logger.debug("Web research query '%s' failed: %s", query, exc)
                continue

            if len(new_nodes) >= self.MAX_NODES_PER_SESSION:
                break

        if new_nodes:
            logger.info(
                "Web research complete: %d new knowledge nodes from internet",
                len(new_nodes),
            )
        return new_nodes

    def research_specific_topic(self, topic: str) -> List[KnowledgeNode]:
        """Research a specific topic when the brain identifies a knowledge gap."""
        try:
            articles = self._search_google_news(topic)
            nodes = []
            for title, summary in articles[:5]:
                node = self._article_to_node(title, summary, topic)
                if node:
                    nodes.append(node)
            return nodes
        except Exception as exc:
            logger.debug("Specific research failed: %s", exc)
            return []

    # ── Private ───────────────────────────────────────────────────────────────

    def _search_google_news(self, query: str) -> List[tuple]:
        """Search Google News RSS. Returns list of (title, summary) tuples."""
        url = GOOGLE_NEWS_TEMPLATE.format(query=quote(query))
        try:
            resp = self._session.get(url, timeout=self.REQUEST_TIMEOUT)
            resp.raise_for_status()
            content = resp.text
        except Exception as exc:
            raise RuntimeError(f"Google News RSS failed: {exc}") from exc

        results = []
        items = re.findall(r"<item>(.*?)</item>", content, re.DOTALL)
        for item in items[:10]:
            title_match = re.search(r"<title>(.*?)</title>", item, re.DOTALL)
            desc_match = re.search(r"<description>(.*?)</description>", item, re.DOTALL)
            if title_match:
                title = re.sub(r"<[^>]+>", "", title_match.group(1)).strip()
                desc = ""
                if desc_match:
                    desc = re.sub(r"<[^>]+>", " ", desc_match.group(1)).strip()[:300]
                    cdata = re.match(r"<!\[CDATA\[(.*?)\]\]>", title, re.DOTALL)
                    if cdata:
                        title = cdata.group(1)
                if title and len(title) > 10:
                    results.append((title, desc))
        return results

    def _article_to_node(
        self, title: str, summary: str, source_query: str
    ) -> Optional[KnowledgeNode]:
        """Convert a news article to a KnowledgeNode if it's relevant."""
        text = (title + " " + summary).lower()

        # Filter: must mention our asset or closely related
        asset_lower = self.asset.lower()
        base_currency = self.asset[:3].lower()
        quote_currency = self.asset[3:].lower()

        relevant = (
            asset_lower in text
            or base_currency in text
            or quote_currency in text
            or "forex" in text
            or "trading" in text
        )
        if not relevant:
            return None

        # Determine direction bias
        direction = self._extract_direction(text)
        tags = self._extract_tags(text)
        tags.append("internet")
        tags.append(self.asset.lower())

        concept = f"[Web] {title[:120]}"
        if direction:
            tags.append(direction + "_signal")

        node = KnowledgeNode(
            node_type=NodeType.MARKET_FACT,
            concept=concept,
            data={
                "title": title,
                "summary": summary[:300],
                "query": source_query,
                "direction_bias": direction,
            },
            confidence=0.40,  # internet knowledge starts with moderate confidence
            evidence_count=1,
            source="internet",
            asset=self.asset,
            tags=tags,
        )
        return node

    @staticmethod
    def _extract_direction(text: str) -> Optional[str]:
        bullish = ["bullish", "buy", "long", "upward", "rally", "surge", "rise",
                   "higher", "support", "positive", "upside", "strength"]
        bearish = ["bearish", "sell", "short", "downward", "drop", "fall", "decline",
                   "lower", "resistance", "negative", "downside", "weakness"]
        bull_score = sum(1 for w in bullish if w in text)
        bear_score = sum(1 for w in bearish if w in text)
        if bull_score > bear_score + 1:
            return "buy"
        if bear_score > bull_score + 1:
            return "sell"
        return None

    @staticmethod
    def _extract_tags(text: str) -> List[str]:
        tag_keywords = {
            "rsi": "rsi", "macd": "macd", "bollinger": "bollinger",
            "support": "support", "resistance": "resistance",
            "trend": "trend", "breakout": "breakout", "reversal": "reversal",
            "volatility": "volatility", "momentum": "momentum",
            "nfp": "nfp", "cpi": "cpi", "interest rate": "interest_rate",
            "fed": "fed", "ecb": "ecb", "central bank": "central_bank",
            "technical": "technical_analysis", "fundamental": "fundamental",
        }
        return [tag for keyword, tag in tag_keywords.items() if keyword in text]
