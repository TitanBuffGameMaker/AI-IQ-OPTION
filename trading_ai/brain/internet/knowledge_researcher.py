"""
KnowledgeResearcher – internet learner for trading TECHNIQUES, STRATEGIES,
and CONCEPTS (not just news).

The original WebResearcher fetches asset-specific news only.  This module
complements it by gathering general trading knowledge: indicator techniques,
strategy frameworks, risk-management principles, and trader psychology.

Free sources used (no API keys required):
  - Wikipedia REST API   → structured concept summaries (high confidence)
  - Google News RSS      → current discussion and opinions
  - DuckDuckGo HTML      → fallback web search (best-effort)

Rotates through 5 topic categories per session so the brain learns broadly:
  technical_analysis → binary_options → price_action → risk_management
  → trading_psychology → (loop)
"""
import logging
import random
import re
import time
from typing import List, Optional, Tuple
from urllib.parse import quote

import requests

from trading_ai.brain.knowledge_node import KnowledgeNode, NodeType

logger = logging.getLogger(__name__)

# ── Topics organised by category (round-robin one category per session) ───────
RESEARCH_TOPICS = {
    "technical_analysis": [
        "RSI divergence trading strategy",
        "MACD histogram crossover signal",
        "Bollinger Band squeeze breakout",
        "Ichimoku Cloud trading rules",
        "Stochastic oscillator overbought oversold",
        "Fibonacci retracement levels forex",
        "support resistance trading strategy",
        "moving average crossover strategy",
        "ADX trend strength indicator",
        "Parabolic SAR trading signals",
    ],
    "binary_options": [
        "binary options 1 minute strategy",
        "binary options OTC weekend trading",
        "binary options entry timing technique",
        "binary options money management rules",
        "binary options technical analysis tips",
        "binary options trend following strategy",
        "binary options support resistance entry",
    ],
    "price_action": [
        "bullish engulfing candlestick pattern",
        "pin bar reversal trading strategy",
        "doji candlestick meaning interpretation",
        "hammer hanging man pattern",
        "morning star evening star reversal",
        "head and shoulders chart pattern",
        "double top double bottom reversal",
        "triangle pattern breakout trading",
        "flag pennant continuation pattern",
    ],
    "risk_management": [
        "forex risk management rules",
        "position sizing Kelly criterion trading",
        "stop loss strategy binary options",
        "risk reward ratio forex trading",
        "drawdown recovery trading strategy",
        "trading capital preservation rules",
    ],
    "trading_psychology": [
        "trader psychology fear greed discipline",
        "revenge trading avoidance",
        "patience in forex trading",
        "trading journal benefits review",
        "emotional control day trading",
        "market sentiment crowd psychology",
        "overconfidence bias trading mistakes",
    ],
}

# Map keywords found in topic text → Wikipedia article title.
# IMPORTANT: keep this ordered most-specific-first.  Iteration matches the first
# keyword found, so "doji" must appear before "candlestick" or it never wins.
WIKI_TITLE_MAP = {
    # Specific candlestick patterns first (before generic "candlestick")
    "engulfing":          "Engulfing_pattern",
    "doji":               "Doji",
    "hammer":             "Hammer_(candlestick_pattern)",
    "morning star":       "Morning_star_(candlestick_pattern)",
    "evening star":       "Morning_star_(candlestick_pattern)",
    "head and shoulders": "Head_and_shoulders_(chart_pattern)",
    "double top":         "Double_top_and_double_bottom",
    "double bottom":      "Double_top_and_double_bottom",
    "triangle":           "Triangle_(chart_pattern)",
    "pin bar":            "Price_action_trading",
    "candlestick":        "Candlestick_pattern",
    # Indicators
    "rsi":                "Relative_strength_index",
    "macd":               "MACD",
    "bollinger":          "Bollinger_Bands",
    "ichimoku":           "Ichimoku_Kink%C5%8D_Hy%C5%8D",
    "stochastic":         "Stochastic_oscillator",
    "fibonacci":          "Fibonacci_retracement",
    "adx":                "Average_directional_movement_index",
    "parabolic sar":      "Parabolic_SAR",
    "moving average":     "Moving_average",
    "support resistance": "Support_and_resistance",
    # Strategy categories
    "scalping":           "Scalping_(trading)",
    "swing trading":      "Swing_trading",
    "day trading":        "Day_trading",
    "mean reversion":     "Mean_reversion_(finance)",
    "price action":       "Price_action_trading",
    "binary option":      "Binary_option",
    "trend":              "Market_trend",
    # Risk + psychology
    "kelly":              "Kelly_criterion",
    "risk management":    "Risk_management",
    "drawdown":           "Drawdown_(economics)",
    "market sentiment":   "Market_sentiment",
    "trader psychology":  "Behavioral_economics",
    "fear greed":         "Behavioral_economics",
    "overconfidence":     "Overconfidence_effect",
}

WIKIPEDIA_API   = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

# Map category → NodeType so the brain can filter by knowledge kind
CATEGORY_NODE_TYPE = {
    "technical_analysis": NodeType.TECHNIQUE,
    "binary_options":     NodeType.STRATEGY_CONCEPT,
    "price_action":       NodeType.PATTERN,
    "risk_management":    NodeType.RISK_CONCEPT,
    "trading_psychology": NodeType.PSYCHOLOGY,
}


class KnowledgeResearcher:
    """
    Periodically gathers trading knowledge from the internet and converts each
    finding into a KnowledgeNode for the brain graph.
    """

    REQUEST_TIMEOUT          = 10
    SEARCH_COOLDOWN          = 120    # 2 min between research sessions
    MAX_NODES_PER_SESSION    = 20
    TOPICS_PER_SESSION       = 6      # sample N topics from the chosen category

    def __init__(self, asset: str = "EURUSD"):
        self.asset = asset
        self._last_research: float = 0.0
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; TradingAI-Researcher/1.0)"
        })
        self._category_cycle = list(RESEARCH_TOPICS.keys())
        self._category_idx   = 0

    def should_research(self) -> bool:
        return (time.time() - self._last_research) > self.SEARCH_COOLDOWN

    def research(self) -> List[KnowledgeNode]:
        """One research session: pick the next category and gather nodes."""
        if not self.should_research():
            return []

        self._last_research = time.time()
        category   = self._category_cycle[self._category_idx % len(self._category_cycle)]
        self._category_idx += 1
        topics     = RESEARCH_TOPICS[category]
        chosen     = random.sample(topics, min(self.TOPICS_PER_SESSION, len(topics)))
        node_type  = CATEGORY_NODE_TYPE.get(category, NodeType.RULE)
        new_nodes: List[KnowledgeNode] = []

        logger.info("KnowledgeResearcher: category=%s topics=%s", category, chosen)

        for topic in chosen:
            # Wikipedia — structured, reliable, high confidence
            try:
                wiki_node = self._wikipedia_summary(topic, category, node_type)
                if wiki_node:
                    new_nodes.append(wiki_node)
            except Exception as exc:
                logger.debug("Wikipedia '%s' failed: %s", topic, exc)

            if len(new_nodes) >= self.MAX_NODES_PER_SESSION:
                break

            # Google News — current discussion, lower confidence
            try:
                articles = self._google_news(topic)
                for title, summary in articles[:2]:
                    art_node = self._article_to_node(
                        title, summary, topic, category, node_type
                    )
                    if art_node:
                        new_nodes.append(art_node)
                        if len(new_nodes) >= self.MAX_NODES_PER_SESSION:
                            break
            except Exception as exc:
                logger.debug("Google News '%s' failed: %s", topic, exc)

            if len(new_nodes) >= self.MAX_NODES_PER_SESSION:
                break

            time.sleep(0.5)   # gentle rate limit

        if new_nodes:
            logger.info(
                "Knowledge research: %d nodes (category=%s, sources=wiki+news)",
                len(new_nodes), category,
            )
        return new_nodes

    # ── Wikipedia ─────────────────────────────────────────────────────────────

    def _wikipedia_summary(
        self, topic: str, category: str, node_type: NodeType,
    ) -> Optional[KnowledgeNode]:
        wiki_title = self._topic_to_wiki_title(topic)
        if not wiki_title:
            return None

        url  = WIKIPEDIA_API.format(title=wiki_title)
        resp = self._session.get(url, timeout=self.REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return None
        try:
            data = resp.json()
        except ValueError:
            return None
        extract = (data.get("extract") or "").strip()
        if not extract or len(extract) < 50:
            return None

        title     = data.get("title", topic)
        direction = self._extract_direction(extract.lower())
        page_url  = (data.get("content_urls", {})
                          .get("desktop", {}).get("page", ""))

        tags = ["wikipedia", category, "knowledge"]
        if direction:
            tags.append(direction + "_signal")

        return KnowledgeNode(
            node_type=node_type,
            concept=f"[Wiki] {title[:100]}",
            data={
                "topic":          topic,
                "title":          title,
                "summary":        extract[:500],
                "category":       category,
                "source_url":     page_url,
                "direction_bias": direction,
            },
            confidence=0.55,   # Wikipedia is reliable → higher starting confidence
            evidence_count=1,
            source="wikipedia",
            asset=self.asset,
            tags=tags,
        )

    @staticmethod
    def _topic_to_wiki_title(topic: str) -> Optional[str]:
        topic_lower = topic.lower()
        for keyword, wiki_title in WIKI_TITLE_MAP.items():
            if keyword in topic_lower:
                return wiki_title
        return None

    # ── Google News ───────────────────────────────────────────────────────────

    def _google_news(self, query: str) -> List[Tuple[str, str]]:
        url  = GOOGLE_NEWS_RSS.format(query=quote(query))
        resp = self._session.get(url, timeout=self.REQUEST_TIMEOUT)
        resp.raise_for_status()
        content = resp.text

        results: List[Tuple[str, str]] = []
        for item in re.findall(r"<item>(.*?)</item>", content, re.DOTALL)[:5]:
            title_match = re.search(r"<title>(.*?)</title>", item, re.DOTALL)
            desc_match  = re.search(r"<description>(.*?)</description>", item, re.DOTALL)
            if not title_match:
                continue
            title = re.sub(r"<[^>]+>", "", title_match.group(1)).strip()
            desc  = ""
            if desc_match:
                desc = re.sub(r"<[^>]+>", " ", desc_match.group(1)).strip()[:300]
            if title and len(title) > 10:
                results.append((title, desc))
        return results

    def _article_to_node(
        self, title: str, summary: str, topic: str,
        category: str, node_type: NodeType,
    ) -> Optional[KnowledgeNode]:
        text = (title + " " + summary).lower()
        keywords = ["trading", "forex", "binary", "strategy", "technique",
                    "indicator", "pattern", "signal", "analysis", "chart",
                    "market", "trader", "candle", "trend"]
        if not any(kw in text for kw in keywords):
            return None

        direction = self._extract_direction(text)
        tags = ["news", category, "knowledge"]
        if direction:
            tags.append(direction + "_signal")

        return KnowledgeNode(
            node_type=node_type,
            concept=f"[News] {title[:120]}",
            data={
                "topic":          topic,
                "title":          title,
                "summary":        summary[:300],
                "category":       category,
                "direction_bias": direction,
            },
            confidence=0.40,
            evidence_count=1,
            source="google_news_research",
            asset=self.asset,
            tags=tags,
        )

    @staticmethod
    def _extract_direction(text: str) -> Optional[str]:
        bullish = ["bullish", "buy", "long", "upward", "rally", "surge", "rise",
                   "higher", "uptrend", "positive", "upside", "strength", "gain"]
        bearish = ["bearish", "sell", "short", "downward", "drop", "fall", "decline",
                   "lower", "downtrend", "negative", "downside", "weakness", "loss"]
        bull = sum(1 for w in bullish if w in text)
        bear = sum(1 for w in bearish if w in text)
        if bull > bear + 1:
            return "buy"
        if bear > bull + 1:
            return "sell"
        return None
