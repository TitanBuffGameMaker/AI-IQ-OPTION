"""
Internet news fetcher – the brain's sensory organ for world events.

Pulls financial news from free RSS feeds and public APIs.
No API key required for basic operation.
Converts news articles into KnowledgeNodes that enter the brain graph.
"""
import logging
import re
import time
from datetime import datetime, timedelta
from typing import List, Optional
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

# Free RSS feeds – no API key needed
# Ordered best-quality-first: forex-specialist sites ranked higher
RSS_FEEDS = {
    # Forex-specialist sources (highest relevance)
    "babypips":        "https://www.babypips.com/news/feed",
    "fxstreet":        "https://rss.fxstreet.com/news",
    "dailyfx":         "https://www.dailyfx.com/feeds/all",
    "forexlive":       "https://www.forexlive.com/feed/news",
    # General financial (broader but still relevant)
    "reuters_forex":   "https://feeds.reuters.com/reuters/businessNews",
    "investing_forex": "https://www.investing.com/rss/news_25.rss",
    "marketwatch":     "https://feeds.marketwatch.com/marketwatch/realtimeheadlines/",
    "cnbc_forex":      "https://www.cnbc.com/id/10037090/device/rss/rss.html",
    "investopedia":    "https://www.investopedia.com/feedbuilder/feed/getfeed?feedName=rss_headline",
}

# Quality weight per source (used when scoring nodes)
SOURCE_QUALITY: dict = {
    "babypips":        0.80,   # forex education + analysis
    "fxstreet":        0.80,   # forex-only news
    "dailyfx":         0.75,   # forex analysis
    "forexlive":       0.70,   # live forex news
    "reuters_forex":   0.65,
    "investing_forex": 0.60,
    "marketwatch":     0.55,
    "cnbc_forex":      0.50,
    "investopedia":    0.70,   # educational finance
}

ASSET_KEYWORDS = {
    "EURUSD":  ["EUR", "USD", "euro", "dollar", "ECB", "Fed", "Federal Reserve",
                "European Central Bank", "eurozone", "EURUSD"],
    "GBPUSD":  ["GBP", "USD", "pound", "sterling", "dollar", "Bank of England",
                "BOE", "GBPUSD"],
    "USDJPY":  ["JPY", "USD", "yen", "dollar", "Bank of Japan", "BOJ", "USDJPY"],
    "EURJPY":  ["EUR", "JPY", "euro", "yen", "ECB", "Bank of Japan", "EURJPY"],
    "USDAED":  ["AED", "USD", "dirham", "dollar", "UAE", "USDAED"],
    "BTCUSD":  ["bitcoin", "BTC", "crypto", "cryptocurrency", "BTCUSD"],
    "GOLD":    ["gold", "XAU", "precious metals", "safe haven"],
}

HIGH_IMPACT_KEYWORDS = [
    "NFP", "non-farm payroll", "CPI", "inflation", "interest rate",
    "rate hike", "rate cut", "Fed decision", "FOMC", "ECB decision",
    "GDP", "unemployment", "recession", "crisis", "crash", "surge",
    "war", "sanctions", "oil", "emergency", "default",
]


class NewsArticle:
    def __init__(self, title: str, summary: str, url: str, published: str, source: str):
        self.title = title
        self.summary = summary
        self.url = url
        self.published = published
        self.source = source
        self.relevance_score: float = 0.0
        self.sentiment: float = 0.0    # -1 (bearish) … +1 (bullish)
        self.is_high_impact: bool = False
        self.matched_assets: List[str] = []

    def __repr__(self):
        return f"[{self.source}] {self.title[:70]}"


class NewsFetcher:
    """
    Fetches financial news from the internet and returns structured articles.
    Handles timeouts, retries, and graceful failure when offline.
    """

    REQUEST_TIMEOUT = 8     # seconds
    MAX_ARTICLES = 30       # per fetch session
    CACHE_TTL = 300         # seconds (5 min) – don't hammer feeds

    def __init__(self, asset: str = "EURUSD"):
        self.asset = asset
        self._cache: List[NewsArticle] = []
        self._last_fetch: float = 0.0
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; TradingAI/1.0)"
        })

    def fetch(self, force: bool = False) -> List[NewsArticle]:
        """
        Fetch and parse news. Returns empty list gracefully if offline.
        Results are cached for CACHE_TTL seconds.
        """
        now = time.time()
        if not force and (now - self._last_fetch) < self.CACHE_TTL and self._cache:
            return self._cache

        articles: List[NewsArticle] = []

        # Try each RSS feed
        for source_name, url in RSS_FEEDS.items():
            try:
                parsed = self._fetch_rss(url, source_name)
                articles.extend(parsed)
                if len(articles) >= self.MAX_ARTICLES:
                    break
            except Exception as exc:
                logger.debug("Feed %s failed: %s", source_name, exc)
                continue

        # Score and filter articles
        for article in articles:
            self._score_article(article)

        # Sort by relevance, keep most relevant
        articles.sort(key=lambda a: a.relevance_score, reverse=True)
        articles = articles[: self.MAX_ARTICLES]

        self._cache = articles
        self._last_fetch = now
        logger.info(
            "News fetch complete: %d articles (%d relevant to %s)",
            len(articles),
            sum(1 for a in articles if self.asset in a.matched_assets),
            self.asset,
        )
        return articles

    def get_relevant(self, min_relevance: float = 0.3) -> List[NewsArticle]:
        """Return articles relevant to the current asset."""
        return [a for a in self.fetch() if a.relevance_score >= min_relevance]

    def has_high_impact_news(self) -> bool:
        """True only when a high-impact article is also relevant to the current asset."""
        recent = self.fetch()
        return any(a.is_high_impact and a.relevance_score >= 0.3 for a in recent)

    def get_sentiment_for_asset(self) -> float:
        """
        Return aggregate sentiment for the asset.
        Positive = bullish news, Negative = bearish news.
        """
        relevant = self.get_relevant(min_relevance=0.2)
        if not relevant:
            return 0.0
        weights = [a.relevance_score for a in relevant]
        sentiments = [a.sentiment for a in relevant]
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        return sum(s * w for s, w in zip(sentiments, weights)) / total_weight

    # ── Private ───────────────────────────────────────────────────────────────

    def _fetch_rss(self, url: str, source: str) -> List[NewsArticle]:
        """Parse RSS feed using only stdlib (no feedparser dependency)."""
        try:
            resp = self._session.get(url, timeout=self.REQUEST_TIMEOUT)
            resp.raise_for_status()
            content = resp.text
        except requests.RequestException as exc:
            raise RuntimeError(f"HTTP error: {exc}") from exc

        articles = []
        # Simple XML parsing without external library
        items = re.findall(r"<item>(.*?)</item>", content, re.DOTALL)
        for item in items[:20]:
            title = self._extract_tag(item, "title")
            desc = self._extract_tag(item, "description") or ""
            link = self._extract_tag(item, "link") or ""
            pub = self._extract_tag(item, "pubDate") or datetime.utcnow().isoformat()

            if not title:
                continue

            # Strip HTML tags from description
            desc_clean = re.sub(r"<[^>]+>", " ", desc).strip()[:500]

            articles.append(NewsArticle(
                title=title,
                summary=desc_clean,
                url=link,
                published=pub,
                source=source,
            ))

        return articles

    @staticmethod
    def _extract_tag(text: str, tag: str) -> Optional[str]:
        match = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
        if match:
            raw = match.group(1)
            # Remove CDATA wrapper
            cdata = re.match(r"<!\[CDATA\[(.*?)\]\]>", raw, re.DOTALL)
            if cdata:
                raw = cdata.group(1)
            return raw.strip()
        return None

    def _score_article(self, article: NewsArticle):
        """Score article for relevance and sentiment."""
        text = (article.title + " " + article.summary).lower()

        # Source quality multiplier (forex-specialist sites score higher)
        quality = SOURCE_QUALITY.get(article.source, 0.50)

        # Relevance: does it mention our asset?
        for asset, keywords in ASSET_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw.lower() in text)
            if matches > 0:
                article.matched_assets.append(asset)
                if asset == self.asset:
                    article.relevance_score += (0.3 + matches * 0.1) * quality

        # High impact check
        high_matches = sum(1 for kw in HIGH_IMPACT_KEYWORDS if kw.lower() in text)
        if high_matches > 0:
            article.is_high_impact = True
            article.relevance_score += (0.2 + high_matches * 0.05) * quality

        # Sentiment analysis (simple lexicon-based)
        article.sentiment = self._score_sentiment(text)

        article.relevance_score = min(1.0, article.relevance_score)

    @staticmethod
    def _score_sentiment(text: str) -> float:
        """Return sentiment score from -1 (very bearish) to +1 (very bullish)."""
        bullish_words = [
            "surge", "rally", "rise", "gain", "bullish", "strong", "buy",
            "increase", "growth", "positive", "boost", "recover", "higher",
            "record high", "exceed", "beat", "outperform", "hawkish",
        ]
        bearish_words = [
            "fall", "drop", "decline", "crash", "bearish", "weak", "sell",
            "decrease", "recession", "negative", "plunge", "lower", "miss",
            "disappoint", "underperform", "dovish", "crisis", "fear", "panic",
        ]
        score = 0.0
        for word in bullish_words:
            if word in text:
                score += 0.1
        for word in bearish_words:
            if word in text:
                score -= 0.1
        return max(-1.0, min(1.0, score))
