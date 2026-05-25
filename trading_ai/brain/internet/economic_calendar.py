"""
Economic calendar fetcher.

Pulls upcoming high-impact economic events (NFP, CPI, interest rate decisions, etc.)
from public sources. Warns the AI when risky events are approaching.
"""
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

# ForexFactory has a JSON calendar endpoint (no key needed)
FOREXFACTORY_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

HIGH_IMPACT_EVENTS = {
    "NFP", "Non-Farm Payrolls", "FOMC", "Federal Funds Rate",
    "CPI", "Consumer Price Index", "GDP", "Gross Domestic Product",
    "Unemployment Rate", "Interest Rate Decision", "ECB Rate Decision",
    "BOE Rate Decision", "BOJ Rate Decision", "PMI", "Retail Sales",
    "PPI", "Producer Price Index", "Trade Balance",
}


@dataclass
class EconomicEvent:
    title: str
    currency: str           # affected currency (USD, EUR, GBP …)
    impact: str             # "High" | "Medium" | "Low"
    datetime_utc: datetime
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None   # filled after release

    @property
    def is_released(self) -> bool:
        return self.actual is not None

    @property
    def minutes_until(self) -> float:
        now = datetime.now(timezone.utc)
        return (self.datetime_utc - now).total_seconds() / 60

    @property
    def is_high_impact(self) -> bool:
        return self.impact == "High" or any(e in self.title for e in HIGH_IMPACT_EVENTS)


class EconomicCalendar:
    """
    Fetches the economic calendar and tells the brain what's coming.
    The brain uses this to be more cautious before major events.
    """

    CACHE_TTL = 3600        # 1 hour
    REQUEST_TIMEOUT = 8

    def __init__(self, currencies: Optional[List[str]] = None):
        self.currencies = currencies or ["USD", "EUR", "GBP", "JPY", "CHF"]
        self._cache: List[EconomicEvent] = []
        self._last_fetch: float = 0.0
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "Mozilla/5.0"})

    def fetch(self, force: bool = False) -> List[EconomicEvent]:
        """Fetch this week's calendar. Returns [] gracefully if offline."""
        now = time.time()
        if not force and (now - self._last_fetch) < self.CACHE_TTL and self._cache:
            return self._cache

        events = []
        try:
            resp = self._session.get(FOREXFACTORY_URL, timeout=self.REQUEST_TIMEOUT)
            resp.raise_for_status()
            raw = resp.json()
            for item in raw:
                ev = self._parse_event(item)
                if ev and ev.currency in self.currencies:
                    events.append(ev)
        except Exception as exc:
            logger.debug("Economic calendar fetch failed: %s", exc)
            return self._cache  # return stale cache

        self._cache = events
        self._last_fetch = now
        high_count = sum(1 for e in events if e.is_high_impact)
        logger.info(
            "Calendar: %d events this week (%d high-impact)", len(events), high_count
        )
        return events

    def upcoming_high_impact(self, within_minutes: float = 120) -> List[EconomicEvent]:
        """Events happening in the next `within_minutes` minutes."""
        return [
            ev for ev in self.fetch()
            if ev.is_high_impact
            and 0 <= ev.minutes_until <= within_minutes
        ]

    def is_danger_zone(self, within_minutes: float = 30) -> bool:
        """True if a high-impact event is less than 30 min away."""
        return bool(self.upcoming_high_impact(within_minutes))

    def get_risk_multiplier(self) -> float:
        """
        Returns a risk multiplier [0.0 – 1.0].
        1.0 = normal conditions
        0.0 = extreme risk, don't trade
        """
        upcoming = self.upcoming_high_impact(within_minutes=120)
        if not upcoming:
            return 1.0
        # Find the closest event
        closest_minutes = min(ev.minutes_until for ev in upcoming)
        if closest_minutes < 15:
            return 0.0   # don't trade
        if closest_minutes < 30:
            return 0.2
        if closest_minutes < 60:
            return 0.5
        if closest_minutes < 120:
            return 0.75
        return 1.0

    def describe_upcoming(self) -> str:
        upcoming = self.upcoming_high_impact(within_minutes=120)
        if not upcoming:
            return "No high-impact events in the next 2 hours"
        lines = []
        for ev in sorted(upcoming, key=lambda e: e.minutes_until):
            lines.append(
                f"  {ev.title} ({ev.currency}) in {ev.minutes_until:.0f} min"
            )
        return "Upcoming events:\n" + "\n".join(lines)

    @staticmethod
    def _parse_event(item: dict) -> Optional[EconomicEvent]:
        try:
            title = item.get("title", "")
            currency = item.get("country", "").upper()
            impact = item.get("impact", "Low").capitalize()
            date_str = item.get("date", "")
            time_str = item.get("time", "00:00am")

            # Parse datetime (ForexFactory uses format like "Jan 17, 2025")
            try:
                dt = datetime.strptime(
                    f"{date_str} {time_str}", "%b %d, %Y %I:%M%p"
                ).replace(tzinfo=timezone.utc)
            except ValueError:
                dt = datetime.now(timezone.utc)

            return EconomicEvent(
                title=title,
                currency=currency,
                impact=impact,
                datetime_utc=dt,
                forecast=item.get("forecast"),
                previous=item.get("previous"),
                actual=item.get("actual") or None,
            )
        except Exception:
            return None
