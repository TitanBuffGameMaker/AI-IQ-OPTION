"""
Episodic memory – remembers specific significant trading events.

Like how humans remember "that one time I lost big on NFP day" –
the brain stores remarkable episodes and learns from them directly.
"""
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """One remarkable trading moment worth remembering."""
    episode_id: str
    timestamp: str
    asset: str
    action: str            # "buy" | "sell" | "hold"
    indicators: dict       # snapshot of key indicator values
    pnl: float
    confidence: float
    win: bool
    notes: str = ""        # why this was remarkable
    tags: List[str] = field(default_factory=list)


class EpisodicMemory:
    """
    Stores significant trading episodes (both good and bad).
    Only records episodes that are remarkable – big wins, big losses,
    or unusual market conditions.
    """

    SIGNIFICANCE_THRESHOLD = 0.5   # reward magnitude to be considered significant

    def __init__(self, base_dir: str = "./knowledge", max_episodes: int = 1000):
        self.base_dir = base_dir
        self.max_episodes = max_episodes
        self._path = os.path.join(base_dir, "episodic_memory.json")
        self._episodes: List[Episode] = []
        self._load()

    def record(
        self,
        asset: str,
        action: str,
        indicators: dict,
        pnl: float,
        confidence: float,
        notes: str = "",
        tags: Optional[List[str]] = None,
    ):
        """Record a significant episode."""
        if abs(pnl) < self.SIGNIFICANCE_THRESHOLD:
            return  # not remarkable enough

        ep = Episode(
            episode_id=datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
            timestamp=datetime.utcnow().isoformat(),
            asset=asset,
            action=action,
            indicators=indicators,
            pnl=pnl,
            confidence=confidence,
            win=pnl > 0,
            notes=notes,
            tags=tags or [],
        )
        self._episodes.append(ep)

        # Keep only the most recent max_episodes
        if len(self._episodes) > self.max_episodes:
            self._episodes = self._episodes[-self.max_episodes:]

        self._save()
        logger.info("Episodic memory: recorded %s episode (pnl=%.2f)", "WIN" if ep.win else "LOSS", pnl)

    def recall_similar(self, indicators: dict, n: int = 5) -> List[Episode]:
        """
        Find past episodes where the market looked similar.
        Simple overlap-based matching on indicator direction.
        """
        if not self._episodes:
            return []

        def similarity(ep: Episode) -> float:
            score = 0.0
            count = 0
            for key in ["rsi", "macd_hist", "bb_position", "adx"]:
                if key in ep.indicators and key in indicators:
                    v1 = ep.indicators[key]
                    v2 = indicators[key]
                    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                        # Same direction
                        if (v1 > 0) == (v2 > 0):
                            score += 1
                        # Close magnitude
                        if abs(v1 - v2) < 0.2:
                            score += 0.5
                        count += 1.5
            return score / max(count, 1)

        scored = [(ep, similarity(ep)) for ep in self._episodes]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [ep for ep, _ in scored[:n]]

    def recall_wins(self, n: int = 10) -> List[Episode]:
        wins = [ep for ep in self._episodes if ep.win]
        wins.sort(key=lambda e: e.pnl, reverse=True)
        return wins[:n]

    def recall_losses(self, n: int = 10) -> List[Episode]:
        losses = [ep for ep in self._episodes if not ep.win]
        losses.sort(key=lambda e: e.pnl)
        return losses[:n]

    def summary(self) -> dict:
        if not self._episodes:
            return {"total": 0, "wins": 0, "losses": 0}
        wins = [e for e in self._episodes if e.win]
        losses = [e for e in self._episodes if not e.win]
        return {
            "total": len(self._episodes),
            "wins": len(wins),
            "losses": len(losses),
            "avg_win_pnl": sum(e.pnl for e in wins) / max(len(wins), 1),
            "avg_loss_pnl": sum(e.pnl for e in losses) / max(len(losses), 1),
        }

    def _save(self):
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump([asdict(e) for e in self._episodes], f, indent=2)

    def _load(self):
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            self._episodes = [Episode(**d) for d in data]
            logger.info("Loaded %d episodic memories", len(self._episodes))
        except Exception as exc:
            logger.error("Episodic memory load error: %s", exc)
