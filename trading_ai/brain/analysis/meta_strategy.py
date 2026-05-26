"""
MetaStrategyEngine — AI learns WHEN each strategy works best

Tracks (asset × hour_bucket × regime × strategy) → performance map.
Returns a multiplier 0.70–1.30 applied to confidence in think().

Like developing judgment — not just executing, but knowing WHEN to act.
"""
import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_HOUR_BUCKETS = [
    (range(0, 6),   "night"),
    (range(6, 10),  "early_morning"),
    (range(10, 14), "morning"),
    (range(14, 18), "afternoon"),
    (range(18, 22), "evening"),
    (range(22, 24), "late_evening"),
]


def _bucket(hour: int) -> str:
    for rng, label in _HOUR_BUCKETS:
        if hour in rng:
            return label
    return "unknown"


class MetaStrategyEngine:
    """
    Context-aware performance map.
    Records trade outcomes per (asset, hour_bucket, regime, strategy) context
    and returns a confidence multiplier based on historical performance.
    """

    MIN_TRADES   = 8     # minimum trades before multiplier is applied
    PERSIST_FILE = "meta_strategy.json"

    def __init__(self, base_dir: str = "."):
        self._base_dir = base_dir
        self._perf: Dict[str, dict] = {}   # key → {wins, total}
        self._total_recorded = 0
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def record(self, asset: str, hour: int, regime: str,
               strategy: str, pnl: float) -> None:
        key = self._key(asset, hour, regime, strategy)
        if key not in self._perf:
            self._perf[key] = {"wins": 0, "total": 0}
        entry = self._perf[key]
        entry["total"] += 1
        if pnl > 0:
            entry["wins"] += 1
        self._total_recorded += 1
        if self._total_recorded % 50 == 0:
            self._save()

    def get_context_multiplier(self, asset: str, hour: int,
                                regime: str, strategy: str) -> float:
        """Confidence multiplier 0.70–1.30 based on this context's history."""
        key = self._key(asset, hour, regime, strategy)
        entry = self._perf.get(key)
        if not entry or entry["total"] < self.MIN_TRADES:
            return 1.0
        wr = entry["wins"] / entry["total"]
        if wr >= 0.70:   return 1.30
        if wr >= 0.62:   return 1.15
        if wr >= 0.55:   return 1.05
        if wr <= 0.28:   return 0.70
        if wr <= 0.36:   return 0.82
        if wr <= 0.44:   return 0.92
        return 1.0

    def top_contexts(self, n: int = 5) -> List[dict]:
        return self._ranked_contexts(reverse=True)[:n]

    def worst_contexts(self, n: int = 5) -> List[dict]:
        return self._ranked_contexts(reverse=False)[:n]

    def stats(self) -> dict:
        qualified = sum(1 for e in self._perf.values() if e["total"] >= self.MIN_TRADES)
        return {
            "total_contexts":     len(self._perf),
            "qualified_contexts": qualified,
            "total_recorded":     self._total_recorded,
            "top_contexts":       self.top_contexts(3),
            "worst_contexts":     self.worst_contexts(3),
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    def _key(self, asset: str, hour: int, regime: str, strategy: str) -> str:
        return f"{asset}|{_bucket(hour)}|{regime}|{strategy}"

    def _ranked_contexts(self, reverse: bool) -> List[dict]:
        scored = []
        for key, e in self._perf.items():
            if e["total"] >= self.MIN_TRADES:
                wr = e["wins"] / e["total"]
                scored.append({"key": key, "win_rate": round(wr, 3), "trades": e["total"]})
        scored.sort(key=lambda x: x["win_rate"], reverse=reverse)
        return scored

    def _path(self) -> str:
        os.makedirs(os.path.join(self._base_dir, "knowledge"), exist_ok=True)
        return os.path.join(self._base_dir, "knowledge", self.PERSIST_FILE)

    def _save(self) -> None:
        try:
            with open(self._path(), "w", encoding="utf-8") as f:
                json.dump({"perf": self._perf, "total": self._total_recorded}, f)
        except Exception as e:
            logger.debug("MetaStrategy save: %s", e)

    def _load(self) -> None:
        path = self._path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._perf = data.get("perf", {})
            self._total_recorded = data.get("total", 0)
            logger.info("MetaStrategyEngine loaded %d contexts", len(self._perf))
        except Exception as e:
            logger.debug("MetaStrategy load: %s", e)
