"""
WeakPointFinder — AI identifies its own systematic weaknesses

Tracks performance across 4 dimensions:
  regime × hour_bucket × asset × strategy

Generates Thai-language improvement suggestions.
Connects to DesireEngine to request help with persistent weak spots.
"""
import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_HOUR_LABELS = [
    (range(0, 6),   "กลางคืน (0–5น.)"),
    (range(6, 10),  "เช้าตรู่ (6–9น.)"),
    (range(10, 14), "เช้า (10–13น.)"),
    (range(14, 18), "บ่าย (14–17น.)"),
    (range(18, 22), "เย็น (18–21น.)"),
    (range(22, 24), "ค่ำ (22–23น.)"),
]


def _hour_label(hour: int) -> str:
    for rng, label in _HOUR_LABELS:
        if hour in rng:
            return label
    return "ไม่ทราบ"


class WeakPointFinder:
    """
    Performance breakdown tracker and weakness detector.
    """

    MIN_TRADES      = 10    # minimum per dimension before flagging
    WEAK_THRESHOLD  = 0.42  # below this win rate → weak point
    DESIRE_COOLDOWN = 3600  # seconds between desire registrations

    def __init__(self):
        self._by_regime:   Dict[str, dict] = defaultdict(lambda: {"wins": 0, "total": 0})
        self._by_hour:     Dict[str, dict] = defaultdict(lambda: {"wins": 0, "total": 0})
        self._by_asset:    Dict[str, dict] = defaultdict(lambda: {"wins": 0, "total": 0})
        self._by_strategy: Dict[str, dict] = defaultdict(lambda: {"wins": 0, "total": 0})
        self._total = 0
        self._last_desire_ts = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def record(self, regime: str, hour: int, asset: str,
               strategy: str, pnl: float) -> None:
        won = pnl > 0
        bucket = _hour_label(hour)
        for dim, key in [
            (self._by_regime,   regime or "unknown"),
            (self._by_hour,     bucket),
            (self._by_asset,    asset or "unknown"),
            (self._by_strategy, strategy or "Unknown"),
        ]:
            dim[key]["total"] += 1
            if won:
                dim[key]["wins"] += 1
        self._total += 1

    def find_weaknesses(self) -> dict:
        return {
            "worst_regime":   self._worst_of(self._by_regime),
            "worst_hour":     self._worst_of(self._by_hour),
            "worst_asset":    self._worst_of(self._by_asset),
            "worst_strategy": self._worst_of(self._by_strategy),
        }

    def get_suggestions(self, n: int = 3) -> List[str]:
        w = self.find_weaknesses()
        suggestions = []

        wr = w.get("worst_regime")
        if wr and wr.get("is_weak"):
            suggestions.append(
                f"หลีกเลี่ยง regime '{wr['key']}' (win rate {wr['win_rate']:.0%}) "
                f"หรือเพิ่ม threshold ใน context นี้"
            )

        wh = w.get("worst_hour")
        if wh and wh.get("is_weak"):
            suggestions.append(
                f"ช่วง{wh['key']} win rate แค่ {wh['win_rate']:.0%} "
                f"ลองลด position หรือ skip ช่วงนี้"
            )

        ws = w.get("worst_strategy")
        if ws and ws.get("is_weak"):
            suggestions.append(
                f"กลยุทธ์ '{ws['key']}' ทำได้ {ws['win_rate']:.0%} "
                f"ควร re-evaluate เงื่อนไขการใช้"
            )

        wa = w.get("worst_asset")
        if wa and wa.get("is_weak"):
            suggestions.append(
                f"สินทรัพย์ {wa['key']} win rate {wa['win_rate']:.0%} "
                f"ต้องการกลยุทธ์เฉพาะทางมากขึ้น"
            )

        return suggestions[:n]

    def maybe_register_desires(self, desire_engine) -> None:
        """Register desires when persistent weaknesses found (1 hr cooldown)."""
        now = time.time()
        if now - self._last_desire_ts < self.DESIRE_COOLDOWN:
            return
        self._last_desire_ts = now

        w = self.find_weaknesses()
        worst_regime = w.get("worst_regime")
        if worst_regime and worst_regime.get("is_weak") and worst_regime["trades"] >= 20:
            try:
                desire_engine.register(
                    title=f"พัฒนาการเทรด regime '{worst_regime['key']}'",
                    description=(
                        f"win rate ใน regime '{worst_regime['key']}' แค่ "
                        f"{worst_regime['win_rate']:.0%} จาก {worst_regime['trades']} ไม้ "
                        f"ต้องการกลยุทธ์หรือข้อมูลเพิ่มเติมสำหรับ regime นี้"
                    ),
                    urgency=6,
                    category="strategy_research",
                )
            except Exception:
                pass

    def get_breakdown(self) -> dict:
        def _fmt(dim: dict) -> list:
            items = []
            for k, v in dim.items():
                if v["total"] > 0:
                    items.append({
                        "key":      k,
                        "win_rate": round(v["wins"] / v["total"], 3),
                        "wins":     v["wins"],
                        "total":    v["total"],
                    })
            items.sort(key=lambda x: x["win_rate"], reverse=True)
            return items
        return {
            "by_regime":   _fmt(self._by_regime),
            "by_hour":     _fmt(self._by_hour),
            "by_asset":    _fmt(self._by_asset),
            "by_strategy": _fmt(self._by_strategy),
        }

    def stats(self) -> dict:
        return {
            "total_recorded": self._total,
            "weaknesses":     self.find_weaknesses(),
            "suggestions":    self.get_suggestions(3),
            "breakdown":      self.get_breakdown(),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _worst_of(self, dim: dict) -> Optional[dict]:
        qualified = [(k, v) for k, v in dim.items() if v["total"] >= self.MIN_TRADES]
        if not qualified:
            return None
        worst_key, worst_v = min(qualified, key=lambda x: x[1]["wins"] / x[1]["total"])
        wr = worst_v["wins"] / worst_v["total"]
        return {
            "key":      worst_key,
            "win_rate": round(wr, 3),
            "trades":   worst_v["total"],
            "is_weak":  wr < self.WEAK_THRESHOLD,
        }
