"""
StrategyLibrary — คลังกลยุทธ์การเทรด

เหมือนมนุษย์ที่รู้หลายสไตล์ — เลือกใช้ strategy ที่เหมาะสมกับสภาพตลาดปัจจุบัน
และอัปเดต win rate จากประสบการณ์จริงอัตโนมัติ
"""
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Strategy:
    name:                 str
    description_th:       str
    entry_conditions:     Dict       # เงื่อนไข indicator
    win_rate:             float = 0.5
    total_trades:         int   = 0
    wins:                 int   = 0
    active:               bool  = True
    confidence_threshold: float = 0.55


class StrategyLibrary:
    """
    คลังกลยุทธ์ 5 แบบ — เลือก strategy ที่เหมาะสมกับ snapshot ปัจจุบัน
    และเรียนรู้จากผลลัพธ์จริง
    """

    _SAVE_FILE = "strategy_library.json"

    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self._strategies: Dict[str, Strategy] = {}
        self._last_selected: Optional[str] = None

        self._init_default_strategies()
        self.load(os.path.join(base_dir, self._SAVE_FILE))
        logger.info("StrategyLibrary loaded: %d strategies", len(self._strategies))

    # ── Init ───────────────────────────────────────────────────────────────────

    def _init_default_strategies(self):
        defaults = [
            Strategy(
                name="TrendFollowing",
                description_th="ตามเทรนด์ — ใช้เมื่อตลาดมีทิศทางชัดเจน",
                entry_conditions={
                    "adx_min": 0.25,       # ADX > 0.25 → มี trend
                    "ema_alignment": True,  # EMA 9 > EMA 21 > EMA 50 → bullish
                    "momentum_min": 0.0,    # momentum positive
                },
                confidence_threshold=0.55,
            ),
            Strategy(
                name="MeanReversion",
                description_th="กลับค่าเฉลี่ย — ใช้เมื่อ RSI overbought/oversold",
                entry_conditions={
                    "rsi_buy_max":  0.35,  # RSI < 0.35 → BUY (oversold)
                    "rsi_sell_min": 0.65,  # RSI > 0.65 → SELL (overbought)
                    "bb_extreme":   True,  # price near BB band
                },
                confidence_threshold=0.58,
            ),
            Strategy(
                name="PatternBreakout",
                description_th="Breakout จาก pattern — ใช้เมื่อราคาหลุด Bollinger Band",
                entry_conditions={
                    "bb_width_max": 0.10,   # BB squeeze (ก่อน breakout)
                    "volume_surge": 1.5,    # volume > 1.5x average
                },
                confidence_threshold=0.60,
            ),
            Strategy(
                name="IchimokuSignal",
                description_th="Ichimoku Cloud — ใช้เมื่อราคาอยู่เหนือ/ใต้ cloud",
                entry_conditions={
                    "ichimoku_cloud_pos_min": 0.0,   # > 0 → above cloud (BUY)
                    "tenkan_kijun_cross":     True,  # tenkan > kijun → bullish
                },
                confidence_threshold=0.60,
            ),
            Strategy(
                name="CandlestickReversal",
                description_th="Candlestick Reversal — Engulfing, Star, Hammer",
                entry_conditions={
                    "pattern_min_score": 0.5,  # is_engulfing, is_star, or is_hammer > 0.5
                },
                confidence_threshold=0.55,
            ),
        ]
        for s in defaults:
            if s.name not in self._strategies:
                self._strategies[s.name] = s

    # ── Selection ──────────────────────────────────────────────────────────────

    def select_strategy(self, snapshot: dict) -> Optional[Strategy]:
        """
        เลือก strategy ที่เหมาะสมกับ market snapshot ปัจจุบัน
        Returns None ถ้าไม่มี strategy ที่เข้าเงื่อนไข
        """
        candidates: List[Tuple[Strategy, float]] = []

        for s in self._strategies.values():
            if not s.active:
                continue
            score = self._match_score(s, snapshot)
            if score > 0:
                candidates.append((s, score))

        if not candidates:
            return None

        # เลือก strategy ที่มี score × win_rate สูงสุด
        candidates.sort(key=lambda x: x[1] * x[0].win_rate, reverse=True)
        best = candidates[0][0]
        self._last_selected = best.name
        logger.debug("StrategyLibrary selected: %s (score×wr=%.3f)",
                     best.name, candidates[0][1] * best.win_rate)
        return best

    def get_strategy_signal(
        self, strategy: Strategy, snapshot: dict
    ) -> Tuple[int, float]:
        """
        คืน (action, confidence) ตาม strategy
        action: 0=hold, 1=buy, 2=sell
        """
        name = strategy.name

        if name == "TrendFollowing":
            return self._trend_following_signal(snapshot, strategy)
        elif name == "MeanReversion":
            return self._mean_reversion_signal(snapshot, strategy)
        elif name == "PatternBreakout":
            return self._pattern_breakout_signal(snapshot, strategy)
        elif name == "IchimokuSignal":
            return self._ichimoku_signal(snapshot, strategy)
        elif name == "CandlestickReversal":
            return self._candlestick_reversal_signal(snapshot, strategy)
        else:
            return 0, 0.5

    # ── Outcome learning ───────────────────────────────────────────────────────

    def record_outcome(self, strategy_name: str, won: bool):
        """อัปเดต win rate ของ strategy จากผลลัพธ์จริง"""
        s = self._strategies.get(strategy_name)
        if s is None:
            return
        s.total_trades += 1
        if won:
            s.wins += 1
        # Recalculate win rate with decay (ให้น้ำหนัก recent trades มากกว่า)
        if s.total_trades >= 5:
            recent_weight = 0.95
            s.win_rate = s.win_rate * recent_weight + (1.0 if won else 0.0) * (1 - recent_weight)
        elif s.total_trades > 0:
            s.win_rate = s.wins / s.total_trades

        # Deactivate ถ้า win rate ต่ำมากหลัง 20 trades
        if s.total_trades >= 20 and s.win_rate < 0.35:
            s.active = False
            logger.warning("Strategy %s deactivated (win_rate=%.2f)", s.name, s.win_rate)

        self.save(os.path.join(self.base_dir, self._SAVE_FILE))

    def get_best_strategy(self) -> Optional[Strategy]:
        """คืน strategy ที่มี win_rate สูงสุด (ต้องมีอย่างน้อย 10 trades)"""
        qualified = [s for s in self._strategies.values()
                     if s.total_trades >= 10 and s.active]
        if not qualified:
            return None
        return max(qualified, key=lambda s: s.win_rate)

    def get_stats(self) -> List[dict]:
        """สถิติทุก strategy"""
        return [
            {
                "name":         s.name,
                "description":  s.description_th,
                "win_rate":     round(s.win_rate, 3),
                "total_trades": s.total_trades,
                "wins":         s.wins,
                "active":       s.active,
            }
            for s in self._strategies.values()
        ]

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str):
        try:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            data = {name: asdict(s) for name, s in self._strategies.items()}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.error("StrategyLibrary save failed: %s", exc)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for name, sdict in data.items():
                if name in self._strategies:
                    s = self._strategies[name]
                    s.win_rate             = sdict.get("win_rate", s.win_rate)
                    s.total_trades         = sdict.get("total_trades", s.total_trades)
                    s.wins                 = sdict.get("wins", s.wins)
                    s.active               = sdict.get("active", s.active)
                    s.confidence_threshold = sdict.get("confidence_threshold", s.confidence_threshold)
            logger.info("StrategyLibrary loaded from %s", path)
        except Exception as exc:
            logger.error("StrategyLibrary load failed: %s", exc)

    # ── Private: match scoring ─────────────────────────────────────────────────

    def _match_score(self, s: Strategy, snapshot: dict) -> float:
        """คืน 0.0 ถ้าไม่เข้าเงื่อนไข, หรือ score > 0 ถ้าเข้า"""
        cond = s.entry_conditions
        name = s.name

        if name == "TrendFollowing":
            adx = snapshot.get("adx", 0.0)
            mom = snapshot.get("momentum", 0.0)
            if adx < cond.get("adx_min", 0.25):
                return 0.0
            score = adx + max(0, mom) * 0.5
            return score

        elif name == "MeanReversion":
            rsi = snapshot.get("rsi", 0.5)
            adx = snapshot.get("adx", 0.0)
            # Mean reversion เหมาะตลาด ranging (ADX ต่ำ)
            if adx > 0.35:
                return 0.0
            if rsi < cond.get("rsi_buy_max", 0.35) or rsi > cond.get("rsi_sell_min", 0.65):
                score = abs(rsi - 0.5) * 2.0
                return score
            return 0.0

        elif name == "PatternBreakout":
            bb_w    = snapshot.get("bb_width", 0.5)
            vol_r   = snapshot.get("volume_ratio", 1.0)
            bb_max  = cond.get("bb_width_max", 0.10)
            vol_min = cond.get("volume_surge", 1.5)
            if bb_w > bb_max:
                return 0.0
            if vol_r < vol_min:
                return 0.0
            return (1.0 - bb_w / bb_max) * vol_r * 0.5

        elif name == "IchimokuSignal":
            cloud = snapshot.get("ichimoku_cloud_pos", 0.0)
            if abs(cloud) < 0.05:  # ราคาใน cloud → ไม่ชัดเจน
                return 0.0
            return abs(cloud)

        elif name == "CandlestickReversal":
            hammer    = abs(snapshot.get("is_hammer", 0.0))
            engulfing = abs(snapshot.get("is_engulfing", 0.0))
            star      = abs(snapshot.get("is_star", 0.0))
            max_pat   = max(hammer, engulfing, star)
            min_score = cond.get("pattern_min_score", 0.5)
            if max_pat < min_score:
                return 0.0
            return max_pat

        return 0.0

    # ── Private: signal generators ─────────────────────────────────────────────

    def _trend_following_signal(
        self, snapshot: dict, s: Strategy
    ) -> Tuple[int, float]:
        ema9  = snapshot.get("ema_9_dist", 0.0)
        ema21 = snapshot.get("ema_21_dist", 0.0)
        mom   = snapshot.get("momentum", 0.0)
        adx   = snapshot.get("adx", 0.0)

        conf = min(0.95, s.confidence_threshold + adx * 0.3)
        if ema9 > 0 and ema21 > 0 and mom > 0:
            return 1, conf   # BUY
        elif ema9 < 0 and ema21 < 0 and mom < 0:
            return 2, conf   # SELL
        return 0, 0.5

    def _mean_reversion_signal(
        self, snapshot: dict, s: Strategy
    ) -> Tuple[int, float]:
        rsi    = snapshot.get("rsi", 0.5)
        bb_pos = snapshot.get("bb_position", 0.5)
        conf   = s.confidence_threshold + abs(rsi - 0.5) * 0.4

        if rsi < 0.35:
            return 1, min(0.90, conf)   # BUY (oversold)
        elif rsi > 0.65:
            return 2, min(0.90, conf)   # SELL (overbought)
        return 0, 0.5

    def _pattern_breakout_signal(
        self, snapshot: dict, s: Strategy
    ) -> Tuple[int, float]:
        bb_pos = snapshot.get("bb_position", 0.5)
        mom    = snapshot.get("momentum", 0.0)
        conf   = s.confidence_threshold + 0.05

        if bb_pos > 0.7 and mom > 0:
            return 1, conf   # BUY breakout up
        elif bb_pos < 0.3 and mom < 0:
            return 2, conf   # SELL breakout down
        elif mom > 0:
            return 1, s.confidence_threshold
        elif mom < 0:
            return 2, s.confidence_threshold
        return 0, 0.5

    def _ichimoku_signal(
        self, snapshot: dict, s: Strategy
    ) -> Tuple[int, float]:
        cloud     = snapshot.get("ichimoku_cloud_pos", 0.0)
        tenkan_d  = snapshot.get("ichimoku_tenkan_dist", 0.0)
        kijun_d   = snapshot.get("ichimoku_kijun_dist", 0.0)
        conf      = s.confidence_threshold + abs(cloud) * 0.2

        if cloud > 0 and tenkan_d > kijun_d:
            return 1, min(0.90, conf)   # BUY: above cloud + tenkan > kijun
        elif cloud < 0 and tenkan_d < kijun_d:
            return 2, min(0.90, conf)   # SELL: below cloud
        elif cloud > 0:
            return 1, s.confidence_threshold
        elif cloud < 0:
            return 2, s.confidence_threshold
        return 0, 0.5

    def _candlestick_reversal_signal(
        self, snapshot: dict, s: Strategy
    ) -> Tuple[int, float]:
        hammer    = snapshot.get("is_hammer", 0.0)
        engulfing = snapshot.get("is_engulfing", 0.0)
        star      = snapshot.get("is_star", 0.0)
        conf      = s.confidence_threshold

        if hammer > 0.5 or engulfing > 0.5 or star > 0.5:
            best = max(hammer, engulfing, star)
            return 1, conf + best * 0.1   # BUY reversal
        elif hammer < -0.5 or engulfing < -0.5 or star < -0.5:
            best = max(abs(hammer), abs(engulfing), abs(star))
            return 2, conf + best * 0.1   # SELL reversal
        return 0, 0.5
