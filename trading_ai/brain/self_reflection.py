"""
SelfReflectionEngine — วิเคราะห์ทุกไม้หลังปิด

เหมือน trader มืออาชีพที่เรียนรู้จากทุก trade:
- วิเคราะห์ว่าแพ้เพราะอะไร
- จำบทเรียนไว้เป็น lesson
- แนะนำการปรับปรุงเป็นภาษาไทย
"""
import json
import logging
import os
import time
import uuid
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TradeReflection:
    trade_id:            str
    timestamp:           str
    action:              str              # "buy", "sell", "hold"
    pnl:                 float
    won:                 bool
    indicators_at_entry: Dict
    strategy_used:       str
    mistake_type:        Optional[str]    # ประเภทข้อผิดพลาด
    lesson:              str              # ข้อสรุปภาษาไทย
    confidence_at_entry: float
    market_regime:       str              # trending/ranging/volatile


class SelfReflectionEngine:
    """
    วิเคราะห์ทุกไม้และเก็บบทเรียนไว้
    """

    _SAVE_FILE   = "self_reflections.json"
    _MAX_ENTRIES = 500

    # ประเภทข้อผิดพลาดที่พบบ่อย
    _MISTAKE_LABELS = {
        "wrong_rsi_reading":   "RSI อ่านผิดทิศ",
        "no_clear_setup":      "ไม่มี setup ที่ชัดเจน",
        "low_confidence_trade":"เทรดทั้งที่ confidence ต่ำ",
        "against_trend":       "เข้า counter-trend",
        "overtrade":           "เทรดถี่เกินไป",
        "wrong_regime":        "market regime ผิด",
        "news_risk":           "มีข่าวสำคัญ",
    }

    def __init__(self, base_dir: str = "."):
        self.base_dir  = base_dir
        self._reflections: List[TradeReflection] = []
        self.load(os.path.join(base_dir, self._SAVE_FILE))
        logger.info("SelfReflectionEngine loaded: %d reflections", len(self._reflections))

    # ── Main: reflect ──────────────────────────────────────────────────────────

    def reflect(
        self,
        action:        int,
        pnl:           float,
        indicators:    Dict,
        strategy_name: str,
        confidence:    float,
        regime:        str,
    ) -> TradeReflection:
        """
        วิเคราะห์ trade หนึ่งไม้ และสร้าง TradeReflection
        """
        won         = pnl > 0
        action_name = {0: "hold", 1: "buy", 2: "sell"}.get(action, "hold")
        mistake     = None
        lesson      = ""

        if won:
            lesson = self._lesson_for_win(action_name, indicators, strategy_name, regime)
        else:
            mistake = self._detect_mistake(action, pnl, indicators, confidence, regime)
            lesson  = self._lesson_for_loss(mistake, action_name, indicators, confidence)

        reflection = TradeReflection(
            trade_id=str(uuid.uuid4())[:8],
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            action=action_name,
            pnl=round(pnl, 4),
            won=won,
            indicators_at_entry={k: round(v, 4) for k, v in indicators.items()},
            strategy_used=strategy_name or "Unknown",
            mistake_type=mistake,
            lesson=lesson,
            confidence_at_entry=round(confidence, 4),
            market_regime=regime,
        )

        self._reflections.append(reflection)

        # trim เก็บแค่ _MAX_ENTRIES ล่าสุด
        if len(self._reflections) > self._MAX_ENTRIES:
            self._reflections = self._reflections[-self._MAX_ENTRIES:]

        logger.debug(
            "Reflection: %s %s pnl=%.2f mistake=%s lesson=%s",
            action_name.upper(), "WIN" if won else "LOSS",
            pnl, mistake, lesson[:40],
        )

        self.save(os.path.join(self.base_dir, self._SAVE_FILE))
        return reflection

    # ── Queries ────────────────────────────────────────────────────────────────

    def get_recent_mistakes(self, n: int = 10) -> List[str]:
        """คืน mistake types ล่าสุด n รายการ"""
        mistakes = [
            r.mistake_type
            for r in self._reflections[-n:]
            if r.mistake_type is not None
        ]
        return mistakes

    def get_improvement_tips(self) -> List[str]:
        """
        วิเคราะห์ mistake ล่าสุด 20 ไม้ และแนะนำภาษาไทยว่าควรแก้ไขอะไร
        """
        recent = self._reflections[-20:]
        mistakes = [r.mistake_type for r in recent if r.mistake_type]
        if not mistakes:
            return ["สถิติดีมาก! ยังคงรักษาวินัยต่อไป"]

        counts = Counter(mistakes)
        tips: List[str] = []

        if counts.get("wrong_rsi_reading", 0) >= 2:
            tips.append("RSI: ควรรอให้ RSI ยืนยันทิศทางก่อนเข้า — อย่าเร่งรีบ")
        if counts.get("no_clear_setup", 0) >= 2:
            tips.append("Setup: ข้ามไม้ที่ไม่มี indicator ยืนยันอย่างน้อย 3 ตัว")
        if counts.get("low_confidence_trade", 0) >= 2:
            tips.append("Confidence: ตั้ง min confidence ให้สูงขึ้น หรือรอสัญญาณที่แน่ใจกว่า")
        if counts.get("against_trend", 0) >= 2:
            tips.append("Trend: อย่าต่อต้านแนวโน้ม — เทรดตาม EMA alignment เสมอ")
        if counts.get("overtrade", 0) >= 2:
            tips.append("Overtrade: ลด frequency ลง — รอเฉพาะ setup คุณภาพสูง")
        if counts.get("wrong_regime", 0) >= 2:
            tips.append("Regime: ตรวจ ADX ก่อนเข้า — ใช้ strategy ที่เหมาะกับ trending/ranging")
        if counts.get("news_risk", 0) >= 1:
            tips.append("News: หลีกเลี่ยงเทรดช่วง 30 นาทีก่อน/หลังข่าวสำคัญ")

        if not tips:
            most_common = counts.most_common(1)[0][0]
            label = self._MISTAKE_LABELS.get(most_common, most_common)
            tips.append(f"แก้ไข: {label} — เกิดซ้ำบ่อยสุด ({counts[most_common]} ครั้ง)")

        return tips[:4]   # คืนสูงสุด 4 tips

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str):
        try:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            data = [asdict(r) for r in self._reflections[-self._MAX_ENTRIES:]]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.error("SelfReflection save failed: %s", exc)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._reflections = [TradeReflection(**d) for d in data]
            logger.info("SelfReflection loaded %d entries from %s", len(self._reflections), path)
        except Exception as exc:
            logger.error("SelfReflection load failed: %s", exc)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _detect_mistake(
        self,
        action:     int,
        pnl:        float,
        indicators: Dict,
        confidence: float,
        regime:     str,
    ) -> Optional[str]:
        """ตรวจสอบว่าแพ้เพราะอะไร"""
        rsi  = indicators.get("rsi", 0.5)
        adx  = indicators.get("adx", 0.25)

        # RSI ผิดทิศ: buy แต่ RSI overbought, หรือ sell แต่ RSI oversold
        if action == 1 and rsi > 0.65:
            return "wrong_rsi_reading"
        if action == 2 and rsi < 0.35:
            return "wrong_rsi_reading"

        # confidence ต่ำแต่ยังเทรด
        if confidence < 0.60 and action != 0:
            return "low_confidence_trade"

        # เทรดสวนเทรนด์: trending market แต่ action ผิดทาง
        if regime == "trending" and adx > 0.30:
            ema9  = indicators.get("ema_9_dist", 0.0)
            ema21 = indicators.get("ema_21_dist", 0.0)
            if action == 1 and ema9 < 0 and ema21 < 0:
                return "against_trend"
            if action == 2 and ema9 > 0 and ema21 > 0:
                return "against_trend"

        # ไม่มี setup ชัดเจน
        momentum = indicators.get("momentum", 0.0)
        macd     = indicators.get("macd_hist", 0.0)
        if abs(momentum) < 0.1 and abs(macd) < 0.05 and abs(rsi - 0.5) < 0.1:
            return "no_clear_setup"

        # volatile regime
        if regime == "volatile":
            return "wrong_regime"

        return None

    def _lesson_for_win(
        self, action: str, indicators: Dict, strategy: str, regime: str
    ) -> str:
        rsi = indicators.get("rsi", 0.5)
        adx = indicators.get("adx", 0.0)

        if strategy and strategy != "Unknown":
            return f"Pattern ใช้ได้ — {strategy} ทำงานดีใน {regime} market จำไว้"
        if regime == "trending" and adx > 0.25:
            return "Trend following ได้ผล — ADX สูง + EMA alignment ถูกต้อง"
        if rsi < 0.35 and action == "buy":
            return "RSI oversold → BUY ถูก — Mean reversion สำเร็จ"
        if rsi > 0.65 and action == "sell":
            return "RSI overbought → SELL ถูก — Mean reversion สำเร็จ"
        return "Pattern ใช้ได้ จำไว้"

    def _lesson_for_loss(
        self,
        mistake:    Optional[str],
        action:     str,
        indicators: Dict,
        confidence: float,
    ) -> str:
        if mistake == "wrong_rsi_reading":
            rsi = indicators.get("rsi", 0.5)
            return f"RSI={rsi:.2f} ผิดทิศ — ตรวจสอบ RSI ให้ละเอียดก่อนเข้า"
        if mistake == "no_clear_setup":
            return "ไม่มี setup ที่ชัดเจน — ควรรอสัญญาณที่แน่ใจกว่าก่อนเข้า"
        if mistake == "low_confidence_trade":
            return f"Confidence={confidence:.0%} ต่ำเกินไป — ตั้ง threshold ให้สูงขึ้น"
        if mistake == "against_trend":
            return "เข้า counter-trend — ตรวจ EMA alignment และ ADX ก่อนเสมอ"
        if mistake == "overtrade":
            return "เทรดถี่เกินไป — รอเฉพาะ setup คุณภาพสูงเท่านั้น"
        if mistake == "wrong_regime":
            return "Market volatile — ควรลด size หรือหยุดเทรดช่วงนี้"
        if mistake == "news_risk":
            return "มีข่าวสำคัญ — หลีกเลี่ยงการเทรดช่วงนี้"
        return "ไม้นี้แพ้ — วิเคราะห์ indicator ใหม่อีกครั้ง"
