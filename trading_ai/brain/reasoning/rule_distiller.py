"""
RuleDistillationEngine – สกัดกฎการเทรดที่ชัดเจนจากการเทรดที่ชนะสะสม

แทนที่จะปล่อยให้ AI เดาตลอดเวลา เราวิเคราะห์ว่า "เงื่อนไข indicator ไหน
ที่ปรากฏบ่อยในการเทรดที่ชนะ?" แล้วสกัดออกมาเป็น rule ที่อ่านได้และนำไปใช้ได้

ตัวอย่าง rule ที่ distill ได้:
  {"indicator": "rsi", "condition": "< 0.32", "direction": "buy",
   "confidence": 0.71, "n": 45}

Rules จะถูก cache ไว้และอัปเดตทุก 50 trades ใหม่ เพื่อไม่ให้ช้า
brain_core สามารถเรียก get_rules() แล้วส่ง rules เข้า KG ได้เอง
"""
import json
import logging
import os
from collections import deque
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# 10 key indicators ที่ใช้ distill rules (ชื่อตรงกับ indicator vector)
_KEY_INDICATORS = [
    "rsi",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "bb_position",
    "bb_width",
    "ema_9_dist",
    "ema_21_dist",
    "ema_50_dist",
    "ema_200_dist",
]

_MAX_LOG_SIZE    = 200   # เก็บ records ล่าสุดสูงสุดกี่ trades
_UPDATE_EVERY    = 50    # distill ใหม่ทุกกี่ records
_STD_THRESHOLD   = 0.15  # ถ้า std < นี้ → indicator นี้ consistent → สร้าง rule
_TOP_RULES       = 5     # คืน rules ที่ consistent ที่สุดกี่ rule
_RULES_FILE      = "distilled_rules.json"

# ป้ายทิศทาง: ถ้า mean ต่ำ → ควร buy (oversold), สูง → ควร sell (overbought)
_BUY_INDICATORS   = {"rsi", "bb_position"}   # ค่าต่ำ = buy signal
_SELL_INDICATORS  = {"rsi", "bb_position"}   # ค่าสูง = sell signal


class RuleDistillationEngine:
    """
    สกัด trading rules จาก winning trades สะสม

    Lifecycle:
      1. record_trade() — เพิ่ม trade record เข้า log
      2. ทุก UPDATE_EVERY records → distill() อัตโนมัติ
      3. get_rules() — คืน rules ที่ distill ล่าสุด (cached)
      4. format_rules() — คำอธิบายภาษาไทยของ rules

    Notes:
      - Rules ไม่ถูกใส่ KG โดยตรง: brain_core รับ list แล้วจัดการเอง
      - Persist ไป knowledge/distilled_rules.json
    """

    def __init__(self, base_dir: str = "./knowledge"):
        self._path = os.path.join(base_dir, _RULES_FILE)
        os.makedirs(base_dir, exist_ok=True)

        # Rolling log ของ trade records
        self._log: deque = deque(maxlen=_MAX_LOG_SIZE)

        # Cache ของ distilled rules
        self._cached_rules: List[dict] = []

        # นับว่าเพิ่ม record มาแล้วกี่ครั้งตั้งแต่ distill ล่าสุด
        self._records_since_distill = 0

        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def record_trade(self, indicators: dict, action: int, won: bool) -> None:
        """
        บันทึก trade record ลงใน rolling log

        Parameters
        ----------
        indicators : dict ของ indicator names → float values
        action     : int (0=hold, 1=buy, 2=sell)
        won        : bool — ผลการเทรด
        """
        record = {
            "indicators": {k: float(v) for k, v in indicators.items()
                           if isinstance(v, (int, float))},
            "action": int(action),
            "won": bool(won),
        }
        self._log.append(record)
        self._records_since_distill += 1

        # Auto-distill ทุก UPDATE_EVERY records
        if self._records_since_distill >= _UPDATE_EVERY:
            try:
                self._cached_rules = self.distill()
                self._save()
            except Exception as exc:
                logger.warning("RuleDistiller: auto-distill failed – %s", exc)
            self._records_since_distill = 0

    def distill(self, min_wins: int = 30) -> List[dict]:
        """
        สกัด rules จาก winning trades ล่าสุด

        Algorithm:
          1. กรองเฉพาะ wins ที่ action != 0
          2. สำหรับแต่ละ indicator ใน 10 ตัว: คำนวณ mean และ std ของค่า wins
          3. ถ้า std < 0.15 (สม่ำเสมอ) → สร้าง rule
          4. คืน top 5 ที่ consistent ที่สุด (std ต่ำสุด)

        Parameters
        ----------
        min_wins : int — ต้องการ wins อย่างน้อยกี่ครั้งก่อน distill

        Returns
        -------
        List[dict] — rules ที่ distill ได้ เรียงจาก consistent ที่สุด
        """
        wins = [r for r in self._log if r.get("won") and r.get("action", 0) != 0]

        if len(wins) < min_wins:
            logger.debug(
                "RuleDistiller: wins=%d < min_wins=%d → ไม่ distill",
                len(wins), min_wins,
            )
            return self._cached_rules  # คืน cache เดิม

        rules: List[dict] = []

        for ind_name in _KEY_INDICATORS:
            values = []
            actions_buy  = []
            actions_sell = []

            for rec in wins:
                val = rec["indicators"].get(ind_name)
                if val is not None:
                    values.append(float(val))
                    if rec["action"] == 1:
                        actions_buy.append(float(val))
                    elif rec["action"] == 2:
                        actions_sell.append(float(val))

            if len(values) < 5:
                continue  # ข้อมูลน้อยเกินไป

            mean_val = float(np.mean(values))
            std_val  = float(np.std(values))

            if std_val >= _STD_THRESHOLD:
                continue  # ค่าไม่สม่ำเสมอ → ไม่ใช่ rule ที่น่าเชื่อถือ

            # กำหนดทิศทางและเงื่อนไข
            direction, condition = self._infer_direction(
                ind_name, mean_val, actions_buy, actions_sell
            )
            if direction is None:
                continue

            # Confidence = 1 - (std / STD_THRESHOLD) ยิ่ง std ต่ำยิ่ง confident
            confidence = round(1.0 - (std_val / _STD_THRESHOLD), 4)
            confidence = float(np.clip(confidence, 0.50, 0.95))

            rules.append({
                "indicator":  ind_name,
                "condition":  condition,
                "direction":  direction,
                "confidence": confidence,
                "n":          len(values),
                "mean":       round(mean_val, 4),
                "std":        round(std_val, 4),
            })

        # เรียงตาม std ต่ำสุด (consistent ที่สุด) แล้วเอา top 5
        rules.sort(key=lambda r: r["std"])
        top_rules = rules[:_TOP_RULES]

        logger.info(
            "RuleDistiller: distilled %d rules จาก %d wins (total log=%d)",
            len(top_rules), len(wins), len(self._log),
        )
        return top_rules

    def get_rules(self) -> List[dict]:
        """
        คืน distilled rules ที่ cache ไว้ล่าสุด

        Returns
        -------
        List[dict] — rules ที่ distill ได้ล่าสุด (อาจเป็น [] ถ้ายังไม่มีข้อมูลพอ)
        """
        return list(self._cached_rules)

    def format_rules(self) -> str:
        """
        คำอธิบายภาษาไทยของ rules ทั้งหมด

        Returns
        -------
        str — human-readable description
        """
        if not self._cached_rules:
            return "ยังไม่มี rule ที่ distill ได้ (ต้องการ wins อย่างน้อย 30 ครั้ง)"

        lines = [f"กฎการเทรดที่สกัดได้ ({len(self._cached_rules)} ข้อ):"]
        for i, rule in enumerate(self._cached_rules, start=1):
            direction_th = "ซื้อ (BUY)" if rule["direction"] == "buy" else "ขาย (SELL)"
            lines.append(
                f"  {i}. {rule['indicator'].upper()} {rule['condition']} "
                f"→ {direction_th} "
                f"(confidence={rule['confidence']:.0%}, n={rule['n']} trades)"
            )

        return "\n".join(lines)

    def stats(self) -> dict:
        """สรุปสถิติ"""
        wins   = sum(1 for r in self._log if r.get("won"))
        losses = sum(1 for r in self._log if not r.get("won"))
        return {
            "log_size":          len(self._log),
            "wins_in_log":       wins,
            "losses_in_log":     losses,
            "distilled_rules":   len(self._cached_rules),
            "records_since_distill": self._records_since_distill,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _infer_direction(
        ind_name: str,
        mean_val: float,
        buy_vals: List[float],
        sell_vals: List[float],
    ) -> tuple:
        """
        อนุมานทิศทาง (buy/sell) และ condition string จากข้อมูล wins

        ถ้า buy wins > sell wins → direction = buy
        เลือก condition จาก mean ของฝั่งที่ชนะ
        """
        if not buy_vals and not sell_vals:
            return None, None

        if len(buy_vals) >= len(sell_vals):
            direction = "buy"
            ref_mean  = float(np.mean(buy_vals)) if buy_vals else mean_val
        else:
            direction = "sell"
            ref_mean  = float(np.mean(sell_vals)) if sell_vals else mean_val

        # สร้าง condition string
        # ถ้า buy และ ค่าต่ำ → < threshold (oversold)
        # ถ้า sell และ ค่าสูง → > threshold (overbought)
        ref_rounded = round(ref_mean, 2)

        if direction == "buy":
            condition = f"< {ref_rounded}"
        else:
            condition = f"> {ref_rounded}"

        return direction, condition

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        """บันทึก cached rules ลง JSON"""
        try:
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._cached_rules, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._path)
            logger.debug("RuleDistiller: saved %d rules", len(self._cached_rules))
        except Exception as exc:
            logger.warning("RuleDistiller: save failed – %s", exc)

    def _load(self) -> None:
        """โหลด cached rules จาก JSON (ถ้ามี)"""
        if not os.path.exists(self._path):
            logger.info("RuleDistiller: ไม่พบไฟล์ rules เริ่มต้นใหม่")
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self._cached_rules = data
            logger.info(
                "RuleDistiller: loaded %d distilled rules",
                len(self._cached_rules),
            )
        except Exception as exc:
            logger.warning("RuleDistiller: load failed – %s", exc)
            self._cached_rules = []
