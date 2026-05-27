"""
Fear System (Amygdala) — จากประสาทวิทยาศาสตร์
══════════════════════════════════════════════

Amygdala คืออะไร?
  → ส่วนของสมองที่ประมวลความกลัวและความเสี่ยง
  → Trigger "fight-or-flight" response
  → ทำให้มนุษย์ระมัดระวังมากขึ้นหลังจากเจ็บปวด
  → ส่งสัญญาณให้ PFC (prefrontal cortex) ชะลอการตัดสินใจ

ทำไมมนุษย์ถึงระวังมากเมื่อเสียเงินจริง?
  → Loss aversion: เจ็บปวดจากการสูญเสีย 2x มากกว่าความสุขจากกำไรเท่ากัน
     (Daniel Kahneman & Amos Tversky — Prospect Theory)
  → Amygdala activate มากกว่าเมื่อ risk สูง
  → ทำให้ตัดสินใจช้าลง ต้องการ evidence มากขึ้น

AI equivalent ในบัญชีจริง:
  → Fear level เพิ่มขึ้นเมื่อขาดทุนติดกัน
  → Fear ลด position size (Kelly-inspired)
  → Fear เพิ่ม confidence threshold ที่ต้องการ
  → Fear trigger cooldown หลังขาดทุนหนัก
  → Fear ลดลงช้าๆ เมื่อชนะกลับมา (กู้ความมั่นใจ)

ไม่มีผลในบัญชีทดลอง — เป็นระบบเฉพาะบัญชีจริงเท่านั้น
"""
import time
import numpy as np
from typing import Tuple


class FearSystem:
    """
    Amygdala-inspired emotional risk regulation for real-money trading.
    Inactive in PRACTICE mode.
    """

    # Fear thresholds
    _LOSS_STREAK_FEAR    = [3, 5, 7]     # streaks that increase fear
    _FEAR_PER_LOSS       = 0.15          # fear += this per consecutive loss
    _FEAR_RECOVERY_RATE  = 0.08          # fear -= this per win
    _FEAR_DECAY_PER_TICK = 0.01          # slow passive decay over time

    # Cooldown rules (consecutive losses → pause minutes)
    _COOLDOWN_RULES = [
        (5, 15,  "ขาดทุน 5 ไม้ติด — พักคิด 15 นาที"),
        (7, 45,  "ขาดทุน 7 ไม้ติด — พักยาว 45 นาที"),
        (10, 120, "ขาดทุน 10 ไม้ติด — หยุดวันนี้"),
    ]

    def __init__(self, account_type: str = "PRACTICE"):
        self.account_type = account_type.upper()
        self.fear_level = 0.0          # 0.0 = calm, 1.0 = max fear
        self._consecutive_losses = 0
        self._session_pnl = 0.0
        self._cooldown_until = 0.0     # epoch timestamp
        self._cooldown_reason = ""
        self._peak_fear = 0.0
        self._history: list = []

    def set_account_type(self, account_type: str) -> None:
        self.account_type = account_type.upper()
        if self.account_type == "PRACTICE":
            self.fear_level = 0.0     # no fear in practice

    def update(self, pnl: float) -> dict:
        """Update fear state after a trade outcome."""
        if self.account_type != "REAL":
            return self._calm_state()

        self._session_pnl += pnl

        if pnl < 0:
            self._consecutive_losses += 1
            self.fear_level = min(1.0, self.fear_level + self._FEAR_PER_LOSS)
            self._check_cooldown_trigger()
        else:
            self._consecutive_losses = 0
            self.fear_level = max(0.0, self.fear_level - self._FEAR_RECOVERY_RATE)

        self._peak_fear = max(self._peak_fear, self.fear_level)

        entry = {
            "fear": round(self.fear_level, 3),
            "consecutive_losses": self._consecutive_losses,
            "pnl": float(pnl),
            "position_mult": round(self.get_position_multiplier(), 3),
        }
        self._history.append(entry)
        if len(self._history) > 100:
            self._history = self._history[-100:]

        return entry

    def _check_cooldown_trigger(self) -> None:
        for streak, minutes, reason in reversed(self._COOLDOWN_RULES):
            if self._consecutive_losses >= streak:
                self._cooldown_until = time.time() + minutes * 60
                self._cooldown_reason = reason
                break

    # ── Position & Confidence adjustments ────────────────────────────────────

    def get_position_multiplier(self) -> float:
        """
        Kelly-inspired: reduce position size when fear is high.
        Fear=0 → 100% normal size
        Fear=1 → 20% minimum size
        """
        if self.account_type != "REAL":
            return 1.0
        return float(np.clip(1.0 - self.fear_level * 0.8, 0.2, 1.0))

    def get_confidence_threshold_boost(self) -> float:
        """
        Fear demands more evidence before trading.
        Returns additional confidence % required (0–25).
        """
        if self.account_type != "REAL":
            return 0.0
        return float(self.fear_level * 25.0)

    def should_cooldown(self) -> Tuple[bool, str, float]:
        """Returns (in_cooldown, reason, remaining_minutes)."""
        if self.account_type != "REAL":
            return False, "", 0.0
        remaining = self._cooldown_until - time.time()
        if remaining > 0:
            return True, self._cooldown_reason, remaining / 60.0
        return False, "", 0.0

    # ── Emotional state description ──────────────────────────────────────────

    def get_emotion_label(self) -> str:
        if self.account_type != "REAL":
            return "ฝึกซ้อม (ไม่มีความกลัว)"
        if self.fear_level < 0.2:
            return "🟢 สงบ — ตัดสินใจได้ดี"
        if self.fear_level < 0.45:
            return "🟡 ระวัง — เริ่มระมัดระวัง"
        if self.fear_level < 0.7:
            return "🟠 กลัว — ลด position อัตโนมัติ"
        return "🔴 กลัวมาก — ต้องการ confidence สูงมาก"

    def _calm_state(self) -> dict:
        return {"fear": 0.0, "consecutive_losses": 0, "position_mult": 1.0}

    def stats(self) -> dict:
        cooldown, reason, remaining = self.should_cooldown()
        return {
            "account_type":      self.account_type,
            "fear_level":        round(self.fear_level, 3),
            "emotion":           self.get_emotion_label(),
            "consecutive_losses": self._consecutive_losses,
            "position_multiplier": round(self.get_position_multiplier(), 3),
            "conf_threshold_boost": round(self.get_confidence_threshold_boost(), 1),
            "in_cooldown":       cooldown,
            "cooldown_reason":   reason,
            "cooldown_remaining_min": round(remaining, 1),
            "peak_fear":         round(self._peak_fear, 3),
            "session_pnl":       round(self._session_pnl, 4),
        }
