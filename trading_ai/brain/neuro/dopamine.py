"""
Dopamine System (Reward Prediction Error) — จากประสาทวิทยาศาสตร์
════════════════════════════════════════════════════════════════

โดปามีนไม่ได้หลั่งเมื่อ "ได้รางวัล" — แต่หลั่งเมื่อ "รางวัลเกินที่คาดไว้"
(Wolfram Schultz, 1997 — งานวิจัยที่นำไปสู่ Nobel Prize)

  RPE = actual_reward − predicted_reward

  RPE > 0 : outcome ดีกว่าที่คิด → dopamine สูง → เรียนรู้เร็ว (reinforce)
  RPE = 0 : outcome ตรงคาด      → dopamine ปกติ → เรียนรู้น้อย
  RPE < 0 : outcome แย่กว่าคาด  → dopamine ต่ำ → เรียนรู้จากความผิดพลาด

ทำไมระบบนี้สำคัญ?
  → สมองมนุษย์เรียนรู้จาก "ความประหลาดใจ" ไม่ใช่ "ผลลัพธ์ธรรมดา"
  → ถ้า outcome ตามคาดเสมอ → ไม่มี signal ให้เรียนรู้
  → ถ้า outcome แปลกใจ (ดีหรือแย่มาก) → เรียนรู้ได้เยอะสุด

AI Equivalent:
  → Amplify gradient update when outcome surprises the model
  → Reduce update when outcome was expected
  → Matches "Prioritized Experience Replay" concept
"""
import time
import numpy as np


class DopamineSystem:
    """
    Reward Prediction Error — amplifies learning from surprising outcomes.
    """

    def __init__(self, alpha: float = 0.1, surprise_scale: float = 2.5):
        # Exponential moving average of recent outcomes (= "prediction")
        self._expected_value = 0.0
        self._alpha = alpha               # EMA speed (how fast prediction updates)
        self._surprise_scale = surprise_scale  # max learning multiplier

        # Stats
        self._history: list = []          # recent RPEs
        self._total_updates = 0
        self._dopamine_level = 0.5        # 0=depleted, 1=peak

    def update(self, actual_pnl: float) -> float:
        """
        Update after observing actual outcome.
        Returns learning_multiplier (>1 = learn more, <1 = learn less).
        """
        rpe = actual_pnl - self._expected_value

        # Update expectation (neocortical slow learning)
        self._expected_value = (
            (1 - self._alpha) * self._expected_value + self._alpha * actual_pnl
        )

        # Dopamine level proportional to RPE magnitude
        normalized_rpe = np.tanh(rpe / (abs(self._expected_value) + 0.5))
        self._dopamine_level = float(np.clip(0.5 + normalized_rpe, 0.0, 1.0))

        # Learning multiplier: surprise → amplify, expected → normal
        multiplier = 1.0 + (abs(normalized_rpe) * (self._surprise_scale - 1.0))
        multiplier = float(np.clip(multiplier, 0.5, self._surprise_scale))

        # Record
        self._history.append({
            "rpe": float(rpe),
            "actual": float(actual_pnl),
            "expected": float(self._expected_value),
            "multiplier": multiplier,
            "dopamine": self._dopamine_level,
        })
        if len(self._history) > 200:
            self._history = self._history[-200:]
        self._total_updates += 1

        return multiplier

    def get_dopamine_level(self) -> float:
        """Current dopamine level 0.0–1.0."""
        return self._dopamine_level

    def get_expected_value(self) -> float:
        """Model's current prediction of outcome."""
        return self._expected_value

    def get_rpe_trend(self) -> str:
        """Trend of recent RPEs."""
        if len(self._history) < 5:
            return "insufficient data"
        recent = self._history[-10:]
        avg_rpe = np.mean([h["rpe"] for h in recent])
        if avg_rpe > 0.3:
            return "positive_surprise"   # doing better than expected
        if avg_rpe < -0.3:
            return "negative_surprise"   # doing worse than expected
        return "as_expected"

    def stats(self) -> dict:
        if not self._history:
            return {
                "dopamine_level": 0.5,
                "expected_value": 0.0,
                "rpe_trend": "no data",
                "total_updates": 0,
            }
        recent = self._history[-20:]
        return {
            "dopamine_level":  round(self._dopamine_level, 3),
            "expected_value":  round(self._expected_value, 4),
            "avg_rpe":         round(float(np.mean([h["rpe"] for h in recent])), 4),
            "avg_multiplier":  round(float(np.mean([h["multiplier"] for h in recent])), 3),
            "rpe_trend":       self.get_rpe_trend(),
            "total_updates":   self._total_updates,
        }
