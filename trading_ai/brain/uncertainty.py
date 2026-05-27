"""
UncertaintyEstimator – แยกความไม่แน่นอน 2 ประเภท

  Epistemic  → ความไม่รู้ตัว (unknown unknowns)
               ตลาดอยู่ในสภาวะที่ AI เคยเห็นน้อยมาก → ไม่ควรเดิมพัน

  Aleatoric  → สัญญาณรบกวนโดยธรรมชาติ (inherent noise)
               สภาวะนี้เคยเห็นบ่อย แต่ผลลัพธ์ไม่แน่นอน → เสี่ยงสูงในตัวเอง

การรวมทั้งสองอย่างช่วย brain_core ตัดสินใจว่าควรข้ามการเทรดหรือลด confidence
"""
import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ชื่อ 10 indicator แรกในเวกเตอร์ 40 มิติ
_INDICATOR_NAMES = [
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


class UncertaintyEstimator:
    """
    ประมาณความไม่แน่นอน epistemic และ aleatoric จาก indicator vector

    ใช้ episodic memory เพื่อถามว่า "เราเคยเห็นสภาวะตลาดนี้บ่อยแค่ไหน?
    และครั้งที่เคยเห็น ผลลัพธ์สม่ำเสมอแค่ไหน?"
    """

    # Thresholds สำหรับ epistemic uncertainty ตามจำนวน similar trades
    _EPI_LEVELS = [
        (3,  0.85),   # < 3 trades known → very unfamiliar
        (7,  0.60),   # 3–7 trades → somewhat unfamiliar
        (15, 0.35),   # 7–15 trades → moderately familiar
    ]
    _EPI_HIGH_FAMILIAR = 0.15  # 15+ trades → well known

    # น้ำหนักในการลด confidence สุดท้าย
    _EPISTEMIC_WEIGHT = 0.30
    _ALEATORIC_WEIGHT = 0.15

    def estimate(
        self,
        indicator_vec: np.ndarray,
        episodic_memory,
        short_term_memory=None,
    ) -> Dict:
        """
        คำนวณ uncertainty จาก indicator vector

        Parameters
        ----------
        indicator_vec     : np.ndarray ขนาด 40 floats (normalized indicators)
        episodic_memory   : EpisodicMemory instance ที่มี recall_similar()
        short_term_memory : ShortTermMemory (optional, ยังไม่ได้ใช้ใน v1)

        Returns
        -------
        dict ที่มี keys: epistemic, aleatoric, total, familiar_trades,
                          conf_multiplier, description
        """
        snapshot = self._vec_to_snapshot(indicator_vec)

        # ── ดึง similar past trades ────────────────────────────────────────────
        similar_episodes: List = []
        try:
            similar_episodes = episodic_memory.recall_similar(snapshot, n=20)
        except Exception as exc:
            logger.warning("UncertaintyEstimator: recall_similar failed – %s", exc)

        familiar_trades = len(similar_episodes)

        # ── Epistemic uncertainty ──────────────────────────────────────────────
        epistemic = self._compute_epistemic(familiar_trades)

        # ── Aleatoric uncertainty ──────────────────────────────────────────────
        aleatoric = self._compute_aleatoric(similar_episodes)

        # ── Combined ───────────────────────────────────────────────────────────
        total = float(np.clip(0.6 * epistemic + 0.4 * aleatoric, 0.0, 1.0))

        # ── Confidence multiplier สำหรับ brain_core ───────────────────────────
        conf_multiplier = round(
            1.0 - self._EPISTEMIC_WEIGHT * epistemic - self._ALEATORIC_WEIGHT * aleatoric,
            4,
        )
        conf_multiplier = float(np.clip(conf_multiplier, 0.40, 1.0))

        description = self._describe(epistemic, aleatoric, familiar_trades)

        result = {
            "epistemic":       round(epistemic, 4),
            "aleatoric":       round(aleatoric, 4),
            "total":           round(total, 4),
            "familiar_trades": familiar_trades,
            "conf_multiplier": conf_multiplier,
            "description":     description,
        }

        logger.debug(
            "Uncertainty → epistemic=%.2f aleatoric=%.2f total=%.2f familiar=%d",
            epistemic, aleatoric, total, familiar_trades,
        )
        return result

    def should_skip_trade(self, epistemic: float, aleatoric: float) -> bool:
        """
        คืน True ถ้าทั้ง epistemic สูงมาก (ไม่รู้จักสภาวะนี้เลย)
        และ aleatoric สูงมาก (สภาวะที่รู้จักก็ยังคาดเดาไม่ได้)
        → ไม่ควรเข้าเทรดในกรณีที่ไม่รู้จริงๆ และสัญญาณก็ไม่ชัดเจน
        """
        return epistemic > 0.75 and aleatoric > 0.70

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _vec_to_snapshot(indicator_vec: np.ndarray) -> dict:
        """แปลง vector 40 มิติ → dict โดยใช้แค่ 10 indicator แรก"""
        snapshot = {}
        for idx, name in enumerate(_INDICATOR_NAMES):
            if idx < len(indicator_vec):
                snapshot[name] = float(indicator_vec[idx])
        return snapshot

    def _compute_epistemic(self, familiar_trades: int) -> float:
        """
        ยิ่ง familiar_trades น้อย → epistemic สูง (AI ไม่คุ้นเคยกับสภาวะนี้)
        """
        for threshold, uncertainty in self._EPI_LEVELS:
            if familiar_trades < threshold:
                return uncertainty
        return self._EPI_HIGH_FAMILIAR

    @staticmethod
    def _compute_aleatoric(similar_episodes: list) -> float:
        """
        ยิ่ง win_rate ใกล้ 50% → aleatoric สูง (ผลลัพธ์คาดเดาไม่ได้แม้จะคุ้นเคย)
        ยิ่ง win_rate ห่างจาก 50% → aleatoric ต่ำ (มี edge ชัดเจน)
        """
        if not similar_episodes:
            return 0.80  # ไม่มีข้อมูลเลย → สมมติว่าเสียงดังมาก

        wins = sum(1 for ep in similar_episodes if getattr(ep, "win", False))
        total = len(similar_episodes)
        win_rate = wins / total

        deviation = abs(win_rate - 0.5)

        # Coin flip zone → very noisy
        if deviation < 0.10:
            return 0.80
        # Moderate edge
        if deviation < 0.20:
            # linear interpolation 0.80 → 0.30 between deviation 0.10 and 0.20
            t = (deviation - 0.10) / 0.10
            return round(0.80 - t * 0.50, 4)
        # Clear edge (deviation >= 0.20) → low noise
        return 0.30

    @staticmethod
    def _describe(epistemic: float, aleatoric: float, familiar_trades: int) -> str:
        """คำอธิบายภาษาไทยของระดับความไม่แน่นอน"""
        if epistemic > 0.75 and aleatoric > 0.70:
            return (
                f"⚠️ ไม่รู้จักสภาวะนี้เลย (เคยเห็น {familiar_trades} ครั้ง) "
                "และสัญญาณก็ยังวุ่นวาย → ควรข้ามการเทรดนี้"
            )
        if epistemic > 0.75:
            return (
                f"สภาวะใหม่มาก (เคยเห็น {familiar_trades} ครั้ง) "
                "→ ลด confidence ลงมาก"
            )
        if aleatoric > 0.70:
            return (
                f"สภาวะคุ้นเคย ({familiar_trades} ครั้ง) แต่ผลลัพธ์สุ่ม "
                "→ ลด confidence ลงปานกลาง"
            )
        if epistemic < 0.20 and aleatoric < 0.35:
            return (
                f"สภาวะที่รู้จักดี ({familiar_trades} ครั้ง) และมี edge ชัดเจน "
                "→ confidence ปกติ"
            )
        return (
            f"ความไม่แน่นอนปานกลาง (epistemic={epistemic:.2f}, "
            f"aleatoric={aleatoric:.2f}, familiar={familiar_trades} ครั้ง)"
        )
