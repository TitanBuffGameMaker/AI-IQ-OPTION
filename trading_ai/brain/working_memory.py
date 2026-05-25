"""
WorkingMemory — ความคิดที่กำลังทำอยู่ ณ ปัจจุบัน

เก็บ 20 trades ล่าสุด พร้อมวิเคราะห์ pattern, attention, และ losing streak
เพื่อปรับ behaviour ของ BrainCore ในเวลาจริง
"""
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MemorySlot:
    timestamp:    str
    obs:          np.ndarray
    action:       int            # 0=hold, 1=buy, 2=sell
    confidence:   float
    pnl:          Optional[float] = None
    pattern_tags: List[str]       = field(default_factory=list)


class WorkingMemory:
    """
    Short-horizon memory of the last `capacity` trades.

    Methods
    -------
    update()               — add new slot
    get_recent_pattern()   — dominant action, consistency, win rate, momentum
    get_active_attention() — cosine similarity to the 5 most recent obs
    detect_losing_streak() — count consecutive recent losses
    should_pause()         — True if AI should stop trading temporarily
    """

    def __init__(self, capacity: int = 20):
        self.capacity = capacity
        self._slots: deque = deque(maxlen=capacity)

    # ── Mutation ───────────────────────────────────────────────────────────────

    def update(
        self,
        obs:        np.ndarray,
        action:     int,
        confidence: float,
        pnl:        Optional[float] = None,
    ):
        """Add a new trade memory slot."""
        tags: List[str] = []
        if pnl is not None:
            tags.append("win" if pnl > 0 else "loss")
        tags.append({0: "hold", 1: "buy", 2: "sell"}.get(action, "unknown"))

        slot = MemorySlot(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            obs=obs.copy(),
            action=action,
            confidence=confidence,
            pnl=pnl,
            pattern_tags=tags,
        )
        self._slots.append(slot)
        logger.debug(
            "WorkingMemory: +slot action=%d conf=%.2f pnl=%s (size=%d)",
            action, confidence, pnl, len(self._slots),
        )

    # ── Analysis ───────────────────────────────────────────────────────────────

    def get_recent_pattern(self) -> Dict:
        """
        Analyse the last `capacity` trades.

        Returns
        -------
        dict with keys:
          dominant_action   : int  — most common action (0/1/2)
          action_consistency: float — fraction of trades that matched dominant action
          recent_win_rate   : float — fraction of finished trades (pnl not None) that won
          momentum          : float — net pnl momentum over last 5 finished trades
        """
        if not self._slots:
            return {
                "dominant_action":    0,
                "action_consistency": 0.0,
                "recent_win_rate":    0.5,
                "momentum":           0.0,
            }

        slots = list(self._slots)

        # Dominant action
        action_counts = {0: 0, 1: 0, 2: 0}
        for s in slots:
            action_counts[s.action] = action_counts.get(s.action, 0) + 1
        dominant = max(action_counts, key=action_counts.get)
        consistency = action_counts[dominant] / len(slots)

        # Win rate (only slots with pnl)
        finished = [s for s in slots if s.pnl is not None]
        if finished:
            wins     = sum(1 for s in finished if s.pnl > 0)
            win_rate = wins / len(finished)
        else:
            win_rate = 0.5

        # Momentum: sum of pnl in last 5 finished trades, normalised to [-1, 1]
        recent_finished = [s for s in finished[-5:] if s.pnl is not None]
        if recent_finished:
            pnls = [s.pnl for s in recent_finished]
            momentum = float(np.clip(sum(pnls) / (abs(sum(pnls)) + 1e-9), -1.0, 1.0))
        else:
            momentum = 0.0

        return {
            "dominant_action":    int(dominant),
            "action_consistency": round(consistency, 3),
            "recent_win_rate":    round(win_rate, 3),
            "momentum":           round(momentum, 3),
        }

    def get_active_attention(self, obs: np.ndarray) -> float:
        """
        Cosine similarity between `obs` and the mean of the 5 most recent obs.
        Returns a float in [0, 1] — higher means current market looks familiar.
        """
        recent = [s.obs for s in list(self._slots)[-5:]]
        if not recent:
            return 0.0

        stacked = np.stack(recent, axis=0)   # (N, obs_size)
        mean_obs = stacked.mean(axis=0)

        norm_a = np.linalg.norm(obs)
        norm_b = np.linalg.norm(mean_obs)
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0

        cosine = float(np.dot(obs, mean_obs) / (norm_a * norm_b))
        # Map from [-1,1] → [0,1]
        return round((cosine + 1.0) / 2.0, 3)

    def detect_losing_streak(self) -> int:
        """
        Count consecutive losses at the tail of the memory (most recent first).
        """
        streak = 0
        for slot in reversed(list(self._slots)):
            if slot.pnl is None:
                continue        # skip trades without outcome yet
            if slot.pnl < 0:
                streak += 1
            else:
                break
        return streak

    def should_pause(self) -> bool:
        """
        Return True if the AI should stop trading temporarily to cut drawdown.
        Conditions:
          - losing streak >= 3  OR
          - recent win rate < 0.30
        """
        streak = self.detect_losing_streak()
        if streak >= 3:
            logger.warning("WorkingMemory: losing streak=%d → PAUSE", streak)
            return True

        pattern = self.get_recent_pattern()
        if pattern["recent_win_rate"] < 0.30:
            logger.warning(
                "WorkingMemory: win_rate=%.2f < 0.30 → PAUSE",
                pattern["recent_win_rate"],
            )
            return True

        return False

    # ── Helpers ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._slots)
