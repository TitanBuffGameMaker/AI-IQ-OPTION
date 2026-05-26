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

    # Recovery thresholds for REAL-account pause
    RECOVERY_COOLDOWN_SEC = 300       # 5-min cool-off after pause triggered
    RECOVERY_STRONG_NEEDED = 3        # need this many high-conf signals observed
    RECOVERY_STRONG_THRESHOLD = 0.55  # what counts as a "strong" signal

    def __init__(self, capacity: int = 20, account_type: str = "PRACTICE"):
        self.capacity = capacity
        self.account_type = (account_type or "PRACTICE").upper()
        self._slots: deque = deque(maxlen=capacity)

        # Recovery / pause state (used only for REAL accounts)
        self._pause_started_at: Optional[float]  = None
        self._pause_reason:     str              = ""
        self._strong_seen:      int              = 0
        self._observations:     int              = 0

    def set_account_type(self, account_type: str) -> None:
        """Switch account type at runtime (PRACTICE ↔ REAL)."""
        new = (account_type or "PRACTICE").upper()
        if new != self.account_type:
            logger.info("WorkingMemory: account_type %s → %s", self.account_type, new)
            self.account_type = new
            # Moving to PRACTICE clears any active pause
            if new == "PRACTICE" and self._pause_started_at is not None:
                self._clear_pause("switched to PRACTICE")

    def record_observation(self, signal_confidence: float) -> None:
        """
        Called every brain.think() cycle. While in REAL-account pause, tracks
        strong-signal observations so we can decide when the AI has 'understood'
        the market again and is ready to resume.
        """
        if self._pause_started_at is None:
            return
        self._observations += 1
        if signal_confidence >= self.RECOVERY_STRONG_THRESHOLD:
            self._strong_seen += 1

    def get_pause_status(self) -> Dict:
        """For UI: snapshot of recovery state."""
        if self._pause_started_at is None:
            return {"paused": False, "account_type": self.account_type}
        elapsed = time.time() - self._pause_started_at
        return {
            "paused":             True,
            "account_type":       self.account_type,
            "reason":             self._pause_reason,
            "elapsed_seconds":    int(elapsed),
            "cooldown_remaining": max(0, self.RECOVERY_COOLDOWN_SEC - int(elapsed)),
            "strong_seen":        self._strong_seen,
            "strong_needed":      self.RECOVERY_STRONG_NEEDED,
            "observations":       self._observations,
        }

    def _clear_pause(self, why: str = "") -> None:
        if self._pause_started_at is None:
            return
        elapsed = time.time() - self._pause_started_at
        logger.info("WorkingMemory: PAUSE cleared after %.0fs (%d strong signals) — %s",
                    elapsed, self._strong_seen, why)
        self._pause_started_at = None
        self._pause_reason     = ""
        self._strong_seen      = 0
        self._observations     = 0

    def _recovery_ready(self) -> bool:
        if self._pause_started_at is None:
            return True
        elapsed = time.time() - self._pause_started_at
        if elapsed < self.RECOVERY_COOLDOWN_SEC:
            return False
        if self._strong_seen < self.RECOVERY_STRONG_NEEDED:
            return False
        return True

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
        Decide whether to stop trading.

        PRACTICE account → NEVER pause.  Practice is the training ground:
        every loss is a learning signal, so the brain must keep trading to
        improve.

        REAL account → pause to protect the balance when things go wrong,
        and only resume once the brain has 'understood' the situation:
          1. Cool-off period (5 min) has elapsed AND
          2. We've observed RECOVERY_STRONG_NEEDED high-confidence signals
             during the pause (showing market clarity has returned).

        Trigger conditions (REAL only):
          - losing streak >= 3  OR
          - recent win rate < 0.30  (requires at least 5 confirmed results)
        """
        # ── PRACTICE: never pause; clear stale state if account switched ───
        if self.account_type == "PRACTICE":
            if self._pause_started_at is not None:
                self._clear_pause("PRACTICE never pauses")
            return False

        # ── Already in pause: check whether recovery conditions are met ────
        if self._pause_started_at is not None:
            if self._recovery_ready():
                self._clear_pause("recovery conditions met — RESUMING")
                return False
            return True

        # ── Evaluate trigger conditions ────────────────────────────────────
        streak = self.detect_losing_streak()
        if streak >= 3:
            self._pause_started_at = time.time()
            self._pause_reason     = f"losing streak={streak}"
            self._strong_seen      = 0
            self._observations     = 0
            logger.warning("WorkingMemory: PAUSE (REAL) — %s. Recovering…", self._pause_reason)
            return True

        finished = [s for s in list(self._slots) if s.pnl is not None]
        if len(finished) >= 5:
            pattern = self.get_recent_pattern()
            if pattern["recent_win_rate"] < 0.30:
                self._pause_started_at = time.time()
                self._pause_reason     = f"win_rate={pattern['recent_win_rate']:.2f} (n={len(finished)})"
                self._strong_seen      = 0
                self._observations     = 0
                logger.warning("WorkingMemory: PAUSE (REAL) — %s. Recovering…", self._pause_reason)
                return True

        return False

    # ── Helpers ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._slots)
