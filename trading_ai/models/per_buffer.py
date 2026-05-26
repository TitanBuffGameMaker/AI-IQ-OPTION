"""
Prioritized Experience Replay (PER)
════════════════════════════════════

ทำไม PER ถึงสำคัญ?

สมองมนุษย์ไม่ได้ replay ทุกประสบการณ์เท่าๆ กัน
→ ประสบการณ์ที่ "น่าประหลาดใจ" หรือ "สำคัญ" ถูก replay บ่อยกว่า
→ ตรงกับ Dopamine RPE theory: surprise → stronger memory formation

AI equivalent:
  Priority = |reward| × rpe_multiplier
  High priority trades → sampled more often → learned more thoroughly

ผลลัพธ์:
  - เรียนรู้จาก trade ที่ชนะ/แพ้มากๆ เร็วกว่า
  - ไม่เสียเวลา replay trade ที่ไม่น่าสนใจ
  - Sample efficiency สูงแม้ trade เยอะ
"""
import numpy as np
from typing import Optional


class PrioritizedReplayBuffer:
    """
    Proportional prioritization PER.
    Priority ∝ |reward| × rpe_multiplier + epsilon
    """

    ALPHA   = 0.6   # priority exponent (0=uniform, 1=full priority)
    BETA    = 0.4   # importance sampling correction (anneals toward 1)
    EPSILON = 0.01  # small constant to prevent zero priority

    def __init__(self, capacity: int = 2000, obs_size: int = 296):
        self.capacity = capacity
        self.obs_size = obs_size
        self._ptr     = 0
        self._size    = 0

        self._obs      = np.zeros((capacity, obs_size), dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self._actions  = np.zeros(capacity, dtype=np.int32)
        self._rewards  = np.zeros(capacity, dtype=np.float32)
        self._priorities= np.ones(capacity,  dtype=np.float32)

        self._max_priority = 1.0
        self._total_added  = 0
        self._total_sampled = 0

    def add(self, obs: np.ndarray, next_obs: np.ndarray,
            action: int, reward: float, rpe_mult: float = 1.0) -> None:
        priority = (abs(reward) * rpe_mult + self.EPSILON) ** self.ALPHA
        priority = min(priority, 10.0)   # cap to avoid runaway priorities

        idx = self._ptr % self.capacity
        self._obs[idx]       = obs[:self.obs_size]
        self._next_obs[idx]  = next_obs[:self.obs_size]
        self._actions[idx]   = action
        self._rewards[idx]   = reward
        self._priorities[idx]= priority
        self._max_priority   = max(self._max_priority, priority)

        self._ptr  += 1
        self._size  = min(self._size + 1, self.capacity)
        self._total_added += 1

    def sample(self, batch_size: int) -> Optional[dict]:
        if self._size < batch_size:
            return None

        probs = self._priorities[:self._size] / self._priorities[:self._size].sum()
        idxs  = np.random.choice(self._size, size=batch_size, replace=False, p=probs)

        # Importance sampling weights (correct for sampling bias)
        weights = (self._size * probs[idxs]) ** (-self.BETA)
        weights /= weights.max()

        self._total_sampled += batch_size
        return {
            "obs":     self._obs[idxs],
            "next_obs":self._next_obs[idxs],
            "actions": self._actions[idxs],
            "rewards": self._rewards[idxs],
            "weights": weights.astype(np.float32),
            "idxs":    idxs,
        }

    def update_priorities(self, idxs: np.ndarray, td_errors: np.ndarray) -> None:
        for i, err in zip(idxs, td_errors):
            self._priorities[i] = (abs(err) + self.EPSILON) ** self.ALPHA

    def stats(self) -> dict:
        if self._size == 0:
            return {"size": 0, "capacity": self.capacity, "fill_pct": 0}
        p = self._priorities[:self._size]
        return {
            "size":          self._size,
            "capacity":      self.capacity,
            "fill_pct":      round(self._size / self.capacity * 100, 1),
            "max_priority":  round(float(self._max_priority), 3),
            "avg_priority":  round(float(p.mean()), 3),
            "total_added":   self._total_added,
            "total_sampled": self._total_sampled,
        }
