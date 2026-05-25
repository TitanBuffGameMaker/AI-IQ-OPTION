"""
Short-term memory – a ring buffer of recent market observations.

Like the brain's working memory: remembers the last N things it saw,
uses them to detect patterns over time before they become long-term knowledge.
"""
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional

import numpy as np


@dataclass
class Observation:
    timestamp: str
    indicator_vec: np.ndarray      # the indicator feature vector
    action_taken: int              # 0=hold, 1=buy, 2=sell
    confidence: float
    reward: Optional[float] = None # filled in after trade closes
    asset: str = "EURUSD"
    tags: List[str] = field(default_factory=list)


class ShortTermMemory:
    """
    Circular buffer holding recent observations.
    Used by the pattern detector to find recurring market states.
    """

    def __init__(self, capacity: int = 200):
        self.capacity = capacity
        self._buffer: Deque[Observation] = deque(maxlen=capacity)

    def record(self, obs: Observation):
        self._buffer.append(obs)

    def update_last_reward(self, reward: float):
        """Call after a trade closes to attach the actual outcome."""
        if self._buffer:
            self._buffer[-1].reward = reward

    def recent(self, n: int = 10) -> List[Observation]:
        return list(self._buffer)[-n:]

    def profitable_observations(self, min_reward: float = 0.0) -> List[Observation]:
        return [o for o in self._buffer if o.reward is not None and o.reward > min_reward]

    def losing_observations(self) -> List[Observation]:
        return [o for o in self._buffer if o.reward is not None and o.reward < 0]

    def win_rate(self) -> float:
        closed = [o for o in self._buffer if o.reward is not None]
        if not closed:
            return 0.0
        wins = sum(1 for o in closed if o.reward > 0)
        return wins / len(closed)

    def average_reward(self) -> float:
        closed = [o for o in self._buffer if o.reward is not None]
        if not closed:
            return 0.0
        return float(np.mean([o.reward for o in closed]))

    def detect_streak(self) -> Dict[str, int]:
        """How many consecutive wins or losses at the end of the buffer."""
        recent = self.recent(20)
        closed = [o for o in recent if o.reward is not None]
        if not closed:
            return {"wins": 0, "losses": 0}
        win_streak = 0
        loss_streak = 0
        for o in reversed(closed):
            if o.reward > 0:
                if loss_streak > 0:
                    break
                win_streak += 1
            else:
                if win_streak > 0:
                    break
                loss_streak += 1
        return {"wins": win_streak, "losses": loss_streak}

    def __len__(self) -> int:
        return len(self._buffer)
