"""
Knowledge persistence manager.

Keeps track of where to save/load the model, manages checkpoint versioning,
and records performance history so you can see the agent improving over time.
"""
import json
import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    Manages all persistent state for the trading AI.

    Directory layout:
        knowledge/
          ├── brain.pt          ← latest model weights (the "brain")
          ├── brain_best.pt     ← best-performing checkpoint
          ├── checkpoints/
          │     ├── ckpt_0001.pt
          │     └── ...
          └── history.json      ← trade/episode performance log
    """

    BRAIN_FILE = "brain.pt"
    BRAIN_BEST_FILE = "brain_best.pt"
    HISTORY_FILE = "history.json"
    CKPT_DIR = "checkpoints"

    def __init__(self, base_dir: str = "./knowledge"):
        self.base_dir = base_dir
        self.ckpt_dir = os.path.join(base_dir, self.CKPT_DIR)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self._history: list = self._load_history()
        self._best_reward: float = float(self._get_best_reward())

    # ── Path helpers ─────────────────────────────────────────────────────────

    @property
    def brain_path(self) -> str:
        return os.path.join(self.base_dir, self.BRAIN_FILE)

    @property
    def best_brain_path(self) -> str:
        return os.path.join(self.base_dir, self.BRAIN_BEST_FILE)

    def ckpt_path(self, episode: int) -> str:
        return os.path.join(self.ckpt_dir, f"ckpt_{episode:05d}.pt")

    # ── Save / load wrappers ─────────────────────────────────────────────────

    def save_brain(self, agent) -> str:
        """Save the agent's current weights as the 'latest brain'."""
        agent.save(self.brain_path)
        return self.brain_path

    def save_checkpoint(self, agent, episode: int) -> str:
        """Save a versioned checkpoint."""
        path = self.ckpt_path(episode)
        agent.save(path)
        return path

    def save_best(self, agent) -> str:
        """Save as the best-performing brain if this episode is best."""
        agent.save(self.best_brain_path)
        logger.info("New best brain saved → %s", self.best_brain_path)
        return self.best_brain_path

    def load_brain(self, agent) -> bool:
        """Load the latest brain into the agent. Returns True on success."""
        return agent.load(self.brain_path)

    # ── Performance history ──────────────────────────────────────────────────

    def record_episode(
        self,
        episode: int,
        total_steps: int,
        episode_pnl: float,
        n_trades: int,
        win_rate: float,
        policy_loss: float,
        value_loss: float,
        entropy: float,
    ):
        """Append episode statistics to the history log."""
        entry = {
            "episode": episode,
            "timestamp": datetime.utcnow().isoformat(),
            "total_steps": total_steps,
            "episode_pnl": round(episode_pnl, 4),
            "n_trades": n_trades,
            "win_rate": round(win_rate, 4),
            "policy_loss": round(policy_loss, 6),
            "value_loss": round(value_loss, 6),
            "entropy": round(entropy, 6),
        }
        self._history.append(entry)
        self._save_history()

        # Check if this is the best episode
        if episode_pnl > self._best_reward:
            self._best_reward = episode_pnl
            return True  # caller should save_best()
        return False

    def get_summary(self) -> dict:
        """Return a summary of all learning progress."""
        if not self._history:
            return {"episodes": 0, "total_pnl": 0.0, "best_episode_pnl": 0.0}
        total_pnl = sum(e["episode_pnl"] for e in self._history)
        best_pnl = max(e["episode_pnl"] for e in self._history)
        recent = self._history[-10:]
        recent_win_rate = (
            sum(e["win_rate"] for e in recent) / len(recent) if recent else 0.0
        )
        return {
            "episodes": len(self._history),
            "total_pnl": round(total_pnl, 2),
            "best_episode_pnl": round(best_pnl, 2),
            "recent_win_rate_10ep": round(recent_win_rate, 3),
            "total_steps": self._history[-1]["total_steps"] if self._history else 0,
        }

    def print_summary(self):
        s = self.get_summary()
        logger.info(
            "── Knowledge Base ──────────────────────────────\n"
            "  Episodes completed : %d\n"
            "  Total P&L          : $%.2f\n"
            "  Best episode P&L   : $%.2f\n"
            "  Recent win rate    : %.1f%%\n"
            "  Total steps        : %d\n"
            "────────────────────────────────────────────────",
            s["episodes"], s["total_pnl"], s["best_episode_pnl"],
            s["recent_win_rate_10ep"] * 100, s["total_steps"],
        )

    # ── Private ──────────────────────────────────────────────────────────────

    def _load_history(self) -> list:
        path = os.path.join(self.base_dir, self.HISTORY_FILE)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save_history(self):
        path = os.path.join(self.base_dir, self.HISTORY_FILE)
        with open(path, "w") as f:
            json.dump(self._history, f, indent=2)

    def _get_best_reward(self) -> float:
        if not self._history:
            return float("-inf")
        return max(e["episode_pnl"] for e in self._history)
