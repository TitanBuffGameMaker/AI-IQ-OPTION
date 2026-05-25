"""
FastLearner — Online Actor-Critic (A3C-style)

Update every single trade immediately (no replay buffer).
Network: small 2-layer MLP with actor + critic heads.
"""
import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

logger = logging.getLogger(__name__)


class _FastNet(nn.Module):
    """Small Actor-Critic network: 296 → 128 → 64 → (actor=3, critic=1)"""

    def __init__(self, obs_size: int, n_actions: int = 3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.actor_head  = nn.Linear(64, n_actions)
        self.critic_head = nn.Linear(64, 1)

        # Initialise weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, x: torch.Tensor):
        h      = self.shared(x)
        logits = self.actor_head(h)
        value  = self.critic_head(h).squeeze(-1)
        return logits, value


class FastLearner:
    """
    Online A3C-style agent.
    Calls update_online() after every trade — no buffer accumulation.
    """

    DISCOUNT = 0.99

    def __init__(self, obs_size: int = 296, n_actions: int = 3, lr: float = 1e-3):
        self.obs_size  = obs_size
        self.n_actions = n_actions
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network   = _FastNet(obs_size, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

        self._total_updates = 0
        logger.info("FastLearner online (obs=%d, lr=%.4f, device=%s)", obs_size, lr, self.device)

    # ── Inference ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> Tuple[int, float]:
        """Return (action, confidence) where confidence is the selected action's probability."""
        obs_t  = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        logits, _ = self.network(obs_t)
        probs  = torch.softmax(logits, dim=-1).squeeze(0)
        action = probs.argmax().item()
        return int(action), float(probs[action].item())

    # ── Online update ──────────────────────────────────────────────────────────

    def update_online(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> Dict[str, float]:
        """
        1-step TD Actor-Critic update.
        Called immediately after every trade outcome (no buffer needed).
        Returns a metrics dict.
        """
        self.network.train()

        obs_t      = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        next_obs_t = torch.from_numpy(next_obs.astype(np.float32)).unsqueeze(0).to(self.device)
        action_t   = torch.tensor([action], dtype=torch.long, device=self.device)
        reward_t   = torch.tensor([reward], dtype=torch.float32, device=self.device)

        logits, value = self.network(obs_t)

        with torch.no_grad():
            _, next_value = self.network(next_obs_t)
            target = reward_t + (0.0 if done else self.DISCOUNT * next_value)

        # TD error (advantage)
        advantage = (target - value).detach()

        # Critic loss
        critic_loss = nn.functional.mse_loss(value, target)

        # Actor loss (policy gradient)
        dist        = Categorical(logits=logits)
        log_prob    = dist.log_prob(action_t)
        actor_loss  = -(log_prob * advantage).mean()

        # Entropy bonus (encourage exploration)
        entropy     = dist.entropy().mean()
        entropy_coef = 0.01

        loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

        self._total_updates += 1
        metrics = {
            "actor_loss":  round(actor_loss.item(), 5),
            "critic_loss": round(critic_loss.item(), 5),
            "entropy":     round(entropy.item(), 5),
            "advantage":   round(advantage.item(), 5),
            "total_updates": self._total_updates,
        }
        logger.debug("FastLearner #%d | %s", self._total_updates, metrics)
        return metrics

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "total_updates": self._total_updates,
        }, path)
        logger.info("FastLearner saved → %s", path)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            logger.info("No FastLearner checkpoint at %s", path)
            return False
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.network.load_state_dict(ckpt["network_state"])
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self._total_updates = ckpt.get("total_updates", 0)
            logger.info("FastLearner loaded ← %s  (updates=%d)", path, self._total_updates)
            return True
        except Exception as exc:
            logger.error("FastLearner load failed: %s", exc)
            return False
