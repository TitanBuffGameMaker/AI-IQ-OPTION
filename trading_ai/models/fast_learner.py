"""
FastLearner v2 — Online Actor-Critic with Priority Experience Replay

Improvements over v1:
  ▸ PriorityBuffer (200 entries): stores experiences weighted by TD-error
    magnitude so surprising outcomes are revisited more often.
  ▸ Mini-batch replay: every 10 online updates, one mini-batch (32 samples)
    is replayed from the priority buffer.  This dramatically improves
    sample efficiency in sparse-reward environments like binary options.
  ▸ N-step (3) return estimate instead of 1-step TD, giving the critic a
    longer horizon and reducing variance.
  ▸ Priority weight IS-correction (beta=0.4) to prevent bias introduced by
    non-uniform sampling.
  ▸ Improved entropy bonus: 0.02 (up from 0.01) to keep exploration alive
    longer given the low sample count per session.
"""
import logging
import os
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

logger = logging.getLogger(__name__)

# ── Priority Experience Replay Buffer ─────────────────────────────────────────

class PriorityBuffer:
    """
    Circular buffer storing (obs, action, reward, next_obs) weighted by
    |TD error|^alpha.  Supports IS-corrected sampling.
    """

    def __init__(self, capacity: int = 200, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha    = alpha
        self.beta     = beta
        self._buf:        List            = []
        self._priorities: List[float]     = []
        self._pos:        int             = 0

    def push(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        td_error: float,
    ) -> None:
        priority = (abs(td_error) + 1e-6) ** self.alpha
        entry    = (obs.copy(), action, reward, next_obs.copy())
        if len(self._buf) < self.capacity:
            self._buf.append(entry)
            self._priorities.append(priority)
        else:
            self._buf[self._pos]        = entry
            self._priorities[self._pos] = priority
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, n: int) -> Tuple[List, np.ndarray]:
        """Returns (experiences, importance_weights)."""
        n   = min(n, len(self._buf))
        pri = np.array(self._priorities[: len(self._buf)], dtype=np.float64)
        prob = pri / pri.sum()
        indices = np.random.choice(len(self._buf), size=n, replace=False, p=prob)
        experiences = [self._buf[i] for i in indices]

        # IS weights (correct bias from non-uniform sampling)
        weights = (len(self._buf) * prob[indices]) ** (-self.beta)
        weights /= weights.max()
        return experiences, weights.astype(np.float32)

    def update_priority(self, idx: int, td_error: float) -> None:
        if idx < len(self._priorities):
            self._priorities[idx] = (abs(td_error) + 1e-6) ** self.alpha

    def __len__(self) -> int:
        return len(self._buf)


# ── Network ───────────────────────────────────────────────────────────────────

class _FastNet(nn.Module):
    """
    Improved Actor-Critic: 296 → 256 → 128 → 64 → (actor=3, critic=1).
    Added LayerNorm for training stability and GELU activation.
    """

    def __init__(self, obs_size: int, n_actions: int = 3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.actor_head  = nn.Linear(64, n_actions)
        self.critic_head = nn.Linear(64, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, x: torch.Tensor):
        h      = self.shared(x)
        logits = self.actor_head(h)
        value  = self.critic_head(h).squeeze(-1)
        return logits, value


# ── FastLearner ───────────────────────────────────────────────────────────────

class FastLearner:
    """
    Online A3C-style agent with Priority Experience Replay.
    Calls update_online() after every trade — no large buffer needed.
    Every 10 updates, a mini-batch replay improves sample efficiency.
    """

    DISCOUNT        = 0.99
    N_STEP          = 3       # N-step return horizon
    ENTROPY_COEF    = 0.02    # higher exploration than v1
    MINI_BATCH_SIZE = 32
    MINI_BATCH_FREQ = 10      # mini-batch update every N online steps

    def __init__(self, obs_size: int = 296, n_actions: int = 3, lr: float = 8e-4):
        self.obs_size  = obs_size
        self.n_actions = n_actions
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network   = _FastNet(obs_size, n_actions).to(self.device)
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=lr, eps=1e-5, weight_decay=1e-5
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, eta_min=1e-5
        )

        self._per_buffer    = PriorityBuffer(capacity=200)
        self._n_step_buf:   deque = deque(maxlen=self.N_STEP)
        self._total_updates = 0

        logger.info(
            "FastLearner v2 (obs=%d, lr=%.4f, device=%s, PER+%d-step)",
            obs_size, lr, self.device, self.N_STEP,
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> Tuple[int, float]:
        obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        logits, _ = self.network(obs_t)
        probs  = torch.softmax(logits, dim=-1).squeeze(0)
        action = probs.argmax().item()
        return int(action), float(probs[action].item())

    # ── Online update ─────────────────────────────────────────────────────────

    def update_online(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> Dict[str, float]:
        """
        N-step TD Actor-Critic update + push to PER buffer.
        Accumulates N steps before training; uses discounted N-step return
        as target, bootstrapping with V(s_{t+N}) when episode continues.
        Triggers a mini-batch replay every MINI_BATCH_FREQ calls.
        """
        self._n_step_buf.append((obs.copy(), action, float(reward), next_obs.copy(), done))

        # Wait until we have N steps or episode ends
        if len(self._n_step_buf) < self.N_STEP and not done:
            return {}

        # Compute N-step discounted return G from the oldest buffered step
        obs_0, act_0, _, _, _ = self._n_step_buf[0]
        G = 0.0
        gamma_k = 1.0
        last_next_obs, last_done = next_obs, done
        for (_, _, r_k, nxt_k, d_k) in self._n_step_buf:
            G += gamma_k * r_k
            gamma_k *= self.DISCOUNT
            last_next_obs, last_done = nxt_k, d_k
            if d_k:
                break

        self.network.train()
        obs_t      = torch.from_numpy(obs_0.astype(np.float32)).unsqueeze(0).to(self.device)
        next_obs_t = torch.from_numpy(last_next_obs.astype(np.float32)).unsqueeze(0).to(self.device)
        action_t   = torch.tensor([act_0], dtype=torch.long, device=self.device)
        reward_t   = torch.tensor([G], dtype=torch.float32, device=self.device)

        logits, value = self.network(obs_t)

        with torch.no_grad():
            _, next_value = self.network(next_obs_t)
            bootstrap = 0.0 if last_done else (gamma_k * next_value)
            target = reward_t + bootstrap

        advantage  = (target - value).detach()
        td_error   = advantage.abs().item()

        critic_loss = nn.functional.mse_loss(value, target)
        dist        = Categorical(logits=logits)
        log_prob    = dist.log_prob(action_t)
        actor_loss  = -(log_prob * advantage).mean()
        entropy     = dist.entropy().mean()
        loss        = actor_loss + 0.5 * critic_loss - self.ENTROPY_COEF * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

        # Push N-step base experience to PER buffer
        self._per_buffer.push(obs_0, act_0, G, last_next_obs, td_error)

        self._total_updates += 1

        # Mini-batch replay from PER buffer
        replay_metrics: Dict[str, float] = {}
        if (
            self._total_updates % self.MINI_BATCH_FREQ == 0
            and len(self._per_buffer) >= self.MINI_BATCH_SIZE
        ):
            replay_metrics = self._mini_batch_update()

        self.scheduler.step()

        metrics = {
            "actor_loss":    round(actor_loss.item(), 5),
            "critic_loss":   round(critic_loss.item(), 5),
            "entropy":       round(entropy.item(), 5),
            "advantage":     round(advantage.item(), 5),
            "td_error":      round(td_error, 5),
            "total_updates": self._total_updates,
            **replay_metrics,
        }
        logger.debug("FastLearner v2 #%d | %s", self._total_updates, metrics)
        return metrics

    # ── Mini-batch replay ─────────────────────────────────────────────────────

    def _mini_batch_update(self) -> Dict[str, float]:
        """Sample from PER buffer and do one supervised mini-batch update."""
        self.network.train()
        experiences, is_weights = self._per_buffer.sample(self.MINI_BATCH_SIZE)

        obs_batch      = np.array([e[0] for e in experiences], dtype=np.float32)
        action_batch   = np.array([e[1] for e in experiences], dtype=np.int64)
        reward_batch   = np.array([e[2] for e in experiences], dtype=np.float32)
        next_obs_batch = np.array([e[3] for e in experiences], dtype=np.float32)

        obs_t      = torch.from_numpy(obs_batch).to(self.device)
        act_t      = torch.from_numpy(action_batch).to(self.device)
        rew_t      = torch.from_numpy(reward_batch).to(self.device)
        next_obs_t = torch.from_numpy(next_obs_batch).to(self.device)
        is_t       = torch.from_numpy(is_weights).to(self.device)

        logits, values = self.network(obs_t)

        with torch.no_grad():
            _, next_values = self.network(next_obs_t)
            targets = rew_t + self.DISCOUNT * next_values

        advantages = (targets - values).detach()

        # IS-weighted critic loss
        critic_loss = (is_t * nn.functional.mse_loss(values, targets, reduction="none")).mean()

        dist        = Categorical(logits=logits)
        log_probs   = dist.log_prob(act_t)
        actor_loss  = -(is_t * log_probs * advantages).mean()
        entropy     = dist.entropy().mean()

        loss = actor_loss + 0.5 * critic_loss - self.ENTROPY_COEF * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

        return {
            "replay_actor":  round(actor_loss.item(), 5),
            "replay_critic": round(critic_loss.item(), 5),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "network_state":   self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "total_updates":   self._total_updates,
        }, path)
        logger.info("FastLearner v2 saved → %s", path)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            logger.info("No FastLearner checkpoint at %s", path)
            return False
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.network.load_state_dict(ckpt["network_state"])
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self._total_updates = ckpt.get("total_updates", 0)
            logger.info("FastLearner v2 loaded ← %s  (updates=%d)",
                        path, self._total_updates)
            return True
        except Exception as exc:
            logger.error("FastLearner v2 load failed: %s", exc)
            return False
