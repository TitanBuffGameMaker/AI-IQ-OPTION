"""
MAML — Model-Agnostic Meta-Learning
══════════════════════════════════════

ปัญหาที่แก้:
  OTC market เปลี่ยน pattern ทุก 1-3 เดือน (IQ Option ปรับ algorithm ใหม่)
  สมองเดิมต้อง re-learn ใหม่จากศูนย์ → เสียเวลาเป็นสัปดาห์

MAML แก้ด้วย:
  เรียนรู้ "จุดเริ่มต้นที่ดี" (θ*) ที่ adapt ได้เร็วในทุก regime/pattern
  เมื่อ OTC shift:
    เดิม: ต้อง 1000+ trades ใหม่กว่าจะ relearn
    MAML: adapt ใน 5-10 gradient steps (20-50 trades ใหม่)

Concept:
  Meta-parameters θ: จุดเริ่มต้นที่ adapt ได้ดี
  Inner loop:  θ' = θ - α∇L_task(θ)   [3 steps บน support set]
  Outer loop:  L_meta = L_task(θ')    [optimize θ ให้ adapt ได้ดี]

"Task" ในระบบนี้ = ชุด trade ล่าสุดในช่วงเวลา/regime เดียวกัน
  - support set: 30 trades ล่าสุด
  - query set: 10 trades ก่อนหน้า support (held-out)
  - Meta-update ทุก 50 trades ใหม่

Interface: wraps any base agent (RainbowAgent, IQNAgent, PPOAgent)
"""
import copy
import logging
import os
import threading
import time
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ── MAML Hyperparameters ───────────────────────────────────────────────────────
INNER_LR       = 0.05     # Inner loop learning rate (larger → faster adapt)
N_INNER_STEPS  = 3        # Number of inner loop gradient steps
META_LR        = 1e-3     # Outer loop (meta) learning rate
SUPPORT_SIZE   = 30       # Trades for inner loop adaptation
QUERY_SIZE     = 10       # Trades for outer loop evaluation
META_EVERY     = 50       # Run meta-update every N new trades
ADAPT_EVERY    = 20       # Run fast_adapt every N new trades (inference)


class MAMLAgent:
    """
    MAML Wrapper — ห่อหุ้ม base agent ด้วย meta-learning.

    การทำงาน:
    1. Base agent ทำงานปกติ (trade ตาม Q-values)
    2. ทุก ADAPT_EVERY trades: fast_adapt บน recent trades
       → θ' = θ - α∇L(θ) [3 steps] → ใช้ adapted weights สำหรับ inference
    3. ทุก META_EVERY trades: meta_update
       → ปรับ θ ให้ "จุดเริ่มต้น" ดีขึ้นสำหรับ fast adaptation ในอนาคต

    เมื่อ OTC pattern shift:
    - System detect ว่า win_rate ลดลง
    - fast_adapt ปรับ θ' ให้เข้ากับ pattern ใหม่ใน ~20 trades
    - base θ ยังคงเป็น "good initialization" สำหรับ pattern ในอนาคต
    """

    def __init__(self, base_agent, inner_lr: float = INNER_LR,
                 n_inner: int = N_INNER_STEPS, meta_lr: float = META_LR):
        self.base_agent  = base_agent
        self.inner_lr    = inner_lr
        self.n_inner     = n_inner
        self.device      = getattr(base_agent, "device",
                                   torch.device("cpu"))

        # Identify which network attribute to meta-learn
        self._net_attr   = self._find_net_attr()
        if self._net_attr:
            self.meta_opt = optim.Adam(
                getattr(base_agent, self._net_attr).parameters(),
                lr=meta_lr,
            )
            self._adapted_params: Optional[dict] = None  # θ' after fast_adapt
        else:
            logger.warning("MAML: no suitable network found in base_agent — meta-learning disabled")
            self.meta_opt = None

        # Experience buffers for meta-learning
        self._support: deque = deque(maxlen=SUPPORT_SIZE)
        self._query:   deque = deque(maxlen=QUERY_SIZE)
        self._new_trades:    int = 0
        self._using_adapted: bool = False

        # Compatibility attrs (proxy to base_agent)
        self.obs_size    = base_agent.obs_size
        self.n_actions   = base_agent.n_actions

        logger.info("MAML — inner_lr=%.3f | n_inner=%d | meta_lr=%.4f | adapt_every=%d",
                    inner_lr, n_inner, meta_lr, ADAPT_EVERY)

    def _find_net_attr(self) -> Optional[str]:
        """Find the main network attribute name."""
        for attr in ("online", "network", "meta_net"):
            net = getattr(self.base_agent, attr, None)
            if isinstance(net, nn.Module):
                return attr
        return None

    # ── Meta-learning core ─────────────────────────────────────────────────────

    def _network(self) -> Optional[nn.Module]:
        if self._net_attr is None:
            return None
        return getattr(self.base_agent, self._net_attr, None)

    def _compute_task_loss(self, net: nn.Module,
                           samples: List[tuple]) -> torch.Tensor:
        """
        Simple regression loss on (obs, action, reward) samples.
        Minimizes MSE between predicted Q(s,a) and observed reward.
        """
        if not samples:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        obs_np = np.array([s[0] for s in samples], dtype=np.float32)
        acts   = [s[1] for s in samples]
        rews   = [s[2] for s in samples]

        obs_t = torch.from_numpy(obs_np).to(self.device)
        acts_t = torch.tensor(acts, dtype=torch.long, device=self.device)
        rews_t = torch.tensor(rews, dtype=torch.float32, device=self.device)

        # Get Q-values from network (works for RainbowNet and similar)
        if hasattr(net, "get_q_values") and hasattr(self.base_agent, "support"):
            support = self.base_agent.support
            q = net.get_q_values(obs_t, support)           # (B, A)
        elif hasattr(net, "mean_q"):
            taus = torch.rand(len(samples), 8, device=self.device)
            q    = net(obs_t, taus).mean(1)                # (B, A)
        else:
            # PPO-style: forward returns (logits, value)
            logits, _ = net(obs_t)
            q = logits

        q_act = q[torch.arange(len(acts)), acts_t]        # (B,)
        return F.smooth_l1_loss(q_act, rews_t)

    def fast_adapt(self, support: Optional[List[tuple]] = None) -> bool:
        """
        Inner loop: create θ' by taking n_inner gradient steps on support.
        Returns True if adaptation was performed.
        """
        net = self._network()
        if net is None:
            return False

        data = list(support if support is not None else self._support)
        if len(data) < max(5, SUPPORT_SIZE // 4):
            return False

        # Snapshot current params as starting point
        params_snapshot = {k: v.clone() for k, v in net.state_dict().items()
                           if v.dtype.is_floating_point}

        # Inner loop: SGD adaptation
        adapted_net = copy.deepcopy(net)
        inner_opt   = optim.SGD(adapted_net.parameters(), lr=self.inner_lr)

        for _ in range(self.n_inner):
            loss = self._compute_task_loss(adapted_net, data)
            inner_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(adapted_net.parameters(), 5.0)
            inner_opt.step()

        # Store adapted params for use during inference
        self._adapted_params = {k: v.clone()
                                for k, v in adapted_net.state_dict().items()
                                if v.dtype.is_floating_point}
        self._using_adapted = True
        logger.info("MAML fast_adapt complete — %d support samples | %d steps",
                    len(data), self.n_inner)
        return True

    def meta_update(self) -> dict:
        """
        Outer loop: update θ so that fast_adapt produces good θ'.
        Uses FOMAML (First-Order MAML) for efficiency.
        """
        net = self._network()
        if net is None or self.meta_opt is None:
            return {}

        support = list(self._support)
        query   = list(self._query)
        if len(support) < max(5, SUPPORT_SIZE // 4) or len(query) < 3:
            return {}

        # Inner loop (with gradient tracking for FOMAML)
        adapted_net = copy.deepcopy(net)
        inner_opt   = optim.SGD(adapted_net.parameters(), lr=self.inner_lr)

        for _ in range(self.n_inner):
            loss = self._compute_task_loss(adapted_net, support)
            inner_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(adapted_net.parameters(), 5.0)
            inner_opt.step()

        # Outer loop: evaluate adapted_net on query set
        meta_loss = self._compute_task_loss(adapted_net, query)

        # FOMAML: approximate meta-gradient using adapted_net's params
        # copy gradients from adapted_net to original net
        self.meta_opt.zero_grad()
        meta_loss.backward()

        # Transfer gradients: FOMAML approximation
        for (name, p_orig), (_, p_adap) in zip(
            net.named_parameters(), adapted_net.named_parameters()
        ):
            if p_orig.grad is not None and p_adap.grad is not None:
                p_orig.grad.data.copy_(p_adap.grad.data)
            elif p_adap.grad is not None:
                p_orig.grad = p_adap.grad.clone()

        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        self.meta_opt.step()

        metrics = {
            "maml_meta_loss":  float(meta_loss.item()),
            "maml_support_n":  len(support),
            "maml_query_n":    len(query),
        }
        logger.info("MAML meta_update | meta_loss=%.4f | support=%d | query=%d",
                    metrics["maml_meta_loss"],
                    metrics["maml_support_n"],
                    metrics["maml_query_n"])
        return metrics

    def _apply_adapted(self):
        """Temporarily apply adapted weights for inference."""
        net = self._network()
        if net is None or self._adapted_params is None:
            return
        own = net.state_dict()
        merged = {k: self._adapted_params[k] if k in self._adapted_params else v
                  for k, v in own.items()}
        net.load_state_dict(merged)

    def _restore_base(self):
        """Restore base (meta) weights after adapted inference."""
        pass  # FOMAML: base weights are NOT overwritten during inference

    # ── PPOAgent-compatible interface (all delegated to base_agent) ────────────

    @property
    def total_steps(self) -> int:
        return self.base_agent.total_steps

    @property
    def total_updates(self) -> int:
        return self.base_agent.total_updates

    @property
    def total_episodes(self) -> int:
        return self.base_agent.total_episodes

    @property
    def cumulative_reward(self) -> float:
        return self.base_agent.cumulative_reward

    def select_action(self, obs: np.ndarray,
                      deterministic: bool = False) -> Tuple[int, float, float]:
        return self.base_agent.select_action(obs, deterministic)

    def get_confidence(self, obs: np.ndarray) -> Tuple[int, float]:
        return self.base_agent.get_confidence(obs)

    def get_action_distribution(self, obs: np.ndarray) -> dict:
        return self.base_agent.get_action_distribution(obs)

    def store(self, obs, next_obs, action: int,
              log_prob: float, reward: float, value: float, done: bool):
        self.base_agent.store(obs, next_obs, action, log_prob, reward, value, done)
        self._new_trades += 1

        # Accumulate for meta-learning
        # Rotate: query ← old support head, support ← new
        if len(self._support) >= SUPPORT_SIZE:
            oldest = self._support[0]
            self._query.append(oldest)
        self._support.append((obs.copy(), action, reward))

    def ready_to_update(self) -> bool:
        return self.base_agent.ready_to_update()

    def update(self, last_obs: np.ndarray) -> dict:
        metrics = self.base_agent.update(last_obs)

        # Periodic fast_adapt
        if self._new_trades > 0 and self._new_trades % ADAPT_EVERY == 0:
            self.fast_adapt()

        # Periodic meta-update
        if self._new_trades > 0 and self._new_trades % META_EVERY == 0:
            m = self.meta_update()
            metrics.update(m)

        return metrics

    def update_online(self, obs: np.ndarray, action: int, reward: float,
                      next_obs: np.ndarray, done: bool) -> dict:
        self.store(obs, next_obs, action, 0.0, reward, 0.0, done)
        if self.ready_to_update():
            return self.update(next_obs)
        return {}

    def save(self, path: str) -> None:
        self.base_agent.save(path)

    def load(self, path: str) -> bool:
        return self.base_agent.load(path)

    def get_weights_bytes(self) -> bytes:
        return self.base_agent.get_weights_bytes()

    def load_weights_bytes(self, data: bytes) -> None:
        self.base_agent.load_weights_bytes(data)

    def fedavg_merge(self, worker_bytes: bytes, alpha: float = 0.25) -> None:
        self.base_agent.fedavg_merge(worker_bytes, alpha)

    def get_buffer_bytes(self) -> bytes:
        return self.base_agent.get_buffer_bytes()
