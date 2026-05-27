"""
Rainbow DQN — RL อัลกอริทึมที่ดีที่สุดสำหรับ Binary Options
══════════════════════════════════════════════════════════════

ทำไม Rainbow ดีกว่า PPO สำหรับ Binary Options?

PPO ออกแบบสำหรับ continuous action (robotics, locomotion)
Binary Options = discrete 3-action (HOLD/BUY/SELL) → Q-learning ชนะเสมอ

Rainbow รวม 6 เทคนิค:
1. Distributional RL (C51) — โมเดล WIN/LOSS distribution แทน mean
   Binary: WIN=+85%, LOSS=-100% → bimodal ไม่ใช่ Gaussian → C51 จับได้ถูกต้อง
2. Noisy Networks — exploration อัจฉริยะ แทน epsilon-greedy สุ่มตัน
   NoisyLinear: w = μ + σ⊙ε, ε sampled ต่าง per-forward pass
3. Dueling Architecture — แยก V(s) และ A(s,a) → เรียนรู้ได้ดีกว่าเมื่อ HOLD ดีที่สุด
4. Double DQN — target network ลด overestimation bias → ลด over-confidence
5. Multi-step Returns (n=3) — เชื่อมโยง signal กับ reward ได้ 3 trade ล่วงหน้า
6. Prioritized Replay — ใช้ per_buffer.py ที่มีอยู่แล้ว (|reward|×RPE priority)

Interface ตรงกับ PPOAgent ทุกประการ → drop-in replacement
"""
import io
import logging
import math
import os
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from trading_ai.models.per_buffer import PrioritizedReplayBuffer

logger = logging.getLogger(__name__)

# ── Distributional RL — atom support ──────────────────────────────────────────
V_MIN   = -1.05   # LOSS -100% + ค่าธรรมเนียมเล็กน้อย
V_MAX   =  0.92   # WIN payout สูงสุด ~89%
N_ATOMS =  51     # 51 จุดบน distribution (ตามงานวิจัย C51 ดั้งเดิม)
DELTA_Z = (V_MAX - V_MIN) / (N_ATOMS - 1)

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_STEP        = 3      # Multi-step return window
GAMMA         = 0.99
TARGET_UPDATE = 10     # อัพเดต target network ทุก 10 gradient steps
WARMUP        = 64     # รอ experience ก่อน update ครั้งแรก
BATCH_SIZE    = 32
LR            = 3e-4
SIGMA_INIT    = 0.5    # Initial std ของ NoisyLinear
LSTM_HIDDEN   = 128


# ── NoisyLinear ────────────────────────────────────────────────────────────────

class NoisyLinear(nn.Module):
    """
    Factorized NoisyLinear — exploration ที่เรียนรู้ได้เอง
    Parameters = 2(p+q) แทน 2pq → ประหยัด memory มาก
    """

    def __init__(self, in_f: int, out_f: int, sigma: float = SIGMA_INIT):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f

        self.w_mu    = nn.Parameter(torch.empty(out_f, in_f))
        self.w_sigma = nn.Parameter(torch.empty(out_f, in_f))
        self.b_mu    = nn.Parameter(torch.empty(out_f))
        self.b_sigma = nn.Parameter(torch.empty(out_f))

        self.register_buffer("w_eps", torch.empty(out_f, in_f))
        self.register_buffer("b_eps", torch.empty(out_f))
        self._sigma = sigma
        self._reset_params()
        self.reset_noise()

    def _reset_params(self):
        r = 1.0 / math.sqrt(self.in_f)
        self.w_mu.data.uniform_(-r, r)
        self.w_sigma.data.fill_(self._sigma / math.sqrt(self.in_f))
        self.b_mu.data.uniform_(-r, r)
        self.b_sigma.data.fill_(self._sigma / math.sqrt(self.out_f))

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        eps_i = self._f(torch.randn(self.in_f,  device=self.w_eps.device))
        eps_j = self._f(torch.randn(self.out_f, device=self.w_eps.device))
        self.w_eps.copy_(eps_j.outer(eps_i))
        self.b_eps.copy_(eps_j)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.w_mu + self.w_sigma * self.w_eps
            b = self.b_mu + self.b_sigma * self.b_eps
        else:
            w, b = self.w_mu, self.b_mu
        return F.linear(x, w, b)


# ── Residual Block (shared with PPO for style consistency) ─────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, size: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size), nn.LayerNorm(size),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(size, size), nn.LayerNorm(size),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


# ── Rainbow Network ────────────────────────────────────────────────────────────

class RainbowNet(nn.Module):
    """
    Rainbow DQN Network:
    obs → ResidualExtractor → LSTM (temporal) → NoisyLinear (Distributional Dueling)
    Output: Q-distribution logits (B, n_actions, N_atoms)
    """

    def __init__(self, obs_size: int, n_actions: int, hidden: int = 384):
        super().__init__()
        self.obs_size  = obs_size
        self.n_actions = n_actions
        self.hidden    = hidden

        # Feature extractor (identical style to PPO ULTRA)
        self.features = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.1),
            ResidualBlock(hidden, 0.1),
            ResidualBlock(hidden, 0.05),
        )

        # LSTM temporal memory
        self.lstm = nn.LSTM(hidden, LSTM_HIDDEN, num_layers=2,
                            batch_first=True, dropout=0.1)
        self._h: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        # Project to core
        core = hidden // 2
        self.proj = nn.Sequential(
            nn.Linear(LSTM_HIDDEN, core),
            nn.LayerNorm(core), nn.GELU(),
        )

        # Distributional Dueling heads — Noisy
        self.val1 = NoisyLinear(core, core // 2)
        self.val2 = NoisyLinear(core // 2, N_ATOMS)

        self.adv1 = NoisyLinear(core, core // 2)
        self.adv2 = NoisyLinear(core // 2, n_actions * N_ATOMS)

        for m in self.features.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)

    def reset_hidden(self, batch: int = 1):
        dev = next(self.parameters()).device
        self._h = (
            torch.zeros(2, batch, LSTM_HIDDEN, device=dev),
            torch.zeros(2, batch, LSTM_HIDDEN, device=dev),
        )

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns raw logits: (B, n_actions, N_atoms)."""
        B = obs.size(0)
        feat = self.features(obs)
        if self._h is None or self._h[0].size(1) != B:
            self.reset_hidden(B)
        out, new_h = self.lstm(feat.unsqueeze(1), self._h)
        self._h = (new_h[0].detach(), new_h[1].detach())
        core = self.proj(out.squeeze(1))              # (B, H//2)

        v = F.gelu(self.val1(core))
        v = self.val2(v)                              # (B, Z)

        a = F.gelu(self.adv1(core))
        a = self.adv2(a).view(B, self.n_actions, N_ATOMS)

        q = v.unsqueeze(1) + a - a.mean(1, keepdim=True)  # Dueling (B,A,Z)
        return q

    def get_probs(self, obs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self(obs), dim=-1)

    def get_q_values(self, obs: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        """Expected Q-values: (B, n_actions)."""
        probs = self.get_probs(obs)                    # (B, A, Z)
        return (probs * support.view(1, 1, -1)).sum(-1)  # (B, A)


# ── Multi-step Buffer ──────────────────────────────────────────────────────────

class NStepBuffer:
    """Accumulates N_STEP transitions, computes discounted return, then emits to PER."""

    def __init__(self, n: int = N_STEP, gamma: float = GAMMA):
        self.n     = n
        self.gamma = gamma
        self._buf: List = []

    def add(self, obs, action: int, reward: float,
            next_obs, done: bool) -> Optional[tuple]:
        self._buf.append((obs, action, reward, next_obs, done))
        if len(self._buf) < self.n:
            return None

        G = 0.0
        for i, (_, _, r, _, d) in enumerate(self._buf):
            G += (self.gamma ** i) * r
            if d:
                break

        obs0,  act0, _, _,    _    = self._buf[0]
        _,     _,    _, obsN, doneN = self._buf[-1]
        self._buf.pop(0)
        return obs0, act0, G, obsN, doneN

    def flush(self):
        self._buf.clear()


# ── Rainbow Agent ──────────────────────────────────────────────────────────────

class RainbowAgent:
    """
    Full Rainbow DQN — drop-in replacement for PPOAgent.

    ความแตกต่างหลัก vs PPO:
    • Q-learning (off-policy) ไม่ใช่ policy gradient (on-policy)
      → เรียนรู้ได้จาก PER buffer ทุก step แทนที่จะรอ 128 trade
    • Distributional: รู้ว่า distribution ของ outcome เป็น bimodal
      ไม่ได้แค่ mean → confidence ที่ออกมา calibrated กว่า PPO
    • Double DQN: target network ป้องกัน overestimation
      → ไม่ trade เพราะ AI คิดว่ามั่นใจแต่จริงๆ ไม่ใช่
    """

    def __init__(self, obs_size: int, n_actions: int = 3, hidden_size: int = 384):
        self.obs_size    = obs_size
        self.n_actions   = n_actions
        self.hidden_size = hidden_size
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online = RainbowNet(obs_size, n_actions, hidden_size).to(self.device)
        self.target = RainbowNet(obs_size, n_actions, hidden_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.support = torch.linspace(V_MIN, V_MAX, N_ATOMS, device=self.device)

        self.optimizer = optim.Adam(self.online.parameters(), lr=LR, eps=1.5e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2, eta_min=1e-5,
        )

        self.per    = PrioritizedReplayBuffer(capacity=2000, obs_size=obs_size)
        self.nstep  = NStepBuffer(N_STEP, GAMMA)

        self.total_steps:       int   = 0
        self.total_updates:     int   = 0
        self.total_episodes:    int   = 0
        self.cumulative_reward: float = 0.0
        self._grad_steps:       int   = 0   # for target network sync

        logger.info("RainbowAgent — device:%s | obs:%d | atoms:%d | n_step:%d",
                    self.device, obs_size, N_ATOMS, N_STEP)

    # ── Inference ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, obs: np.ndarray,
                      deterministic: bool = False) -> Tuple[int, float, float]:
        """Returns (action, log_prob, value) — same signature as PPOAgent."""
        self.online.eval()
        self.online.reset_noise()
        obs_t  = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
        q      = self.online.get_q_values(obs_t, self.support).squeeze(0)  # (A,)
        probs  = torch.softmax(q / 0.10, dim=-1)
        action = int(q.argmax().item()) if deterministic \
                 else int(torch.multinomial(probs, 1).item())
        log_p  = float(torch.log(probs[action] + 1e-9).item())
        value  = float(q.max().item())
        self.online.train()
        return action, log_p, value

    @torch.no_grad()
    def get_confidence(self, obs: np.ndarray) -> Tuple[int, float]:
        self.online.eval()
        obs_t = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
        q     = self.online.get_q_values(obs_t, self.support).squeeze(0)
        probs = torch.softmax(q / 0.10, dim=-1)
        best  = int(probs.argmax().item())
        conf  = float(probs[best].item())
        self.online.train()
        return best, conf

    @torch.no_grad()
    def get_action_distribution(self, obs: np.ndarray) -> dict:
        self.online.eval()
        obs_t = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
        q     = self.online.get_q_values(obs_t, self.support).squeeze(0)
        probs = torch.softmax(q / 0.10, dim=-1).cpu().numpy()
        self.online.train()
        return {"hold": float(probs[0]), "buy": float(probs[1]), "sell": float(probs[2])}

    # ── Learning ───────────────────────────────────────────────────────────────

    def store(self, obs, next_obs, action: int,
              log_prob: float, reward: float, value: float, done: bool):
        """log_prob and value unused (Q-learning) — kept for interface compat."""
        result = self.nstep.add(obs, action, reward, next_obs, done)
        if result is not None:
            o0, a0, G, oN, dN = result
            self.per.add(o0, oN, a0, G, rpe_mult=abs(G) + 1.0)
        self.total_steps       += 1
        self.cumulative_reward += reward
        if done:
            self.nstep.flush()
            self.total_episodes += 1

    def ready_to_update(self) -> bool:
        return self.per._size >= WARMUP

    def update(self, last_obs: np.ndarray) -> dict:
        """One Rainbow gradient step (call every trade when buffer is ready)."""
        batch = self.per.sample(BATCH_SIZE)
        if batch is None:
            return {}

        obs    = torch.from_numpy(batch["obs"]).float().to(self.device)
        next_o = torch.from_numpy(batch["next_obs"]).float().to(self.device)
        acts   = torch.from_numpy(batch["actions"]).long().to(self.device)
        rews   = torch.from_numpy(batch["rewards"]).float().to(self.device)
        wts    = torch.from_numpy(batch["weights"]).float().to(self.device)
        idxs   = batch["idxs"]
        B      = obs.size(0)

        self.online.reset_noise()
        self.target.reset_noise()

        # ── Double DQN target distribution ────────────────────────────────────
        with torch.no_grad():
            # Online selects next action
            q_next  = self.online.get_q_values(next_o, self.support)   # (B, A)
            acts_n  = q_next.argmax(1)                                  # (B,)

            # Target evaluates
            probs_n = self.target.get_probs(next_o)                     # (B,A,Z)
            p_best  = probs_n[torch.arange(B), acts_n]                 # (B,Z)

            # Distributional Bellman projection onto fixed support
            gn  = GAMMA ** N_STEP
            Tz  = (rews.unsqueeze(1) + gn * self.support.unsqueeze(0)).clamp(V_MIN, V_MAX)
            b   = (Tz - V_MIN) / DELTA_Z
            lo  = b.floor().long().clamp(0, N_ATOMS - 1)
            hi  = b.ceil().long().clamp(0, N_ATOMS - 1)

            m   = torch.zeros(B, N_ATOMS, device=self.device)
            off = torch.arange(B, device=self.device).unsqueeze(1) * N_ATOMS
            m.view(-1).scatter_add_(0, (lo + off).view(-1),
                                    (p_best * (hi.float() - b)).view(-1))
            m.view(-1).scatter_add_(0, (hi + off).view(-1),
                                    (p_best * (b - lo.float())).view(-1))

        # ── Online prediction ─────────────────────────────────────────────────
        log_p  = F.log_softmax(self.online(obs), dim=-1)       # (B, A, Z)
        log_pa = log_p[torch.arange(B), acts]                  # (B, Z)

        # Cross-entropy loss (KL divergence between m and online)
        loss_elem = -(m * log_pa).sum(-1)                      # (B,)
        loss      = (loss_elem * wts).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        # Update PER priorities
        self.per.update_priorities(idxs, loss_elem.detach().cpu().numpy())

        # Target network sync
        self._grad_steps += 1
        if self._grad_steps % TARGET_UPDATE == 0:
            self.target.load_state_dict(self.online.state_dict())

        self.scheduler.step()
        self.total_updates += 1

        metrics = {
            "rainbow_loss":   float(loss.item()),
            "rainbow_lr":     self.optimizer.param_groups[0]["lr"],
            "rainbow_buf":    self.per._size,
            "rainbow_td_mean": float(loss_elem.detach().mean().item()),
        }
        logger.info("Rainbow #%d | loss=%.4f | lr=%.6f | buf=%d",
                    self.total_updates, metrics["rainbow_loss"],
                    metrics["rainbow_lr"], metrics["rainbow_buf"])
        return metrics

    def update_online(self, obs: np.ndarray, action: int, reward: float,
                      next_obs: np.ndarray, done: bool) -> dict:
        """Single-step online update (called every trade)."""
        self.store(obs, next_obs, action, 0.0, reward, 0.0, done)
        if self.ready_to_update():
            return self.update(next_obs)
        return {}

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "online":     self.online.state_dict(),
            "target":     self.target.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "scheduler":  self.scheduler.state_dict(),
            "steps":      self.total_steps,
            "updates":    self.total_updates,
            "episodes":   self.total_episodes,
            "cum_reward": self.cumulative_reward,
        }, path)
        logger.info("Rainbow saved → %s (steps=%d)", path, self.total_steps)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            logger.info("No Rainbow checkpoint — starting fresh")
            return False
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.online.load_state_dict(ckpt["online"])
            self.target.load_state_dict(ckpt["target"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            self.total_steps       = ckpt.get("steps", 0)
            self.total_updates     = ckpt.get("updates", 0)
            self.total_episodes    = ckpt.get("episodes", 0)
            self.cumulative_reward = ckpt.get("cum_reward", 0.0)
            logger.info("Rainbow loaded ← %s (steps=%d)", path, self.total_steps)
            return True
        except Exception as exc:
            logger.error("Rainbow load failed: %s", exc)
            return False

    def get_weights_bytes(self) -> bytes:
        buf = io.BytesIO()
        torch.save(self.online.state_dict(), buf)
        return buf.getvalue()

    def load_weights_bytes(self, data: bytes) -> None:
        buf   = io.BytesIO(data)
        state = torch.load(buf, map_location=self.device, weights_only=True)
        self.online.load_state_dict(state)

    def fedavg_merge(self, worker_bytes: bytes, alpha: float = 0.25) -> None:
        buf    = io.BytesIO(worker_bytes)
        worker = torch.load(buf, map_location=self.device, weights_only=True)
        own    = self.online.state_dict()
        merged = {
            k: (1 - alpha) * own[k].float() + alpha * worker[k].float()
            if k in worker else own[k]
            for k in own
        }
        self.online.load_state_dict(merged)
        logger.info("Rainbow FedAvg alpha=%.2f", alpha)

    def get_buffer_bytes(self) -> bytes:
        import pickle
        buf = io.BytesIO()
        pickle.dump({
            "obs":     self.per._obs[:self.per._size],
            "actions": self.per._actions[:self.per._size],
            "rewards": self.per._rewards[:self.per._size],
            "size":    self.per._size,
        }, buf)
        return buf.getvalue()
