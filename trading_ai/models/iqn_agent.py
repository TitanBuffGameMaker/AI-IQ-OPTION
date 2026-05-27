"""
IQN — Implicit Quantile Network
══════════════════════════════════

ดีกว่า C51 (Rainbow) อย่างไร?

C51 (Rainbow):  กำหนด 51 atoms คงที่ [V_min, V_max] ล่วงหน้า
IQN:            sample quantile τ ~ U(0,1) แบบ random ทุก forward pass
               → ไม่ต้องกำหนด atom ล่วงหน้า
               → model quantile function โดยตรง F⁻¹(τ|s,a)
               → ยืดหยุ่นกว่าสำหรับ distribution ที่เปลี่ยนตาม OTC pattern

สำหรับ Binary Options:
  Distribution มีสองรูปแบบที่ต่างกันมาก:
  - ชั่วโมงดี: bimodal กว้าง (WIN ชัด/LOSS ชัด)
  - ชั่วโมงแย่: กอง loss ที่ -1.0 มาก → IQN จับรูปแบบนี้ได้ชัดกว่า C51
  - OTC shift: distribution ย้ายตำแหน่ง → IQN adapt ได้เร็วกว่า (no fixed atoms)

IQN ยังให้ risk-aware trading:
  - Conservative: ใช้ quantile τ ต่ำ (เชื่อ worst-case scenario)
  - Aggressive: ใช้ quantile τ สูง (เชื่อ best-case scenario)
  - เหมาะกับ REAL account ที่ต้องการ conservative mode

Interface เหมือน PPOAgent และ RainbowAgent
"""
import io
import logging
import math
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from trading_ai.models.per_buffer import PrioritizedReplayBuffer
from trading_ai.models.rainbow_dqn import (
    NoisyLinear, ResidualBlock, NStepBuffer,
    N_STEP, GAMMA, WARMUP, BATCH_SIZE, LR, LSTM_HIDDEN, TARGET_UPDATE,
)

logger = logging.getLogger(__name__)

# ── IQN Hyperparameters ────────────────────────────────────────────────────────
N_COS       = 64     # Cosine features for quantile embedding
K_TRAIN     = 32    # Quantile samples during training
K_ACT       = 8     # Quantile samples during action selection (speed vs accuracy)
KAPPA       = 1.0   # Huber loss threshold
TAU_RISK    = 0.25  # Conservative mode: use lower quantile (REAL account)


class IQNNet(nn.Module):
    """
    Implicit Quantile Network:
    Q(s,a,τ) = f(state_embed(s) ⊙ quantile_embed(τ))

    แทนที่จะ output distribution ที่ fixed atoms (C51),
    IQN output Q-value ที่ quantile τ ที่ random sample มา
    """

    def __init__(self, obs_size: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.hidden    = hidden
        self.n_actions = n_actions
        self.n_cos     = N_COS

        # State encoder (deep feature extraction)
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.1),
            ResidualBlock(hidden, 0.1),
        )

        # LSTM for temporal context
        self.lstm = nn.LSTM(hidden, LSTM_HIDDEN, num_layers=2,
                            batch_first=True, dropout=0.1)
        self._h: Optional[Tuple] = None
        self.proj = nn.Sequential(
            nn.Linear(LSTM_HIDDEN, hidden),
            nn.LayerNorm(hidden), nn.GELU(),
        )

        # Quantile embedding: cos(π·i·τ) for i=1..N_COS
        self.quantile_embed = nn.Sequential(
            nn.Linear(N_COS, hidden), nn.GELU(),
        )

        # Dueling heads (Noisy)
        core = hidden
        self.val1 = NoisyLinear(core, core // 2)
        self.val2 = NoisyLinear(core // 2, 1)
        self.adv1 = NoisyLinear(core, core // 2)
        self.adv2 = NoisyLinear(core // 2, n_actions)

        for m in self.encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)

    def reset_hidden(self, B: int = 1):
        dev = next(self.parameters()).device
        self._h = (
            torch.zeros(2, B, LSTM_HIDDEN, device=dev),
            torch.zeros(2, B, LSTM_HIDDEN, device=dev),
        )

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def _state_feature(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to state vector: (B, hidden)."""
        B = obs.size(0)
        feat = self.encoder(obs)
        if self._h is None or self._h[0].size(1) != B:
            self.reset_hidden(B)
        out, new_h = self.lstm(feat.unsqueeze(1), self._h)
        self._h = (new_h[0].detach(), new_h[1].detach())
        return self.proj(out.squeeze(1))   # (B, hidden)

    def _quantile_feature(self, taus: torch.Tensor) -> torch.Tensor:
        """
        Embed quantiles τ via cosine features.
        taus: (B, K) → returns (B, K, hidden)
        """
        B, K = taus.shape
        i    = torch.arange(1, self.n_cos + 1, device=taus.device, dtype=torch.float32)
        # cos(π × i × τ): (B, K, N_COS)
        cos  = torch.cos(taus.unsqueeze(-1) * i.unsqueeze(0).unsqueeze(0) * math.pi)
        return self.quantile_embed(cos)    # (B, K, hidden)

    def forward(self, obs: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
        """
        obs:  (B, obs_size)
        taus: (B, K) — K quantile samples from U(0,1)
        Returns Q(s,a,τ): (B, K, n_actions)
        """
        B, K    = taus.shape
        s_feat  = self._state_feature(obs)          # (B, hidden)
        q_feat  = self._quantile_feature(taus)      # (B, K, hidden)

        # Hadamard product: state × quantile embedding
        x = s_feat.unsqueeze(1) * q_feat            # (B, K, hidden)
        x = x.reshape(B * K, self.hidden)

        # Dueling
        v = F.gelu(self.val1(x))
        v = self.val2(v)                             # (BK, 1)
        a = F.gelu(self.adv1(x))
        a = self.adv2(a)                             # (BK, n_actions)

        q = v + a - a.mean(-1, keepdim=True)         # (BK, n_actions)
        return q.view(B, K, self.n_actions)          # (B, K, n_actions)

    @torch.no_grad()
    def mean_q(self, obs: torch.Tensor, k: int = K_ACT,
               tau_bias: Optional[float] = None) -> torch.Tensor:
        """
        Mean Q-value over K quantile samples: (B, n_actions).
        tau_bias: if set, sample from U(tau_bias, 1) for conservative mode.
        """
        B    = obs.size(0)
        taus = torch.rand(B, k, device=obs.device)
        if tau_bias is not None:
            taus = taus * (1.0 - tau_bias) + tau_bias   # shift up/down
        q = self(obs, taus)          # (B, k, A)
        return q.mean(1)             # (B, A)


# ── IQN Agent ──────────────────────────────────────────────────────────────────

class IQNAgent:
    """
    IQN Agent — Implicit Quantile Network for Binary Options.
    Drop-in replacement for PPOAgent and RainbowAgent.

    ข้อดีพิเศษ:
    - Conservative mode (REAL account): ใช้ quantile ต่ำ → ระวังกว่า
    - Fast adaptation: ไม่มี fixed atom → adapt distribution เมื่อ OTC shift
    - Risk-aware confidence: width ของ quantile spread = uncertainty measure
    """

    def __init__(self, obs_size: int, n_actions: int = 3, hidden_size: int = 256,
                 conservative: bool = False):
        self.obs_size     = obs_size
        self.n_actions    = n_actions
        self.conservative = conservative   # True สำหรับ REAL account
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online = IQNNet(obs_size, n_actions, hidden_size).to(self.device)
        self.target = IQNNet(obs_size, n_actions, hidden_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.optimizer = optim.Adam(self.online.parameters(), lr=LR, eps=1.5e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2, eta_min=1e-5,
        )

        self.per   = PrioritizedReplayBuffer(capacity=2000, obs_size=obs_size)
        self.nstep = NStepBuffer(N_STEP, GAMMA)

        self.total_steps:       int   = 0
        self.total_updates:     int   = 0
        self.total_episodes:    int   = 0
        self.cumulative_reward: float = 0.0
        self._grad_steps:       int   = 0

        mode = "conservative" if conservative else "standard"
        logger.info("IQNAgent [%s] — device:%s | obs:%d | K_train:%d | K_act:%d",
                    mode, self.device, obs_size, K_TRAIN, K_ACT)

    # ── Inference ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, obs: np.ndarray,
                      deterministic: bool = False) -> Tuple[int, float, float]:
        self.online.eval()
        self.online.reset_noise()
        obs_t  = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
        tau_b  = TAU_RISK if self.conservative else None
        q      = self.online.mean_q(obs_t, K_ACT, tau_b).squeeze(0)  # (A,)
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
        obs_t  = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
        tau_b  = TAU_RISK if self.conservative else None
        q      = self.online.mean_q(obs_t, K_ACT, tau_b).squeeze(0)
        probs  = torch.softmax(q / 0.10, dim=-1)
        best   = int(probs.argmax().item())
        conf   = float(probs[best].item())
        self.online.train()
        return best, conf

    @torch.no_grad()
    def get_uncertainty(self, obs: np.ndarray) -> float:
        """Epistemic uncertainty = std over quantile samples (wider → less certain)."""
        self.online.eval()
        obs_t  = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
        B, K   = 1, 32
        taus   = torch.rand(B, K, device=self.device)
        q      = self.online(obs_t, taus).squeeze(0)  # (K, A)
        std    = float(q.std(0).max().item())
        self.online.train()
        return std

    @torch.no_grad()
    def get_action_distribution(self, obs: np.ndarray) -> dict:
        self.online.eval()
        obs_t = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
        q     = self.online.mean_q(obs_t, K_ACT).squeeze(0)
        probs = torch.softmax(q / 0.10, dim=-1).cpu().numpy()
        self.online.train()
        return {"hold": float(probs[0]), "buy": float(probs[1]), "sell": float(probs[2])}

    # ── Learning ───────────────────────────────────────────────────────────────

    def store(self, obs, next_obs, action: int,
              log_prob: float, reward: float, value: float, done: bool):
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
        """IQN quantile Huber (QR-Huber) loss update."""
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

        # ── Target Q-values (Double DQN) ──────────────────────────────────────
        with torch.no_grad():
            taus_act = torch.rand(B, K_ACT, device=self.device)
            q_next_online = self.online(next_o, taus_act).mean(1)  # (B, A)
            acts_n        = q_next_online.argmax(1)                # (B,)

            taus_target = torch.rand(B, K_TRAIN, device=self.device)
            q_target    = self.target(next_o, taus_target)         # (B, K, A)
            q_best      = q_target[torch.arange(B), :, acts_n]    # (B, K)

            # Bellman target: z = r + γⁿ Q(s', a*)
            gamma_n  = GAMMA ** N_STEP
            targets  = rews.unsqueeze(1) + gamma_n * q_best        # (B, K)

        # ── Online prediction ─────────────────────────────────────────────────
        taus_online = torch.rand(B, K_TRAIN, device=self.device)
        q_online    = self.online(obs, taus_online)                 # (B, K, A)
        q_pred      = q_online[torch.arange(B), :, acts]           # (B, K_online)

        # ── Quantile Huber loss ───────────────────────────────────────────────
        # u = target - pred: (B, K_target, K_online)
        u    = targets.unsqueeze(2) - q_pred.unsqueeze(1)          # (B, Kt, Ko)
        tau  = taus_online.unsqueeze(1)                             # (B, 1, Ko)
        huber = F.huber_loss(q_pred.unsqueeze(1).expand_as(u),
                             targets.unsqueeze(2).expand_as(u),
                             reduction="none", delta=KAPPA)
        qr_loss  = (torch.abs(tau - (u < 0).float()) * huber / KAPPA)  # (B,Kt,Ko)
        loss_per = qr_loss.mean(dim=(1, 2))                        # (B,)
        loss     = (loss_per * wts).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        self.per.update_priorities(idxs, loss_per.detach().cpu().numpy())

        self._grad_steps += 1
        if self._grad_steps % TARGET_UPDATE == 0:
            self.target.load_state_dict(self.online.state_dict())

        self.scheduler.step()
        self.total_updates += 1

        metrics = {
            "iqn_loss":    float(loss.item()),
            "iqn_lr":      self.optimizer.param_groups[0]["lr"],
            "iqn_buf":     self.per._size,
        }
        logger.info("IQN #%d | loss=%.4f | lr=%.6f | buf=%d",
                    self.total_updates, metrics["iqn_loss"],
                    metrics["iqn_lr"], metrics["iqn_buf"])
        return metrics

    def update_online(self, obs: np.ndarray, action: int, reward: float,
                      next_obs: np.ndarray, done: bool) -> dict:
        self.store(obs, next_obs, action, 0.0, reward, 0.0, done)
        if self.ready_to_update():
            return self.update(next_obs)
        return {}

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "online":        self.online.state_dict(),
            "target":        self.target.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "scheduler":     self.scheduler.state_dict(),
            "steps":         self.total_steps,
            "updates":       self.total_updates,
            "episodes":      self.total_episodes,
            "cum_reward":    self.cumulative_reward,
            "conservative":  self.conservative,
        }, path)
        logger.info("IQN saved → %s (steps=%d)", path, self.total_steps)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            logger.info("No IQN checkpoint — starting fresh")
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
            logger.info("IQN loaded ← %s (steps=%d)", path, self.total_steps)
            return True
        except Exception as exc:
            logger.error("IQN load failed: %s", exc)
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
        merged = {k: (1 - alpha) * own[k].float() + alpha * worker[k].float()
                  if k in worker else own[k] for k in own}
        self.online.load_state_dict(merged)

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
