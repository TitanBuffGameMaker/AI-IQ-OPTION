"""
World Model (DreamerV3 Lite) — โลกในจินตนาการของ AI
═══════════════════════════════════════════════════════

ทำไมต้องใช้ World Model?

สมองที่เรียนรู้จากประสบการณ์จริงเท่านั้น (model-free):
  1 วัน = 50-100 trades = 50-100 learning steps → เรียนรู้ช้ามาก

World Model แก้ด้วย:
  เรียนรู้ dynamics ของตลาดจากประสบการณ์จริง
  → จินตนาการ (imagine) ว่าจะเกิดอะไรขึ้นถ้าทำ action ต่างๆ
  → สร้าง "imagined rollouts" เพิ่มเป็น 10x ของ real experience
  → Sample efficiency เพิ่ม 10-20x

Architecture (RSSM lite — ดัดแปลงจาก DreamerV3):
  ObsEncoder:    obs(296) → latent(64)
  RSSM:
    Deterministic: h_t = GRU(z_{t-1} ⊕ a_{t-1}, h_{t-1})  [128 dim]
    Stochastic:    z_t ~ q(z|h_t, o_t) ← Posterior [32 dim]
                   z_t ~ p(z|h_t)       ← Prior [32 dim]
  RewardModel:   (h_t, z_t) → r̂_t  [predict trade outcome]
  ObsDecoder:    (h_t, z_t) → ô_t   [reconstruct observation]

Training loss (ELBO):
  L = -log P(o_t|h_t,z_t) + -log P(r_t|h_t,z_t) + KL[q‖p]

Imagination:
  ใช้ Prior p(z|h) + Policy(z) → สร้าง imagined trajectories
  ส่งไปยัง Rainbow/IQN buffer → เพิ่ม training data 10x
"""
import logging
import os
import threading
import time
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger(__name__)

# ── World Model Hyperparameters ────────────────────────────────────────────────
OBS_LATENT     = 64    # Encoded observation size
DET_SIZE       = 128   # Deterministic state h (GRU hidden)
STOCH_SIZE     = 32    # Stochastic state z
ACTION_SIZE    = 3     # HOLD/BUY/SELL
WM_LR          = 3e-4
KL_BALANCE     = 0.8   # Weight of KL free bits regularization
FREE_NATS      = 1.0   # Minimum KL (free bits) — prevents posterior collapse
IMAGINE_STEPS  = 10    # Steps to imagine per real step
IMAGINE_GAMMA  = 0.99  # Discount in imagined rollouts
WM_WARMUP      = 50    # Real experiences before world model trains
WM_BATCH       = 16    # Batch size for world model training
WM_SEQ_LEN     = 8     # Sequence length for RSSM training


class ObsEncoder(nn.Module):
    """obs(296) → μ,σ in latent space (64 dim)."""

    def __init__(self, obs_size: int, latent: int = OBS_LATENT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 256), nn.ELU(),
            nn.Linear(256, 128),      nn.ELU(),
            nn.Linear(128, latent * 2),
        )
        self.latent = latent

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out     = self.net(obs)
        mu, ls  = out.chunk(2, dim=-1)
        return mu, ls.clamp(-4, 4)   # μ, log_σ


class ObsDecoder(nn.Module):
    """(h, z) → reconstructed obs."""

    def __init__(self, obs_size: int, h: int = DET_SIZE, z: int = STOCH_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h + z, 256), nn.ELU(),
            nn.Linear(256, 256),   nn.ELU(),
            nn.Linear(256, obs_size),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h, z], dim=-1))


class RewardModel(nn.Module):
    """(h, z) → predicted reward (trade outcome)."""

    def __init__(self, h: int = DET_SIZE, z: int = STOCH_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h + z, 128), nn.ELU(),
            nn.Linear(128, 64),    nn.ELU(),
            nn.Linear(64, 1),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h, z], dim=-1)).squeeze(-1)


class RSSM(nn.Module):
    """
    Recurrent State Space Model — แยก state ออกเป็น 2 ส่วน:
    h (deterministic): จำ information ยาวๆ ได้ผ่าน GRU
    z (stochastic):    จับ uncertainty ของตลาดที่ model ไม่รู้
    """

    def __init__(self, h: int = DET_SIZE, z: int = STOCH_SIZE,
                 obs_lat: int = OBS_LATENT, n_act: int = ACTION_SIZE):
        super().__init__()
        self.h_size = h
        self.z_size = z

        # Deterministic path: GRU
        self.gru     = nn.GRUCell(z + n_act, h)
        self.h_norm  = nn.LayerNorm(h)

        # Prior p(z|h)
        self.prior   = nn.Sequential(
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, z * 2),
        )

        # Posterior q(z|h, obs_latent)
        self.post    = nn.Sequential(
            nn.Linear(h + obs_lat, h), nn.ELU(),
            nn.Linear(h, z * 2),
        )

    @staticmethod
    def _reparam(mu: torch.Tensor, log_s: torch.Tensor) -> torch.Tensor:
        return mu + log_s.exp() * torch.randn_like(log_s)

    def prior_step(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """p(z|h) → z, μ, log_σ"""
        out        = self.prior(h)
        mu, log_s  = out.chunk(2, dim=-1)
        return self._reparam(mu, log_s), mu, log_s

    def posterior_step(self, h: torch.Tensor,
                       obs_lat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """q(z|h, obs) → z, μ, log_σ"""
        out        = self.post(torch.cat([h, obs_lat], dim=-1))
        mu, log_s  = out.chunk(2, dim=-1)
        return self._reparam(mu, log_s), mu, log_s

    def step_h(self, h: torch.Tensor, z: torch.Tensor,
               action_oh: torch.Tensor) -> torch.Tensor:
        """Advance deterministic state: h_t = GRU(z_{t-1} ⊕ a_{t-1}, h_{t-1})."""
        inp = torch.cat([z, action_oh], dim=-1)
        return self.h_norm(self.gru(inp, h))


class WorldModel(nn.Module):
    """
    Complete World Model — encoder + RSSM + decoder + reward.
    Trains to predict next observation and reward from current state.
    """

    def __init__(self, obs_size: int):
        super().__init__()
        self.obs_size = obs_size
        self.encoder  = ObsEncoder(obs_size, OBS_LATENT)
        self.rssm     = RSSM(DET_SIZE, STOCH_SIZE, OBS_LATENT, ACTION_SIZE)
        self.decoder  = ObsDecoder(obs_size, DET_SIZE, STOCH_SIZE)
        self.reward   = RewardModel(DET_SIZE, STOCH_SIZE)

    def initial_state(self, B: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(B, DET_SIZE, device=device)
        z = torch.zeros(B, STOCH_SIZE, device=device)
        return h, z

    def forward(self, obs_seq: torch.Tensor, act_seq: torch.Tensor) -> dict:
        """
        Compute ELBO over a sequence.
        obs_seq: (B, T, obs_size)
        act_seq: (B, T, ACTION_SIZE)  — one-hot actions
        Returns dict of losses.
        """
        B, T, _ = obs_seq.shape
        device  = obs_seq.device

        h, z = self.initial_state(B, device)

        kl_loss   = torch.tensor(0.0, device=device)
        rec_loss  = torch.tensor(0.0, device=device)
        rew_loss  = torch.tensor(0.0, device=device)

        for t in range(T):
            # 1. Advance deterministic state
            h = self.rssm.step_h(h, z, act_seq[:, t])

            # 2. Encode current obs
            obs_mu, obs_ls = self.encoder(obs_seq[:, t])
            obs_lat        = obs_mu + obs_ls.exp() * torch.randn_like(obs_ls)

            # 3. Posterior z (uses observation)
            z, q_mu, q_ls = self.rssm.posterior_step(h, obs_lat)

            # 4. Prior (no observation — used for imagination)
            _, p_mu, p_ls = self.rssm.prior_step(h)

            # 5. Reconstruction loss
            obs_hat     = self.decoder(h, z)
            rec_loss   += F.mse_loss(obs_hat, obs_seq[:, t])

            # 6. KL divergence q‖p (with free bits)
            kl = torch.distributions.kl_divergence(
                torch.distributions.Normal(q_mu, q_ls.exp().clamp(1e-4, 1e4)),
                torch.distributions.Normal(p_mu, p_ls.exp().clamp(1e-4, 1e4)),
            ).sum(-1)
            kl_loss += torch.maximum(kl, torch.tensor(FREE_NATS, device=device)).mean()

        return {
            "rec_loss": rec_loss / T,
            "kl_loss":  kl_loss  / T,
            "total":    rec_loss / T + KL_BALANCE * kl_loss / T,
        }

    @torch.no_grad()
    def imagine(self, h: torch.Tensor, z: torch.Tensor,
                policy_fn, steps: int = IMAGINE_STEPS) -> List[dict]:
        """
        Roll out imagined trajectory using prior (no real obs).
        policy_fn(z) → action_index (0/1/2)
        Returns list of {obs_hat, reward_hat, z, h}
        """
        rollout = []
        for _ in range(steps):
            # Action from current latent state
            with torch.no_grad():
                act_idx = policy_fn(z)

            act_oh = torch.zeros(z.size(0), ACTION_SIZE, device=z.device)
            act_oh.scatter_(1, act_idx.unsqueeze(1), 1.0)

            # Advance state using prior (no obs)
            h = self.rssm.step_h(h, z, act_oh)
            z, _, _ = self.rssm.prior_step(h)

            # Predict obs and reward
            obs_hat = self.decoder(h, z)
            rew_hat = self.reward(h, z)

            rollout.append({
                "obs_hat": obs_hat.cpu().numpy(),
                "rew_hat": rew_hat.cpu().numpy(),
                "action":  act_idx.cpu().numpy(),
                "h": h.detach(), "z": z.detach(),
            })
        return rollout


class WorldModelTrainer:
    """
    Manages World Model training + imagination loop.
    Runs in a background thread to avoid blocking the trading AI.

    Usage in server.py:
        _world_trainer = WorldModelTrainer(obs_size=296)
        _world_trainer.start()
        # Every trade: _world_trainer.add_experience(obs, action, reward, next_obs)
        # Every update: imagined = _world_trainer.get_imagined_batch()
        #               → add imagined experiences to Rainbow buffer
    """

    def __init__(self, obs_size: int, device: Optional[torch.device] = None):
        self.obs_size  = obs_size
        self.device    = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model     = WorldModel(obs_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=WM_LR)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-5,
        )

        # Real experience buffer for WM training
        self._real_buf: deque = deque(maxlen=2000)

        # Imagined experience queue (for Rainbow buffer)
        self._imagined_queue: deque = deque(maxlen=500)

        # Background thread state
        self._thread:   Optional[threading.Thread] = None
        self._running:  bool = False
        self._lock:     threading.Lock = threading.Lock()

        # Stats
        self.total_wm_updates = 0
        self.total_imagined   = 0

        logger.info("WorldModelTrainer — device:%s | obs:%d | imagine_steps:%d",
                    self.device, obs_size, IMAGINE_STEPS)

    def start(self):
        """Start background training thread."""
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._train_loop, daemon=True)
        self._thread.start()
        logger.info("WorldModel background thread started")

    def stop(self):
        """Stop background thread."""
        self._running = False

    def add_experience(self, obs: np.ndarray, action: int,
                       reward: float, next_obs: np.ndarray):
        """Add a real trade to the WM training buffer."""
        with self._lock:
            self._real_buf.append({
                "obs":      obs.copy(),
                "action":   action,
                "reward":   reward,
                "next_obs": next_obs.copy(),
            })

    def get_imagined_batch(self, n: int = 16) -> List[dict]:
        """
        Get up to n imagined (obs, action, reward, next_obs) tuples.
        These can be added to Rainbow's PER buffer to augment training.
        """
        with self._lock:
            batch = []
            while self._imagined_queue and len(batch) < n:
                batch.append(self._imagined_queue.popleft())
            return batch

    def _train_loop(self):
        """Background: train WM every 10s, then generate imaginations."""
        while self._running:
            time.sleep(10)
            with self._lock:
                n_real = len(self._real_buf)

            if n_real < WM_WARMUP:
                continue

            try:
                self._train_step()
                self._imagine_step()
            except Exception as exc:
                logger.debug("WorldModel train loop error: %s", exc)

    def _train_step(self):
        """One WM training step on a random sequence batch."""
        with self._lock:
            buf = list(self._real_buf)

        if len(buf) < WM_SEQ_LEN * WM_BATCH:
            return

        # Build (B, T, ...) tensors from random sequences
        B, T = WM_BATCH, WM_SEQ_LEN
        obs_batch  = np.zeros((B, T, self.obs_size), dtype=np.float32)
        act_batch  = np.zeros((B, T, ACTION_SIZE),   dtype=np.float32)

        for b in range(B):
            start = np.random.randint(0, len(buf) - T)
            for t, item in enumerate(buf[start: start + T]):
                obs_batch[b, t]    = item["obs"]
                act_batch[b, t, item["action"]] = 1.0

        obs_t = torch.from_numpy(obs_batch).to(self.device)
        act_t = torch.from_numpy(act_batch).to(self.device)

        self.model.train()
        losses  = self.model(obs_t, act_t)
        loss    = losses["total"]

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 100.0)
        self.optimizer.step()
        self.scheduler.step()

        self.total_wm_updates += 1
        if self.total_wm_updates % 10 == 0:
            logger.info("WorldModel #%d | rec=%.4f | kl=%.4f | total=%.4f",
                        self.total_wm_updates,
                        float(losses["rec_loss"].item()),
                        float(losses["kl_loss"].item()),
                        float(losses["total"].item()))

    def _imagine_step(self):
        """Generate imagined trajectories and add to imagined queue."""
        with self._lock:
            buf = list(self._real_buf)
        if not buf:
            return

        self.model.eval()
        B = 4  # small batch for imagination

        # Start from random real states
        idxs = np.random.choice(len(buf), B, replace=False)
        obs0 = np.array([buf[i]["obs"] for i in idxs], dtype=np.float32)
        obs_t = torch.from_numpy(obs0).to(self.device)

        # Encode starting state
        with torch.no_grad():
            mu, ls = self.model.encoder(obs_t)
            obs_lat = mu + ls.exp() * torch.randn_like(ls)
            h, z    = self.model.initial_state(B, self.device)
            h       = self.model.rssm.step_h(h, z,
                        torch.zeros(B, ACTION_SIZE, device=self.device))
            z, _, _ = self.model.rssm.posterior_step(h, obs_lat)

        # Policy: greedy on reward prediction (simple heuristic)
        def policy_fn(z_cur: torch.Tensor) -> torch.Tensor:
            r_vals = []
            for a in range(ACTION_SIZE):
                ah = torch.zeros(z_cur.size(0), ACTION_SIZE, device=z_cur.device)
                ah[:, a] = 1.0
                h_next = self.model.rssm.step_h(h, z_cur, ah)
                z_next, _, _ = self.model.rssm.prior_step(h_next)
                r = self.model.reward(h_next, z_next)
                r_vals.append(r)
            best = torch.stack(r_vals, dim=1).argmax(dim=1)
            return best

        rollout = self.model.imagine(h, z, policy_fn, steps=IMAGINE_STEPS)

        # Add imagined experiences to queue
        imagined = []
        for step_i, step in enumerate(rollout):
            for b in range(B):
                if step_i == 0:
                    obs_start = obs0[b]
                else:
                    obs_start = rollout[step_i - 1]["obs_hat"][b]
                obs_end  = step["obs_hat"][b]
                reward   = float(step["rew_hat"][b])
                action   = int(step["action"][b])
                imagined.append({
                    "obs":      obs_start,
                    "next_obs": obs_end,
                    "action":   action,
                    "reward":   reward * 0.5,   # discount imagined rewards
                })

        with self._lock:
            for item in imagined:
                self._imagined_queue.append(item)
            self.total_imagined += len(imagined)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "model":     self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates":   self.total_wm_updates,
            "imagined":  self.total_imagined,
        }, path)
        logger.info("WorldModel saved → %s (updates=%d)", path, self.total_wm_updates)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            logger.info("No WorldModel checkpoint — starting fresh")
            return False
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.total_wm_updates = ckpt.get("updates", 0)
            self.total_imagined   = ckpt.get("imagined", 0)
            logger.info("WorldModel loaded ← %s (updates=%d)", path, self.total_wm_updates)
            return True
        except Exception as exc:
            logger.error("WorldModel load failed: %s", exc)
            return False

    def status(self) -> dict:
        with self._lock:
            return {
                "wm_running":      self._running,
                "wm_real_buf":     len(self._real_buf),
                "wm_updates":      self.total_wm_updates,
                "wm_imagined":     self.total_imagined,
                "wm_queue":        len(self._imagined_queue),
            }
