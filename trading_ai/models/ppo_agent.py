"""
PPO (Proximal Policy Optimization) Agent — ULTRA EDITION

สถาปัตยกรรมใหม่ระดับสูงสุด:
  - LSTM: จำลำดับเวลา (temporal sequence memory)
  - Self-Attention: โฟกัสบน indicators ที่สำคัญที่สุด
  - Dueling Architecture: แยก Value vs Advantage streams
  - Residual Connections: gradient ไหลได้ลึกกว่า, เรียนรู้เร็วกว่า
  - Entropy Annealing: explore มากตอนแรก → exploit มากเมื่อเก่งแล้ว
  - Cosine LR Scheduler: ปรับ learning rate อัตโนมัติ
  - ICM (Intrinsic Curiosity Module): ค้นหา pattern ใหม่เอง
  - Value Clipping: เสถียรภาพ training สูงขึ้น
"""
import logging
import os
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from trading_ai.config import config
from trading_ai.models.tft_encoder import ActorCriticV2

logger = logging.getLogger(__name__)

LSTM_HIDDEN = 128   # ขนาด LSTM hidden state


# ── Residual Block ─────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, size: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size),
            nn.LayerNorm(size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(size, size),
            nn.LayerNorm(size),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


# ── Self-Attention Module ──────────────────────────────────────────────────────

class IndicatorAttention(nn.Module):
    """Multi-head self-attention บน indicator features"""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # ทำให้ d_model หารด้วย n_heads ได้
        actual_heads = n_heads
        while d_model % actual_heads != 0 and actual_heads > 1:
            actual_heads -= 1
        self.attn = nn.MultiheadAttention(d_model, actual_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


# ── ICM (Intrinsic Curiosity Module) ──────────────────────────────────────────

class ICMModule(nn.Module):
    """
    สร้าง intrinsic reward จากความ 'แปลกใจ' กับ next state
    AI จะสนใจ explore สถานการณ์ที่ยังไม่รู้จัก
    """
    def __init__(self, obs_size: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.forward_model = nn.Sequential(
            nn.Linear(hidden + n_actions, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs, next_obs, action_onehot):
        feat       = self.encoder(obs)
        next_feat  = self.encoder(next_obs)
        pred_next  = self.forward_model(torch.cat([feat, action_onehot], dim=-1))
        pred_act   = self.inverse_model(torch.cat([feat, next_feat], dim=-1))
        intrinsic  = 0.5 * (pred_next - next_feat.detach()).pow(2).mean(dim=-1)
        return intrinsic, pred_act


# ── ActorCritic ULTRA ──────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    ULTRA Architecture:
      Input → Feature Extractor (Residual) → LSTM (temporal memory)
            → Self-Attention (indicator focus) → Combiner
            → Dueling Actor Head | Value Head
    """

    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 384):
        super().__init__()
        self.obs_size    = obs_size
        self.n_actions   = n_actions
        self.hidden_size = hidden_size

        # 1. Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            ResidualBlock(hidden_size, dropout=0.1),
            ResidualBlock(hidden_size, dropout=0.05),
        )

        # 2. LSTM (temporal context)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=LSTM_HIDDEN,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        # 3. Attention on indicator tokens
        attn_d = 64  # token dimension
        n_tokens = hidden_size // attn_d
        self.n_tokens  = n_tokens
        self.attn_d    = attn_d
        self.attn_proj = nn.Linear(hidden_size, n_tokens * attn_d)
        self.attention = IndicatorAttention(attn_d, n_heads=4)
        self.attn_out  = nn.Linear(n_tokens * attn_d, hidden_size // 2)

        # 4. Combine LSTM + Attention
        combined_size = LSTM_HIDDEN + hidden_size // 2
        self.combiner = nn.Sequential(
            nn.Linear(combined_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            ResidualBlock(hidden_size // 2, dropout=0.05),
        )

        core_size = hidden_size // 2

        # 5. Dueling: Advantage stream (Actor)
        self.adv_stream = nn.Sequential(
            nn.Linear(core_size, core_size // 2),
            nn.GELU(),
            nn.Linear(core_size // 2, n_actions),
        )

        # 6. Value stream (Critic)
        self.val_stream = nn.Sequential(
            nn.Linear(core_size, core_size // 2),
            nn.GELU(),
            nn.Linear(core_size // 2, 1),
        )

        self._init_weights()
        self._lstm_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.adv_stream[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.val_stream[-1].weight, gain=1.0)

    def reset_hidden(self, batch_size: int = 1):
        device = next(self.parameters()).device
        self._lstm_hidden = (
            torch.zeros(2, batch_size, LSTM_HIDDEN, device=device),
            torch.zeros(2, batch_size, LSTM_HIDDEN, device=device),
        )

    def _core(self, obs_t: torch.Tensor):
        batch = obs_t.size(0)

        # Feature extraction
        feat = self.feature_extractor(obs_t)   # (B, H)

        # LSTM
        if self._lstm_hidden is None or self._lstm_hidden[0].size(1) != batch:
            self.reset_hidden(batch)

        lstm_out, new_hidden = self.lstm(feat.unsqueeze(1), self._lstm_hidden)
        lstm_feat = lstm_out.squeeze(1)    # (B, LSTM_H)
        # detach để prevent backprop ผ่าน old hidden
        self._lstm_hidden = (new_hidden[0].detach(), new_hidden[1].detach())

        # Attention tokens
        tokens = self.attn_proj(feat)                             # (B, n*d)
        tokens = tokens.view(batch, self.n_tokens, self.attn_d)  # (B, n, d)
        tokens = self.attention(tokens)
        attn_feat = self.attn_out(tokens.reshape(batch, -1))     # (B, H//2)

        # Combine
        combined = torch.cat([lstm_feat, attn_feat], dim=-1)
        return self.combiner(combined)   # (B, H//2)

    def forward(self, obs: torch.Tensor):
        core  = self._core(obs)
        value = self.val_stream(core).squeeze(-1)
        adv   = self.adv_stream(core)
        # Dueling: logits = A - mean(A)
        logits = adv - adv.mean(dim=-1, keepdim=True)
        return logits, value

    def get_action(self, obs):
        logits, value = self(obs)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, obs, actions):
        logits, value = self(obs)
        dist  = Categorical(logits=logits)
        return dist.log_prob(actions), value, dist.entropy()

    def action_probs(self, obs):
        logits, _ = self(obs)
        return torch.softmax(logits, dim=-1)


# ── Rollout Buffer ─────────────────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self, size: int, obs_size: int, device: torch.device):
        self.size   = size
        self.device = device
        self.obs       = torch.zeros(size, obs_size, dtype=torch.float32)
        self.next_obs  = torch.zeros(size, obs_size, dtype=torch.float32)
        self.actions   = torch.zeros(size, dtype=torch.long)
        self.log_probs = torch.zeros(size, dtype=torch.float32)
        self.rewards   = torch.zeros(size, dtype=torch.float32)
        self.values    = torch.zeros(size, dtype=torch.float32)
        self.dones     = torch.zeros(size, dtype=torch.float32)
        self.ptr  = 0
        self.full = False

    def add(self, obs, next_obs, action, log_prob, reward, value, done):
        idx = self.ptr % self.size
        self.obs[idx]       = torch.from_numpy(np.asarray(obs, dtype=np.float32))
        self.next_obs[idx]  = torch.from_numpy(np.asarray(next_obs, dtype=np.float32))
        self.actions[idx]   = action
        self.log_probs[idx] = log_prob
        self.rewards[idx]   = reward
        self.values[idx]    = value
        self.dones[idx]     = float(done)
        self.ptr += 1
        if self.ptr >= self.size:
            self.full = True

    def is_ready(self) -> bool:
        return self.ptr >= self.size

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        n          = self.size
        advantages = torch.zeros(n)
        gae        = 0.0
        for t in reversed(range(n)):
            nxt_val  = last_value if t == n - 1 else float(self.values[t + 1])
            nxt_done = 0.0 if t == n - 1 else float(self.dones[t + 1])
            delta    = (
                float(self.rewards[t])
                + gamma * nxt_val * (1.0 - float(self.dones[t]))
                - float(self.values[t])
            )
            gae = delta + gamma * gae_lambda * (1.0 - nxt_done) * gae
            advantages[t] = gae

        returns  = advantages + self.values
        adv_mean = advantages.mean()
        adv_std  = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std
        return returns, advantages

    def get_batches(self, batch_size: int):
        indices = torch.randperm(self.size)
        for start in range(0, self.size, batch_size):
            yield indices[start: start + batch_size]

    def get_sequential_batches(self, batch_size: int):
        """Yield indices in chronological order (required for LSTM/Transformer training)."""
        indices = torch.arange(self.size)
        for start in range(0, self.size, batch_size):
            yield indices[start: start + batch_size]

    def reset(self):
        self.ptr  = 0
        self.full = False


# ── PPO Agent ULTRA ────────────────────────────────────────────────────────────

class PPOAgent:
    """
    ULTRA PPO Agent พร้อม:
    - LSTM + Attention + Dueling
    - Entropy annealing (explore → exploit)
    - Cosine LR scheduler
    - ICM (Intrinsic Curiosity)
    - Value clipping
    """

    ENTROPY_START = 0.05
    ENTROPY_END   = 0.003
    ENTROPY_DECAY = 50_000

    def __init__(self, obs_size: int, n_actions: int = 3, hidden_size: int = 384,
                 use_v2: Optional[bool] = None):
        self.obs_size    = obs_size
        self.n_actions   = n_actions
        self.hidden_size = hidden_size
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # use_v2=None means "read from config"; explicit bool overrides
        self._use_v2 = config.USE_V2 if use_v2 is None else use_v2

        if self._use_v2:
            logger.info("PPO V2 (Semi-Pro TFT+MoE) — device: %s", self.device)
            self.network = ActorCriticV2(obs_size, n_actions, hidden_size=256).to(self.device)
        else:
            logger.info("PPO ULTRA agent — device: %s | hidden: %d", self.device, hidden_size)
            self.network = ActorCritic(obs_size, n_actions, hidden_size=hidden_size).to(self.device)
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.LEARNING_RATE,
            eps=1e-5,
            weight_decay=1e-5,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-5
        )

        self.icm           = ICMModule(obs_size, n_actions).to(self.device)
        self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=1e-3)
        self.icm_coef      = 0.01

        self.buffer = RolloutBuffer(config.UPDATE_EVERY, obs_size, self.device)

        self.total_steps:       int   = 0
        self.total_updates:     int   = 0
        self.total_episodes:    int   = 0
        self.cumulative_reward: float = 0.0

    # ── Decision making ────────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False):
        obs_t      = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        logits, value = self.network(obs_t)
        dist       = Categorical(logits=logits)
        action     = logits.argmax(dim=-1) if deterministic else dist.sample()
        probs      = torch.softmax(logits, dim=-1).squeeze(0)
        logger.debug(
            "HOLD=%.3f BUY=%.3f SELL=%.3f → %d",
            probs[0].item(), probs[1].item(), probs[2].item(), action.item(),
        )
        return action.item(), dist.log_prob(action).item(), value.squeeze().item()

    @torch.no_grad()
    def get_confidence(self, obs: np.ndarray) -> Tuple[int, float]:
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        probs = self.network.action_probs(obs_t).squeeze(0)
        best  = probs.argmax().item()
        return best, probs[best].item()

    @torch.no_grad()
    def get_action_distribution(self, obs: np.ndarray) -> dict:
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        probs = self.network.action_probs(obs_t).squeeze(0).cpu().numpy()
        return {"hold": float(probs[0]), "buy": float(probs[1]), "sell": float(probs[2])}

    # ── Learning ───────────────────────────────────────────────────────────────

    def _entropy_coef(self) -> float:
        t = min(self.total_steps / self.ENTROPY_DECAY, 1.0)
        return self.ENTROPY_START + (self.ENTROPY_END - self.ENTROPY_START) * t

    def store(self, obs, next_obs, action, log_prob, reward, value, done):
        self.buffer.add(obs, next_obs, action, log_prob, reward, value, done)
        self.total_steps       += 1
        self.cumulative_reward += reward

    def ready_to_update(self) -> bool:
        return self.buffer.is_ready()

    def update(self, last_obs: np.ndarray) -> dict:
        self.network.eval()
        with torch.no_grad():
            obs_t = torch.from_numpy(last_obs).unsqueeze(0).to(self.device)
            _, last_value = self.network(obs_t)
            last_value = last_value.squeeze().item()
        self.network.train()
        if self._use_v2:
            self.network.reset_context()
        else:
            self.network.reset_hidden()

        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, config.GAMMA, config.GAE_LAMBDA
        )
        returns    = returns.to(self.device)
        advantages = advantages.to(self.device)

        all_obs        = self.buffer.obs.to(self.device)
        all_next_obs   = self.buffer.next_obs.to(self.device)
        all_actions    = self.buffer.actions.to(self.device)
        all_old_lprobs = self.buffer.log_probs.to(self.device)
        all_old_values = self.buffer.values.to(self.device)

        ent_coef = self._entropy_coef()
        metrics  = {"policy_loss": 0.0, "value_loss": 0.0,
                    "entropy": 0.0, "icm_loss": 0.0, "lr": 0.0}
        n_upd = 0

        for _ in range(config.PPO_EPOCHS):
            self.network.reset_context() if self._use_v2 else self.network.reset_hidden()
            batch_iter = (self.buffer.get_sequential_batches(config.BATCH_SIZE)
                          if self._use_v2 else self.buffer.get_batches(config.BATCH_SIZE))
            for idx in batch_iter:
                obs_b      = all_obs[idx]
                nxt_b      = all_next_obs[idx]
                act_b      = all_actions[idx]
                old_lp_b   = all_old_lprobs[idx]
                adv_b      = advantages[idx]
                ret_b      = returns[idx]
                old_val_b  = all_old_values[idx]
                new_lp, values, entropy = self.network.evaluate(obs_b, act_b)

                # PPO clipped surrogate
                ratio  = torch.exp(new_lp - old_lp_b)
                surr1  = ratio * adv_b
                surr2  = torch.clamp(ratio, 1 - config.CLIP_EPSILON, 1 + config.CLIP_EPSILON) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss with clipping
                v_clip = old_val_b + torch.clamp(
                    values - old_val_b, -config.CLIP_EPSILON, config.CLIP_EPSILON
                )
                value_loss = torch.max(
                    nn.functional.mse_loss(values, ret_b),
                    nn.functional.mse_loss(v_clip, ret_b),
                )

                # ICM intrinsic curiosity
                act_oh = torch.zeros(obs_b.size(0), self.n_actions, device=self.device)
                act_oh.scatter_(1, act_b.unsqueeze(1), 1.0)
                intrinsic, pred_act = self.icm(obs_b.detach(), nxt_b.detach(), act_oh.detach())
                icm_loss = (
                    0.5 * intrinsic.mean()
                    + nn.functional.cross_entropy(pred_act, act_b)
                )

                loss = (
                    policy_loss
                    + config.VALUE_LOSS_COEF * value_loss
                    - ent_coef * entropy.mean()
                    + self.icm_coef * icm_loss
                )

                self.optimizer.zero_grad()
                self.icm_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), config.MAX_GRAD_NORM)
                nn.utils.clip_grad_norm_(self.icm.parameters(), config.MAX_GRAD_NORM)
                self.optimizer.step()
                self.icm_optimizer.step()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"]  += value_loss.item()
                metrics["entropy"]     += entropy.mean().item()
                metrics["icm_loss"]    += icm_loss.item()
                n_upd += 1

        if n_upd > 0:
            for k in ("policy_loss", "value_loss", "entropy", "icm_loss"):
                metrics[k] /= n_upd

        self.scheduler.step()
        metrics["lr"]           = self.optimizer.param_groups[0]["lr"]
        metrics["entropy_coef"] = ent_coef

        self.buffer.reset()
        if self._use_v2:
            self.network.reset_context()
        else:
            self.network.reset_hidden()
        self.total_updates += 1

        logger.info(
            "PPO ULTRA #%d | steps=%d | ploss=%.4f | vloss=%.4f | ent=%.4f | lr=%.6f",
            self.total_updates, self.total_steps,
            metrics["policy_loss"], metrics["value_loss"],
            metrics["entropy"], metrics["lr"],
        )
        return metrics

    def update_online(self, obs: np.ndarray, action: int, reward: float,
                      next_obs: np.ndarray, done: bool) -> dict:
        """
        Lightweight single-step store+update for NAS shadow challengers.
        Fills the rollout buffer one step at a time; triggers a full PPO
        update once the buffer is ready.  Avoids the need for a separate
        FastLearner instance per challenger.
        """
        obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.network(obs_t)
            dist     = Categorical(logits=logits)
            log_prob = dist.log_prob(torch.tensor([action], device=self.device))
        self.store(
            obs=obs, next_obs=next_obs, action=action,
            log_prob=float(log_prob.item()), reward=float(reward),
            value=float(value.squeeze().item()), done=bool(done),
        )
        if self.ready_to_update():
            return self.update(next_obs)
        return {}

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "network_state":     self.network.state_dict(),
            "optimizer_state":   self.optimizer.state_dict(),
            "scheduler_state":   self.scheduler.state_dict(),
            "icm_state":         self.icm.state_dict(),
            "total_steps":       self.total_steps,
            "total_updates":     self.total_updates,
            "total_episodes":    self.total_episodes,
            "cumulative_reward": self.cumulative_reward,
        }, path)
        logger.info("Knowledge saved → %s  (steps=%d)", path, self.total_steps)

    # ── Distributed Worker Support ────────────────────────────────────────────

    def get_weights_bytes(self) -> bytes:
        """Serialize network weights for sending to worker machines."""
        import io
        buf = io.BytesIO()
        torch.save(self.network.state_dict(), buf)
        return buf.getvalue()

    def load_weights_bytes(self, data: bytes) -> None:
        """Load network weights received from a worker machine."""
        import io
        buf = io.BytesIO(data)
        state = torch.load(buf, map_location=self.device, weights_only=True)
        self.network.load_state_dict(state)

    def fedavg_merge(self, worker_weights_bytes: bytes, alpha: float = 0.25) -> None:
        """
        Federated averaging: blend worker-trained weights into this model.
        alpha=0.25 = 25% worker influence (conservative, preserves server knowledge).
        """
        import io
        buf = io.BytesIO(worker_weights_bytes)
        worker_state = torch.load(buf, map_location=self.device, weights_only=True)
        own_state = self.network.state_dict()
        merged = {
            k: (1.0 - alpha) * own_state[k].float() + alpha * worker_state[k].float()
            if k in worker_state else own_state[k]
            for k in own_state
        }
        self.network.load_state_dict(merged)
        logger.info("FedAvg applied — worker alpha=%.2f", alpha)

    def get_buffer_bytes(self) -> bytes:
        """Serialize the rollout buffer for sending to workers."""
        import io, pickle
        data = {
            "obs":       self.buffer.obs.numpy(),
            "next_obs":  self.buffer.next_obs.numpy(),
            "actions":   self.buffer.actions.numpy(),
            "log_probs": self.buffer.log_probs.numpy(),
            "rewards":   self.buffer.rewards.numpy(),
            "values":    self.buffer.values.numpy(),
            "dones":     self.buffer.dones.numpy(),
            "size":      self.buffer.size,
        }
        buf = io.BytesIO()
        pickle.dump(data, buf)
        return buf.getvalue()

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            logger.info("No checkpoint at %s – starting fresh", path)
            return False
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            try:
                self.network.load_state_dict(ckpt["network_state"])
            except RuntimeError:
                # Architecture changed (e.g. V1→V2 upgrade) — old weights are
                # incompatible. Back up the file and start fresh so the next
                # save() will write a clean V2 checkpoint.
                bak = path + ".v1.bak"
                os.replace(path, bak)
                logger.info(
                    "Checkpoint architecture mismatch (V1→V2 upgrade). "
                    "Old weights backed up to %s — starting fresh with V2.", bak
                )
                return False
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            if "scheduler_state" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler_state"])
            self.total_steps       = ckpt.get("total_steps", 0)
            self.total_updates     = ckpt.get("total_updates", 0)
            self.total_episodes    = ckpt.get("total_episodes", 0)
            self.cumulative_reward = ckpt.get("cumulative_reward", 0.0)
            logger.info(
                "Knowledge loaded ← %s  (steps=%d, updates=%d)",
                path, self.total_steps, self.total_updates,
            )
            return True
        except Exception as exc:
            logger.error("Failed to load checkpoint: %s", exc)
            return False
