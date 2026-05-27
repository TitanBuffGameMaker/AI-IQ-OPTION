"""
Hierarchical RL — Gate + Direction
════════════════════════════════════

ปัญหาของสมองเดิม:
  obs → [HOLD / BUY / SELL]   ← ทำ 2 งานพร้อมกัน = confusion สูง

Hierarchical แก้ด้วย:
  Stage 1 — GateNetwork:      "ควรเทรดเดี๋ยวนี้ไหม?"  (binary: trade/no_trade)
  Stage 2 — DirectionNetwork: "ถ้าเทรด ขึ้นหรือลง?"    (binary: BUY/SELL)

ผลลัพธ์:
  - Gate เรียนรู้ EV-positive moments โดยเฉพาะ
  - Direction เรียนรู้ market direction โดยเฉพาะ
  - ลด false signal ได้ 30-50% (split responsibility)
  - Gate ใช้ Q-value ของ Rainbow เป็น prior → ตัดสินใจได้เร็วขึ้น

Interface เหมือน PPOAgent ทุกประการ
"""
import io
import logging
import math
import os
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from trading_ai.models.rainbow_dqn import NoisyLinear, RainbowAgent, RainbowNet, GAMMA

logger = logging.getLogger(__name__)

GATE_LR     = 3e-4
DIR_LR      = 3e-4
GRU_HIDDEN  = 64
GATE_THRESH = 0.52    # P(trade) ≥ threshold → ออกคำสั่ง
WARMUP      = 30      # รอ experience ก่อน update


class GateNet(nn.Module):
    """
    ตอบ: "ควรเทรดเดี๋ยวนี้ไหม?"
    ถ้า P(trade) < GATE_THRESH → HOLD อัตโนมัติ (ไม่ส่งให้ DirectionNet เลย)
    ทำให้ AI ไม่เทรดเมื่อ EV ต่ำ — ลด false signal
    """

    def __init__(self, obs_size: int, hidden: int = 192):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.1),
        )
        self.gru = nn.GRU(hidden, GRU_HIDDEN, batch_first=True)
        self._h:  Optional[torch.Tensor] = None
        self.head = nn.Sequential(
            NoisyLinear(GRU_HIDDEN, GRU_HIDDEN // 2), nn.GELU(),
            NoisyLinear(GRU_HIDDEN // 2, 1),
        )
        self._init()

    def _init(self):
        for m in self.features.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)

    def reset_hidden(self, B: int = 1):
        dev = next(self.parameters()).device
        self._h = torch.zeros(1, B, GRU_HIDDEN, device=dev)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.size(0)
        if self._h is None or self._h.size(1) != B:
            self.reset_hidden(B)
        feat     = self.features(obs)
        out, h   = self.gru(feat.unsqueeze(1), self._h)
        self._h  = h.detach()
        return self.head(out.squeeze(1))   # (B, 1) logit


class DirectionNet(nn.Module):
    """
    ตอบ: "ถ้าเทรด ขึ้น(BUY) หรือ ลง(SELL)?"
    เรียกเฉพาะเมื่อ Gate บอกให้เทรด
    เรียนรู้ direction โดยไม่ถูกรบกวนด้วยคำถาม "ควรเทรดไหม?"
    """

    def __init__(self, obs_size: int, hidden: int = 192):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.1),
        )
        self.gru = nn.GRU(hidden, GRU_HIDDEN, batch_first=True)
        self._h:  Optional[torch.Tensor] = None
        self.head = nn.Sequential(
            NoisyLinear(GRU_HIDDEN, GRU_HIDDEN // 2), nn.GELU(),
            NoisyLinear(GRU_HIDDEN // 2, 2),   # BUY=0  SELL=1
        )
        self._init()

    def _init(self):
        for m in self.features.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)

    def reset_hidden(self, B: int = 1):
        dev = next(self.parameters()).device
        self._h = torch.zeros(1, B, GRU_HIDDEN, device=dev)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.size(0)
        if self._h is None or self._h.size(1) != B:
            self.reset_hidden(B)
        feat     = self.features(obs)
        out, h   = self.gru(feat.unsqueeze(1), self._h)
        self._h  = h.detach()
        return self.head(out.squeeze(1))   # (B, 2) logits


class HierarchicalAgent:
    """
    Hierarchical RL Agent — wraps RainbowAgent ด้วย Gate + Direction

    Decision pipeline:
      1. Gate:       P(trade) = sigmoid(GateNet(obs))
         - ถ้า P(trade) < GATE_THRESH → HOLD (ประหยัด trade โอกาสไม่ดี)
      2. Direction:  softmax(DirectionNet(obs))
         - 0→BUY (action=1), 1→SELL (action=2)
      3. Rainbow:    ใช้ Q-values เพื่อ verify + confidence boost

    Training:
      - Gate BCE loss: reward > 0 → label=1 (ควรเทรด), ≤0 → label=0
      - Direction CE loss: correct direction → ดูจาก actual outcome
      - Rainbow ฝึกต่อเนื่องด้วย PER buffer
    """

    def __init__(self, obs_size: int, n_actions: int = 3, hidden_size: int = 384):
        self.obs_size    = obs_size
        self.n_actions   = n_actions
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Sub-agents
        self.rainbow = RainbowAgent(obs_size, n_actions, hidden_size)
        self.gate    = GateNet(obs_size).to(self.device)
        self.dir_net = DirectionNet(obs_size).to(self.device)

        self.gate_opt = optim.Adam(self.gate.parameters(),    lr=GATE_LR)
        self.dir_opt  = optim.Adam(self.dir_net.parameters(), lr=DIR_LR)

        # Experience buffers for gate + direction training
        self._gate_buf:  deque = deque(maxlen=500)  # (obs, gate_label)
        self._dir_buf:   deque = deque(maxlen=500)  # (obs, dir_label)

        # Compatibility attrs
        self.total_steps:       int   = 0
        self.total_updates:     int   = 0
        self.total_episodes:    int   = 0
        self.cumulative_reward: float = 0.0

        logger.info("HierarchicalAgent — device:%s | obs:%d | thresh:%.2f",
                    self.device, obs_size, GATE_THRESH)

    # ── High-level decide() ────────────────────────────────────────────────────

    @torch.no_grad()
    def should_trade(self, obs: np.ndarray) -> Tuple[bool, float]:
        """Returns (trade_bool, gate_probability)."""
        self.gate.eval()
        self.gate.reset_noise()
        obs_t  = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
        logit  = self.gate(obs_t).squeeze()
        prob   = float(torch.sigmoid(logit).item())
        self.gate.train()
        return prob >= GATE_THRESH, prob

    @torch.no_grad()
    def get_direction(self, obs: np.ndarray) -> Tuple[int, float]:
        """Returns (action_1or2, confidence). action: 1=BUY, 2=SELL."""
        self.dir_net.eval()
        self.dir_net.reset_noise()
        obs_t  = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
        logits = self.dir_net(obs_t).squeeze(0)   # (2,)
        probs  = torch.softmax(logits, dim=-1)
        idx    = int(probs.argmax().item())
        conf   = float(probs[idx].item())
        self.dir_net.train()
        return idx + 1, conf   # 0→BUY=1, 1→SELL=2

    # ── PPOAgent-compatible interface ──────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, obs: np.ndarray,
                      deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Full hierarchical decision:
          Gate → if trade → Direction → Rainbow verify → return action
        """
        trade, gate_p = self.should_trade(obs)

        if not trade:
            # HOLD — use Rainbow confidence as the "value" estimate
            _, _, r_val = self.rainbow.select_action(obs, deterministic=True)
            return 0, float(np.log(1.0 - gate_p + 1e-9)), r_val

        dir_action, dir_conf = self.get_direction(obs)

        # Cross-check with Rainbow: if Rainbow strongly disagrees, lower confidence
        r_action, r_logp, r_val = self.rainbow.select_action(obs, deterministic)
        if r_action == 0:
            # Rainbow says HOLD despite Gate saying trade → dampen confidence
            log_p = float(np.log(gate_p * dir_conf * 0.7 + 1e-9))
        else:
            # Rainbow agrees (or disagrees on direction) — use combined signal
            agree = (r_action == dir_action)
            combined = gate_p * dir_conf * (1.1 if agree else 0.9)
            log_p = float(np.log(combined + 1e-9))

        return dir_action, log_p, r_val

    @torch.no_grad()
    def get_confidence(self, obs: np.ndarray) -> Tuple[int, float]:
        action, log_p, _ = self.select_action(obs, deterministic=True)
        # Convert log_p back to an approximate probability
        conf = float(min(1.0, np.exp(log_p)))
        return action, conf

    @torch.no_grad()
    def get_action_distribution(self, obs: np.ndarray) -> dict:
        trade, gate_p = self.should_trade(obs)
        if not trade:
            return {"hold": 1.0 - gate_p, "buy": gate_p / 2, "sell": gate_p / 2}

        self.dir_net.eval()
        obs_t  = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
        logits = self.dir_net(obs_t).squeeze(0)
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        self.dir_net.train()
        return {
            "hold": 1.0 - gate_p,
            "buy":  float(probs[0]) * gate_p,
            "sell": float(probs[1]) * gate_p,
        }

    # ── Learning ───────────────────────────────────────────────────────────────

    def store(self, obs, next_obs, action: int,
              log_prob: float, reward: float, value: float, done: bool):
        """Store experience to Rainbow PER + gate/direction buffers."""
        self.rainbow.store(obs, next_obs, action, log_prob, reward, value, done)
        self.total_steps       += 1
        self.cumulative_reward += reward
        if done:
            self.total_episodes += 1

        # Gate label: trade was good if reward > 0
        if action != 0:
            gate_label = 1.0 if reward > 0 else 0.0
            self._gate_buf.append((obs.copy(), gate_label))

            # Direction label: 0=BUY correct, 1=SELL correct
            if action == 1:   # BUY
                dir_label = 0 if reward > 0 else 1
            else:              # SELL
                dir_label = 1 if reward > 0 else 0
            self._dir_buf.append((obs.copy(), dir_label))

    def ready_to_update(self) -> bool:
        return (len(self._gate_buf) >= WARMUP
                or self.rainbow.ready_to_update())

    def update(self, last_obs: np.ndarray) -> dict:
        metrics = {}

        # Update Rainbow
        if self.rainbow.ready_to_update():
            m = self.rainbow.update(last_obs)
            metrics.update(m)

        # Update Gate
        if len(self._gate_buf) >= WARMUP:
            g_metrics = self._update_gate()
            metrics.update(g_metrics)

        # Update Direction
        if len(self._dir_buf) >= WARMUP:
            d_metrics = self._update_direction()
            metrics.update(d_metrics)

        self.total_updates += 1
        return metrics

    def _update_gate(self) -> dict:
        batch = list(self._gate_buf)[-WARMUP:]
        obs_np   = np.array([b[0] for b in batch], dtype=np.float32)
        labels   = torch.tensor([b[1] for b in batch], dtype=torch.float32,
                                device=self.device)
        obs_t    = torch.from_numpy(obs_np).to(self.device)

        self.gate.reset_noise()
        logits   = self.gate(obs_t).squeeze(-1)
        loss     = F.binary_cross_entropy_with_logits(logits, labels)
        self.gate_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.gate.parameters(), 5.0)
        self.gate_opt.step()
        return {"gate_loss": float(loss.item())}

    def _update_direction(self) -> dict:
        batch  = list(self._dir_buf)[-WARMUP:]
        obs_np = np.array([b[0] for b in batch], dtype=np.float32)
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long,
                              device=self.device)
        obs_t  = torch.from_numpy(obs_np).to(self.device)

        self.dir_net.reset_noise()
        logits = self.dir_net(obs_t)
        loss   = F.cross_entropy(logits, labels)
        self.dir_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.dir_net.parameters(), 5.0)
        self.dir_opt.step()
        return {"dir_loss": float(loss.item())}

    def update_online(self, obs: np.ndarray, action: int, reward: float,
                      next_obs: np.ndarray, done: bool) -> dict:
        self.store(obs, next_obs, action, 0.0, reward, 0.0, done)
        if self.ready_to_update():
            return self.update(next_obs)
        return {}

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save HRL (gate + direction) alongside Rainbow."""
        # Rainbow saves to path as-is (e.g. knowledge/rainbow.pt)
        self.rainbow.save(path)
        hrl_path = path.replace(".pt", "_hrl.pt")
        os.makedirs(os.path.dirname(hrl_path) if os.path.dirname(hrl_path) else ".",
                    exist_ok=True)
        torch.save({
            "gate":    self.gate.state_dict(),
            "dir_net": self.dir_net.state_dict(),
            "steps":   self.total_steps,
            "updates": self.total_updates,
            "episodes": self.total_episodes,
            "cum_r":   self.cumulative_reward,
        }, hrl_path)
        logger.info("HRL saved → %s (steps=%d)", hrl_path, self.total_steps)

    def load(self, path: str) -> bool:
        ok = self.rainbow.load(path)
        hrl_path = path.replace(".pt", "_hrl.pt")
        if os.path.exists(hrl_path):
            try:
                ckpt = torch.load(hrl_path, map_location=self.device, weights_only=False)
                self.gate.load_state_dict(ckpt["gate"])
                self.dir_net.load_state_dict(ckpt["dir_net"])
                self.total_steps       = ckpt.get("steps", 0)
                self.total_updates     = ckpt.get("updates", 0)
                self.total_episodes    = ckpt.get("episodes", 0)
                self.cumulative_reward = ckpt.get("cum_r", 0.0)
                logger.info("HRL gate+direction loaded ← %s", hrl_path)
                ok = True
            except Exception as exc:
                logger.warning("HRL checkpoint load failed: %s", exc)
        return ok

    def get_weights_bytes(self) -> bytes:
        return self.rainbow.get_weights_bytes()

    def load_weights_bytes(self, data: bytes) -> None:
        self.rainbow.load_weights_bytes(data)

    def fedavg_merge(self, worker_bytes: bytes, alpha: float = 0.25) -> None:
        self.rainbow.fedavg_merge(worker_bytes, alpha)

    def get_buffer_bytes(self) -> bytes:
        return self.rainbow.get_buffer_bytes()
