"""
AI Trading Worker — รันบนเครื่องอื่นเพื่อช่วยประมวลผล PPO Training

วิธีใช้:
    python worker.py --server ws://192.168.1.100:8000 --name "PC-2"

    IP คือ IP ของเครื่องหลักที่รัน server อยู่
    ดูได้จาก: ipconfig (Windows) หรือ ifconfig (Mac/Linux)

ต้องการ:
    pip install websockets torch numpy

ไม่ต้องการ:
    - IQ Option account
    - Browser / GUI
    - GPU (CPU ก็ได้)
"""
import argparse
import asyncio
import base64
import io
import json
import logging
import os
import pickle
import sys
import time
from typing import Dict, Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("worker")


# ── Minimal PPO network (mirrors ppo_agent.py) ────────────────────────────────

def _build_network(obs_size: int, n_actions: int, hidden_size: int):
    """สร้าง PPO network โดยไม่ต้อง import ทั้งระบบ"""
    import torch
    import torch.nn as nn

    class ResidualBlock(nn.Module):
        def __init__(self, size):
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(size, size), nn.LayerNorm(size), nn.GELU(),
                nn.Dropout(0.1), nn.Linear(size, size), nn.LayerNorm(size),
            )
            self.act = nn.GELU()
        def forward(self, x):
            return self.act(x + self.block(x))

    class WorkerNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Sequential(
                nn.Linear(obs_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU(),
                ResidualBlock(hidden_size), ResidualBlock(hidden_size),
            )
            lstm_hidden = min(128, hidden_size // 2)
            self.lstm = nn.LSTM(hidden_size, lstm_hidden, batch_first=True)
            self.policy_head = nn.Linear(lstm_hidden, n_actions)
            self.value_head  = nn.Linear(lstm_hidden, 1)
            self._lstm_h = None
            self._lstm_c = None

        def reset_hidden(self, batch_size: int = 1):
            dev = next(self.parameters()).device
            lstm_hidden = self.lstm.hidden_size
            self._lstm_h = torch.zeros(1, batch_size, lstm_hidden, device=dev)
            self._lstm_c = torch.zeros(1, batch_size, lstm_hidden, device=dev)

        def forward(self, x):
            feat = self.embed(x)
            if feat.dim() == 2:
                feat = feat.unsqueeze(1)
            if self._lstm_h is None or self._lstm_h.size(1) != feat.size(0):
                self.reset_hidden(feat.size(0))
            out, (h, c) = self.lstm(feat, (self._lstm_h, self._lstm_c))
            self._lstm_h, self._lstm_c = h.detach(), c.detach()
            out = out.squeeze(1)
            return self.policy_head(out), self.value_head(out).squeeze(-1)

        def evaluate(self, obs, actions):
            logits, values = self.forward(obs)
            dist    = torch.distributions.Categorical(logits=logits)
            log_p   = dist.log_prob(actions)
            entropy = dist.entropy()
            return log_p, values, entropy

    return WorkerNetwork()


def _load_weights(network, weights_bytes: bytes, device):
    import torch
    buf   = io.BytesIO(weights_bytes)
    # Load only keys that exist in our simplified network
    full  = torch.load(buf, map_location=device, weights_only=True)
    own   = network.state_dict()
    match = {k: v for k, v in full.items() if k in own and own[k].shape == v.shape}
    own.update(match)
    network.load_state_dict(own, strict=False)
    logger.debug("Loaded %d/%d weight tensors", len(match), len(full))


def _train_on_buffer(
    network,
    buffer_bytes: bytes,
    hypers: Dict[str, Any],
    device,
) -> bytes:
    """รัน PPO training บน buffer ที่ได้รับ แล้วคืน updated weights"""
    import torch
    import torch.nn as nn
    import torch.optim as optim

    data = pickle.loads(buffer_bytes)
    size = data["size"]

    obs       = torch.tensor(data["obs"][:size],       dtype=torch.float32).to(device)
    next_obs  = torch.tensor(data["next_obs"][:size],  dtype=torch.float32).to(device)
    actions   = torch.tensor(data["actions"][:size],   dtype=torch.long).to(device)
    log_probs = torch.tensor(data["log_probs"][:size], dtype=torch.float32).to(device)
    rewards   = torch.tensor(data["rewards"][:size],   dtype=torch.float32).to(device)
    values    = torch.tensor(data["values"][:size],    dtype=torch.float32).to(device)
    dones     = torch.tensor(data["dones"][:size],     dtype=torch.float32).to(device)

    # GAE returns & advantages
    gamma      = hypers.get("gamma", 0.99)
    gae_lambda = hypers.get("gae_lambda", 0.95)
    advantages = torch.zeros(size, device=device)
    gae = 0.0
    for t in reversed(range(size)):
        nxt_val  = float(values[t + 1]) if t < size - 1 else 0.0
        nxt_done = float(dones[t + 1])  if t < size - 1 else 0.0
        delta    = float(rewards[t]) + gamma * nxt_val * (1.0 - float(dones[t])) - float(values[t])
        gae      = delta + gamma * gae_lambda * (1.0 - nxt_done) * gae
        advantages[t] = gae

    returns    = advantages + values
    adv_mean   = advantages.mean()
    adv_std    = advantages.std() + 1e-8
    advantages = (advantages - adv_mean) / adv_std

    clip_eps   = hypers.get("clip_epsilon", 0.2)
    epochs     = hypers.get("ppo_epochs", 4)
    batch_size = hypers.get("batch_size", 64)
    lr         = hypers.get("lr", 3e-4)

    optimizer = optim.Adam(network.parameters(), lr=lr, eps=1e-5)
    network.train()

    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    n_upd   = 0

    for _ in range(epochs):
        network.reset_hidden()
        indices = torch.randperm(size, device=device)
        for start in range(0, size, batch_size):
            idx    = indices[start: start + batch_size]
            obs_b  = obs[idx];  act_b = actions[idx]
            lp_b   = log_probs[idx]; adv_b = advantages[idx]
            ret_b  = returns[idx];   val_b = values[idx]

            new_lp, new_val, entropy = network.evaluate(obs_b, act_b)
            ratio  = torch.exp(new_lp - lp_b)
            surr1  = ratio * adv_b
            surr2  = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_b
            p_loss = -torch.min(surr1, surr2).mean()

            v_clip  = val_b + torch.clamp(new_val - val_b, -clip_eps, clip_eps)
            v_loss  = torch.max(
                nn.functional.mse_loss(new_val, ret_b),
                nn.functional.mse_loss(v_clip, ret_b),
            )
            loss = p_loss + 0.5 * v_loss - 0.01 * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 0.5)
            optimizer.step()

            metrics["policy_loss"] += p_loss.item()
            metrics["value_loss"]  += v_loss.item()
            metrics["entropy"]     += entropy.mean().item()
            n_upd += 1

    if n_upd > 0:
        for k in metrics:
            metrics[k] /= n_upd

    # Serialize updated weights
    buf = io.BytesIO()
    torch.save(network.state_dict(), buf)
    return buf.getvalue(), metrics


# ── Main worker loop ──────────────────────────────────────────────────────────

async def run_worker(server_url: str, worker_name: str):
    import websockets

    logger.info("=" * 55)
    logger.info("  🤖 AI Trading Worker — %s", worker_name)
    logger.info("  Server: %s", server_url)
    logger.info("=" * 55)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    network = None
    reconnect_delay = 3

    while True:
        try:
            logger.info("กำลังเชื่อมต่อ %s …", server_url)
            async with websockets.connect(
                server_url,
                ping_interval=20,
                ping_timeout=30,
                max_size=200 * 1024 * 1024,   # 200 MB max message
            ) as ws:
                reconnect_delay = 3
                logger.info("เชื่อมต่อสำเร็จ! ส่ง hello…")

                await ws.send(json.dumps({
                    "type": "hello",
                    "name": worker_name,
                }))

                async for raw in ws:
                    msg = json.loads(raw)
                    mtype = msg.get("type")

                    if mtype == "welcome":
                        wid = msg.get("worker_id", "?")
                        logger.info("✅ %s | ID=%s", msg.get("message", ""), wid)

                    elif mtype == "work_request":
                        logger.info("📥 ได้รับงาน training จาก server…")
                        t0 = time.time()

                        hypers       = msg.get("hypers", {})
                        weights_b64  = msg.get("weights", "")
                        buffer_b64   = msg.get("buffer",  "")

                        if not weights_b64 or not buffer_b64:
                            logger.warning("งานว่างเปล่า — ข้ามไป")
                            continue

                        weights_bytes = base64.b64decode(weights_b64)
                        buffer_bytes  = base64.b64decode(buffer_b64)

                        # Build or reuse network
                        obs_size    = hypers.get("obs_size", 296)
                        n_actions   = hypers.get("n_actions", 3)
                        hidden_size = hypers.get("hidden_size", 384)

                        if network is None:
                            network = _build_network(obs_size, n_actions, hidden_size).to(device)
                            logger.info("สร้าง network (obs=%d, act=%d, hid=%d)", obs_size, n_actions, hidden_size)

                        _load_weights(network, weights_bytes, device)

                        try:
                            updated_weights, metrics = _train_on_buffer(
                                network, buffer_bytes, hypers, device
                            )
                            elapsed = time.time() - t0
                            logger.info(
                                "✅ Training เสร็จใน %.1fs | ploss=%.4f vloss=%.4f ent=%.4f",
                                elapsed,
                                metrics["policy_loss"],
                                metrics["value_loss"],
                                metrics["entropy"],
                            )
                            await ws.send(json.dumps({
                                "type":    "work_result",
                                "weights": base64.b64encode(updated_weights).decode(),
                                "metrics": metrics,
                            }))
                            logger.info("📤 ส่ง weights กลับ server แล้ว")
                        except Exception as train_exc:
                            logger.error("Training ล้มเหลว: %s", train_exc)

        except (ConnectionRefusedError, OSError) as exc:
            logger.warning("เชื่อมต่อไม่ได้: %s — รอ %ds แล้วลองใหม่", exc, reconnect_delay)
        except Exception as exc:
            logger.warning("Connection หลุด: %s — รอ %ds แล้วลองใหม่", exc, reconnect_delay)

        await asyncio.sleep(reconnect_delay)
        reconnect_delay = min(reconnect_delay * 2, 60)


def main():
    parser = argparse.ArgumentParser(description="AI Trading Worker")
    parser.add_argument(
        "--server", default="ws://localhost:8000/ws/worker",
        help="WebSocket URL ของ server หลัก (เช่น ws://192.168.1.100:8000/ws/worker)",
    )
    parser.add_argument(
        "--name", default=f"Worker-{os.getpid()}",
        help="ชื่อเครื่อง worker นี้",
    )
    args = parser.parse_args()

    # Ensure URL has the right path
    url = args.server
    if not url.endswith("/ws/worker"):
        url = url.rstrip("/") + "/ws/worker"

    asyncio.run(run_worker(url, args.name))


if __name__ == "__main__":
    main()
