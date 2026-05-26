"""
AI Trading Worker — รันบนเครื่องอื่นเพื่อช่วย 2 อย่าง:
  1. PPO Training  — รับ experience buffer จาก server, train, ส่ง weights กลับ
  2. Research      — ค้นหาความรู้จาก Wikipedia + Google News, ส่ง knowledge nodes กลับ

วิธีใช้:
    python worker.py --server ws://192.168.1.100:8000 --name "PC-2"

    IP คือ IP ของเครื่องหลักที่รัน server อยู่
    ดูได้จาก: ipconfig (Windows) หรือ ifconfig (Mac/Linux)

ต้องการ:
    pip install websockets torch numpy requests

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
import random
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("worker")


# ── Knowledge Research (embedded — no need to import full trading_ai) ─────────

RESEARCH_TOPICS = {
    "technical_analysis": [
        "RSI divergence trading strategy", "MACD histogram crossover signal",
        "Bollinger Band squeeze breakout", "Ichimoku Cloud trading rules",
        "Stochastic oscillator overbought oversold", "Fibonacci retracement levels forex",
        "support resistance trading strategy", "moving average crossover strategy",
        "ADX trend strength indicator", "Parabolic SAR trading signals",
    ],
    "binary_options": [
        "binary options 1 minute strategy", "binary options OTC weekend trading",
        "binary options entry timing technique", "binary options money management rules",
        "binary options technical analysis tips", "binary options trend following strategy",
    ],
    "price_action": [
        "bullish engulfing candlestick pattern", "pin bar reversal trading strategy",
        "doji candlestick meaning interpretation", "hammer hanging man pattern",
        "morning star evening star reversal", "head and shoulders chart pattern",
        "double top double bottom reversal", "triangle pattern breakout trading",
        "flag pennant continuation pattern",
    ],
    "risk_management": [
        "forex risk management rules", "position sizing Kelly criterion trading",
        "stop loss strategy binary options", "risk reward ratio forex trading",
        "drawdown recovery trading strategy", "trading capital preservation rules",
    ],
    "trading_psychology": [
        "trader psychology fear greed discipline", "revenge trading avoidance",
        "patience in forex trading", "trading journal benefits review",
        "emotional control day trading", "overconfidence bias trading mistakes",
    ],
}

WIKI_TITLE_MAP = {
    "engulfing": "Engulfing_pattern", "doji": "Doji",
    "hammer": "Hammer_(candlestick_pattern)",
    "morning star": "Morning_star_(candlestick_pattern)",
    "evening star": "Morning_star_(candlestick_pattern)",
    "head and shoulders": "Head_and_shoulders_(chart_pattern)",
    "double top": "Double_top_and_double_bottom",
    "double bottom": "Double_top_and_double_bottom",
    "triangle": "Triangle_(chart_pattern)",
    "pin bar": "Price_action_trading",
    "candlestick": "Candlestick_pattern",
    "rsi": "Relative_strength_index", "macd": "MACD",
    "bollinger": "Bollinger_Bands",
    "ichimoku": "Ichimoku_Kink%C5%8D_Hy%C5%8D",
    "stochastic": "Stochastic_oscillator",
    "fibonacci": "Fibonacci_retracement",
    "adx": "Average_directional_movement_index",
    "parabolic sar": "Parabolic_SAR",
    "moving average": "Moving_average",
    "support resistance": "Support_and_resistance",
    "kelly": "Kelly_criterion", "risk management": "Risk_management",
    "drawdown": "Drawdown_(economics)",
    "binary option": "Binary_option", "trend": "Market_trend",
    "price action": "Price_action_trading",
    "trader psychology": "Behavioral_economics",
    "overconfidence": "Overconfidence_effect",
}

CATEGORY_NODE_TYPE = {
    "technical_analysis": "technique",
    "binary_options":     "strategy_concept",
    "price_action":       "pattern",
    "risk_management":    "risk_concept",
    "trading_psychology": "psychology",
}

WIKIPEDIA_API   = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
RESEARCH_COOLDOWN = 120    # 2 min between research sessions


class WorkerResearcher:
    """ค้นหาความรู้จาก internet แล้วคืน list ของ node dicts"""

    def __init__(self):
        self._category_idx   = random.randint(0, len(RESEARCH_TOPICS) - 1)
        self._categories     = list(RESEARCH_TOPICS.keys())
        self._last_research  = 0.0
        try:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": "Mozilla/5.0 (compatible; TradingAI-Worker/1.0)"
            })
            self._ok = True
        except ImportError:
            logger.warning("requests ไม่ได้ติดตั้ง — ไม่สามารถทำ research ได้")
            self._ok = False

    def should_research(self) -> bool:
        return self._ok and (time.time() - self._last_research) > RESEARCH_COOLDOWN

    def research(self) -> List[dict]:
        """รัน 1 session ค้นหาความรู้ → คืน list of serializable node dicts"""
        if not self._ok:
            return []
        self._last_research = time.time()
        category  = self._categories[self._category_idx % len(self._categories)]
        self._category_idx += 1
        topics    = RESEARCH_TOPICS[category]
        chosen    = random.sample(topics, min(6, len(topics)))
        node_type = CATEGORY_NODE_TYPE.get(category, "rule")
        nodes: List[dict] = []

        logger.info("🔍 Research: category=%s topics=%s", category, chosen)

        for topic in chosen:
            node = self._wikipedia(topic, category, node_type)
            if node:
                nodes.append(node)
            news = self._google_news_nodes(topic, category, node_type)
            nodes.extend(news[:3])
            if len(nodes) >= 20:
                break
            time.sleep(0.3)

        logger.info("🔍 Research เสร็จ: %d nodes (category=%s)", len(nodes), category)
        return nodes

    def _wikipedia(self, topic: str, category: str, node_type: str) -> Optional[dict]:
        wiki_title = self._topic_to_wiki(topic)
        if not wiki_title:
            return None
        try:
            url  = WIKIPEDIA_API.format(title=wiki_title)
            resp = self._session.get(url, timeout=10)
            if resp.status_code != 200:
                return None
            data    = resp.json()
            extract = (data.get("extract") or "").strip()
            if not extract or len(extract) < 50:
                return None
            title     = data.get("title", topic)
            direction = self._direction(extract.lower())
            return {
                "node_type": node_type, "concept": f"[Wiki] {title[:100]}",
                "data": {
                    "topic": topic, "title": title, "summary": extract[:500],
                    "category": category, "direction_bias": direction,
                    "source_url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                },
                "confidence": 0.55, "evidence_count": 1, "contradiction_count": 0,
                "activation": 1.0, "source": "wikipedia_worker",
                "asset": None,
                "tags": ["wikipedia", category, "knowledge", "from_worker"],
                "edges": [],
            }
        except Exception as exc:
            logger.debug("Wikipedia '%s': %s", topic, exc)
            return None

    def _google_news_nodes(self, topic: str, category: str, node_type: str) -> List[dict]:
        try:
            url  = GOOGLE_NEWS_RSS.format(query=quote(topic))
            resp = self._session.get(url, timeout=10)
            resp.raise_for_status()
            results = []
            for item in re.findall(r"<item>(.*?)</item>", resp.text, re.DOTALL)[:3]:
                tm = re.search(r"<title>(.*?)</title>", item, re.DOTALL)
                dm = re.search(r"<description>(.*?)</description>", item, re.DOTALL)
                if not tm:
                    continue
                title = re.sub(r"<[^>]+>", "", tm.group(1)).strip()
                desc  = re.sub(r"<[^>]+>", " ", dm.group(1) if dm else "").strip()[:300]
                if not title or len(title) < 10:
                    continue
                text = (title + " " + desc).lower()
                kws  = ["trading", "forex", "binary", "strategy", "indicator",
                        "pattern", "signal", "analysis", "chart", "market"]
                if not any(k in text for k in kws):
                    continue
                direction = self._direction(text)
                results.append({
                    "node_type": node_type, "concept": f"[News] {title[:120]}",
                    "data": {
                        "topic": topic, "title": title, "summary": desc,
                        "category": category, "direction_bias": direction,
                    },
                    "confidence": 0.40, "evidence_count": 1, "contradiction_count": 0,
                    "activation": 1.0, "source": "google_news_worker",
                    "asset": None,
                    "tags": ["news", category, "knowledge", "from_worker"],
                    "edges": [],
                })
            return results
        except Exception as exc:
            logger.debug("Google News '%s': %s", topic, exc)
            return []

    @staticmethod
    def _topic_to_wiki(topic: str) -> Optional[str]:
        t = topic.lower()
        for kw, title in WIKI_TITLE_MAP.items():
            if kw in t:
                return title
        return None

    @staticmethod
    def _direction(text: str) -> Optional[str]:
        bull = sum(1 for w in ["bullish","buy","long","upward","rally","surge","rise","uptrend"] if w in text)
        bear = sum(1 for w in ["bearish","sell","short","downward","drop","fall","decline","downtrend"] if w in text)
        if bull > bear + 1: return "buy"
        if bear > bull + 1: return "sell"
        return None


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
            dist    = __import__('torch').distributions.Categorical(logits=logits)
            log_p   = dist.log_prob(actions)
            entropy = dist.entropy()
            return log_p, values, entropy

    return WorkerNetwork()


def _load_weights(network, weights_bytes: bytes, device):
    import torch
    buf   = io.BytesIO(weights_bytes)
    full  = torch.load(buf, map_location=device, weights_only=True)
    own   = network.state_dict()
    match = {k: v for k, v in full.items() if k in own and own[k].shape == v.shape}
    own.update(match)
    network.load_state_dict(own, strict=False)


def _train_on_buffer(network, buffer_bytes: bytes, hypers: Dict[str, Any], device) -> tuple:
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

    gamma = hypers.get("gamma", 0.99); gae_lambda = hypers.get("gae_lambda", 0.95)
    advantages = torch.zeros(size, device=device)
    gae = 0.0
    for t in reversed(range(size)):
        nxt_val  = float(values[t + 1]) if t < size - 1 else 0.0
        nxt_done = float(dones[t + 1])  if t < size - 1 else 0.0
        delta    = float(rewards[t]) + gamma * nxt_val * (1.0 - float(dones[t])) - float(values[t])
        gae      = delta + gamma * gae_lambda * (1.0 - nxt_done) * gae
        advantages[t] = gae

    returns    = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    clip_eps   = hypers.get("clip_epsilon", 0.2)
    epochs     = hypers.get("ppo_epochs", 4)
    batch_size = hypers.get("batch_size", 64)
    lr         = hypers.get("lr", 3e-4)
    optimizer  = optim.Adam(network.parameters(), lr=lr, eps=1e-5)
    network.train()

    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    n_upd   = 0

    for _ in range(epochs):
        network.reset_hidden()
        indices = torch.randperm(size, device=device)
        for start in range(0, size, batch_size):
            idx    = indices[start: start + batch_size]
            new_lp, new_val, entropy = network.evaluate(obs[idx], actions[idx])
            ratio  = torch.exp(new_lp - log_probs[idx])
            adv_b  = advantages[idx]
            surr1  = ratio * adv_b
            surr2  = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_b
            p_loss = -torch.min(surr1, surr2).mean()
            v_clip  = values[idx] + torch.clamp(new_val - values[idx], -clip_eps, clip_eps)
            v_loss  = torch.max(
                nn.functional.mse_loss(new_val, returns[idx]),
                nn.functional.mse_loss(v_clip, returns[idx]),
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

    buf = io.BytesIO()
    import torch as _t
    _t.save(network.state_dict(), buf)
    return buf.getvalue(), metrics


# ── Main worker loop ──────────────────────────────────────────────────────────

async def run_worker(server_url: str, worker_name: str):
    import websockets
    import torch

    logger.info("=" * 55)
    logger.info("  🤖 AI Trading Worker — %s", worker_name)
    logger.info("  Server: %s", server_url)
    logger.info("  ช่วย: PPO Training + Knowledge Research")
    logger.info("=" * 55)

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    network    = None
    researcher = WorkerResearcher()
    reconnect_delay = 3

    while True:
        try:
            logger.info("กำลังเชื่อมต่อ %s …", server_url)
            async with websockets.connect(
                server_url,
                ping_interval=20,
                ping_timeout=30,
                max_size=200 * 1024 * 1024,
            ) as ws:
                reconnect_delay = 3
                logger.info("เชื่อมต่อสำเร็จ!")

                await ws.send(json.dumps({"type": "hello", "name": worker_name}))

                # ── Background research task ───────────────────────────────
                async def research_loop():
                    """ค้นหาความรู้ทุก 2 นาที และส่งให้ server อัตโนมัติ"""
                    while True:
                        await asyncio.sleep(10)   # wait 10s after connect before first research
                        if researcher.should_research():
                            try:
                                nodes = await asyncio.get_event_loop().run_in_executor(
                                    None, researcher.research
                                )
                                if nodes:
                                    await ws.send(json.dumps({
                                        "type":  "knowledge_result",
                                        "nodes": nodes,
                                        "count": len(nodes),
                                    }))
                                    logger.info("📚 ส่ง %d knowledge nodes ให้ server", len(nodes))
                            except Exception as exc:
                                logger.error("Research loop error: %s", exc)
                        await asyncio.sleep(30)

                research_task = asyncio.create_task(research_loop())

                # ── Keepalive: ส่ง ping ทุก 15s ป้องกัน server timeout ───
                async def keepalive_loop():
                    while True:
                        await asyncio.sleep(15)
                        try:
                            await ws.send(json.dumps({"type": "ping"}))
                        except Exception:
                            break  # ws ปิดแล้ว

                keepalive_task = asyncio.create_task(keepalive_loop())

                try:
                    async for raw in ws:
                        msg   = json.loads(raw)
                        mtype = msg.get("type")

                        if mtype == "welcome":
                            logger.info("✅ %s | ID=%s", msg.get("message", ""), msg.get("worker_id"))

                        elif mtype == "pong":
                            pass  # server ตอบรับ ping ของเรา — connection ยังอยู่

                        elif mtype == "work_request":
                            logger.info("📥 ได้รับงาน PPO training…")
                            t0 = time.time()
                            hypers      = msg.get("hypers", {})
                            weights_b64 = msg.get("weights", "")
                            buffer_b64  = msg.get("buffer",  "")
                            if not weights_b64 or not buffer_b64:
                                continue

                            obs_size    = hypers.get("obs_size", 296)
                            n_actions   = hypers.get("n_actions", 3)
                            hidden_size = hypers.get("hidden_size", 384)

                            if network is None:
                                network = _build_network(obs_size, n_actions, hidden_size).to(device)

                            _load_weights(network, base64.b64decode(weights_b64), device)
                            try:
                                updated, metrics = _train_on_buffer(
                                    network, base64.b64decode(buffer_b64), hypers, device
                                )
                                logger.info(
                                    "✅ Training %.1fs | ploss=%.4f vloss=%.4f",
                                    time.time() - t0, metrics["policy_loss"], metrics["value_loss"],
                                )
                                await ws.send(json.dumps({
                                    "type":    "work_result",
                                    "weights": base64.b64encode(updated).decode(),
                                    "metrics": metrics,
                                }))
                            except Exception as exc:
                                logger.error("Training failed: %s", exc)

                        elif mtype == "research_request":
                            # Server สั่งให้ทำ research ทันที
                            category = msg.get("category")
                            logger.info("🔍 Server ขอ research: category=%s", category or "auto")
                            try:
                                if category and category in RESEARCH_TOPICS:
                                    researcher._category_idx = list(RESEARCH_TOPICS.keys()).index(category)
                                researcher._last_research = 0  # force research now
                                nodes = await asyncio.get_event_loop().run_in_executor(
                                    None, researcher.research
                                )
                                if nodes:
                                    await ws.send(json.dumps({
                                        "type":  "knowledge_result",
                                        "nodes": nodes,
                                        "count": len(nodes),
                                    }))
                                    logger.info("📚 ส่ง %d nodes ให้ server", len(nodes))
                            except Exception as exc:
                                logger.error("On-demand research failed: %s", exc)
                finally:
                    research_task.cancel()
                    keepalive_task.cancel()

        except (ConnectionRefusedError, OSError) as exc:
            logger.warning("เชื่อมต่อไม่ได้: %s — รอ %ds", exc, reconnect_delay)
        except Exception as exc:
            logger.warning("Connection หลุด: %s — รอ %ds", exc, reconnect_delay)

        await asyncio.sleep(reconnect_delay)
        reconnect_delay = min(reconnect_delay * 2, 60)


def main():
    parser = argparse.ArgumentParser(description="AI Trading Worker")
    parser.add_argument("--server", default="ws://localhost:8000/ws/worker",
                        help="WebSocket URL ของ server หลัก")
    parser.add_argument("--name", default=f"Worker-{os.getpid()}",
                        help="ชื่อเครื่อง worker นี้")
    args = parser.parse_args()
    url  = args.server
    if not url.endswith("/ws/worker"):
        url = url.rstrip("/") + "/ws/worker"
    asyncio.run(run_worker(url, args.name))


if __name__ == "__main__":
    main()
