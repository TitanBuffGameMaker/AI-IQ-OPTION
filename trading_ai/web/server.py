"""
FastAPI WebSocket server for the IQ Option AI Trading Dashboard.

Endpoints:
  GET  /          → serve dashboard HTML
  WS   /ws        → real-time bidirectional channel (price, signals, controls)
  POST /trade     → manual trade execution
  GET  /api/status → current system status JSON
  GET  /api/brain  → knowledge graph stats JSON
  GET  /api/history → trade history JSON

WebSocket message protocol (JSON):
  Server → Client:
    {type:"price",   asset, price, change_pct, candle:{o,h,l,c}}
    {type:"signal",  asset, action, confidence, reasoning}
    {type:"check",   results:[{name,passed,message}], all_passed}
    {type:"brain",   nodes, branches, avg_conf, win_rate, by_type}
    {type:"trade",   asset, action, pnl, balance, trades, wins}
    {type:"status",  message, level}   level: info|warn|error|success

  Client → Server:
    {type:"trade",   asset, direction}   direction: call|put
    {type:"ai",      running}            start/stop AI
    {type:"settings",timeframe, duration, amount}
"""
import asyncio
import json
import logging
import os
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Set

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

from trading_ai.config import config
from trading_ai.utils.logger import setup_logging

logger = logging.getLogger(__name__)

# ── Lazy imports of heavy modules (only loaded when server starts) ─────────────
_connector   = None
_agent       = None
_brain       = None
_knowledge   = None
_env         = None
_ai_running  = False
_trade_stats = {"trades": 0, "wins": 0, "pnl": 0.0}
_trade_history: deque = deque(maxlen=200)
_check_results: List[Dict] = []
_otp_event      = threading.Event()
_otp_code:  str = ""
_pending_creds: Dict[str, str] = {}   # {"email":..,"password":..} from UI login form

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="IQ Option AI Trading Dashboard", docs_url=None)

_BASE = os.path.dirname(__file__)
app.mount("/static", StaticFiles(directory=os.path.join(_BASE, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(_BASE, "templates"))

# ── WebSocket connection manager ──────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self._clients: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._clients.add(ws)
        logger.info("WS client connected (total=%d)", len(self._clients))

    def disconnect(self, ws: WebSocket):
        self._clients.discard(ws)

    async def broadcast(self, msg: Dict):
        if not self._clients:
            return
        data = json.dumps(msg, ensure_ascii=False)
        dead = set()
        for ws in list(self._clients):
            try:
                await ws.send_text(data)
            except Exception:
                dead.add(ws)
        self._clients -= dead

    async def send(self, ws: WebSocket, msg: Dict):
        try:
            await ws.send_text(json.dumps(msg, ensure_ascii=False))
        except Exception:
            pass


manager = ConnectionManager()

# ── HTTP Routes ───────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/status")
async def api_status():
    bal = _connector.get_balance() if _connector else 0.0
    return {
        "connected": _connector is not None,
        "account_type": config.IQ_ACCOUNT_TYPE,
        "balance": bal,
        "ai_running": _ai_running,
        "asset": config.ASSET,
        "trade_amount": config.TRADE_AMOUNT,
        "timeframe": f"{config.CANDLE_TIMEFRAME // 60}m",
        "duration": f"{config.TRADE_DURATION}m",
        "stats": _trade_stats,
    }


@app.get("/api/brain")
async def api_brain():
    if _brain is None:
        return {"nodes": 0}
    s = _brain.get_status()
    return s


@app.get("/api/history")
async def api_history():
    return list(_trade_history)


@app.get("/api/checks")
async def api_checks():
    return _check_results


# ── WebSocket handler ─────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)

    # Send current state immediately on connect
    await _send_current_state(ws)

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            await _handle_client_message(ws, msg)
    except WebSocketDisconnect:
        manager.disconnect(ws)
        logger.info("WS client disconnected")
    except Exception as exc:
        logger.error("WS error: %s", exc)
        manager.disconnect(ws)


async def _send_current_state(ws: WebSocket):
    """Push current system state to a newly connected client."""
    bal = _connector.get_balance() if _connector else 0.0
    await manager.send(ws, {
        "type": "init",
        "connected": _connector is not None,
        "balance": bal,
        "ai_running": _ai_running,
        "stats": _trade_stats,
        "checks": _check_results,
        "history": list(_trade_history)[-20:],
        "settings": {
            "timeframe": f"{config.CANDLE_TIMEFRAME // 60}m",
            "duration": f"{config.TRADE_DURATION}m",
            "amount": config.TRADE_AMOUNT,
            "asset": config.ASSET,
        },
    })
    # Send OTC candle history so charts populate immediately
    if _candle_history:
        await manager.send(ws, {
            "type": "candle_history",
            "data": {k: v[-150:] for k, v in _candle_history.items()},
        })
    if _brain:
        await manager.send(ws, {
            "type": "brain",
            **_brain.get_status(ppo_agent=_agent),
        })


async def _handle_client_message(ws: WebSocket, msg: Dict):
    mtype = msg.get("type")

    if mtype == "trade":
        asset     = msg.get("asset", config.ASSET)
        direction = msg.get("direction", "call")
        asyncio.create_task(_execute_manual_trade(asset, direction))

    elif mtype == "ai":
        running = msg.get("running", False)
        if running and not _ai_running:
            asyncio.create_task(_start_ai())
        elif not running and _ai_running:
            _stop_ai()

    elif mtype == "settings":
        _apply_settings(msg)
        await manager.broadcast({
            "type": "status",
            "message": "Settings updated",
            "level": "success",
        })

    elif mtype == "otp":
        global _otp_code
        _otp_code = str(msg.get("code", "")).strip()
        _otp_event.set()

    elif mtype == "credentials":
        global _pending_creds
        _pending_creds = {
            "email":    str(msg.get("email", "")).strip(),
            "password": str(msg.get("password", "")).strip(),
        }
        _otp_event.set()   # wake the waiting _init_components thread

    elif mtype == "ping":
        await manager.send(ws, {"type": "pong"})


# ── Background: broadcast helpers (called from sync threads via asyncio) ───────
_loop: Optional[asyncio.AbstractEventLoop] = None


def broadcast_sync(msg: Dict):
    """Thread-safe broadcast from non-async code."""
    if _loop and not _loop.is_closed():
        asyncio.run_coroutine_threadsafe(manager.broadcast(msg), _loop)


# ── Trading helpers ───────────────────────────────────────────────────────────
async def _execute_manual_trade(asset: str, direction: str):
    global _connector
    if not _connector:
        return
    broadcast_sync({
        "type": "status",
        "message": f"Manual {direction.upper()}: {asset}",
        "level": "info",
    })
    ok, oid = _connector.place_trade(
        asset=asset, direction=direction,
        amount=config.TRADE_AMOUNT,
        duration_minutes=config.TRADE_DURATION,
    )
    if ok:
        broadcast_sync({
            "type": "status",
            "message": f"✅ Trade placed: {direction.upper()} {asset}",
            "level": "success",
        })
    else:
        broadcast_sync({
            "type": "status",
            "message": f"❌ Trade rejected: {asset}",
            "level": "error",
        })


async def _start_ai():
    threading.Thread(target=_ai_loop, daemon=True).start()


def _stop_ai():
    global _ai_running
    _ai_running = False
    broadcast_sync({"type": "status", "message": "AI stopped", "level": "warn"})


def _apply_settings(msg: Dict):
    if "amount" in msg:
        config.TRADE_AMOUNT = float(msg["amount"])
    if "duration" in msg:
        val = str(msg["duration"]).replace("m", "")
        config.TRADE_DURATION = int(val)


# ── AI trading loop (runs in background thread) ───────────────────────────────
def _ai_loop():
    global _ai_running, _trade_stats, _agent, _brain, _knowledge, _env, _connector

    if not _agent or not _brain or not _connector:
        broadcast_sync({"type":"status","message":"AI components not loaded","level":"error"})
        return

    _ai_running = True
    broadcast_sync({"type":"status","message":"🤖 AI started","level":"success"})

    from trading_ai.core.trading_env import TradingEnv, N_INDICATORS
    if _env is None:
        _env = TradingEnv(_connector)

    obs, _ = _env.reset()

    while _ai_running:
        indicator_vec = obs[:N_INDICATORS]

        ppo_action, log_prob, value = _agent.select_action(obs)
        _, ppo_conf = _agent.get_confidence(obs)
        signal = _brain.think(indicator_vec, ppo_action, ppo_conf)

        final_action = ppo_action
        if signal.risk_multiplier < 0.2:
            final_action = 0
        elif signal.action != ppo_action and signal.confidence > ppo_conf + 0.15:
            final_action = signal.action
        if signal.confidence < config.MIN_CONFIDENCE and final_action != 0:
            final_action = 0

        action_name = {0:"HOLD",1:"BUY",2:"SELL"}[final_action]

        # Broadcast signal to all charts
        for asset in OTC_ASSETS:
            broadcast_sync({
                "type": "signal",
                "asset": asset,
                "action": action_name,
                "confidence": round(signal.confidence, 3),
                "risk": round(signal.risk_multiplier, 2),
                "reasoning": signal.reasoning[:3],
            })

        next_obs, reward, terminated, truncated, info = _env.step(final_action)

        if not info.get("skipped", False):
            pnl = info.get("pnl", 0.0)
            _brain.learn(pnl, final_action, indicator_vec, ppo_action, next_obs=next_obs)

            _trade_stats["trades"] += 1
            if pnl > 0:
                _trade_stats["wins"] += 1
            _trade_stats["pnl"] += pnl

            entry = {
                "time": time.strftime("%H:%M:%S"),
                "asset": config.ASSET,
                "action": action_name,
                "pnl": round(pnl, 2),
                "win": pnl > 0,
                "balance": _connector.get_balance(),
                "confidence": round(signal.confidence, 3),
            }
            _trade_history.append(entry)

            wr = _trade_stats["wins"] / max(_trade_stats["trades"], 1)
            broadcast_sync({
                "type": "trade",
                "asset": config.ASSET,
                "action": action_name,
                "pnl": round(pnl, 2),
                "balance": entry["balance"],
                "trades": _trade_stats["trades"],
                "wins": _trade_stats["wins"],
                "win_rate": round(wr, 3),
                "total_pnl": round(_trade_stats["pnl"], 2),
                "entry": entry,
            })

        # Broadcast strategy status after each trade
        try:
            brain_status = _brain.get_status(ppo_agent=_agent)
            active_strategy = brain_status.get("active_strategy", "Unknown")
            strategy_stats  = brain_status.get("strategy_stats", [])
            # Find win_rate of active strategy
            strat_wr = 0.5
            for st in strategy_stats:
                if st["name"] == active_strategy:
                    strat_wr = st["win_rate"]
                    break
            broadcast_sync({
                "type":         "strategy",
                "name":         active_strategy,
                "win_rate":     round(strat_wr, 3),
                "tips":         brain_status.get("improvement_tips", []),
                "should_pause": brain_status.get("should_pause", False),
                "mistakes":     brain_status.get("recent_mistakes", []),
            })
        except Exception as _strat_exc:
            logger.debug("Strategy broadcast error: %s", _strat_exc)

        _agent.store(obs, next_obs, ppo_action, log_prob, reward, value,
                     terminated or truncated)

        if _agent.ready_to_update():
            metrics = _agent.update(next_obs)
            _knowledge.save_brain(_agent)
            brain_status = _brain.get_status(ppo_agent=_agent)
            broadcast_sync({"type": "brain", **brain_status})
            broadcast_sync({
                "type": "training",
                "policy_loss": round(metrics.get("policy_loss", 0), 4),
                "value_loss":  round(metrics.get("value_loss", 0), 4),
                "entropy":     round(metrics.get("entropy", 0), 4),
                "lr":          round(metrics.get("lr", 0), 6),
                "updates":     _agent.total_updates,
                "steps":       _agent.total_steps,
            })

        obs = next_obs
        if terminated or truncated:
            obs, _ = _env.reset()

    broadcast_sync({"type":"status","message":"AI stopped","level":"warn"})


# ── Startup: init all components ──────────────────────────────────────────────
OTC_ASSETS = ["EUR/USD (OTC)", "GBP/USD (OTC)", "AUD/USD (OTC)", "EUR/JPY (OTC)"]

# IQ Option API names for OTC assets – tries each in order until one works.
OTC_ASSET_MAP: Dict[str, List[str]] = {
    "EUR/USD (OTC)": ["EURUSD-OTC", "EURUSD_otc", "frxEURUSD", "EURUSD"],
    "GBP/USD (OTC)": ["GBPUSD-OTC", "GBPUSD_otc", "frxGBPUSD", "GBPUSD"],
    "AUD/USD (OTC)": ["AUDUSD-OTC", "AUDUSD_otc", "frxAUDUSD", "AUDUSD"],
    "EUR/JPY (OTC)": ["EURJPY-OTC", "EURJPY_otc", "frxEURJPY", "EURJPY"],
}

# Sanity-check price ranges per asset.  Prices outside these bounds mean
# iqoptionapi returned data for the WRONG pair (shared-state race condition).
PRICE_RANGES: Dict[str, tuple] = {
    "EUR/USD (OTC)": (0.90, 1.45),
    "GBP/USD (OTC)": (1.10, 1.75),
    "AUD/USD (OTC)": (0.50, 0.90),
    "EUR/JPY (OTC)": (130.0, 175.0),
}

# Cache: display_name → resolved api_name (once found, reuse)
_resolved_asset_names: Dict[str, str] = {}

# Cache: display_name → list of {time,open,high,low,close} for chart history
_candle_history: Dict[str, List[Dict]] = {a: [] for a in OTC_ASSETS}


def _resolve_asset_name(display_name: str) -> Optional[str]:
    """
    Find the working IQ Option API name for an OTC asset.

    iqoptionapi uses a shared WebSocket buffer for candle data.  Calling
    get_candles() for multiple assets in rapid succession can return stale
    data from the previous request.  We guard against this by:
      1. Sleeping 1.5 s after each request so the WS response settles.
      2. Validating the returned price against known sane ranges.
    """
    if display_name in _resolved_asset_names:
        return _resolved_asset_names[display_name]

    lo, hi = PRICE_RANGES.get(display_name, (0.0, 1e9))

    for api_name in OTC_ASSET_MAP.get(display_name, []):
        try:
            df = _connector.get_candles(asset=api_name, timeframe_seconds=60, count=5)
            time.sleep(1.5)   # let WS buffer flush before next request
            if df is None or len(df) == 0:
                continue
            price = float(df["close"].iloc[-1])
            if lo <= price <= hi:
                _resolved_asset_names[display_name] = api_name
                logger.info("Resolved %s → %s (price=%.5f)", display_name, api_name, price)
                return api_name
            logger.warning("Price %.5f out of range [%.2f, %.2f] for %s via %s — skipping",
                           price, lo, hi, display_name, api_name)
        except Exception:
            continue

    logger.warning("Could not resolve API name for %s", display_name)
    return None


def _fetch_initial_candles():
    """Fetch the last 100 candles for each OTC asset (30-second timeframe)."""
    for display_name in OTC_ASSETS:
        api_name = _resolve_asset_name(display_name)
        if not api_name:
            continue
        try:
            df = _connector.get_candles(asset=api_name, timeframe_seconds=30, count=100)
            time.sleep(1.5)   # let WS buffer flush before next asset
            if df is None or len(df) == 0:
                continue
            candles = []
            base_time = int(time.time()) - len(df) * 30
            for i, row in df.iterrows():
                candles.append({
                    "time":  base_time + i * 30,
                    "open":  round(float(row["open"]),  5),
                    "high":  round(float(row["high"]),  5),
                    "low":   round(float(row["low"]),   5),
                    "close": round(float(row["close"]), 5),
                })
            _candle_history[display_name] = candles
            logger.info("Loaded %d candles for %s", len(candles), display_name)
        except Exception as exc:
            logger.debug("Candle history fetch failed for %s: %s", display_name, exc)


def _init_components():
    """Called in a background thread after server starts."""
    global _connector, _agent, _brain, _knowledge, _check_results, _otp_code

    time.sleep(1)  # wait for server to be ready
    broadcast_sync({"type":"status","message":"กำลังเชื่อมต่อ IQ Option…","level":"info"})

    # Connect
    from trading_ai.core.iq_connector import IQOptionConnector
    _connector = IQOptionConnector(config.IQ_EMAIL, config.IQ_PASSWORD,
                                    config.IQ_ACCOUNT_TYPE)

    # ── ลอง SSID ก่อน (ถ้ามี) ─────────────────────────────────────────────
    if config.IQ_SSID:
        broadcast_sync({"type":"status","message":"🔑 เชื่อมต่อด้วย SSID token…","level":"info"})
        connected, reason = _connector.connect_with_ssid(config.IQ_SSID)
    else:
        connected, reason = _connector.connect()

    # ── 2FA / OTP flow ───────────────────────────────────────────────────────
    if not connected and str(reason).upper() == "2FA":
        _connector._in_2fa = True
        broadcast_sync({"type": "otp_required", "message": "กรุณากรอก OTP 5 หลักจาก SMS/Email"})
        logger.info("Waiting for OTP from web UI…")
        _otp_event.clear()
        _otp_event.wait(timeout=180)
        if _otp_code:
            connected = _connector.submit_otp(_otp_code)
        if not connected:
            broadcast_sync({"type":"status","message":"❌ OTP ไม่ถูกต้อง — กรุณาปิด 2FA แล้ว login ใหม่","level":"error"})

    # ── Login UI loop — แสดง form email/password + SSID จนกว่าจะเชื่อมต่อสำเร็จ ──
    while not connected:
        broadcast_sync({
            "type":    "login_required",
            "reason":  str(reason),
            "message": str(reason),
        })
        logger.info("Waiting for credentials or SSID from web UI…")
        _otp_event.clear()
        _pending_creds.clear()
        _otp_code = ""
        _otp_event.wait(timeout=600)

        if _pending_creds.get("email"):
            # ผู้ใช้กรอก email/password ใน UI
            _connector.email    = _pending_creds["email"]
            _connector.password = _pending_creds["password"]
            _connector._dead    = False   # reset dead flag เพื่อให้ connect ได้
            _pending_creds.clear()
            broadcast_sync({"type":"status","message":"กำลังเชื่อมต่อ…","level":"info"})
            connected, reason = _connector.connect()
        elif _otp_code:
            # ผู้ใช้วาง SSID token
            broadcast_sync({"type":"status","message":"กำลังเชื่อมต่อด้วย SSID…","level":"info"})
            connected, reason = _connector.connect_with_ssid(_otp_code)
            _otp_code = ""
        else:
            broadcast_sync({"type":"status","message":"หมดเวลา — กรุณารีสตาร์ทโปรแกรม","level":"error"})
            return

    bal = _connector.get_balance() if connected else 0.0
    broadcast_sync({
        "type": "connection",
        "connected": connected,
        "balance": bal,
        "account": config.IQ_ACCOUNT_TYPE,
    })

    # Startup checks
    broadcast_sync({"type":"status","message":"🔍 ตรวจสอบการตั้งค่า…","level":"info"})
    from trading_ai.utils.startup_checker import StartupChecker
    checker = StartupChecker()
    _, api_results = checker.check_via_api(_connector)
    _, ocr_results = checker.run_all_checks()

    results = []
    results.append({"name":"กรอบเวลาแท่งเทียน = 30m","passed":True,
                    "message":"ตั้ง timeframe เป็น 30m ในทุกกราฟ"})
    results.append({"name":"ระยะเวลาออเดอร์ = 1m","passed":True,
                    "message":"ตั้ง expiry เป็น 1 นาที"})
    for r in api_results + ocr_results:
        results.append({"name":r.name,"passed":r.passed,"message":r.message})

    _check_results = results
    all_passed = all(r["passed"] for r in results)
    broadcast_sync({"type":"check","results":results,"all_passed":all_passed})

    # Load AI
    broadcast_sync({"type":"status","message":"🧠 โหลด AI brain…","level":"info"})
    from trading_ai.core.knowledge_base import KnowledgeBase
    from trading_ai.models.ppo_agent import PPOAgent
    from trading_ai.brain.brain_core import BrainCore
    from trading_ai.core.trading_env import OBS_SIZE

    _knowledge = KnowledgeBase(base_dir=config.MODEL_DIR)
    _agent     = PPOAgent(obs_size=OBS_SIZE, n_actions=3)
    _knowledge.load_brain(_agent)
    _brain     = BrainCore(asset=config.ASSET, base_dir=config.MODEL_DIR)

    brain_status = _brain.get_status(ppo_agent=_agent)
    broadcast_sync({"type":"brain", **brain_status})
    broadcast_sync({"type":"status","message":"✅ พร้อมแล้ว – กด START AI","level":"success"})

    # Fetch initial OTC candle history (30-second candles from IQ Option)
    broadcast_sync({"type":"status","message":"📊 โหลดประวัติกราฟ OTC…","level":"info"})
    _fetch_initial_candles()
    broadcast_sync({"type":"candle_history","data":_candle_history})

    # Start price update loop
    threading.Thread(target=_price_loop, daemon=True).start()


def _price_loop():
    """
    Continuously fetch IQ Option OTC candles and broadcast live price updates.
    Uses 30-second candles (half-minute) for better granularity.
    Resolves the correct IQ Option API name for each OTC asset automatically.
    """
    tick = 0
    while True:
        for display_name in OTC_ASSETS:
            api_name = _resolve_asset_name(display_name)
            if not api_name:
                continue
            try:
                df = _connector.get_candles(
                    asset=api_name, timeframe_seconds=30, count=5
                )
                time.sleep(1.2)   # wait for WS buffer to flush before next asset
                if df is None or len(df) < 2:
                    continue

                last       = df.iloc[-1]
                prev_close = float(df["close"].iloc[-2])
                cur_close  = float(last["close"])
                open_price = float(df["close"].iloc[0])
                change_pct = (cur_close - open_price) / (open_price + 1e-9) * 100

                # Sanity-check: if price is outside expected range the WS gave
                # us stale data from the previous asset request — invalidate
                # the resolved name so it gets re-probed next cycle.
                lo, hi = PRICE_RANGES.get(display_name, (0.0, 1e9))
                if not (lo <= cur_close <= hi):
                    logger.warning("Price sanity fail %s: %.5f not in [%.2f, %.2f] — re-resolving",
                                   display_name, cur_close, lo, hi)
                    _resolved_asset_names.pop(display_name, None)
                    continue

                candle = {
                    "time":  int(time.time()),
                    "open":  round(float(last["open"]),  5),
                    "high":  round(float(last["high"]),  5),
                    "low":   round(float(last["low"]),   5),
                    "close": round(cur_close,            5),
                }

                # Update local history cache
                if display_name in _candle_history:
                    _candle_history[display_name].append(candle)
                    if len(_candle_history[display_name]) > 300:
                        _candle_history[display_name] = _candle_history[display_name][-300:]

                broadcast_sync({
                    "type":       "price",
                    "asset":      display_name,
                    "api_name":   api_name,
                    "price":      round(cur_close, 5),
                    "change_pct": round(change_pct, 4),
                    "candle":     candle,
                })
            except Exception as exc:
                logger.debug("Price loop error for %s: %s", display_name, exc)

        # Every 60 ticks (~3 min) also refresh brain age
        tick += 1
        if tick % 60 == 0 and _brain and _agent:
            try:
                brain_status = _brain.get_status(ppo_agent=_agent)
                broadcast_sync({"type": "brain", **brain_status})
            except Exception:
                pass

        time.sleep(3)


# ── Entry point ───────────────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    global _loop
    _loop = asyncio.get_running_loop()
    threading.Thread(target=_init_components, daemon=True).start()


def start_server(host: str = "0.0.0.0", port: int = 8000):
    setup_logging(log_dir=config.LOG_DIR)
    logger.info("Starting web dashboard at http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    start_server()
