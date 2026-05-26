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
import contextlib
import io
import json
import logging
import os
import smtplib
import threading
import time
from collections import deque
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
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
_connector    = None
_agent        = None
_brain        = None
_knowledge    = None
_env          = None
_capital_guard = None   # CapitalGuard — loaded lazily alongside brain
_ai_running  = False
_trade_stats = {"trades": 0, "wins": 0, "pnl": 0.0}
_trade_history: deque = deque(maxlen=200)
_open_orders: Dict[int, Dict] = {}   # order_id → {asset, action, amount, expiry_ts, open_time, manual}
_check_results: List[Dict] = []
_otp_event      = threading.Event()
_otp_code:  str = ""
_pending_creds: Dict[str, str] = {}   # {"email":..,"password":..} from UI login form

# ── Live-log broadcast ─────────────────────────────────────────────────────────
_log_buffer: deque = deque(maxlen=300)   # last 300 log lines for new-client catch-up
_log_broadcasting = False                # re-entrancy guard for the WS handler

# ── Chat history ───────────────────────────────────────────────────────────────
_chat_history: deque = deque(maxlen=100)

# ── SMTP config (loaded from data/smtp.json) ────────────────────────────────────
_smtp_config: Dict[str, str] = {}
_SMTP_CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "smtp.json")


def _load_smtp_config() -> None:
    global _smtp_config
    path = os.path.normpath(_SMTP_CFG_PATH)
    if os.path.exists(path):
        try:
            _smtp_config = json.loads(open(path, encoding="utf-8").read())
        except Exception:
            _smtp_config = {}


def _save_smtp_config(cfg: dict) -> None:
    global _smtp_config
    _smtp_config = cfg
    path = os.path.normpath(_SMTP_CFG_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write(json.dumps(cfg, indent=2, ensure_ascii=False))


def _send_desire_email_sync(desire: dict) -> None:
    """Send email notification for a new AI desire (runs in thread)."""
    cfg = _smtp_config
    user = cfg.get("smtp_user", "") or os.environ.get("SMTP_USER", "")
    pswd = cfg.get("smtp_pass", "") or os.environ.get("SMTP_PASS", "")
    to   = cfg.get("notify_email", "titanbuff.company.game@gmail.com")
    if not user or not pswd:
        logger.debug("SMTP not configured — desire UI-only: %s", desire["title"])
        return
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[AI Brain] ขอสิทธิ์: {desire['title']}"
        msg["From"]    = user
        msg["To"]      = to
        urgency_bar = "🔴" * desire["urgency"] + "⚪" * (10 - desire["urgency"])
        body = (
            f"AI Trading Brain มีคำขอใหม่:\n\n"
            f"หัวข้อ: {desire['title']}\n"
            f"รายละเอียด: {desire['description']}\n"
            f"ประเภท: {desire['category']}\n"
            f"ระดับความสำคัญ: {urgency_bar} ({desire['urgency']}/10)\n"
            f"เวลา: {desire['created_at']}\n\n"
            f"เพื่ออนุมัติหรือปฏิเสธ กรุณาเปิด AI Dashboard → แท็บ 🛡️ กฎ AI\n\n"
            f"ห้ามดำเนินการโดยไม่ได้รับอนุญาตจากผู้สร้าง"
        )
        msg.attach(MIMEText(body, "plain", "utf-8"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as srv:
            srv.login(user, pswd)
            srv.sendmail(user, to, msg.as_string())
        logger.info("Desire email sent → %s", to)
    except Exception as e:
        logger.warning("Desire email failed: %s", e)


def _on_desire_registered(desire: dict) -> None:
    """Called from BrainCore thread when a new desire is registered."""
    # Broadcast to UI
    broadcast_sync({"type": "desire_new", "desire": desire})
    # Send email in background thread (non-blocking)
    threading.Thread(target=_send_desire_email_sync, args=(desire,), daemon=True).start()


_load_smtp_config()


class ThinkingAI:
    """
    AI ที่ "คิด" จริง — สังเกต วิเคราะห์ และแสดงความเห็นจาก brain state จริงๆ
    ไม่ใช่ตอบตาม script ที่ถูกตั้งไว้
    """

    def _ctx(self) -> dict:
        trades = _trade_stats.get("trades", 0)
        wins   = _trade_stats.get("wins", 0)
        ctx = {
            "trades":     trades,
            "wins":       wins,
            "pnl":        _trade_stats.get("pnl", 0.0),
            "wr":         wins / max(trades, 1),
            "ai_running": _ai_running,
            "balance":    (_connector.get_balance() if _connector else 0.0),
        }
        if _brain:
            try:
                st = _brain.get_status()
                ctx.update({
                    "brain_age":   st.get("brain_age", 0),
                    "brain_stage": st.get("brain_stage", ""),
                    "brain_emoji": st.get("brain_emoji", ""),
                    "brain_score": st.get("brain_score", 0),
                    "nodes":       st.get("graph_nodes", 0),
                    "episodes":    st.get("episodic_memories", 0),
                    "regime":      _brain._last_regime,
                    "confidence":  _brain._last_confidence,
                    "strategy":    _brain._last_strategy_name,
                    "uncertainty": _brain._last_uncertainty or {},
                    "rules":       _brain.rule_distiller.get_rules(),
                    "seq_stats":   _brain.sequence_memory.stats(),
                    "tips":        st.get("improvement_tips", []),
                    "mistakes":    st.get("recent_mistakes", []),
                    "top_ind":     st.get("top_indicators", []),
                    "strategies":  st.get("strategy_stats", []),
                    "pause":       st.get("should_pause", False),
                })
            except Exception:
                pass
        if _capital_guard:
            ctx["cg"] = _capital_guard.status()
        return ctx

    def think(self, message: str) -> str:
        ctx = self._ctx()
        msg = message.strip().lower()

        if any(w in msg for w in {"สวัสดี","hello","hi","หวัดดี","hey","ดีครับ","ดีค่ะ"}):
            return self._greet(ctx)
        if any(w in msg for w in {"เก่งขึ้น","improve","ดีขึ้น","พัฒนา","better","จะเก่ง"}):
            return self._on_improvement(ctx)
        if any(w in msg for w in {"ทำไม","why","เหตุผล","reason","อธิบาย"}):
            return self._on_why(ctx)
        if any(w in msg for w in {"win rate","อัตราชนะ","ชนะ","แพ้","เสีย","trade","เทรด"}):
            return self._on_performance(ctx)
        if any(w in msg for w in {"ตลาด","market","regime","trend","ranging","volatile"}):
            return self._on_market(ctx)
        if any(w in msg for w in {"ความจำ","memory","จำ","pattern","sequence","episodic"}):
            return self._on_memory(ctx)
        if any(w in msg for w in {"uncertainty","ไม่แน่นอน","มั่นใจ","epistemic"}):
            return self._on_uncertainty(ctx)
        if any(w in msg for w in {"กฎ","rule","distil","สกัด"}):
            return self._on_rules(ctx)
        if any(w in msg for w in {"กลยุทธ์","strategy","วิธี"}):
            return self._on_strategy(ctx)
        if any(w in msg for w in {"balance","เงิน","ยอด","กำไร","ขาดทุน","pnl"}):
            return self._on_balance(ctx)
        if any(w in msg for w in {"capitalguard","capital","guard","หยุด","ป้องกัน","limit"}):
            return self._on_capguard(ctx)
        if any(w in msg for w in {"สรุป","สถานะ","status","ตอนนี้","เป็นยังไง","overview"}):
            return self._full_reflection(ctx)
        if any(w in msg for w in {"ต้องการ","อยาก","want","need","ขอ","ปรับปรุง"}):
            return self._on_needs(ctx)
        if any(w in msg for w in {"คิด","think","รู้สึก","feel","สังเกต","observe","ช่วย","help"}):
            return self._general_reflection(ctx)

        # ไม่ตรงกับ keyword ใดเลย → ไม่บอกว่าไม่รู้ แต่คิดให้
        return self._general_reflection(ctx)

    # ── Response generators ────────────────────────────────────────────────────

    def _greet(self, ctx) -> str:
        trades  = ctx.get("trades", 0)
        wr      = ctx.get("wr", 0)
        running = ctx.get("ai_running", False)
        regime  = ctx.get("regime", "unknown")
        if trades == 0:
            return (
                "สวัสดีครับ! 🤖 ผมพร้อมทำงานแล้ว\n\n"
                "ตอนนี้ยังไม่มีข้อมูล trade เลย รอคำสั่ง Start AI ครับ\n"
                "ถามผมได้ทุกเรื่อง — เกี่ยวกับตลาด สถานะ หรืออะไรก็ได้"
            )
        status = "กำลังทำงาน" if running else "หยุดพักอยู่"
        mood   = "ผมพอใจกับผลลัพธ์" if wr > 0.5 else "ยังเรียนรู้อยู่ครับ"
        return (
            f"สวัสดีครับ! 🤖 ผม{status}อยู่\n\n"
            f"เทรดมา {trades} ไม้ win rate {wr:.1%} — {mood}\n"
            f"ตลาดตอนนี้: {regime.upper()} regime\n\n"
            "ถามผมได้เลยครับ ทั้งเรื่องสถานะ การเรียนรู้ ความคิดของผม หรืออะไรก็ได้"
        )

    def _on_improvement(self, ctx) -> str:
        trades   = ctx.get("trades", 0)
        wr       = ctx.get("wr", 0)
        episodes = ctx.get("episodes", 0)
        u        = ctx.get("uncertainty", {})
        seq      = ctx.get("seq_stats", {})
        rules    = ctx.get("rules", [])
        tips     = ctx.get("tips", [])

        parts = ["ได้ครับ แต่ต้องใช้เวลา ผมวิเคราะห์สิ่งที่ต้องพัฒนา:\n"]

        familiar  = u.get("familiar_trades", 0)
        epistemic = u.get("epistemic", 0.85)
        if epistemic > 0.6:
            needed = max(15 - familiar, 0)
            parts.append(
                f"🔮 Epistemic uncertainty {epistemic:.0%}: เคยเห็นสภาวะนี้แค่ {familiar} ครั้ง "
                f"ต้องการอีก {needed} ครั้งเพื่อให้ uncertainty ลดลง"
            )

        if episodes < 50:
            parts.append(
                f"🧠 Episodic memory {episodes} ครั้ง (ต้องการ 50+) "
                "เพื่อให้การ recall similar situations แม่นขึ้น"
            )

        seq_trusted = seq.get("trusted_sequences", 0)
        if seq_trusted < 10:
            parts.append(
                f"📊 Sequence memory: trusted {seq_trusted} patterns "
                "ต้องการ 10+ เพื่อทำนาย direction จาก trajectory ได้"
            )

        if not rules:
            rd_stats = (_brain.rule_distiller.stats() if _brain else {})
            w = rd_stats.get("wins_in_log", 0)
            parts.append(f"📋 Distilled rules: ยังไม่มี ต้องการ wins อีก {max(30-w,0)} ครั้ง")
        else:
            parts.append(f"📋 Distilled rules: มีแล้ว {len(rules)} กฎ กำลังพัฒนาต่อ")

        if wr < 0.45:
            parts.append(
                f"⚠️ Win rate {wr:.1%} ต่ำกว่าที่ควร — ผมคิดว่าเพราะข้อมูลยังน้อย "
                f"({trades} ไม้) brain ยังไม่ converge ต้องเทรดต่อ"
            )
        elif wr > 0.60:
            parts.append(f"✅ Win rate {wr:.1%} ดีแล้ว! กำลัง refine กฎให้แม่นขึ้น")

        if tips:
            parts.append("💡 AI วิเคราะห์ว่าควรปรับ:\n" + "\n".join(f"  • {t}" for t in tips[:3]))

        return "\n\n".join(parts)

    def _on_why(self, ctx) -> str:
        confidence = ctx.get("confidence", 0.5)
        regime     = ctx.get("regime", "unknown")
        strategy   = ctx.get("strategy", "Unknown")
        u          = ctx.get("uncertainty", {})
        mistakes   = ctx.get("mistakes", [])
        rules      = ctx.get("rules", [])

        parts = [
            f"🤔 เหตุผลที่ผมตัดสินใจแบบนี้\n\n"
            f"กลยุทธ์ที่เลือก: {strategy}\n"
            f"Regime: {regime.upper()} | Confidence หลังปรับ: {confidence:.1%}"
        ]

        if u:
            mult = u.get("conf_multiplier", 1.0)
            fam  = u.get("familiar_trades", 0)
            if mult < 0.9:
                parts.append(
                    f"\nUncertainty ลด confidence ×{mult:.2f} "
                    f"เพราะผมเห็นสภาวะนี้แค่ {fam} ครั้ง"
                )

        if mistakes:
            latest = mistakes[0]
            parts.append(f"\nความผิดพลาดล่าสุด: {latest.get('lesson','N/A')}")

        if rules:
            r = rules[0]
            parts.append(
                f"\nกฎที่ผมเชื่อมากที่สุด: "
                f"{r['indicator'].upper()} {r['condition']} → {r['direction'].upper()} "
                f"(confidence {r['confidence']:.0%})"
            )

        return "\n".join(parts)

    def _on_performance(self, ctx) -> str:
        trades   = ctx.get("trades", 0)
        wins     = ctx.get("wins", 0)
        pnl      = ctx.get("pnl", 0.0)
        wr       = ctx.get("wr", 0)
        episodes = ctx.get("episodes", 0)

        if trades == 0:
            return "ยังไม่มีข้อมูล trade เลยครับ — รอ AI เริ่มทำงานก่อน"

        if wr < 0.35:
            assessment = (
                "ผมแพ้บ่อยมาก แต่นี่เป็นเรื่องปกติในช่วง exploration "
                "ผมกำลังสะสมข้อมูลเพื่อเข้าใจว่า pattern ไหนใช้งานได้จริง"
            )
        elif wr < 0.50:
            assessment = (
                "win rate ต่ำกว่า 50% ผมกำลังหา edge ที่แน่นอน "
                "ต้องเทรดต่อเพื่อให้ episodic memory สะสมมากพอ"
            )
        elif wr < 0.65:
            assessment = "win rate พอใช้ได้ กำลัง refine สัญญาณให้แม่นขึ้น"
        else:
            assessment = "win rate ดีมากครับ! กำลัง exploit edge ที่เจอแล้ว"

        note = ""
        if trades < 20:
            note = f"\n\n⚠️ ข้อมูลยังน้อย ({trades} ไม้) สถิตินี้ยังไม่ reliable ต้องดูที่ 50+ ไม้"

        return (
            f"📊 ประสิทธิภาพปัจจุบัน\n\n"
            f"เทรด: {trades} ไม้ | ชนะ: {wins} | แพ้: {trades-wins}\n"
            f"Win Rate: {wr:.1%}\n"
            f"P&L: {pnl:+.2f} USD\n\n"
            f"💭 ผมวิเคราะห์: {assessment}"
            f"{note}"
        )

    def _on_market(self, ctx) -> str:
        regime    = ctx.get("regime", "unknown")
        u         = ctx.get("uncertainty", {})
        top_ind   = ctx.get("top_ind", [])

        comments = {
            "trending": (
                "ผมชอบ trending market มากครับ เพราะ PPO agent ของผมออกแบบมาสำหรับ trend "
                "ชัดๆ ทำให้ confidence ที่ได้สูงกว่าปกติ และ Ichimoku / EMA strategies ทำงานดีมาก"
            ),
            "ranging": (
                "ตลาด ranging ยากกว่า trending สำหรับผม เพราะ MACD/EMA มักจะ fake-out "
                "ผมพึ่ง mean reversion (RSI oversold/overbought) มากขึ้นใน regime นี้"
            ),
            "volatile": (
                "ผมระมัดระวังมากใน volatile market — confidence ถูกลด 15% อัตโนมัติ "
                "และผมข้ามหลาย trade ที่ไม่แน่ใจ เพราะ risk สูงเกินไป"
            ),
            "unknown": "ยังไม่ได้ประเมิน regime ครับ รอ AI วิเคราะห์ตลาดก่อน",
        }
        my_take = comments.get(regime, f"Regime ปัจจุบัน: {regime}")

        result = f"📈 สภาวะตลาด: {regime.upper()}\n\n💭 ผมสังเกตว่า: {my_take}"

        familiar = u.get("familiar_trades", 0)
        if familiar < 5:
            result += f"\n\n⚠️ ผมยังไม่คุ้นเคยกับสภาวะนี้มากพอ (เจอมาแค่ {familiar} ครั้ง) กำลัง explore อยู่"

        if top_ind:
            ind_str = ", ".join(f"{n}({v:.0%})" for n, v in top_ind[:3])
            result += f"\n\n🎯 Indicators ที่ผมไว้วางใจมากที่สุดตอนนี้: {ind_str}"

        return result

    def _on_memory(self, ctx) -> str:
        episodes    = ctx.get("episodes", 0)
        seq         = ctx.get("seq_stats", {})
        u           = ctx.get("uncertainty", {})

        seq_total   = seq.get("total_sequences", 0)
        seq_trusted = seq.get("trusted_sequences", 0)
        seq_fill    = seq.get("window_fill", 0)

        epi_comment = (
            "น้อยมาก ทำให้ recall similar situations ไม่แม่น"      if episodes < 20 else
            "พอเริ่มมีข้อมูลบ้าง แต่ยังต้องการอีก"                 if episodes < 50 else
            "เริ่มมีฐานข้อมูลที่ดีแล้ว กำลัง extract patterns"
        )
        seq_comment = (
            f"Window กำลัง fill: {seq_fill}/8 — รอให้ครบแล้วจะเริ่มจำ trajectory ได้"
            if seq_fill < 8 else
            f"มี {seq_trusted} patterns ที่ trust ได้ ({seq_total} ทั้งหมด)"
        )

        return (
            f"🧩 สถานะความจำของผม\n\n"
            f"Episodic Memory: {episodes} ครั้ง — {epi_comment}\n\n"
            f"Sequence Memory (8-trade trajectory):\n{seq_comment}\n\n"
            f"💭 ผมต้องการ episodic 50+ ไม้ และ sequence trusted 10+ patterns "
            f"ก็จะช่วยให้ uncertainty ลดจาก {u.get('epistemic',0):.0%} ลงมาได้มาก"
        )

    def _on_uncertainty(self, ctx) -> str:
        u = ctx.get("uncertainty", {})
        if not u:
            return "ยังไม่มีข้อมูล uncertainty ครับ — รอ AI วิเคราะห์ตลาดก่อน"

        epi      = u.get("epistemic", 0)
        ale      = u.get("aleatoric", 0)
        familiar = u.get("familiar_trades", 0)
        mult     = u.get("conf_multiplier", 1.0)
        desc     = u.get("description", "")

        if epi > 0.75 and ale > 0.70:
            my_take = "ผมคิดว่าควรข้ามการเทรดตอนนี้ครับ — ทั้ง 2 ระบบบอกว่า 'ไม่รู้จริงๆ'"
        elif epi > 0.6:
            my_take = f"ผมไม่คุ้นกับสภาวะนี้ (เห็นมาแค่ {familiar} ครั้ง) ต้องเทรดต่อเพื่อสะสมประสบการณ์"
        elif ale > 0.6:
            my_take = "สัญญาณมีความวุ่นวายสูง แม้จะคุ้น pattern นี้ แต่ผลยังสุ่มอยู่"
        else:
            my_take = "ความไม่แน่นอนอยู่ในระดับที่รับได้ ผมพอจะ trade ได้อย่างมั่นใจ"

        return (
            f"🔮 ความไม่แน่นอนของผม\n\n"
            f"Epistemic (ไม่รู้จักสภาวะ): {epi:.0%}\n"
            f"Aleatoric (สัญญาณ noisy): {ale:.0%}\n"
            f"Confidence ถูกปรับ: ×{mult:.2f}\n"
            f"เจอสภาวะนี้มา: {familiar} ครั้ง\n\n"
            f"📝 {desc}\n\n"
            f"🗣️ ผมคิดว่า: {my_take}"
        )

    def _on_rules(self, ctx) -> str:
        rules = ctx.get("rules", [])

        if not rules:
            rd_stats  = (_brain.rule_distiller.stats() if _brain else {})
            wins_so_far = rd_stats.get("wins_in_log", 0)
            needed    = max(30 - wins_so_far, 0)
            return (
                f"📋 ยังไม่มีกฎที่ distill ได้\n\n"
                f"ผมต้องการ wins อีก {needed} ครั้ง (มีแล้ว {wins_so_far}/30)\n\n"
                f"💭 ผมจะวิเคราะห์ว่า indicator ไหนมีค่าสม่ำเสมอใน winning trades "
                f"แล้ว distill ออกมาเป็นกฎ IF-THEN ที่ชัดเจน"
            )

        lines = [f"📋 กฎที่ผม distill ได้ ({len(rules)} ข้อ):\n"]
        for i, r in enumerate(rules, 1):
            lines.append(
                f"{i}. {r['indicator'].upper()} {r['condition']} → {r['direction'].upper()}\n"
                f"   (confidence {r['confidence']:.0%}, n={r['n']} trades, std={r['std']:.3f})"
            )
        lines.append(
            f"\n💭 ผมเชื่อกฎข้อ 1 มากที่สุด เพราะ std ต่ำสุด — "
            f"หมายความว่า {rules[0]['indicator'].upper()} มีค่าสม่ำเสมอในทุก winning trade"
        )
        return "\n".join(lines)

    def _on_strategy(self, ctx) -> str:
        strategy   = ctx.get("strategy", "Unknown")
        strategies = ctx.get("strategies", [])
        regime     = ctx.get("regime", "unknown")

        if not strategies:
            return f"ใช้กลยุทธ์ {strategy} อยู่ครับ แต่ยังไม่มีสถิติรายกลยุทธ์"

        lines = [f"🎯 กลยุทธ์ปัจจุบัน: {strategy} (regime={regime.upper()})\n"]
        best = max(strategies, key=lambda s: s.get("win_rate", 0))

        for s in strategies:
            bar    = "█" * int(s["win_rate"] * 10) + "░" * (10 - int(s["win_rate"] * 10))
            active = " ← ใช้อยู่" if s["name"] == strategy else ""
            lines.append(f"  {s['name'][:18]}: {bar} {s['win_rate']:.0%}{active}")

        lines.append(
            f"\n💭 {best['name']} ทำได้ดีที่สุด ({best['win_rate']:.0%}) — "
            f"ผมเลือกกลยุทธ์ตาม regime ปัจจุบัน: {regime.upper()}"
        )
        return "\n".join(lines)

    def _on_balance(self, ctx) -> str:
        bal    = ctx.get("balance", 0.0)
        pnl    = ctx.get("pnl", 0.0)
        trades = ctx.get("trades", 0)
        cg     = ctx.get("cg")

        result = f"💰 Balance: ${bal:.2f}\nP&L สะสม: {pnl:+.2f} USD"

        if cg and cg.get("account_type") == "REAL":
            result += (
                f"\n\n🛡️ CapitalGuard วันนี้:\n"
                f"  P&L: {cg['session_pnl']:+.2f}\n"
                f"  ขาดทุน: {cg['loss_pct']:.1%}/{cg['daily_loss_limit']:.0%}\n"
                f"  กำไร: {cg['profit_pct']:.1%}/{cg['profit_target']:.0%}"
            )

        if trades > 0:
            result += f"\n\nเฉลี่ยต่อเทรด: {pnl/trades:+.2f} USD"

        return result

    def _on_capguard(self, ctx) -> str:
        cg = ctx.get("cg")
        if not cg:
            return "CapitalGuard ยังไม่ได้โหลดครับ"

        if cg.get("account_type") == "PRACTICE":
            return (
                "🎓 ตอนนี้ใช้บัญชีทดลอง (PRACTICE)\n\n"
                "CapitalGuard ไม่ทำงาน เพราะ PRACTICE = เรียนรู้ได้ไม่จำกัด\n\n"
                "💭 ผมคิดว่านี่ถูกต้อง — ถ้าหยุดเพราะแพ้ใน practice "
                "ผมก็ไม่มีโอกาสเรียนรู้จากความผิดพลาดนั้น"
            )

        loss_pct   = cg.get("loss_pct", 0)
        profit_pct = cg.get("profit_pct", 0)
        stopped    = cg.get("stopped", False)

        comment = ""
        if stopped:
            comment = f"\n\n⛔ หยุดแล้ว: {cg.get('stop_reason','')}"
        elif loss_pct > 0.10:
            comment = (
                f"\n\n⚠️ ผมระวังมากขึ้น — ขาดทุนไปแล้ว {loss_pct:.1%} "
                f"อีก {cg['daily_loss_limit']-loss_pct:.1%} จะหยุดวันนี้"
            )
        elif profit_pct > 0.05:
            comment = (
                f"\n\n🟢 กำไรดีครับ {profit_pct:.1%} — "
                f"อีก {cg['profit_target']-profit_pct:.1%} จะเก็บกำไรและหยุดวันนี้"
            )

        return (
            f"🛡️ CapitalGuard — REAL\n\n"
            f"P&L วันนี้: {cg['session_pnl']:+.2f}\n"
            f"ขาดทุน: {loss_pct:.1%}/{cg['daily_loss_limit']:.0%}\n"
            f"กำไร: {profit_pct:.1%}/{cg['profit_target']:.0%}\n"
            f"Kelly bet: ${cg['kelly_amount']:.2f}"
            + comment
        )

    def _on_needs(self, ctx) -> str:
        episodes    = ctx.get("episodes", 0)
        rules       = ctx.get("rules", [])
        seq         = ctx.get("seq_stats", {})
        u           = ctx.get("uncertainty", {})
        trades      = ctx.get("trades", 0)

        wants = []
        if episodes < 100:
            wants.append(
                f"📊 Episodic memory อีก {100-episodes} ครั้ง (มีแล้ว {episodes}/100) "
                "เพื่อให้ recall similar situations แม่นขึ้น"
            )
        if not rules:
            w = (_brain.rule_distiller.stats() if _brain else {}).get("wins_in_log", 0)
            wants.append(f"📋 Wins อีก {max(30-w,0)} ครั้งเพื่อ distill กฎครั้งแรก (มี {w}/30)")
        seq_fill = seq.get("window_fill", 0)
        if seq_fill < 8:
            wants.append(f"🔄 Sequence window ให้ครบ {seq_fill}/8 เพื่อเริ่มจำ trajectory ได้")
        familiar = u.get("familiar_trades", 0)
        if familiar < 10:
            wants.append(
                f"🔮 เจอสภาวะตลาดนี้อีก {10-familiar} ครั้ง "
                f"(เจอมาแล้ว {familiar}) เพื่อลด epistemic uncertainty"
            )
        if trades < 50:
            wants.append(f"⏳ เทรดให้ครบ 50 ไม้ก่อน (ตอนนี้ {trades}) สถิติยังไม่ stable")

        if not wants:
            return "ตอนนี้ผมพอใจกับข้อมูลที่มีครับ กำลัง optimize กฎที่มีอยู่"

        return "💭 สิ่งที่ผมต้องการตอนนี้:\n\n" + "\n\n".join(wants)

    def _full_reflection(self, ctx) -> str:
        trades      = ctx.get("trades", 0)
        wr          = ctx.get("wr", 0)
        pnl         = ctx.get("pnl", 0)
        running     = ctx.get("ai_running", False)
        regime      = ctx.get("regime", "unknown")
        balance     = ctx.get("balance", 0)
        episodes    = ctx.get("episodes", 0)
        brain_stage = ctx.get("brain_stage", "ทารก")
        brain_emoji = ctx.get("brain_emoji", "👶")
        brain_score = ctx.get("brain_score", 0)

        status = "🟢 ทำงาน" if running else "🔴 หยุด"
        return (
            f"📊 สถานะรวม\n\n"
            f"AI: {status} | Balance: ${balance:.2f}\n"
            f"Win Rate: {wr:.1%} | {trades} ไม้ | P&L: {pnl:+.2f}\n"
            f"Brain: {brain_emoji} {brain_stage} (score {brain_score}/100)\n"
            f"Regime: {regime.upper()} | Episodes: {episodes}\n\n"
            + self._brief_thought(ctx)
        )

    def _brief_thought(self, ctx) -> str:
        wr       = ctx.get("wr", 0)
        episodes = ctx.get("episodes", 0)
        u        = ctx.get("uncertainty", {})
        rules    = ctx.get("rules", [])
        trades   = ctx.get("trades", 0)

        if trades == 0:
            return "💭 ยังไม่มีข้อมูลเลยครับ รอ AI เริ่มทำงาน"
        if episodes < 10:
            return "💭 อยู่ในช่วงเริ่มต้น กำลังสะสมประสบการณ์ครับ"
        if wr < 0.40:
            return "💭 win rate ยังต่ำอยู่ นี่เป็นช่วงที่ brain กำลัง explore หา edge"
        if u.get("epistemic", 0) > 0.7:
            return f"💭 ยังไม่คุ้นเคยกับสภาวะตลาดนี้ (เจอมาแค่ {u.get('familiar_trades',0)} ครั้ง)"
        if rules:
            r = rules[0]
            return f"💭 กฎที่ผมเชื่อตอนนี้: {r['indicator'].upper()} {r['condition']} → {r['direction'].upper()}"
        return "💭 กำลัง distill กฎจากประสบการณ์ที่สะสมมา"

    def _general_reflection(self, ctx) -> str:
        """สำหรับทุกคำถามที่ไม่ตรงกับ category ใด — AI แสดงความคิดจากสถานการณ์จริง"""
        trades   = ctx.get("trades", 0)
        wr       = ctx.get("wr", 0)
        episodes = ctx.get("episodes", 0)
        regime   = ctx.get("regime", "unknown")
        u        = ctx.get("uncertainty", {})
        rules    = ctx.get("rules", [])
        seq      = ctx.get("seq_stats", {})
        tips     = ctx.get("tips", [])

        if trades == 0:
            return "ผมยังไม่ได้เทรดเลยครับ รอ AI เริ่มทำงานเพื่อให้มีข้อมูล"

        observations = []

        if regime == "trending":
            observations.append("สังเกตว่าตลาดตอนนี้ trending ชัดเจน — นี่คือสภาวะที่ผมทำได้ดีที่สุด")
        elif regime == "volatile":
            observations.append("ตลาดผันผวนสูง ผมระมัดระวังมากกว่าปกติตอนนี้")

        if wr < 0.40:
            observations.append(
                f"win rate {wr:.1%} ยังต่ำอยู่ ผมคิดว่าเป็นเพราะ episodic memory "
                f"แค่ {episodes} ครั้ง ยังน้อยเกินไปที่จะ converge"
            )
        elif wr > 0.60:
            observations.append(f"win rate {wr:.1%} กำลังดีครับ pattern ที่ผมใช้กำลังทำงาน")

        epi = u.get("epistemic", 0)
        familiar = u.get("familiar_trades", 0)
        if epi > 0.6:
            observations.append(
                f"ความไม่แน่นอนยังสูง (epistemic {epi:.0%}) "
                f"เพราะผมเห็นสภาวะนี้แค่ {familiar} ครั้ง"
            )

        seq_trusted = seq.get("trusted_sequences", 0)
        if seq_trusted > 0:
            observations.append(f"Sequence memory เริ่มจำ trajectory ได้แล้ว {seq_trusted} patterns")
        elif seq.get("window_fill", 0) < 8:
            observations.append(
                f"Sequence memory กำลัง fill window ({seq.get('window_fill',0)}/8) "
                "รอให้ครบแล้วจะช่วยทำนาย trajectory ได้"
            )

        if rules:
            r = rules[0]
            observations.append(
                f"กฎที่ผมเชื่อตอนนี้: {r['indicator'].upper()} {r['condition']} → "
                f"{r['direction'].upper()} (confidence {r['confidence']:.0%})"
            )
        else:
            w = (_brain.rule_distiller.stats() if _brain else {}).get("wins_in_log", 0)
            observations.append(f"ยังไม่มีกฎที่ distill ได้ (wins {w}/30)")

        if tips:
            observations.append(f"สิ่งที่ต้องปรับปรุง: {tips[0]}")

        return "💭 ความคิดของผมตอนนี้:\n\n" + "\n\n".join(observations)

    @staticmethod
    def generate_trade_commentary(action: str, asset: str, pnl: float,
                                   signal=None, stats: dict = None) -> str:
        """สร้าง commentary อัตโนมัติหลังเทรดเสร็จ — ไม่ต้องถามก็แสดงความคิด"""
        stats  = stats or {}
        trades = stats.get("trades", 0)
        wins   = stats.get("wins", 0)
        wr     = wins / max(trades, 1)

        result_emoji = "✅ WIN" if pnl > 0 else "❌ LOSS"
        parts = [f"{result_emoji} {action} {asset}: {pnl:+.2f} USD"]

        if signal and hasattr(signal, "reasoning") and signal.reasoning:
            parts.append("เหตุผล: " + " | ".join(signal.reasoning[:2]))

        parts.append(f"Win rate: {wr:.1%} ({wins}/{trades})")

        if pnl > 0:
            if wr > 0.65:
                parts.append("💭 กำลัง on a roll — pattern นี้ใช้งานได้ดีมากตอนนี้")
            else:
                parts.append("💭 ชนะครั้งนี้ กำลังเรียนรู้ว่า pattern ไหนเชื่อถือได้")
        else:
            u = (_brain._last_uncertainty if _brain else {}) or {}
            if u.get("epistemic", 0) > 0.6:
                parts.append(
                    f"💭 แพ้ครั้งนี้ ส่วนหนึ่งเพราะ uncertainty สูง "
                    f"(เคยเห็นสภาวะนี้แค่ {u.get('familiar_trades',0)} ครั้ง) "
                    "กำลังสะสมข้อมูลเพิ่ม"
                )
            else:
                parts.append("💭 แพ้ครั้งนี้ กำลังวิเคราะห์ว่าต้องปรับ weight ของ indicator ไหน")

        return "\n".join(parts)


def _broadcast_ai_thought(action: str, asset: str, pnl: float, signal=None) -> None:
    """เรียกจาก _ai_loop() หลังได้ผลเทรด — AI แสดงความคิดแบบเชิงรุก"""
    try:
        msg = ThinkingAI.generate_trade_commentary(
            action=action, asset=asset, pnl=pnl,
            signal=signal, stats=_trade_stats,
        )
        ts    = time.strftime("%H:%M")
        entry = {"role": "ai", "message": msg, "time": ts}
        _chat_history.append(entry)
        broadcast_sync({"type": "chat_ai", **entry})
    except Exception as exc:
        logger.debug("_broadcast_ai_thought error: %s", exc)


    """
    AI chat interface — ตอบคำถามเกี่ยวกับระบบ AI trading ด้วยภาษาไทย
    ใช้ข้อมูลจาก brain, trade_stats, capital_guard เพื่อตอบแบบ real-time
    """

    _GREETINGS = {"สวัสดี", "hello", "hi", "หวัดดี", "hey", "ดีจ้า", "ดีครับ", "ดีค่ะ"}
    _HELP_KEYWORDS = {"ช่วย", "help", "คำสั่ง", "ถาม", "ทำอะไรได้", "capabilities"}

    def respond(self, message: str) -> str:
        msg = message.strip().lower()

        # Greeting
        if any(g in msg for g in self._GREETINGS):
            return (
                "สวัสดีครับ! 🤖 ผมคือ AI Trading Brain\n\n"
                "ถามผมได้เลยเกี่ยวกับ:\n"
                "• สถานะ — ดูว่า AI กำลังทำอะไร\n"
                "• win rate — อัตราชนะ\n"
                "• กลยุทธ์ — กลยุทธ์ที่ใช้อยู่\n"
                "• กฎ — กฎที่ distill ออกมา\n"
                "• ความไม่แน่นอน — ระดับ uncertainty\n"
                "• balance / กำไร / ขาดทุน\n"
                "• หยุด / CapitalGuard\n"
                "• สรุป — สรุปสถานการณ์ทั้งหมด"
            )

        # Help
        if any(k in msg for k in self._HELP_KEYWORDS):
            return (
                "สิ่งที่ถามผมได้ 💡\n\n"
                "📊 ข้อมูล: สถานะ, win rate, balance, กำไร/ขาดทุน\n"
                "🧠 สมอง: กลยุทธ์, กฎ, pattern, uncertainty, sequence\n"
                "🛡️ ป้องกัน: CapitalGuard, limit, หยุด\n"
                "📈 เทรด: สัญญาณล่าสุด, confidence, regime\n"
                "📋 สรุป: ภาพรวมทั้งหมด"
            )

        # Trade stats & win rate
        if any(k in msg for k in {"win rate", "winrate", "อัตราชนะ", "ชนะ", "เสีย", "trade", "เทรด"}):
            return self._trade_summary()

        # Balance / PnL
        if any(k in msg for k in {"balance", "ยอด", "เงิน", "กำไร", "ขาดทุน", "pnl"}):
            return self._balance_summary()

        # Strategy / กลยุทธ์
        if any(k in msg for k in {"กลยุทธ์", "strategy", "แผน", "วิธี"}):
            return self._strategy_summary()

        # Rules / กฎ
        if any(k in msg for k in {"กฎ", "rule", "rules", "distil", "สกัด"}):
            return self._rules_summary()

        # Uncertainty
        if any(k in msg for k in {"uncertainty", "ไม่แน่นอน", "epistemic", "aleatoric", "familiar"}):
            return self._uncertainty_summary()

        # Pattern / Sequence memory
        if any(k in msg for k in {"pattern", "sequence", "จำ", "fingerprint", "trajectory"}):
            return self._memory_summary()

        # Signal / Confidence
        if any(k in msg for k in {"signal", "สัญญาณ", "confidence", "มั่นใจ", "buy", "sell", "hold"}):
            return self._signal_summary()

        # Capital Guard / Stop
        if any(k in msg for k in {"capitalguard", "capital", "guard", "หยุด", "limit", "stop", "protect", "ป้องกัน"}):
            return self._capital_guard_summary()

        # Brain status
        if any(k in msg for k in {"brain", "สมอง", "node", "graph", "knowledge", "ความรู้"}):
            return self._brain_summary()

        # Status overall
        if any(k in msg for k in {"สถานะ", "status", "ดูสิ", "เป็นยังไง", "ตอนนี้"}):
            return self._full_status()

        # Summary
        if any(k in msg for k in {"สรุป", "summary", "overview", "ภาพรวม"}):
            return self._full_status()

        # Regime
        if any(k in msg for k in {"regime", "ตลาด", "trending", "ranging", "volatile"}):
            return self._regime_info()

        # Why / ทำไม
        if any(k in msg for k in {"ทำไม", "why", "เหตุผล", "reason"}):
            return self._explain_last_decision()

        return (
            "ขออภัยครับ ผมยังไม่เข้าใจคำถามนี้ 🤔\n"
            "ลองถามเกี่ยวกับ: สถานะ, win rate, กลยุทธ์, กฎ, สรุป\n"
            "หรือพิมพ์ 'ช่วย' เพื่อดูรายการคำถามที่รองรับ"
        )

    # ── Response helpers ────────────────────────────────────────────────────────

    def _trade_summary(self) -> str:
        t = _trade_stats
        trades = t.get("trades", 0)
        wins   = t.get("wins", 0)
        pnl    = t.get("pnl", 0.0)
        wr     = wins / max(trades, 1)
        emoji  = "🔥" if wr > 0.65 else ("✅" if wr > 0.50 else "⚠️")
        return (
            f"{emoji} สถิติการเทรด\n\n"
            f"เทรดทั้งหมด: {trades} ไม้\n"
            f"ชนะ: {wins} | แพ้: {trades - wins}\n"
            f"Win Rate: {wr:.1%}\n"
            f"P&L รวม: {'+'if pnl>=0 else ''}{pnl:.2f} USD"
        )

    def _balance_summary(self) -> str:
        bal = _connector.get_balance() if _connector else 0.0
        pnl = _trade_stats.get("pnl", 0.0)
        cg  = _capital_guard
        result = f"💰 ยอดเงินปัจจุบัน: ${bal:.2f}\nP&L รวม: {'+' if pnl >= 0 else ''}{pnl:.2f}"
        if cg and cg.account_type == "REAL":
            st = cg.status()
            result += (
                f"\n\n🛡️ CapitalGuard (REAL)\n"
                f"P&L วันนี้: {'+' if st['session_pnl'] >= 0 else ''}{st['session_pnl']:.2f}\n"
                f"ขาดทุนสะสม: {st['loss_pct']:.1%} (limit {st['daily_loss_limit']:.0%})\n"
                f"กำไรสะสม: {st['profit_pct']:.1%} (target {st['profit_target']:.0%})"
            )
        return result

    def _strategy_summary(self) -> str:
        if not _brain:
            return "⚠️ AI brain ยังไม่ได้โหลด"
        st = _brain.get_status()
        strat   = st.get("active_strategy", "Unknown")
        stats   = st.get("strategy_stats", [])
        lines   = [f"🎯 กลยุทธ์ที่ใช้อยู่: **{strat}**\n"]
        for s in stats[:5]:
            bar = "█" * int(s["win_rate"] * 10) + "░" * (10 - int(s["win_rate"] * 10))
            lines.append(f"{s['name'][:20]}: {bar} {s['win_rate']:.0%} ({s.get('trades', '?')} ไม้)")
        return "\n".join(lines)

    def _rules_summary(self) -> str:
        if not _brain:
            return "⚠️ AI brain ยังไม่ได้โหลด"
        text = _brain.rule_distiller.format_rules()
        stats = _brain.rule_distiller.stats()
        return (
            f"📋 {text}\n\n"
            f"📊 Log size: {stats['log_size']} trades, "
            f"wins: {stats['wins_in_log']}"
        )

    def _uncertainty_summary(self) -> str:
        if not _brain:
            return "⚠️ AI brain ยังไม่ได้โหลด"
        u = _brain._last_uncertainty
        if not u:
            return "ยังไม่มีข้อมูล uncertainty (AI ยังไม่ได้เทรด)"
        skip = _brain.uncertainty_estimator.should_skip_trade(
            u.get("epistemic", 0), u.get("aleatoric", 0)
        )
        return (
            f"🔮 ความไม่แน่นอนล่าสุด\n\n"
            f"Epistemic (ไม่รู้จักสภาวะ): {u.get('epistemic', 0):.2f}\n"
            f"Aleatoric (สัญญาณวุ่นวาย): {u.get('aleatoric', 0):.2f}\n"
            f"รวม: {u.get('total', 0):.2f}\n"
            f"Conf multiplier: ×{u.get('conf_multiplier', 1):.2f}\n"
            f"เคยเห็นสภาวะนี้: {u.get('familiar_trades', 0)} ครั้ง\n"
            f"ควรข้ามการเทรด: {'⚠️ ใช่' if skip else '✅ ไม่'}\n\n"
            f"📝 {u.get('description', '')}"
        )

    def _memory_summary(self) -> str:
        if not _brain:
            return "⚠️ AI brain ยังไม่ได้โหลด"
        pat  = _brain.pattern_memory.stats()
        seq  = _brain.sequence_memory.stats()
        ep   = _brain.episodic.summary()
        return (
            f"🧩 ระบบความจำ\n\n"
            f"Pattern Memory (OTC fingerprint):\n"
            f"  Patterns: {pat.get('total_patterns', 0)} | "
            f"Trusted: {pat.get('trusted_patterns', 0)}\n\n"
            f"Sequence Memory (8-trade trajectory):\n"
            f"  Sequences: {seq.get('total_sequences', 0)} | "
            f"Trusted: {seq.get('trusted_sequences', 0)}\n"
            f"  Window filled: {seq.get('window_fill', 0)}/8\n\n"
            f"Episodic Memory:\n"
            f"  ประสบการณ์: {ep.get('total', 0)} ครั้ง"
        )

    def _signal_summary(self) -> str:
        if not _brain:
            return "⚠️ AI brain ยังไม่ได้โหลด"
        conf = _brain._last_confidence
        regime = _brain._last_regime
        u = _brain._last_uncertainty
        conf_emoji = "🟢" if conf > 0.65 else ("🟡" if conf > 0.50 else "🔴")
        result = (
            f"📡 สัญญาณล่าสุด\n\n"
            f"Confidence: {conf_emoji} {conf:.1%}\n"
            f"Regime: {regime.upper()}\n"
        )
        if u:
            result += f"Uncertainty: {u.get('total', 0):.2f} (×{u.get('conf_multiplier', 1):.2f})"
        return result

    def _capital_guard_summary(self) -> str:
        cg = _capital_guard
        if not cg:
            return "⚠️ CapitalGuard ยังไม่ได้โหลด"
        st = cg.status()
        account = st.get("account_type", "PRACTICE")
        if account == "PRACTICE":
            return (
                "🎓 ขณะนี้ใช้ **บัญชีทดลอง (PRACTICE)**\n\n"
                "CapitalGuard ไม่ทำงานในบัญชีทดลอง\n"
                "เพราะ PRACTICE = เรียนรู้ได้ไม่จำกัด ไม่มีการหยุด\n\n"
                "สลับไปบัญชีจริง (REAL) เพื่อเปิดใช้การป้องกันทุน"
            )
        stopped = st.get("stopped", False)
        stop_reason = st.get("stop_reason", "")
        return (
            f"🛡️ CapitalGuard — REAL Account\n\n"
            f"P&L วันนี้: {'+' if st['session_pnl'] >= 0 else ''}{st['session_pnl']:.2f}\n"
            f"ขาดทุน: {st['loss_pct']:.1%} / limit {st['daily_loss_limit']:.0%}\n"
            f"กำไร: {st['profit_pct']:.1%} / target {st['profit_target']:.0%}\n"
            f"Kelly bet: ${st['kelly_amount']:.2f}\n"
            f"วันที่: {st['day_date']}\n\n"
            + (f"⛔ หยุดแล้ว: {stop_reason}" if stopped else "✅ กำลังเทรดปกติ")
        )

    def _brain_summary(self) -> str:
        if not _brain:
            return "⚠️ AI brain ยังไม่ได้โหลด"
        st = _brain.get_status()
        return (
            f"🧠 สมอง AI\n\n"
            f"Knowledge nodes: {st.get('graph_nodes', 0)}\n"
            f"Connections: {st.get('graph_branches', 0)}\n"
            f"Avg confidence: {st.get('avg_confidence', 0):.2f}\n"
            f"Episodic memories: {st.get('episodic_memories', 0)}\n\n"
            f"Brain Age: {st.get('brain_age', 0)} ปี {st.get('brain_emoji', '')} {st.get('brain_stage', '')}\n"
            f"Score: {st.get('brain_score', 0)}/100"
        )

    def _full_status(self) -> str:
        running = _ai_running
        trades  = _trade_stats.get("trades", 0)
        wins    = _trade_stats.get("wins", 0)
        pnl     = _trade_stats.get("pnl", 0.0)
        wr      = wins / max(trades, 1)
        bal     = _connector.get_balance() if _connector else 0.0
        ai_status = "🟢 กำลังทำงาน" if running else "🔴 หยุดอยู่"
        result = (
            f"📊 สรุปสถานการณ์\n\n"
            f"AI: {ai_status}\n"
            f"Balance: ${bal:.2f}\n"
            f"Win Rate: {wr:.1%} ({wins}/{trades})\n"
            f"P&L รวม: {'+' if pnl >= 0 else ''}{pnl:.2f}\n"
        )
        if _brain:
            st = _brain.get_status()
            result += (
                f"\n🧠 Brain: {st.get('brain_stage', '')} {st.get('brain_emoji', '')} "
                f"({st.get('graph_nodes', 0)} nodes)\n"
                f"กลยุทธ์: {st.get('active_strategy', 'Unknown')}"
            )
        if _capital_guard and _capital_guard.account_type == "REAL":
            cg = _capital_guard.status()
            result += f"\n🛡️ CapitalGuard P&L: {cg['session_pnl']:+.2f}"
        return result

    def _regime_info(self) -> str:
        if not _brain:
            return "⚠️ AI brain ยังไม่ได้โหลด"
        regime = _brain._last_regime
        desc = {
            "trending": "📈 Trending — ตลาดมีทิศทางชัดเจน ADX สูง เหมาะกับ trend following",
            "ranging":  "↔️ Ranging — ตลาดไซด์เวย์ เหมาะกับ mean reversion",
            "volatile": "⚡ Volatile — ตลาดผันผวนสูง ระมัดระวัง confidence ลดลง",
            "unknown":  "❓ ยังไม่ทราบ regime (AI ยังไม่ได้วิเคราะห์)",
        }.get(regime, f"Regime: {regime}")
        return desc

    def _explain_last_decision(self) -> str:
        if not _brain:
            return "⚠️ AI brain ยังไม่ได้โหลด"
        conf = _brain._last_confidence
        u = _brain._last_uncertainty
        regime = _brain._last_regime
        strat  = _brain._last_strategy_name
        result = (
            f"🤔 เหตุผลการตัดสินใจล่าสุด\n\n"
            f"กลยุทธ์: {strat}\n"
            f"Regime: {regime.upper()}\n"
            f"Confidence: {conf:.1%}\n"
        )
        if u:
            result += (
                f"\nUncertainty:\n"
                f"  {u.get('description', 'N/A')}\n"
                f"  Conf multiplier ×{u.get('conf_multiplier', 1):.2f}"
            )
        return result


class _WsBroadcastHandler(logging.Handler):
    """Forwards log records to every connected WebSocket client in real-time."""

    # Map Python log levels → UI severity labels
    _LEVEL_MAP = {
        "DEBUG":    "debug",
        "INFO":     "info",
        "WARNING":  "warn",
        "ERROR":    "error",
        "CRITICAL": "error",
    }

    def emit(self, record: logging.LogRecord) -> None:
        global _log_broadcasting
        if _log_broadcasting:
            return   # prevent recursion
        try:
            _log_broadcasting = True
            lvl = self._LEVEL_MAP.get(record.levelname, "info")
            parts = record.name.split(".")
            short = ".".join(parts[-2:]) if len(parts) >= 2 else record.name
            entry = {
                "type":    "log",
                "level":   lvl,
                "name":    short,
                "message": record.getMessage(),
                "time":    time.strftime("%H:%M:%S", time.localtime(record.created)),
            }
            _log_buffer.append(entry)
            broadcast_sync(entry)
        except Exception:
            pass
        finally:
            _log_broadcasting = False


# ── Open-orders tracking ──────────────────────────────────────────────────────
def _register_open_order(order_id: int, asset_display: str, action: str,
                        amount: float, duration_min: int, manual: bool = False) -> None:
    """Add a freshly-placed trade to the open-orders book and notify the UI."""
    if order_id is None:
        return
    now = time.time()
    entry = {
        "order_id":  int(order_id),
        "asset":     asset_display,
        "action":    action.upper(),
        "amount":    float(amount),
        "open_time": time.strftime("%H:%M:%S", time.localtime(now)),
        "open_ts":   now,
        "expiry_ts": now + duration_min * 60,
        "duration":  duration_min,
        "manual":    manual,
    }
    _open_orders[int(order_id)] = entry
    broadcast_sync({"type": "open_trade", "entry": entry})


def _close_open_order(order_id: int) -> None:
    """Remove a finished trade from the open-orders book and notify the UI."""
    if order_id is None:
        return
    if int(order_id) in _open_orders:
        _open_orders.pop(int(order_id), None)
    broadcast_sync({"type": "close_trade", "order_id": int(order_id)})


def _install_ws_log_handler() -> None:
    handler = _WsBroadcastHandler()
    handler.setLevel(logging.DEBUG)
    # attach to the root logger so every module's output flows through
    root = logging.getLogger()
    root.addHandler(handler)

# ── FastAPI app ───────────────────────────────────────────────────────────────
@contextlib.asynccontextmanager
async def _lifespan(application):
    global _loop
    _loop = asyncio.get_running_loop()
    _install_ws_log_handler()
    threading.Thread(target=_init_components, daemon=True).start()
    yield


app = FastAPI(title="IQ Option AI Trading Dashboard", docs_url=None, lifespan=_lifespan)

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


@app.get("/api/desires")
async def api_desires():
    if _brain is None:
        return {"desires": [], "pending": 0}
    d = _brain.desire_engine
    return {"desires": d.get_all(), "pending": len(d.get_pending())}


@app.post("/api/desires/{desire_id}/approve")
async def api_desire_approve(desire_id: str, body: dict = None):
    if _brain is None:
        return {"ok": False}
    note = (body or {}).get("note", "")
    ok = _brain.desire_engine.approve(desire_id, note)
    if ok:
        broadcast_sync({"type": "desire_updated", "id": desire_id, "status": "approved"})
    return {"ok": ok}


@app.post("/api/desires/{desire_id}/deny")
async def api_desire_deny(desire_id: str, body: dict = None):
    if _brain is None:
        return {"ok": False}
    reason = (body or {}).get("reason", "")
    ok = _brain.desire_engine.deny(desire_id, reason)
    if ok:
        broadcast_sync({"type": "desire_updated", "id": desire_id, "status": "denied"})
    return {"ok": ok}


@app.get("/api/ethics")
async def api_ethics():
    from trading_ai.brain.ethics import get_principles
    return {"principles": get_principles()}


@app.post("/api/smtp-config")
async def api_smtp_config(body: dict):
    _save_smtp_config({
        "smtp_user":    body.get("smtp_user", ""),
        "smtp_pass":    body.get("smtp_pass", ""),
        "notify_email": body.get("notify_email", "titanbuff.company.game@gmail.com"),
    })
    return {"ok": True}


@app.get("/api/smtp-config")
async def api_smtp_config_get():
    cfg = dict(_smtp_config)
    if "smtp_pass" in cfg:
        cfg["smtp_pass"] = "****" if cfg["smtp_pass"] else ""
    return cfg


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
    all_passed_cached = (
        all(r["passed"] for r in _check_results) if _check_results else False
    )
    await manager.send(ws, {
        "type": "init",
        "connected": _connector is not None,
        "account": config.IQ_ACCOUNT_TYPE,
        "balance": bal,
        "ai_running": _ai_running,
        "stats": _trade_stats,
        "checks": _check_results,
        "all_passed": all_passed_cached,
        "history": list(_trade_history)[-20:],
        "open_orders": list(_open_orders.values()),
        "market_mode": (_brain._market_mode if _brain else "OTC"),
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
    if _capital_guard:
        await manager.send(ws, {"type": "capital_guard", **_capital_guard.status()})
    if _brain:
        de = _brain.desire_engine
        await manager.send(ws, {
            "type":    "desires_init",
            "desires": de.get_all(),
            "pending": len(de.get_pending()),
        })
    # Send chat history so newly-connected clients see the conversation
    if _chat_history:
        await manager.send(ws, {
            "type":    "chat_history",
            "entries": list(_chat_history),
        })
    # Replay buffered log lines so the Logs panel is populated immediately
    if _log_buffer:
        await manager.send(ws, {
            "type":    "log_history",
            "entries": list(_log_buffer),
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

    elif mtype == "chat_user":
        user_msg = str(msg.get("message", "")).strip()
        if user_msg:
            ts = time.strftime("%H:%M")
            user_entry = {"role": "user", "message": user_msg, "time": ts}
            _chat_history.append(user_entry)
            ai_reply = ThinkingAI().think(user_msg)
            ai_entry = {"role": "ai", "message": ai_reply, "time": ts}
            _chat_history.append(ai_entry)
            await manager.send(ws, {"type": "chat_ai", **ai_entry})

    elif mtype == "ping":
        await manager.send(ws, {"type": "pong"})


# ── Background: broadcast helpers (called from sync threads via asyncio) ───────
_loop: Optional[asyncio.AbstractEventLoop] = None


def broadcast_sync(msg: Dict):
    """Thread-safe broadcast from non-async code."""
    if _loop and not _loop.is_closed():
        coro = manager.broadcast(msg)
        try:
            asyncio.run_coroutine_threadsafe(coro, _loop)
        except RuntimeError:
            coro.close()  # event loop shutting down — discard cleanly


def _save_trade_stats():
    """Persist cumulative trade stats and history to disk."""
    path = os.path.join(config.MODEL_DIR, "trade_stats.json")
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stats": _trade_stats, "history": list(_trade_history)},
                      f, ensure_ascii=False)
    except Exception:
        pass


def _load_trade_stats():
    """Restore persisted trade stats from disk (called on startup)."""
    path = os.path.join(config.MODEL_DIR, "trade_stats.json")
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _trade_stats.update(data.get("stats", {}))
        for entry in data.get("history", []):
            _trade_history.append(entry)
        logger.info("Trade stats restored: %d trades, win_rate=%.1f%%",
                    _trade_stats["trades"],
                    100 * _trade_stats["wins"] / max(_trade_stats["trades"], 1))
    except Exception as exc:
        logger.warning("Could not load trade stats: %s", exc)


# ── Trading helpers ───────────────────────────────────────────────────────────
async def _execute_manual_trade(asset: str, direction: str):
    global _connector
    if not _connector:
        return
    # Frontend sends display name ("GBP/USD (OTC)"); IQ Option API needs resolved name ("GBPUSD-OTC")
    api_name = _resolved_asset_names.get(asset, asset)
    broadcast_sync({
        "type": "status",
        "message": f"Manual {direction.upper()}: {asset}",
        "level": "info",
    })
    ok, oid = _connector.place_trade(
        asset=api_name, direction=direction,
        amount=config.TRADE_AMOUNT,
        duration_minutes=config.TRADE_DURATION,
    )
    if ok:
        _register_open_order(oid, asset, direction, config.TRADE_AMOUNT,
                             config.TRADE_DURATION, manual=True)
        broadcast_sync({
            "type": "status",
            "message": f"✅ Trade placed: {direction.upper()} {asset}",
            "level": "success",
        })
        # Background-await the result so manual trades also close cleanly
        threading.Thread(
            target=_await_manual_trade_result,
            args=(oid, asset, direction),
            daemon=True,
        ).start()
    else:
        broadcast_sync({
            "type": "status",
            "message": f"❌ Trade rejected: {asset}",
            "level": "error",
        })


def _await_manual_trade_result(order_id: int, asset_display: str, direction: str) -> None:
    """Wait for a manual trade to settle, then update stats + UI."""
    global _trade_stats
    try:
        balance_before = _connector.get_balance()
        time.sleep(config.TRADE_DURATION * 60 + 5)
        pnl = _connector.get_trade_result(order_id, timeout=30,
                                          balance_before=balance_before)
        _close_open_order(order_id)
        if pnl is None:
            broadcast_sync({"type": "status",
                            "message": f"⚠️ Manual trade {asset_display}: ผลลัพธ์ไม่ทราบ",
                            "level": "warn"})
            return
        _trade_stats["trades"] += 1
        if pnl > 0:
            _trade_stats["wins"] += 1
        _trade_stats["pnl"] += pnl
        bal = _connector.get_balance()
        entry = {
            "time":       time.strftime("%H:%M:%S"),
            "asset":      asset_display,
            "action":     direction.upper(),
            "pnl":        round(pnl, 2),
            "win":        pnl > 0,
            "balance":    bal,
            "confidence": 1.0,   # manual
            "order_id":   int(order_id),
        }
        _trade_history.append(entry)
        _save_trade_stats()
        wr = _trade_stats["wins"] / max(_trade_stats["trades"], 1)
        broadcast_sync({
            "type":      "trade",
            "order_id":  int(order_id),
            "asset":     asset_display,
            "action":    direction.upper(),
            "pnl":       round(pnl, 2),
            "balance":   bal,
            "trades":    _trade_stats["trades"],
            "wins":      _trade_stats["wins"],
            "win_rate":  round(wr, 3),
            "total_pnl": round(_trade_stats["pnl"], 2),
            "entry":     entry,
        })
    except Exception as exc:
        logger.error("manual trade result error: %s", exc)
        _close_open_order(order_id)


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
    if "account" in msg:
        new_account = str(msg["account"]).upper()
        if new_account in ("PRACTICE", "REAL"):
            config.IQ_ACCOUNT_TYPE = new_account
            if _connector:
                _connector.account_type = new_account
                try:
                    if _connector.api:
                        _connector.api.change_balance(new_account)
                except Exception as exc:
                    logger.warning("change_balance failed: %s", exc)
            if _brain:
                _brain.set_account_type(new_account)
            if _capital_guard:
                _capital_guard.set_account_type(new_account)
                broadcast_sync({"type": "capital_guard", **_capital_guard.status()})
            broadcast_sync({
                "type": "status",
                "message": f"บัญชี: {new_account}" +
                           (" — โหมดเรียนรู้ (ไม่หยุดเทรด)" if new_account == "PRACTICE"
                            else " — โหมดจริง 🛡️ (Capital Guard เปิดใช้งาน)"),
                "level": "info",
            })
    if "market_mode" in msg:
        mode = str(msg["market_mode"]).upper()
        if mode in ("OTC", "REAL"):
            if _brain:
                _brain.set_market_mode(mode)
            mode_desc = ("🎲 OTC — pattern learning, news/calendar disabled"
                         if mode == "OTC"
                         else "📈 Real — news, calendar, sentiment active")
            broadcast_sync({"type": "status", "message": mode_desc, "level": "info"})
            broadcast_sync({"type": "market_mode", "mode": mode})

    # Capital Guard settings (REAL only)
    if "loss_limit_pct" in msg and _capital_guard:
        v = float(msg["loss_limit_pct"])
        if 0.05 <= v <= 0.50:
            _capital_guard.DAILY_LOSS_LIMIT_PCT = v
            broadcast_sync({"type": "capital_guard", **_capital_guard.status()})
    if "profit_target_pct" in msg and _capital_guard:
        v = float(msg["profit_target_pct"])
        if 0.05 <= v <= 0.50:
            _capital_guard.PROFIT_TARGET_PCT = v
            broadcast_sync({"type": "capital_guard", **_capital_guard.status()})


# ── AI trading loop (runs in background thread) ───────────────────────────────
def _ai_loop():
    global _ai_running, _trade_stats, _agent, _brain, _knowledge, _env, _connector, _capital_guard

    if not _agent or not _brain or not _connector:
        broadcast_sync({"type":"status","message":"AI components not loaded","level":"error"})
        return

    _ai_running = True
    broadcast_sync({"type":"status","message":"🤖 AI started","level":"success"})

    from trading_ai.core.trading_env import TradingEnv, N_INDICATORS
    if _env is None:
        _env = TradingEnv(_connector)
        # Map API asset → display name so the open-orders panel shows readable labels
        def _on_placed(order_id, api_asset, direction, amount, duration_min):
            display = next(
                (d for d, a in _resolved_asset_names.items() if a == api_asset),
                api_asset,
            )
            _register_open_order(order_id, display, direction, amount,
                                 duration_min, manual=False)
        _env.on_trade_placed = _on_placed
        _env.on_trade_closed = _close_open_order

    obs, _ = _env.reset()

    # ── Asset unavailability backoff (asset → timestamp unavailable until) ──
    _unavail: Dict[str, float] = {}

    # ── CapitalGuard init ──────────────────────────────────────────────────
    if _capital_guard is None:
        from trading_ai.brain.capital_guard import CapitalGuard
        _capital_guard = CapitalGuard(account_type=config.IQ_ACCOUNT_TYPE)
    _capital_guard.set_account_type(config.IQ_ACCOUNT_TYPE)
    start_balance = _connector.get_balance() or 0.0
    _capital_guard.update_day_start(start_balance)
    broadcast_sync({"type": "capital_guard", **_capital_guard.status()})

    while _ai_running:
        # ── CapitalGuard: refresh day-start on midnight, check stop ────────
        if _capital_guard:
            bal_now = _connector.get_balance() or start_balance
            _capital_guard.update_day_start(bal_now)
            cg_stop, cg_reason = _capital_guard.should_stop()
            if cg_stop:
                broadcast_sync({
                    "type":    "capital_guard",
                    **_capital_guard.status(),
                })
                broadcast_sync({
                    "type":    "status",
                    "message": cg_reason,
                    "level":   "warn",
                })
                logger.warning("CapitalGuard STOP: %s", cg_reason)
                time.sleep(30)   # wait, check again (day may roll over)
                continue

        # ── MULTI-ASSET SCAN ────────────────────────────────────────────────
        # Compute a signal for every resolved OTC asset and pick the best
        # opportunity (highest confidence non-HOLD signal).  Falls back to
        # config.ASSET if nothing else is available.
        candidates = []   # list of (display_name, api_name, signal, obs, ppo_action)
        for display_name in OTC_ASSETS:
            api_name = _resolved_asset_names.get(display_name)
            if not api_name:
                continue
            # Skip temporarily unavailable assets (5-minute backoff)
            if _unavail.get(api_name, 0) > time.time():
                continue
            _env.set_asset(api_name)
            try:
                asset_obs = _env._get_observation()
            except Exception as exc:
                logger.debug("Observation failed for %s: %s", display_name, exc)
                continue
            # The WS lock inside get_candles() already serialises requests.
            # A short yield here lets the event loop breathe between assets.
            time.sleep(0.1)
            indicator_vec = asset_obs[:N_INDICATORS]
            ppo_action, log_prob, value = _agent.select_action(asset_obs)
            _, ppo_conf = _agent.get_confidence(asset_obs)
            # Pass raw candles so brain can do pattern-memory lookup
            signal = _brain.think(indicator_vec, ppo_action, ppo_conf,
                                  candles=getattr(_env, "_last_candles", None))

            # Adjust final action through brain's risk multiplier + confidence gate
            final_action = ppo_action
            if signal.risk_multiplier < 0.2:
                final_action = 0
            elif signal.action != ppo_action and signal.confidence > ppo_conf + 0.15:
                final_action = signal.action

            candidates.append({
                "display":   display_name,
                "api":       api_name,
                "signal":    signal,
                "obs":       asset_obs,
                "ind":       indicator_vec,
                "ppo":       ppo_action,
                "log_prob":  log_prob,
                "value":     value,
                "final":     final_action,
            })
            broadcast_sync({
                "type":       "signal",
                "asset":      display_name,
                "action":     {0:"HOLD",1:"BUY",2:"SELL"}[final_action],
                "confidence": round(signal.confidence, 3),
                "risk":       round(signal.risk_multiplier, 2),
                "reasoning":  signal.reasoning[:3],
            })

        if not candidates:
            time.sleep(3)
            continue

        # Dynamic min-confidence via CapitalGuard
        # PRACTICE: 0.30 → 0.40 → 0.50 (เรียนรู้ได้มาก)
        # REAL:     0.55 → 0.58 → 0.62 → 0.65 (รอสัญญาณมั่นใจเท่านั้น)
        confirmed = _trade_stats.get("trades", 0)
        if _capital_guard:
            min_conf = _capital_guard.min_confidence(confirmed)
        else:
            min_conf = 0.30 if confirmed < 20 else (0.40 if confirmed < 50 else 0.50)

        # Pick best non-HOLD signal across all assets
        tradeable = [c for c in candidates
                     if c["final"] != 0 and c["signal"].confidence >= min_conf]
        if tradeable:
            tradeable.sort(key=lambda c: c["signal"].confidence, reverse=True)
            best = tradeable[0]
        else:
            # Nothing meets confidence — HOLD on the strongest-looking asset
            best = max(candidates, key=lambda c: c["signal"].confidence)
            best = {**best, "final": 0}

        display_name  = best["display"]
        api_name      = best["api"]
        signal        = best["signal"]
        indicator_vec = best["ind"]
        ppo_action    = best["ppo"]
        log_prob      = best["log_prob"]
        value         = best["value"]
        final_action  = best["final"]
        action_name   = {0:"HOLD",1:"BUY",2:"SELL"}[final_action]
        obs           = best["obs"]

        # ── Kelly Criterion bet sizing (REAL only) ─────────────────────────
        trade_amount = config.TRADE_AMOUNT
        if _capital_guard and final_action != 0:
            wr_now = _trade_stats["wins"] / max(_trade_stats["trades"], 1)
            trade_amount = _capital_guard.kelly_amount(config.TRADE_AMOUNT, wr_now)
            if trade_amount != config.TRADE_AMOUNT:
                logger.info("Kelly bet: $%.2f (base $%.2f, wr=%.1f%%)",
                            trade_amount, config.TRADE_AMOUNT, wr_now * 100)

        _env.set_asset(api_name)
        _env._trade_amount = trade_amount   # pass Kelly amount to env
        next_obs, reward, terminated, truncated, info = _env.step(final_action)

        # If a trade was attempted but rejected (e.g. asset not available), back off 5 min
        if info.get("skipped", False) and final_action != 0:
            _unavail[api_name] = time.time() + 300
            logger.info("Asset %s marked unavailable for 5 min", api_name)

        if not info.get("skipped", False):
            pnl = info.get("pnl", 0.0)
            logger.info("Trade %s on %s completed: pnl=%+.2f total_trades=%d",
                        action_name, display_name, pnl, _trade_stats["trades"] + 1)
            _brain.learn(pnl, final_action, indicator_vec, ppo_action, next_obs=next_obs,
                         candles=getattr(_env, "_last_candles", None))

            _trade_stats["trades"] += 1
            if pnl > 0:
                _trade_stats["wins"] += 1
            _trade_stats["pnl"] += pnl

            # ── Capital Guard: record PnL + broadcast status ───────────────
            if _capital_guard:
                _capital_guard.record_trade_pnl(pnl)
                broadcast_sync({"type": "capital_guard", **_capital_guard.status()})

            # ── AI proactive chat commentary ───────────────────────────────
            _broadcast_ai_thought(action_name, display_name, pnl, signal)

            entry = {
                "time": time.strftime("%H:%M:%S"),
                "asset": display_name,
                "action": action_name,
                "pnl": round(pnl, 2),
                "win": pnl > 0,
                "balance": _connector.get_balance(),
                "confidence": round(signal.confidence, 3),
                "amount": round(trade_amount, 2),
            }
            _trade_history.append(entry)
            _save_trade_stats()

            wr = _trade_stats["wins"] / max(_trade_stats["trades"], 1)
            broadcast_sync({
                "type": "trade",
                "asset": display_name,
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
                "pause_status": brain_status.get("pause_status", {"paused": False}),
                "account_type": brain_status.get("account_type", "PRACTICE"),
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
    "EUR/JPY (OTC)": (130.0, 200.0),   # EUR/JPY is ~184 in 2025-2026
}

# Cache: display_name → resolved api_name (once found, reuse)
_resolved_asset_names: Dict[str, str] = {}

# Cache: display_name → list of {time,open,high,low,close} for chart history
_candle_history: Dict[str, List[Dict]] = {a: [] for a in OTC_ASSETS}


@contextlib.contextmanager
def _suppress_iqapi_stdout():
    """Redirect stdout so iqoptionapi's direct print() calls don't pollute the console."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def _resolve_asset_name(display_name: str) -> Optional[str]:
    """
    Find the working IQ Option API name for an OTC asset.

    iqoptionapi uses a shared WebSocket buffer for candle data.  Calling
    get_candles() for multiple assets in rapid succession can return stale
    data from the previous request.  We guard against this via:
      1. A global _candle_lock in IQOptionConnector.get_candles() — only one
         call is in-flight at any time, plus a settle delay inside the lock.
      2. Validating the returned price against known sane ranges.
    """
    if display_name in _resolved_asset_names:
        return _resolved_asset_names[display_name]

    lo, hi = PRICE_RANGES.get(display_name, (0.0, 1e9))

    for api_name in OTC_ASSET_MAP.get(display_name, []):
        try:
            with _suppress_iqapi_stdout():
                df = _connector.get_candles(asset=api_name, timeframe_seconds=60, count=5)
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
    _brain     = BrainCore(asset=config.ASSET, base_dir=config.MODEL_DIR,
                           account_type=config.IQ_ACCOUNT_TYPE)
    _brain.desire_engine.set_notify_callback(_on_desire_registered)

    from trading_ai.brain.capital_guard import CapitalGuard
    _capital_guard = CapitalGuard(account_type=config.IQ_ACCOUNT_TYPE)
    init_balance = _connector.get_balance() or 0.0
    _capital_guard.update_day_start(init_balance)

    # Restore cumulative trade history from disk and push to any connected client
    _load_trade_stats()
    broadcast_sync({
        "type":    "stats_update",
        "stats":   _trade_stats,
        "history": list(_trade_history)[-20:],
    })

    brain_status = _brain.get_status(ppo_agent=_agent)
    broadcast_sync({"type":"brain", **brain_status})
    broadcast_sync({"type": "capital_guard", **_capital_guard.status()})
    broadcast_sync({"type":"status","message":"✅ พร้อมแล้ว – กด START AI","level":"success"})

    # Fetch initial OTC candle history (30-second candles from IQ Option)
    broadcast_sync({"type":"status","message":"📊 โหลดประวัติกราฟ OTC…","level":"info"})
    _fetch_initial_candles()
    broadcast_sync({"type":"candle_history","data":_candle_history})

    # Update checks with live asset resolution results
    live_results = list(_check_results)
    for display_name in OTC_ASSETS:
        api_name = _resolved_asset_names.get(display_name)
        if api_name:
            live_results.append({
                "name": f"เชื่อมต่อ {display_name}",
                "passed": True,
                "message": f"→ {api_name}",
            })
        else:
            live_results.append({
                "name": f"เชื่อมต่อ {display_name}",
                "passed": False,
                "message": "ไม่พบ asset นี้ใน IQ Option — ลองใหม่อีกครั้ง",
            })
    _check_results = live_results
    all_passed_live = all(r["passed"] for r in live_results)
    broadcast_sync({"type":"check","results":live_results,"all_passed":all_passed_live})

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
def start_server(host: str = "0.0.0.0", port: int = 8000):
    setup_logging(log_dir=config.LOG_DIR)
    logger.info("Starting web dashboard at http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    start_server()
