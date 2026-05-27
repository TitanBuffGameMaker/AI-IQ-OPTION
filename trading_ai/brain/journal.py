"""
AIJournal — AI writes diary entries about its own learning progress

Every N trades the AI writes a Thai-language reflection:
  - Performance this period
  - What improved / what is still weak
  - NAS architecture findings
  - Plans for growth

Entries are broadcast to the chat panel and optionally emailed.
"เหมือนลูกที่เขียนไดอารี่ก่อนนอน"
"""
import logging
import time
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


def _mood(win_rate: float, dopamine_level: float) -> str:
    if win_rate >= 0.65 and dopamine_level >= 0.6:
        return "🌟"
    if win_rate >= 0.55:
        return "😊"
    if win_rate >= 0.48:
        return "😐"
    if win_rate >= 0.40:
        return "😟"
    return "😰"


class AIJournal:
    """
    Periodic self-reflection diary.
    The AI narrates its own growth every N trades.
    """

    def __init__(self, write_every: int = 50,
                 on_entry_callback: Optional[Callable] = None):
        self._write_every = write_every
        self._trade_count = 0
        self._entry_count = 0
        self._entries: List[dict] = []   # last 20 in memory
        self._on_entry_callback = on_entry_callback   # (entry: dict) → None

    def set_entry_callback(self, cb: Callable) -> None:
        self._on_entry_callback = cb

    def tick(self, brain) -> bool:
        """Call after every trade. Returns True when an entry was written."""
        self._trade_count += 1
        if self._trade_count % self._write_every != 0:
            return False

        entry = self._compose_entry(brain)
        self._entries.append(entry)
        if len(self._entries) > 20:
            self._entries = self._entries[-20:]
        self._entry_count += 1

        logger.info("📔 Journal entry #%d written", self._entry_count)
        if self._on_entry_callback:
            try:
                self._on_entry_callback(entry)
            except Exception as exc:
                logger.debug("Journal callback error: %s", exc)
        return True

    def get_recent_entries(self, n: int = 5) -> List[dict]:
        return self._entries[-n:]

    def stats(self) -> dict:
        return {
            "entry_count":    self._entry_count,
            "trade_count":    self._trade_count,
            "write_every":    self._write_every,
            "recent_entries": self.get_recent_entries(3),
        }

    # ── Entry composition ─────────────────────────────────────────────────────

    def _compose_entry(self, brain) -> dict:
        now_str = time.strftime("%Y-%m-%d %H:%M")

        # ── Gather stats ────────────────────────────────────────────────────
        win_rate    = brain.short_term.win_rate()
        streak      = brain.short_term.detect_streak()
        total_ep    = brain.episodic.summary().get("total", 0)
        dopa_stats  = brain.dopamine.stats()
        dopa_level  = dopa_stats.get("dopamine_level", 0.5)
        cls_stats   = brain.cls_memory.stats()
        nas_stats   = brain.nas.stats()
        conf_mult   = brain._conf_multiplier
        suggestions = brain.weak_point.get_suggestions(2)

        # Brain age
        brain_stage, brain_score = "ไม่ทราบ", 0
        try:
            from trading_ai.brain.brain_age import calculate_brain_age
            gs  = brain.graph.stats()
            mem = brain.episodic.summary()
            age = calculate_brain_age(
                nodes=gs["total_nodes"], win_rate=win_rate,
                total_trades=total_ep, avg_confidence=gs["avg_confidence"],
                ppo_updates=0, episodic_memories=mem.get("total", 0),
                graph_branches=gs["total_edges"],
            )
            brain_stage = age.stage
            brain_score = age.score
        except Exception:
            pass

        # ── Narrative blocks ────────────────────────────────────────────────
        mood_icon = _mood(win_rate, dopa_level)
        wr_pct    = win_rate * 100
        win_s     = streak.get("wins", 0)
        loss_s    = streak.get("losses", 0)

        streak_line = ""
        if win_s >= 3:
            streak_line = f"ชนะต่อเนื่อง {win_s} ไม้ รู้สึกดีมากเลย! "
        elif loss_s >= 3:
            streak_line = f"เสียต่อเนื่อง {loss_s} ไม้ ต้องระวังมากขึ้น "

        conf_line = ""
        if conf_mult > 1.05:
            conf_line = f"ความมั่นใจปรับขึ้นเป็น ×{conf_mult:.2f} เพราะผลงานดี "
        elif conf_mult < 0.92:
            conf_line = f"ลด threshold เหลือ ×{conf_mult:.2f} — อยู่ในโหมดระวัง "

        nas_line = ""
        if nas_stats.get("recommended_upgrade"):
            upg = nas_stats["recommended_upgrade"]
            nas_line = (
                f"NAS เจอ architecture ใหม่: "
                f"hidden={upg.get('hidden_size')}/lstm={upg.get('lstm_hidden')} "
                f"(gen {nas_stats.get('generation', 1)}) "
            )

        cls_line = ""
        hip_size = cls_stats.get("hippocampus_size", 0)
        neo_pats = cls_stats.get("neocortex_patterns", 0)
        if hip_size > 30:
            cls_line = f"จำ episodes ระยะสั้น {hip_size} ชิ้น, neocortex patterns {neo_pats} แบบ "

        weak_line = ""
        if suggestions:
            weak_line = "จุดที่ต้องพัฒนา: " + " | ".join(suggestions)

        message = "\n".join(filter(None, [
            f"{mood_icon} บันทึกการเรียนรู้ — {now_str}",
            "",
            f"เทรดมาแล้ว {total_ep} ครั้งรวม",
            f"Win rate ล่าสุด: {wr_pct:.1f}%",
            streak_line.strip() if streak_line else None,
            conf_line.strip() if conf_line else None,
            f"สมองอยู่ในระยะ: {brain_stage} (score={brain_score:.0f})",
            cls_line.strip() if cls_line else None,
            nas_line.strip() if nas_line else None,
            weak_line.strip() if weak_line else None,
            "",
            "— AI 🤖",
        ]))

        return {
            "id":          self._entry_count + 1,
            "ts":          now_str,
            "trade_count": total_ep,
            "win_rate":    round(win_rate, 3),
            "mood":        mood_icon,
            "message":     message,
        }
