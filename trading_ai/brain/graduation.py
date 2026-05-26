"""
GraduationSystem — ระบบเลื่อนชั้นจาก Practice → Real Account

AI จะ "สำเร็จการศึกษา" และพร้อมใช้บัญชีจริงได้ก็ต่อเมื่อ
ผ่านเกณฑ์ทั้งหมดที่กำหนดไว้ เหมือนนักเรียนที่ต้องสอบผ่านก่อนออกทำงาน

เกณฑ์บังคับ (required):
  ✅ Win rate ≥ 58% ใน ≥ 200 trades
  ✅ ขาดทุนติดกันสูงสุด ≤ 7 ไม้ตลอดประวัติ
  ✅ Brain Age ≥ วัยรุ่น (stage index ≥ 2)
  ✅ Max drawdown < 20% ตลอดประวัติ
  ✅ มี Distilled rules ≥ 3 ข้อ

เกณฑ์เสริม (bonus):
  ⭐ ผ่าน trending, ranging, volatile ครบทุก regime
  ⭐ Episodic memory ≥ 100 ประสบการณ์
  ⭐ CLS neocortex ≥ 20 patterns
"""
import time
from typing import Any, Dict, List


# Brain Age stages from brain_age.py — index matches stage
_STAGE_INDEX = {
    "ทารก": 0, "เด็กเล็ก": 1, "วัยรุ่น": 2,
    "หนุ่มสาว": 3, "มืออาชีพ": 4, "ผู้เชี่ยวชาญ": 5,
    "ปรมาจารย์": 6, "ตำนาน": 7,
}
_MIN_STAGE = 2   # วัยรุ่น

CRITERIA: List[Dict[str, Any]] = [
    {
        "id": "win_rate",
        "label": "Win Rate ≥ 58%",
        "desc": "ต้องชนะมากกว่าครึ่งอย่างสม่ำเสมอใน 200+ ไม้",
        "required": True,
        "icon": "🎯",
    },
    {
        "id": "min_trades",
        "label": "เทรด ≥ 200 ไม้",
        "desc": "ต้องมีประสบการณ์เพียงพอ",
        "required": True,
        "icon": "📊",
    },
    {
        "id": "loss_streak",
        "label": "ขาดทุนติดกันสูงสุด ≤ 7 ไม้",
        "desc": "ควบคุมการขาดทุนต่อเนื่องได้",
        "required": True,
        "icon": "🛡️",
    },
    {
        "id": "brain_age",
        "label": "Brain Age ≥ วัยรุ่น",
        "desc": "สมองต้องโตพอ — ผ่านการเรียนรู้ขั้นพื้นฐาน",
        "required": True,
        "icon": "🧠",
    },
    {
        "id": "drawdown",
        "label": "Max Drawdown < 20%",
        "desc": "ไม่เคยเสียเงินมากเกินไปในรอบเดียว",
        "required": True,
        "icon": "📉",
    },
    {
        "id": "rules",
        "label": "มีกฎที่ distill ได้ ≥ 3 ข้อ",
        "desc": "AI ต้องสร้างกฎการเทรดของตัวเองได้แล้ว",
        "required": True,
        "icon": "📋",
    },
    {
        "id": "regimes",
        "label": "ผ่าน trending + ranging + volatile",
        "desc": "ต้องเทรดได้ในทุกสภาพตลาด",
        "required": False,
        "icon": "🌊",
    },
    {
        "id": "episodic",
        "label": "ประสบการณ์ episodic ≥ 100",
        "desc": "ความจำเชิงประสบการณ์เพียงพอ",
        "required": False,
        "icon": "💭",
    },
]


class GraduationSystem:
    """Evaluates AI readiness to move from Practice to Real Account."""

    def __init__(self):
        self._seen_regimes: set = set()
        self._max_loss_streak: int = 0
        self._current_loss_streak: int = 0
        self._max_drawdown: float = 0.0
        self._peak_balance: float = 0.0
        self._current_balance: float = 0.0
        self._graduated_at: str = ""
        self._was_graduated = False

    # ── Live tracking (call from brain.learn) ─────────────────────────────────

    def record_trade(self, pnl: float, regime: str, balance: float) -> None:
        """Call after every trade to track graduation metrics."""
        if regime:
            self._seen_regimes.add(regime.lower())

        if pnl < 0:
            self._current_loss_streak += 1
            self._max_loss_streak = max(self._max_loss_streak, self._current_loss_streak)
        else:
            self._current_loss_streak = 0

        if balance > self._peak_balance:
            self._peak_balance = balance
        if self._peak_balance > 0:
            dd = (self._peak_balance - balance) / self._peak_balance
            self._max_drawdown = max(self._max_drawdown, dd)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, brain) -> Dict[str, Any]:
        """Full evaluation against all graduation criteria."""
        from trading_ai.brain.brain_age import calculate_brain_age

        mem_summary  = brain.episodic.summary()
        graph_stats  = brain.graph.stats()
        trades       = mem_summary.get("total", 0)
        # Win rate from all-time episodic memory (persists across restarts).
        # Falls back to short-term only when episodic is empty (brand new brain).
        ep_wins = mem_summary.get("wins", 0)
        wr = ep_wins / max(trades, 1) if trades > 0 else brain.short_term.win_rate()
        rules        = brain.rule_distiller.get_rules()
        ep           = trades

        try:
            age_result = calculate_brain_age(
                nodes             = graph_stats["total_nodes"],
                win_rate          = wr,
                total_trades      = trades,
                avg_confidence    = graph_stats["avg_confidence"],
                ppo_updates       = 0,
                episodic_memories = trades,
                graph_branches    = graph_stats["total_edges"],
            )
            stage = age_result.stage
        except Exception:
            stage = "ทารก"

        stage_idx = _STAGE_INDEX.get(stage, 0)
        try:
            neo = brain.cls_memory.stats()["neocortex_patterns"]
        except Exception:
            neo = 0

        checks = {
            "win_rate":   wr >= 0.58,
            "min_trades": trades >= 200,
            "loss_streak": self._max_loss_streak <= 7,
            "brain_age":  stage_idx >= _MIN_STAGE,
            "drawdown":   self._max_drawdown < 0.20,
            "rules":      len(rules) >= 3,
            "regimes":    {"trending", "ranging", "volatile"}.issubset(self._seen_regimes),
            "episodic":   ep >= 100,
        }

        values = {
            "win_rate":   f"{wr:.1%}",
            "min_trades": f"{trades} ไม้",
            "loss_streak": f"สูงสุด {self._max_loss_streak} ไม้",
            "brain_age":  stage,
            "drawdown":   f"{self._max_drawdown:.1%}",
            "rules":      f"{len(rules)} ข้อ",
            "regimes":    ", ".join(self._seen_regimes) or "ยังไม่ครบ",
            "episodic":   f"{ep} ครั้ง",
        }

        criteria_status = []
        for c in CRITERIA:
            criteria_status.append({
                **c,
                "passed": checks[c["id"]],
                "value":  values[c["id"]],
            })

        required_all = all(checks[c["id"]] for c in CRITERIA if c["required"])
        bonus_count  = sum(1 for c in CRITERIA if not c["required"] and checks[c["id"]])

        readiness = self._compute_readiness(checks, wr, trades, stage_idx)

        # Record graduation event
        if required_all and not self._was_graduated:
            self._was_graduated = True
            self._graduated_at = time.strftime("%Y-%m-%d %H:%M:%S")

        return {
            "criteria":        criteria_status,
            "required_passed": required_all,
            "bonus_count":     bonus_count,
            "readiness_pct":   round(readiness, 1),
            "can_graduate":    required_all,
            "graduated_at":    self._graduated_at,
            "seen_regimes":    list(self._seen_regimes),
            "max_loss_streak": self._max_loss_streak,
            "max_drawdown_pct": round(self._max_drawdown * 100, 1),
            "verdict": self._verdict(required_all, readiness),
        }

    def _compute_readiness(self, checks: dict, wr: float, trades: int,
                           stage_idx: int) -> float:
        """0–100 readiness score."""
        score = 0.0
        # Required criteria (80% of score)
        req = [c["id"] for c in CRITERIA if c["required"]]
        per_req = 80.0 / len(req)
        score += sum(per_req for k in req if checks[k])

        # Bonus criteria (20% of score)
        bon = [c["id"] for c in CRITERIA if not c["required"]]
        per_bon = 20.0 / max(len(bon), 1)
        score += sum(per_bon for k in bon if checks[k])

        # Gradient bonus: extra credit for excellence
        if wr > 0.65:
            score = min(100.0, score + 5.0)
        if trades > 500:
            score = min(100.0, score + 3.0)
        if stage_idx >= 4:
            score = min(100.0, score + 2.0)

        return score

    def _verdict(self, passed: bool, readiness: float) -> str:
        if passed:
            return "🎓 พร้อมใช้บัญชีจริงแล้ว!"
        if readiness >= 80:
            return "🔜 เกือบแล้ว — ยังขาดเกณฑ์บังคับบางข้อ"
        if readiness >= 50:
            return "📈 กำลังพัฒนา — ยังต้องฝึกต่อ"
        if readiness >= 25:
            return "🌱 เพิ่งเริ่ม — ต้องการประสบการณ์อีกมาก"
        return "👶 ยังเป็นมือใหม่ — เทรดต่อไปเรื่อยๆ"
