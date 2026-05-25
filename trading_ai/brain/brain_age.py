"""
Brain Age Calculator

วัดความฉลาดของ AI เทียบกับอายุคน:
  ทารก (0-3 ปี)       → เพิ่งสร้าง ยังไม่รู้อะไร
  เด็กเล็ก (3-10 ปี)  → เริ่มจำรูปแบบได้
  วัยรุ่น (10-18 ปี)  → เริ่มมีประสบการณ์
  หนุ่มสาว (18-30 ปี) → เทรดได้พอสมควร
  มืออาชีพ (30-45 ปี) → เก่ง มีระบบ
  ผู้เชี่ยวชาญ (45-60)→ ประสบการณ์สูงมาก
  ปรมาจารย์ (60-80)   → ระดับสูงสุด
  ตำนาน (80+)         → สุดยอด
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class BrainAgeResult:
    age: float                  # อายุเป็นปี (0-100+)
    score: float                # คะแนนดิบ (0-100)
    stage: str                  # ชื่อขั้น
    stage_en: str
    emoji: str
    description: str            # คำอธิบาย
    description_en: str
    next_milestone: str         # ต้องทำอะไรเพื่อเก่งขึ้น
    pct_to_next: float          # % ความก้าวหน้าไปขั้นถัดไป (0-1)
    breakdown: dict             # คะแนนแต่ละด้าน


# Age stages definition
_STAGES = [
    # (score_min, age_range, stage_th, stage_en, emoji, desc_th, desc_en)
    (0,  (0,  3),  "ทารก",         "Newborn",      "👶",
     "เพิ่งเกิด ยังไม่รู้จักตลาดเลย",
     "Just born, no market knowledge yet"),
    (5,  (3,  10), "เด็กเล็ก",     "Toddler",      "🧒",
     "เริ่มจดจำรูปแบบบางอย่าง แต่ยังผิดบ่อย",
     "Starting to recognize patterns, still makes many mistakes"),
    (15, (10, 18), "วัยรุ่น",      "Teenager",     "🧑",
     "เข้าใจ indicator หลักๆ เริ่มมีวินัย",
     "Understands key indicators, developing discipline"),
    (28, (18, 30), "หนุ่มสาว",     "Young Adult",  "👨",
     "เทรดได้ดีในสภาวะตลาดชัดเจน ยังสับสนตอน volatile",
     "Trades well in clear conditions, struggles in volatility"),
    (45, (30, 45), "มืออาชีพ",    "Professional", "👨‍💼",
     "มีระบบชัดเจน win rate เริ่มสม่ำเสมอ",
     "Clear system, consistent win rate"),
    (62, (45, 60), "ผู้เชี่ยวชาญ", "Expert",       "🧓",
     "อ่านตลาดได้แม่นยำ ควบคุมความเสี่ยงดีมาก",
     "Reads market accurately, excellent risk management"),
    (78, (60, 80), "ปรมาจารย์",   "Master",       "👴",
     "เข้าใจตลาดในระดับลึก สัญชาตญาณเฉียบแหลม",
     "Deep market understanding, sharp intuition"),
    (92, (80, 99), "ตำนาน",       "Legend",       "🏆",
     "AI ระดับนี้หายาก win rate สม่ำเสมอสูงมาก",
     "Rare AI level, consistently very high win rate"),
]


def calculate_brain_age(
    nodes: int,
    win_rate: float,
    total_trades: int,
    avg_confidence: float,
    ppo_updates: int,
    episodic_memories: int,
    graph_branches: int,
) -> BrainAgeResult:
    """
    คำนวณ 'อายุสมอง' จากตัวชี้วัดต่างๆ

    Returns a BrainAgeResult with age, stage, description, and breakdown.
    """

    # ── Component scores (each 0-100, then weighted) ────────────────────

    # 1. Knowledge breadth: nodes + branches (25 pts)
    node_raw   = min(nodes / 1500, 1.0)
    branch_raw = min(graph_branches / 5000, 1.0)
    knowledge_score = (node_raw * 0.7 + branch_raw * 0.3) * 25

    # 2. Win rate performance (30 pts) – baseline 50%, ceiling 72%
    if total_trades < 5:
        wr_score = 0.0          # not enough data
    else:
        wr_norm = max(0.0, (win_rate - 0.50) / 0.22)   # 50%→0, 72%→1
        wr_score = min(wr_norm, 1.0) * 30

    # 3. Experience: trades + episodic memories (25 pts)
    trade_raw   = min(total_trades / 10_000, 1.0)
    episode_raw = min(episodic_memories / 500, 1.0)
    experience_score = (trade_raw * 0.8 + episode_raw * 0.2) * 25

    # 4. Learning depth: PPO updates (10 pts)
    update_score = min(ppo_updates / 500, 1.0) * 10

    # 5. Decision quality: avg confidence in meaningful range (10 pts)
    #    confidence 0.55-0.80 is ideal; too high or too low is penalised
    conf_ideal = 1.0 - abs(avg_confidence - 0.67) / 0.25
    confidence_score = max(0.0, min(conf_ideal, 1.0)) * 10

    total_score = (
        knowledge_score + wr_score + experience_score +
        update_score + confidence_score
    )
    total_score = min(total_score, 100.0)

    # ── Map score → age ────────────────────────────────────────────────
    stage_data = _STAGES[0]
    for sd in _STAGES:
        if total_score >= sd[0]:
            stage_data = sd

    score_min, age_range, stage_th, stage_en, emoji, desc_th, desc_en = stage_data

    # Interpolate within the stage
    stage_idx = _STAGES.index(stage_data)
    if stage_idx < len(_STAGES) - 1:
        next_stage = _STAGES[stage_idx + 1]
        score_span = next_stage[0] - score_min
        score_pos  = (total_score - score_min) / max(score_span, 1)
        age = age_range[0] + score_pos * (age_range[1] - age_range[0])
        pct_to_next = score_pos
        next_milestone = _next_milestone_text(stage_idx, nodes, win_rate,
                                              total_trades, ppo_updates)
    else:
        # Max stage
        score_pos = min((total_score - score_min) / (100 - score_min), 1.0)
        age = age_range[0] + score_pos * (age_range[1] - age_range[0])
        pct_to_next = 1.0
        next_milestone = "คุณถึงระดับสูงสุดแล้ว 🏆"

    return BrainAgeResult(
        age=round(age, 1),
        score=round(total_score, 1),
        stage=stage_th,
        stage_en=stage_en,
        emoji=emoji,
        description=desc_th,
        description_en=desc_en,
        next_milestone=next_milestone,
        pct_to_next=round(pct_to_next, 3),
        breakdown={
            "knowledge":   round(knowledge_score,   1),
            "performance": round(wr_score,           1),
            "experience":  round(experience_score,   1),
            "learning":    round(update_score,       1),
            "confidence":  round(confidence_score,   1),
            "total":       round(total_score,        1),
        },
    )


def _next_milestone_text(
    stage_idx: int, nodes: int, win_rate: float,
    trades: int, updates: int
) -> str:
    targets = [
        f"เพิ่ม knowledge nodes ถึง 50 (ตอนนี้ {nodes})",
        f"เพิ่ม trades ถึง 200 และ win rate > 52% (ตอนนี้ {trades} trades, {win_rate:.1%})",
        f"เพิ่ม trades ถึง 1,000 และ win rate > 55% (ตอนนี้ {win_rate:.1%})",
        f"เพิ่ม PPO updates ถึง 100 และ win rate > 58% (ตอนนี้ {updates} updates)",
        f"win rate ต้องถึง 62%+ และ nodes > 500 (ตอนนี้ {win_rate:.1%}, {nodes} nodes)",
        f"win rate ต้องถึง 66%+ และ trades > 5,000 (ตอนนี้ {trades} trades)",
        f"win rate ต้องถึง 70%+ อย่างสม่ำเสมอ (ตอนนี้ {win_rate:.1%})",
    ]
    idx = min(stage_idx, len(targets) - 1)
    return targets[idx]
