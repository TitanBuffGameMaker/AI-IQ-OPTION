"""
NASEngine — Neural Architecture Search แบบ Evolutionary

Search Space (สิ่งที่ AI วิวัฒนาการ):
  hidden_size  : ความกว้างของ network    [192, 256, 384, 512, 640, 768]
  lstm_hidden  : ความจำ LSTM             [64, 128, 192, 256]
  dropout      : ความต้านทาน overfitting [0.05, 0.10, 0.15, 0.20]

Evolutionary Algorithm:
  1. เริ่มต้นด้วย champion (ค่า default ปัจจุบัน)
  2. สร้าง N challengers ด้วย mutated configs
  3. ทุก trade: challengers observe obs เดียวกัน → predict → score
  4. ทุก EVAL_PERIOD trades: เปรียบ challenger score vs champion
  5. ถ้า challenger ชนะ → บันทึกเป็น "recommended upgrade"
  6. Mutate challengers ที่แพ้ → evolution ไม่หยุด

ข้อสำคัญ:
  - Challengers ไม่เทรดจริง (shadow mode)
  - ทุก challenger อัปเดต weights ด้วย gradient เดียวกัน (เรียนรู้คู่ขนาน)
  - การ upgrade architecture จริงต้อง reset และ retrain ด้วย winner config
"""
import logging
import math
import random
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Search space ──────────────────────────────────────────────────────────────

HIDDEN_SIZES   = [192, 256, 384, 512, 640, 768]
LSTM_HIDDENS   = [64, 128, 192, 256]
DROPOUTS       = [0.05, 0.10, 0.15, 0.20]

EVAL_PERIOD        = 100    # trades to evaluate each challenger
PROMOTE_THRESHOLD  = 0.04   # challenger must beat champion by 4% accuracy
N_CHALLENGERS      = 2      # number of shadow models
MAX_HISTORY        = 20     # keep last N generation records


@dataclass
class NASConfig:
    hidden_size: int   = 384
    lstm_hidden: int   = 128
    dropout:     float = 0.10
    generation:  int   = 1

    def mutate(self) -> "NASConfig":
        """Randomly change one parameter."""
        gene = random.choice(["hidden_size", "lstm_hidden", "dropout"])
        new = NASConfig(
            hidden_size=self.hidden_size,
            lstm_hidden=self.lstm_hidden,
            dropout=self.dropout,
            generation=self.generation + 1,
        )
        if gene == "hidden_size":
            others = [h for h in HIDDEN_SIZES if h != self.hidden_size]
            new.hidden_size = random.choice(others)
        elif gene == "lstm_hidden":
            others = [h for h in LSTM_HIDDENS if h != self.lstm_hidden]
            new.lstm_hidden = random.choice(others)
        else:
            others = [d for d in DROPOUTS if d != self.dropout]
            new.dropout = round(random.choice(others), 2)
        return new

    def label(self) -> str:
        return f"h{self.hidden_size}/l{self.lstm_hidden}/d{int(self.dropout*100)}"


@dataclass
class Challenger:
    config:   NASConfig
    agent:    Any                = field(default=None, repr=False)
    correct:  int                = 0
    trades:   int                = 0
    score:    float              = 0.5
    promoted: bool               = False


class NASEngine:
    """
    Evolutionary Neural Architecture Search.
    Runs shadow models in parallel with the main agent
    to discover better architectures automatically.
    """

    def __init__(self, obs_size: int, n_actions: int,
                 champion_config: Optional[NASConfig] = None):
        self._obs_size   = obs_size
        self._n_actions  = n_actions
        self._generation = 1
        self._total_obs  = 0

        self.champion_config: NASConfig = champion_config or NASConfig()
        self.champion_score: float      = 0.50   # baseline
        self.best_ever_config: NASConfig = NASConfig(**asdict(self.champion_config))
        self.best_ever_score:  float     = 0.50
        self.recommended_upgrade: Optional[NASConfig] = None

        self._challengers: List[Challenger] = []
        self._history: List[Dict] = []

        self._init_challengers()
        logger.info(
            "NASEngine online | champion=%s | challengers=%d",
            self.champion_config.label(), len(self._challengers),
        )

    # ── Init ─────────────────────────────────────────────────────────────────

    def _init_challengers(self) -> None:
        for _ in range(N_CHALLENGERS):
            cfg = self.champion_config.mutate()
            self._challengers.append(self._make_challenger(cfg))

    def _make_challenger(self, cfg: NASConfig) -> Challenger:
        agent = self._create_agent(cfg)
        return Challenger(config=cfg, agent=agent)

    def _create_agent(self, cfg: NASConfig):
        """Lazy import to avoid circular dependency."""
        try:
            from trading_ai.models.ppo_agent import PPOAgent
            return PPOAgent(
                obs_size=self._obs_size,
                n_actions=self._n_actions,
                hidden_size=cfg.hidden_size,
            )
        except Exception as e:
            logger.warning("NAS challenger creation failed: %s", e)
            return None

    # ── Per-trade observation ─────────────────────────────────────────────────

    def observe(self, obs: np.ndarray, actual_action: int, pnl: float) -> None:
        """
        Called after each trade.
        Challengers predict on same obs → score vs actual outcome.
        Triggers evaluation every EVAL_PERIOD trades.
        """
        if pnl == 0.0:
            return

        self._total_obs += 1

        for ch in self._challengers:
            if ch.agent is None:
                continue
            try:
                # get_confidence() returns (action, prob) — correct PPOAgent method
                pred_action, _prob = ch.agent.get_confidence(obs.astype(np.float32))
                if int(pred_action) == actual_action:
                    ch.correct += 1
            except Exception as e:
                logger.debug("NAS challenger predict error: %s", e)
            ch.trades += 1

            if ch.trades >= EVAL_PERIOD:
                self._evaluate(ch)
                ch.correct = 0
                ch.trades  = 0

    def update_challengers(self, obs: np.ndarray, action: int, pnl: float,
                           next_obs: np.ndarray) -> None:
        """Online gradient update for all shadow challengers (they learn too)."""
        for ch in self._challengers:
            if ch.agent is None:
                continue
            try:
                ch.agent.update_online(
                    obs=obs.astype(np.float32),
                    action=action,
                    reward=float(pnl),
                    next_obs=next_obs.astype(np.float32),
                    done=False,
                )
            except Exception:
                pass

    # ── Evaluation & Evolution ────────────────────────────────────────────────

    def _evaluate(self, ch: Challenger) -> None:
        accuracy = ch.correct / EVAL_PERIOD
        ch.score = accuracy

        logger.debug(
            "NAS eval | %s accuracy=%.1f%% vs champion=%.1f%%",
            ch.config.label(), accuracy * 100, self.champion_score * 100,
        )

        if accuracy > self.champion_score + PROMOTE_THRESHOLD:
            self._record_win(ch, accuracy)

        if accuracy < self.champion_score - PROMOTE_THRESHOLD:
            # Challenger is significantly worse → mutate it
            self._mutate_challenger(ch)

        # Update best ever
        if accuracy > self.best_ever_score:
            self.best_ever_score = accuracy
            self.best_ever_config = NASConfig(**asdict(ch.config))
            self.recommended_upgrade = self.best_ever_config
            logger.info(
                "NAS NEW BEST: %s — %.1f%%",
                self.best_ever_config.label(), accuracy * 100,
            )

    def _record_win(self, ch: Challenger, accuracy: float) -> None:
        self._generation += 1
        record = {
            "generation":   self._generation,
            "winner":       ch.config.label(),
            "winner_hidden": ch.config.hidden_size,
            "winner_lstm":   ch.config.lstm_hidden,
            "winner_dropout": ch.config.dropout,
            "accuracy":     round(accuracy, 4),
            "vs_champion":  round(self.champion_score, 4),
            "ts":           time.strftime("%H:%M:%S"),
        }
        self._history.append(record)
        if len(self._history) > MAX_HISTORY:
            self._history = self._history[-MAX_HISTORY:]

        logger.info(
            "🧬 NAS Generation %d | %s promoted! %.1f%% > %.1f%%",
            self._generation, ch.config.label(),
            accuracy * 100, self.champion_score * 100,
        )

        # Update champion record (not swapping agent — need rebuild)
        old_champion_config = NASConfig(**asdict(self.champion_config))
        self.champion_config = NASConfig(**asdict(ch.config))
        self.champion_score  = accuracy
        self.recommended_upgrade = self.champion_config

        # Mutate the winning challenger into new territory
        ch.config = old_champion_config.mutate()
        if ch.agent:
            try:
                ch.agent = self._create_agent(ch.config)
            except Exception:
                pass

    def _mutate_challenger(self, ch: Challenger) -> None:
        """Replace underperforming challenger with a mutation of the champion."""
        ch.config  = self.champion_config.mutate()
        ch.agent   = self._create_agent(ch.config)
        ch.correct = 0
        ch.trades  = 0
        ch.score   = 0.5

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        return {
            "generation":         self._generation,
            "champion":           self.champion_config.label(),
            "champion_hidden":    self.champion_config.hidden_size,
            "champion_lstm":      self.champion_config.lstm_hidden,
            "champion_dropout":   self.champion_config.dropout,
            "champion_score":     round(self.champion_score, 3),
            "best_ever":          self.best_ever_config.label(),
            "best_ever_score":    round(self.best_ever_score, 3),
            "recommended_upgrade": (
                asdict(self.recommended_upgrade)
                if self.recommended_upgrade else None
            ),
            "challengers": [
                {
                    "label":    ch.config.label(),
                    "hidden":   ch.config.hidden_size,
                    "lstm":     ch.config.lstm_hidden,
                    "dropout":  ch.config.dropout,
                    "score":    round(ch.score, 3),
                    "trades":   ch.trades,
                    "progress": round(ch.trades / EVAL_PERIOD * 100),
                }
                for ch in self._challengers
            ],
            "history":      self._history[-5:],
            "total_obs":    self._total_obs,
            "eval_period":  EVAL_PERIOD,
        }
