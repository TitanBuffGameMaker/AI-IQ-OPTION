"""
BrainCore — ULTRA EDITION (Living Brain)

เพิ่ม:
  - INDICATOR_NAMES สำหรับ 40 features
  - Market regime tracking
  - Performance analytics
  - Indicator reliability tracking
  - WorkingMemory — จำ 20 trades ล่าสุด
  - StrategyLibrary — คลังกลยุทธ์ 5 แบบ
  - SelfReflectionEngine — วิเคราะห์ทุกไม้หลังปิด
  - FastLearner — Online A3C update ทุกไม้ทันที
"""
import logging
import threading
import time
from typing import List, Optional

import numpy as np

from trading_ai.brain.knowledge_graph import KnowledgeGraph
from trading_ai.brain.knowledge_node import KnowledgeNode, NodeType
from trading_ai.brain.memory.short_term import ShortTermMemory, Observation
from trading_ai.brain.memory.episodic import EpisodicMemory
from trading_ai.brain.internet.news_fetcher import NewsFetcher
from trading_ai.brain.internet.economic_calendar import EconomicCalendar
from trading_ai.brain.internet.web_researcher import WebResearcher
from trading_ai.brain.reasoning.brain_reasoner import BrainReasoner, BrainSignal
from trading_ai.brain.working_memory import WorkingMemory
from trading_ai.brain.strategy_library import StrategyLibrary
from trading_ai.brain.self_reflection import SelfReflectionEngine
from trading_ai.models.fast_learner import FastLearner
from trading_ai.config import config

logger = logging.getLogger(__name__)

# 40 indicator names — ต้องตรงกับ IndicatorEngine.compute() output order
INDICATOR_NAMES = [
    "rsi",
    "macd_line", "macd_signal", "macd_hist",
    "bb_position", "bb_width",
    "ema_9_dist", "ema_21_dist", "ema_50_dist", "ema_200_dist",
    "stoch_k", "stoch_d",
    "atr", "adx", "cci", "williams_r", "obv_norm",
    "momentum", "volume_ratio",
    "ichimoku_tenkan_dist", "ichimoku_kijun_dist", "ichimoku_cloud_pos",
    "parabolic_sar",
    "rsi_divergence",
    "support_proximity", "resistance_proximity",
    "fibonacci_position",
    "is_doji", "is_hammer", "is_engulfing", "is_star",
    "market_regime",
    "higher_high", "lower_low",
    "momentum_3", "candle_body_ratio",
    "volume_acceleration", "macd_momentum", "stoch_crossover",
    "volatility_regime",
]

OBS_SIZE = 296  # 40 indicators + 256 CNN features


class BrainCore:
    """
    ULTRA Brain — orchestrates all AI subsystems
    """

    INTERNET_REFRESH_INTERVAL = 1800   # 30 min
    RESEARCH_INTERVAL         = 3600   # 1 hour

    def __init__(self, asset: str = None, base_dir: str = None):
        self.asset    = asset or config.ASSET
        self.base_dir = base_dir or config.MODEL_DIR

        self.graph      = KnowledgeGraph(base_dir=self.base_dir)
        self.short_term = ShortTermMemory(capacity=1000)
        self.episodic   = EpisodicMemory(base_dir=self.base_dir)

        self.news       = NewsFetcher(asset=self.asset)
        self.calendar   = EconomicCalendar(currencies=self._asset_currencies(self.asset))
        self.researcher = WebResearcher(asset=self.asset)

        self.reasoner = BrainReasoner(
            graph=self.graph,
            episodic=self.episodic,
            short_term=self.short_term,
            news=self.news,
            calendar=self.calendar,
            asset=self.asset,
        )

        # ── Living Brain subsystems ─────────────────────────────────────────
        self.working_memory = WorkingMemory(capacity=20)
        self.strategy_lib   = StrategyLibrary(base_dir=self.base_dir)
        self.reflection     = SelfReflectionEngine(base_dir=self.base_dir)
        self.fast_learner   = FastLearner(obs_size=OBS_SIZE)

        # ── State tracking ──────────────────────────────────────────────────
        self._last_strategy_name: str = "Unknown"
        self._last_confidence:    float = 0.5
        self._last_indicator_vec: Optional[np.ndarray] = None
        self._last_regime:        str = "unknown"

        self._stop_event      = threading.Event()
        self._internet_thread = threading.Thread(
            target=self._internet_loop, daemon=True
        )
        self._internet_thread.start()

        logger.info(
            "BrainCore ULTRA (Living Brain) online | asset=%s | nodes=%d",
            self.asset, self.graph.stats()["total_nodes"]
        )

    # ── Main interface ─────────────────────────────────────────────────────────

    def think(
        self,
        indicator_vec: np.ndarray,
        ppo_action: int,
        ppo_confidence: float,
    ) -> BrainSignal:
        # ── Build snapshot ──────────────────────────────────────────────────
        snapshot = {
            name: float(indicator_vec[i]) if i < len(indicator_vec) else 0.0
            for i, name in enumerate(INDICATOR_NAMES)
        }

        # ── Working memory context ──────────────────────────────────────────
        wm_pattern = self.working_memory.get_recent_pattern()
        wm_attention = self.working_memory.get_active_attention(indicator_vec[:40]
                       if len(indicator_vec) >= 40 else indicator_vec)
        should_pause = self.working_memory.should_pause()

        # ── Strategy selection ──────────────────────────────────────────────
        strategy = self.strategy_lib.select_strategy(snapshot)
        strategy_signal = None
        strategy_name   = "Unknown"

        if strategy is not None:
            strategy_name   = strategy.name
            strategy_signal = self.strategy_lib.get_strategy_signal(strategy, snapshot)

        self._last_strategy_name = strategy_name

        # ── Reasoner with strategy signal ───────────────────────────────────
        signal = self.reasoner.think(
            indicator_vec=indicator_vec,
            indicator_names=INDICATOR_NAMES,
            ppo_action=ppo_action,
            ppo_confidence=ppo_confidence,
            strategy_signal=strategy_signal,
            strategy_name=strategy_name,
        )

        # ── Should-pause override ────────────────────────────────────────────
        if should_pause and signal.action != 0:
            signal.action     = 0
            signal.confidence = 0.9
            signal.reasoning.append("WorkingMemory: PAUSE — too many losses")
            logger.warning("BrainCore: WorkingMemory recommends PAUSE")

        # ── Update state ─────────────────────────────────────────────────────
        self._last_confidence    = signal.confidence
        self._last_indicator_vec = indicator_vec.copy()
        self._last_regime        = signal.regime

        obs = Observation(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            indicator_vec=indicator_vec,
            action_taken=signal.action,
            confidence=signal.confidence,
            asset=self.asset,
        )
        self.short_term.record(obs)
        return signal

    def learn(
        self,
        pnl: float,
        action_taken: int,
        indicator_vec: np.ndarray,
        ppo_action: int,
        next_obs: Optional[np.ndarray] = None,
    ):
        self.short_term.update_last_reward(pnl)

        snapshot = {
            name: float(indicator_vec[i]) if i < len(indicator_vec) else 0.0
            for i, name in enumerate(INDICATOR_NAMES)
        }

        # ── FastLearner online update ────────────────────────────────────────
        if next_obs is not None and self._last_indicator_vec is not None:
            try:
                obs_arr = self._last_indicator_vec.astype(np.float32)
                nxt_arr = next_obs.astype(np.float32)
                if len(obs_arr) == OBS_SIZE and len(nxt_arr) == OBS_SIZE:
                    self.fast_learner.update_online(
                        obs=obs_arr,
                        action=action_taken,
                        reward=float(pnl),
                        next_obs=nxt_arr,
                        done=False,
                    )
            except Exception as exc:
                logger.debug("FastLearner update error: %s", exc)

        # ── Working memory update ────────────────────────────────────────────
        self.working_memory.update(
            obs=indicator_vec,
            action=action_taken,
            confidence=self._last_confidence,
            pnl=pnl,
        )

        # ── Self reflection ──────────────────────────────────────────────────
        try:
            self.reflection.reflect(
                action=action_taken,
                pnl=pnl,
                indicators=snapshot,
                strategy_name=self._last_strategy_name,
                confidence=self._last_confidence,
                regime=self._last_regime,
            )
        except Exception as exc:
            logger.debug("Reflection error: %s", exc)

        # ── Strategy outcome ─────────────────────────────────────────────────
        self.strategy_lib.record_outcome(self._last_strategy_name, pnl > 0)

        # ── Existing reasoner learning ───────────────────────────────────────
        self.reasoner.learn_from_outcome(
            pnl=pnl,
            action_taken=action_taken,
            indicator_snapshot=snapshot,
            ppo_action=ppo_action,
        )
        self._log_brain_state(pnl)

    # ── Status ─────────────────────────────────────────────────────────────────

    def get_status(self, ppo_agent=None) -> dict:
        from trading_ai.brain.brain_age import calculate_brain_age

        graph_stats  = self.graph.stats()
        mem_summary  = self.episodic.summary()
        streak       = self.short_term.detect_streak()
        win_rate     = self.short_term.win_rate()
        trades       = mem_summary.get("total", 0)
        ppo_updates  = ppo_agent.total_updates if ppo_agent else 0

        age_result = calculate_brain_age(
            nodes             = graph_stats["total_nodes"],
            win_rate          = win_rate,
            total_trades      = trades,
            avg_confidence    = graph_stats["avg_confidence"],
            ppo_updates       = ppo_updates,
            episodic_memories = mem_summary.get("total", 0),
            graph_branches    = graph_stats["total_edges"],
        )

        # Indicator reliability (top 5)
        top_indicators = self.reasoner.reliability.top_indicators(5)

        # Living Brain extras
        wm_pattern       = self.working_memory.get_recent_pattern()
        should_pause     = self.working_memory.should_pause()
        strategy_stats   = self.strategy_lib.get_stats()
        recent_mistakes  = self.reflection.get_recent_mistakes(5)
        improvement_tips = self.reflection.get_improvement_tips()

        return {
            "graph_nodes":        graph_stats["total_nodes"],
            "graph_branches":     graph_stats["total_edges"],
            "avg_confidence":     graph_stats["avg_confidence"],
            "knowledge_by_type":  graph_stats["by_type"],
            "episodic_memories":  mem_summary["total"],
            "short_term_size":    len(self.short_term),
            "recent_win_rate":    round(win_rate, 3),
            "win_streak":         streak["wins"],
            "loss_streak":        streak["losses"],
            # Brain age
            "brain_age":          age_result.age,
            "brain_score":        age_result.score,
            "brain_stage":        age_result.stage,
            "brain_stage_en":     age_result.stage_en,
            "brain_emoji":        age_result.emoji,
            "brain_desc":         age_result.description,
            "brain_next":         age_result.next_milestone,
            "brain_pct_next":     age_result.pct_to_next,
            "brain_breakdown":    age_result.breakdown,
            # ULTRA extras
            "top_indicators":     top_indicators,
            "ppo_updates":        ppo_updates,
            # Living Brain extras
            "strategy_stats":           strategy_stats,
            "recent_mistakes":          recent_mistakes,
            "improvement_tips":         improvement_tips,
            "working_memory_pattern":   wm_pattern,
            "should_pause":             should_pause,
            "active_strategy":          self._last_strategy_name,
        }

    def shutdown(self):
        self._stop_event.set()
        self.graph.save()
        logger.info("BrainCore ULTRA shutdown complete.")

    # ── Background ─────────────────────────────────────────────────────────────

    def _internet_loop(self):
        time.sleep(30)
        last_news  = 0.0
        last_res   = 0.0

        while not self._stop_event.is_set():
            now = time.time()

            if (now - last_news) >= self.INTERNET_REFRESH_INTERVAL:
                try:
                    self.reasoner.absorb_news_as_nodes()
                    last_news = now
                    logger.info("Brain absorbed latest news")
                except Exception as exc:
                    logger.debug("News error: %s", exc)

            if (now - last_res) >= self.RESEARCH_INTERVAL:
                try:
                    new_nodes = self.researcher.research()
                    if new_nodes:
                        self.reasoner.absorb_internet_knowledge(new_nodes)
                    last_res = now
                except Exception as exc:
                    logger.debug("Research error: %s", exc)

            self._stop_event.wait(timeout=60)

    def _log_brain_state(self, pnl: float):
        direction = "WIN" if pnl > 0 else "LOSS"
        logger.info(
            "Brain updated after %s (pnl=%.2f) | nodes=%d | win_rate=%.1f%% | strategy=%s",
            direction, pnl,
            self.graph.stats()["total_nodes"],
            self.short_term.win_rate() * 100,
            self._last_strategy_name,
        )

    @staticmethod
    def _asset_currencies(asset: str) -> List[str]:
        if len(asset) >= 6:
            return [asset[:3].upper(), asset[3:6].upper()]
        return ["USD", "EUR"]
