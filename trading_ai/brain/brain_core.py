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
from trading_ai.brain.internet.knowledge_researcher import KnowledgeResearcher
from trading_ai.brain.memory.pattern_memory import CandlePatternMemory
from trading_ai.brain.memory.sequence_memory import TemporalSequenceMemory
from trading_ai.brain.reasoning.brain_reasoner import BrainReasoner, BrainSignal
from trading_ai.brain.reasoning.rule_distiller import RuleDistillationEngine
from trading_ai.brain.uncertainty import UncertaintyEstimator
from trading_ai.brain.ethics import get_principles, evaluate_desire
from trading_ai.brain.desire import DesireEngine
from trading_ai.brain.graduation import GraduationSystem
from trading_ai.brain.neuro import CLSMemory, DopamineSystem, FearSystem, SleepCycle
from trading_ai.brain.nas import NASEngine
from trading_ai.brain.nas.nas_engine import NASConfig
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

    INTERNET_REFRESH_INTERVAL = 1800   # 30 min — news fetch
    RESEARCH_INTERVAL         = 3600   # 1 hour — asset-specific web research
    KNOWLEDGE_INTERVAL        = 1500   # 25 min — trading-knowledge research
                                       # (techniques, strategies, concepts)

    def __init__(self, asset: str = None, base_dir: str = None,
                 account_type: str = None):
        self.asset        = asset or config.ASSET
        self.base_dir     = base_dir or config.MODEL_DIR
        self.account_type = (account_type or config.IQ_ACCOUNT_TYPE or "PRACTICE").upper()

        self.graph      = KnowledgeGraph(base_dir=self.base_dir)
        self.short_term = ShortTermMemory(capacity=1000)
        self.episodic   = EpisodicMemory(base_dir=self.base_dir)

        self.news       = NewsFetcher(asset=self.asset)
        self.calendar   = EconomicCalendar(currencies=self._asset_currencies(self.asset))
        self.researcher           = WebResearcher(asset=self.asset)
        self.knowledge_researcher = KnowledgeResearcher(asset=self.asset)

        self.reasoner = BrainReasoner(
            graph=self.graph,
            episodic=self.episodic,
            short_term=self.short_term,
            news=self.news,
            calendar=self.calendar,
            asset=self.asset,
        )

        # ── Living Brain subsystems ─────────────────────────────────────────
        self.working_memory = WorkingMemory(capacity=20, account_type=self.account_type)
        self.strategy_lib   = StrategyLibrary(base_dir=self.base_dir)
        self.reflection     = SelfReflectionEngine(base_dir=self.base_dir)
        self.fast_learner   = FastLearner(obs_size=OBS_SIZE)
        self.pattern_memory  = CandlePatternMemory(base_dir=self.base_dir)
        self._market_mode    = "OTC"   # "OTC" or "REAL"

        # ── Phase 3 additions ────────────────────────────────────────────────
        self.uncertainty_estimator = UncertaintyEstimator()
        self.sequence_memory       = TemporalSequenceMemory(base_dir=self.base_dir)
        self.rule_distiller        = RuleDistillationEngine(base_dir=self.base_dir)

        # ── Ethics & Desire system ───────────────────────────────────────────
        self.ethics        = get_principles()
        self.desire_engine = DesireEngine(base_dir=self.base_dir)
        self._pnl_log: list = []   # rolling pnl for desire trigger logic

        # ── Neuroscience-inspired modules ────────────────────────────────────
        self.cls_memory   = CLSMemory(hippocampus_size=500)
        self.dopamine     = DopamineSystem(alpha=0.1, surprise_scale=2.5)
        self.fear_system  = FearSystem(account_type=self.account_type)
        self.sleep_cycle  = SleepCycle(trigger_every=50,
                                       on_sleep_callback=self._on_sleep_state)
        self._is_sleeping = False

        # ── Graduation system ────────────────────────────────────────────────
        self.graduation   = GraduationSystem()

        # ── Neural Architecture Search ───────────────────────────────────────
        self.nas = NASEngine(
            obs_size=OBS_SIZE,
            n_actions=3,
            champion_config=NASConfig(hidden_size=384, lstm_hidden=128, dropout=0.10),
        )

        # ── State tracking ──────────────────────────────────────────────────
        self._last_strategy_name: str = "Unknown"
        self._last_confidence:    float = 0.5
        self._last_indicator_vec: Optional[np.ndarray] = None
        self._last_regime:        str = "unknown"
        self._last_uncertainty:   dict = {}

        self._stop_event      = threading.Event()
        self._internet_thread = threading.Thread(
            target=self._internet_loop, daemon=True
        )
        self._internet_thread.start()

        logger.info(
            "BrainCore ULTRA (Living Brain) online | asset=%s | nodes=%d",
            self.asset, self.graph.stats()["total_nodes"]
        )

    # ── Account-type plumbing ─────────────────────────────────────────────────

    def set_account_type(self, account_type: str) -> None:
        """Propagate account type to subsystems that change behaviour by it."""
        new = (account_type or "PRACTICE").upper()
        self.account_type = new
        self.working_memory.set_account_type(new)
        self.fear_system.set_account_type(new)

    def set_market_mode(self, mode: str) -> None:
        """Switch between 'OTC' and 'REAL' market mode.
        OTC: skip news/calendar, rely on patterns + indicators.
        REAL: full news/calendar/sentiment active.
        """
        self._market_mode = mode.upper()
        self.reasoner.set_market_mode(self._market_mode)
        logger.info("Market mode → %s", self._market_mode)

    # ── Main interface ─────────────────────────────────────────────────────────

    def think(
        self,
        indicator_vec: np.ndarray,
        ppo_action: int,
        ppo_confidence: float,
        candles=None,
    ) -> BrainSignal:
        # ── Build snapshot ──────────────────────────────────────────────────
        snapshot = {
            name: float(indicator_vec[i]) if i < len(indicator_vec) else 0.0
            for i, name in enumerate(INDICATOR_NAMES)
        }

        # Pattern memory lookup (OTC: main signal; REAL: supplementary)
        pattern_result = None
        if candles is not None:
            pattern_result = self.pattern_memory.lookup(candles)

        # Multi-timeframe trend: synthetic 5-min trend from 1-min candles
        mtf_trend = self._compute_mtf_trend(candles)

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

        # ── Uncertainty estimation ───────────────────────────────────────────
        uncertainty = self.uncertainty_estimator.estimate(
            indicator_vec=indicator_vec[:40] if len(indicator_vec) >= 40 else indicator_vec,
            episodic_memory=self.episodic,
        )
        self._last_uncertainty = uncertainty

        # ── Sequence memory lookup (trajectory-based hint) ───────────────────
        seq_result = self.sequence_memory.lookup(
            indicator_vec[:40] if len(indicator_vec) >= 40 else indicator_vec
        )

        # ── Reasoner with strategy signal ───────────────────────────────────
        signal = self.reasoner.think(
            indicator_vec=indicator_vec,
            indicator_names=INDICATOR_NAMES,
            ppo_action=ppo_action,
            ppo_confidence=ppo_confidence,
            strategy_signal=strategy_signal,
            strategy_name=strategy_name,
            pattern_result=pattern_result,
            mtf_trend=mtf_trend,
        )

        # ── Apply uncertainty to confidence ──────────────────────────────────
        if self.uncertainty_estimator.should_skip_trade(
            uncertainty["epistemic"], uncertainty["aleatoric"]
        ) and signal.action != 0:
            signal.action     = 0
            signal.confidence = 0.7
            signal.reasoning.append(
                f"⚠️ Uncertainty too high — skipping trade "
                f"(epistemic={uncertainty['epistemic']:.2f}, "
                f"aleatoric={uncertainty['aleatoric']:.2f}, "
                f"familiar={uncertainty['familiar_trades']} trades)"
            )
        else:
            signal.confidence = round(
                float(np.clip(signal.confidence * uncertainty["conf_multiplier"], 0.0, 0.95)),
                3,
            )
            if uncertainty["epistemic"] > 0.55:
                signal.reasoning.append(uncertainty["description"])

        # ── Sequence memory boost ────────────────────────────────────────────
        if seq_result is not None and signal.action != 0:
            seq_wr, seq_conf, seq_n = seq_result
            if seq_wr > 0.60:
                signal.reasoning.append(
                    f"Sequence: trajectory → BUY bias ({seq_wr:.0%} win rate, n={seq_n})"
                )
                if signal.action == 1:
                    signal.confidence = min(0.95, signal.confidence + seq_conf * 0.05)
            elif seq_wr < 0.40:
                signal.reasoning.append(
                    f"Sequence: trajectory → SELL bias ({1-seq_wr:.0%} win rate, n={seq_n})"
                )
                if signal.action == 2:
                    signal.confidence = min(0.95, signal.confidence + seq_conf * 0.05)

        # ── Record this cycle's signal strength for recovery tracking ───────
        # (no-op on PRACTICE; on REAL it counts toward resume-readiness)
        self.working_memory.record_observation(signal.confidence)

        # ── Should-pause override (REAL accounts only) ──────────────────────
        if should_pause and signal.action != 0:
            status = self.working_memory.get_pause_status()
            signal.action     = 0
            signal.confidence = 0.9
            signal.reasoning.append(
                f"PAUSE (REAL) — {status.get('reason','')}; "
                f"cool-off {status.get('cooldown_remaining',0)}s, "
                f"strong {status.get('strong_seen',0)}/{status.get('strong_needed',0)}"
            )
            logger.info("BrainCore: PAUSE active — recovery %d/%d strong, %ds cool-off left",
                        status.get("strong_seen", 0),
                        status.get("strong_needed", 0),
                        status.get("cooldown_remaining", 0))

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
        candles=None,
    ):
        # Record candle pattern outcome for OTC pattern memory
        if candles is not None:
            try:
                self.pattern_memory.record(candles, pnl)
            except Exception:
                pass

        self.short_term.update_last_reward(pnl)

        snapshot = {
            name: float(indicator_vec[i]) if i < len(indicator_vec) else 0.0
            for i, name in enumerate(INDICATOR_NAMES)
        }

        # ── FastLearner online update (reward scaled by Dopamine RPE) ──────────
        rpe_mult = self.dopamine.update(pnl)   # surprise → amplify learning
        if next_obs is not None and self._last_indicator_vec is not None:
            try:
                obs_arr = self._last_indicator_vec.astype(np.float32)
                nxt_arr = next_obs.astype(np.float32)
                if len(obs_arr) == OBS_SIZE and len(nxt_arr) == OBS_SIZE:
                    self.fast_learner.update_online(
                        obs=obs_arr,
                        action=action_taken,
                        reward=float(pnl) * rpe_mult,   # dopamine-scaled reward
                        next_obs=nxt_arr,
                        done=False,
                    )
            except Exception as exc:
                logger.warning("FastLearner update error: %s", exc)  # was DEBUG

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

        # ── Sequence memory + Rule distillation ─────────────────────────────
        if len(indicator_vec) >= 20:
            try:
                self.sequence_memory.record(
                    indicator_vec[:40] if len(indicator_vec) >= 40 else indicator_vec,
                    pnl,
                )
            except Exception as exc:
                logger.debug("SequenceMemory.record error: %s", exc)
        try:
            self.rule_distiller.record_trade(snapshot, action_taken, pnl > 0)
        except Exception as exc:
            logger.debug("RuleDistiller.record_trade error: %s", exc)

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

        # ── NAS: evaluate challengers on same obs ───────────────────────────
        if self._last_indicator_vec is not None:
            self.nas.observe(
                obs=self._last_indicator_vec,
                actual_action=action_taken,
                pnl=pnl,
            )
            if next_obs is not None:
                self.nas.update_challengers(
                    obs=self._last_indicator_vec,
                    action=action_taken,
                    pnl=pnl,
                    next_obs=next_obs,
                )

        # ── Neuroscience modules ─────────────────────────────────────────────
        # CLS hippocampus: encode this episode fast
        self.cls_memory.encode(
            indicators=indicator_vec[:40] if len(indicator_vec) >= 40 else indicator_vec,
            action=action_taken,
            pnl=pnl,
            confidence=self._last_confidence,
        )

        # Fear system: update emotional state (active only in REAL mode)
        self.fear_system.update(pnl)

        # Sleep cycle: trigger consolidation every 50 trades
        self.sleep_cycle.tick(self)

        # Graduation: track this trade
        self.graduation.record_trade(
            pnl=pnl,
            regime=self._last_regime,
            balance=0.0,   # updated by server when balance known
        )

        # Desire engine: check if brain should express a want
        self._pnl_log.append(pnl)
        if len(self._pnl_log) > 50:
            self._pnl_log = self._pnl_log[-50:]
        self._maybe_generate_desire()

    # ── Sleep callback ─────────────────────────────────────────────────────────

    def _on_sleep_state(self, sleeping: bool) -> None:
        self._is_sleeping = sleeping

    # ── Desire generation ──────────────────────────────────────────────────────

    def _maybe_generate_desire(self) -> None:
        """Introspect current state and register a desire if brain senses a gap."""
        pnl_log = self._pnl_log
        if len(pnl_log) < 20:
            return

        recent = pnl_log[-20:]
        wr  = sum(1 for p in recent if p > 0) / len(recent)
        u   = self._last_uncertainty
        ep  = self.episodic.summary().get("total", 0)

        if wr < 0.38:
            self.desire_engine.register(
                title="ขอวิเคราะห์ข้อมูล Historical เพิ่มเติม",
                description=(
                    f"Win rate ล่าสุด {wr:.0%} ต่ำกว่าเกณฑ์ใน {len(recent)} ไม้ติดต่อกัน "
                    "ต้องการดึง historical price data เพิ่มเพื่อค้นหารูปแบบที่พลาดไป"
                ),
                urgency=7,
                category="historical_data",
            )
        elif u.get("epistemic", 0) > 0.72 and ep > 30:
            self.desire_engine.register(
                title="ขอเรียนรู้สภาวะตลาดแบบใหม่",
                description=(
                    f"พบสภาวะตลาดที่ไม่คุ้นเคย (epistemic uncertainty {u.get('epistemic', 0):.2f}) "
                    "อยากได้ข้อมูลหรือตัวอย่างการเทรดใน regime นี้เพิ่มเติม"
                ),
                urgency=6,
                category="market_knowledge",
            )
        elif len(self.rule_distiller.get_rules()) == 0 and ep > 100:
            self.desire_engine.register(
                title="ต้องการทดสอบกลยุทธ์ใหม่",
                description=(
                    f"เทรดมา {ep} ครั้งแล้ว แต่ยังไม่สามารถสกัดกฎที่ชัดเจนได้ "
                    "อยากลองกลยุทธ์รูปแบบอื่นเพื่อหา edge ที่ดีกว่า"
                ),
                urgency=5,
                category="strategy_research",
            )

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
        pause_status     = self.working_memory.get_pause_status()
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
            "pause_status":             pause_status,
            "account_type":             self.account_type,
            "active_strategy":          self._last_strategy_name,
            "pattern_memory":           self.pattern_memory.stats(),
            "market_mode":              self._market_mode,
            "uncertainty":              self._last_uncertainty,
            "sequence_memory":          self.sequence_memory.stats(),
            "distilled_rules":          self.rule_distiller.get_rules(),
            "distilled_rules_text":     self.rule_distiller.format_rules(),
            # Neuroscience modules
            "neuro_cls":                self.cls_memory.stats(),
            "neuro_dopamine":           self.dopamine.stats(),
            "neuro_fear":               self.fear_system.stats(),
            "neuro_sleep":              self.sleep_cycle.stats(),
            "is_sleeping":              self._is_sleeping,
            # Graduation
            "graduation":               self.graduation.evaluate(self),
            # NAS
            "nas":                      self.nas.stats(),
        }

    def shutdown(self):
        self._stop_event.set()
        self.graph.save()
        logger.info("BrainCore ULTRA shutdown complete.")

    # ── Background ─────────────────────────────────────────────────────────────

    def _internet_loop(self):
        time.sleep(30)
        last_news      = 0.0
        last_res       = 0.0
        last_knowledge = 0.0

        while not self._stop_event.is_set():
            now = time.time()

            # In OTC mode, news and asset-specific research are irrelevant —
            # skip them to avoid noise and save bandwidth.
            if self._market_mode != "OTC":
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
            else:
                # Keep last_news/last_res updated to avoid burst when switching modes
                last_news = now
                last_res  = now

            # Knowledge research (techniques/strategies) runs in both modes
            if (now - last_knowledge) >= self.KNOWLEDGE_INTERVAL:
                try:
                    new_nodes = self.knowledge_researcher.research()
                    if new_nodes:
                        self.reasoner.absorb_internet_knowledge(new_nodes)
                        logger.info("Brain learned %d new trading concepts from internet",
                                    len(new_nodes))
                    last_knowledge = now
                except Exception as exc:
                    logger.debug("KnowledgeResearcher error: %s", exc)

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
    def _compute_mtf_trend(candles) -> int:
        """
        Compute a synthetic 5-minute trend from 1-minute candle data.

        Groups every 5 consecutive 1-min candles into one synthetic 5-min bar
        (using the last close of each group), then computes EMA(3) over the
        resulting bars.  Returns:
          +1  if current price > EMA(3) by > 0.03 %  → up-trend
          -1  if current price < EMA(3) by > 0.03 %  → down-trend
           0  otherwise (neutral / not enough data)
        """
        if candles is None or len(candles) < 25:
            return 0
        try:
            closes = candles["close"].values[-25:]
            # Build 5 synthetic 5-min bar closes (index 4, 9, 14, 19, 24)
            bars = [float(closes[i * 5 + 4]) for i in range(5)]
            # EMA(3) with alpha = 2/(3+1) = 0.5
            alpha = 0.5
            ema = bars[0]
            for b in bars[1:]:
                ema = alpha * b + (1.0 - alpha) * ema
            current = bars[-1]
            threshold = ema * 0.0003   # 0.03 % to filter micro-noise
            if current > ema + threshold:
                return 1
            if current < ema - threshold:
                return -1
        except Exception:
            pass
        return 0

    @staticmethod
    def _asset_currencies(asset: str) -> List[str]:
        if len(asset) >= 6:
            return [asset[:3].upper(), asset[3:6].upper()]
        return ["USD", "EUR"]
