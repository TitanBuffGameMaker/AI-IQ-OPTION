"""
BrainReasoner — ULTRA EDITION

ปรับปรุงจากเดิม:
  - Multi-layer signal fusion (7 layers)
  - Market Regime Detection (Trending/Ranging/Volatile)
  - Adaptive confidence weighting ตาม regime
  - Pattern clustering: รวม pattern ที่คล้ายกัน
  - Meta-learning signal: เรียนรู้ว่า indicator ไหนเชื่อถือได้
  - Contradiction detection: จัดการ signal ขัดแย้ง
  - Dynamic risk calculation
  - Confidence calibration
"""
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from trading_ai.brain.knowledge_graph import KnowledgeGraph
from trading_ai.brain.knowledge_node import KnowledgeNode, NodeType
from trading_ai.brain.memory.episodic import EpisodicMemory
from trading_ai.brain.memory.short_term import ShortTermMemory
from trading_ai.brain.internet.news_fetcher import NewsFetcher
from trading_ai.brain.internet.economic_calendar import EconomicCalendar

logger = logging.getLogger(__name__)


@dataclass
class BrainSignal:
    action:          int           # 0=hold 1=buy 2=sell
    confidence:      float         # 0-1
    risk_multiplier: float         # 0-1
    reasoning:       List[str]
    regime:          str = "unknown"  # trending/ranging/volatile
    signal_strength: float = 0.5      # raw signal strength

    def __repr__(self):
        a = {0:"HOLD",1:"BUY",2:"SELL"}[self.action]
        return f"BrainSignal({a}, conf={self.confidence:.2f}, risk={self.risk_multiplier:.2f}, {self.regime})"


class MarketRegimeDetector:
    """ตรวจจับ regime ตลาด: Trending / Ranging / Volatile"""

    @staticmethod
    def detect(snapshot: dict) -> str:
        adx   = snapshot.get("adx", 0.0) * 100   # undo norm
        atr   = snapshot.get("atr", 0.0)
        bb_w  = snapshot.get("bb_width", 0.0)

        # Trending: ADX > 0.25 (normalized)
        if adx > 0.25:
            return "trending"
        # Volatile: BB width กว้าง
        if bb_w > 0.6:
            return "volatile"
        return "ranging"


class IndicatorReliabilityTracker:
    """
    Meta-learning: ติดตามว่า indicator ไหนทำนายถูกบ่อยที่สุด
    ปรับ weight ของแต่ละ indicator อัตโนมัติ
    """
    DECAY = 0.95

    def __init__(self):
        self._wins:   defaultdict = defaultdict(float)
        self._total:  defaultdict = defaultdict(float)

    def record(self, indicator_name: str, correct: bool):
        self._wins[indicator_name]  = self._wins[indicator_name] * self.DECAY + (1.0 if correct else 0.0)
        self._total[indicator_name] = self._total[indicator_name] * self.DECAY + 1.0

    def reliability(self, indicator_name: str) -> float:
        total = self._total.get(indicator_name, 0)
        if total < 3:
            return 0.5  # ยังไม่มีข้อมูลพอ → neutral
        return self._wins[indicator_name] / total

    def top_indicators(self, n: int = 5) -> List[Tuple[str, float]]:
        scores = {k: self._wins[k] / max(self._total[k], 1) for k in self._total}
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]


class BrainReasoner:
    """
    ULTRA reasoning layer:
    7-layer signal fusion ที่ฉลาดที่สุด
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        episodic: EpisodicMemory,
        short_term: ShortTermMemory,
        news: NewsFetcher,
        calendar: EconomicCalendar,
        asset: str = "EURUSD",
    ):
        self.graph      = graph
        self.episodic   = episodic
        self.short_term = short_term
        self.news       = news
        self.calendar   = calendar
        self.asset      = asset

        self._last_fired_nodes: List[str] = []
        self.regime_detector   = MarketRegimeDetector()
        self.reliability       = IndicatorReliabilityTracker()

        # ประวัติ signals สำหรับ smoothing
        self._signal_history:  deque = deque(maxlen=5)
        self._pnl_history:     deque = deque(maxlen=20)
        self._market_mode = "OTC"   # set via set_market_mode()

    # ── Main: think ────────────────────────────────────────────────────────────

    def think(
        self,
        indicator_vec: np.ndarray,
        indicator_names: List[str],
        ppo_action: int,
        ppo_confidence: float,
        strategy_signal: Optional[Tuple[int, float]] = None,
        strategy_name: str = "",
        pattern_result=None,
    ) -> BrainSignal:
        """7-layer signal fusion + optional strategy signal boost"""
        reasoning: List[str] = []
        self._last_fired_nodes = []
        snapshot = self._vec_to_snapshot(indicator_vec, indicator_names)

        # ── Layer 0: Market Regime ──────────────────────────────────────────
        regime = self.regime_detector.detect(snapshot)
        reasoning.append(f"Regime: {regime.upper()}")

        # ── Layer 1: Knowledge Graph ────────────────────────────────────────
        kg_buy, kg_sell, kg_hold, fired = self._query_graph_ultra(snapshot, regime)
        self._last_fired_nodes.extend(fired)

        # ── Layer 2: Episodic Memory ────────────────────────────────────────
        ep_buy, ep_sell, ep_reason = self._recall_episodes_ultra(snapshot, ppo_action)
        if ep_reason:
            reasoning.append(ep_reason)

        # ── Layer 3: Candlestick Patterns ───────────────────────────────────
        cs_buy, cs_sell, cs_reason = self._candlestick_signals(snapshot)
        if cs_reason:
            reasoning.append(cs_reason)

        # ── Layer 4: News Sentiment ─────────────────────────────────────────
        news_buy, news_sell, news_hold, news_reason = self._news_signals()
        if news_reason:
            reasoning.append(news_reason)

        # ── Layer 5: Economic Calendar ──────────────────────────────────────
        risk_mult, cal_reason = self._calendar_risk()
        if cal_reason:
            reasoning.append(cal_reason)

        # ── Layer 6: Recent Performance ─────────────────────────────────────
        perf_boost, perf_reason = self._performance_signals()
        if perf_reason:
            reasoning.append(perf_reason)

        # ── Layer 7: Pattern Memory (OTC-specific) ──────────────────────────
        pattern_buy = pattern_sell = pattern_hold = 0.0
        pattern_reason = ""
        if pattern_result is not None:
            pat_wr, pat_conf, pat_n = pattern_result
            if pat_wr > 0.58:        # pattern historically resolves UP
                pattern_buy  = pat_conf * 2.0
                pattern_reason = f"Pattern memory: {pat_wr:.0%} BUY ({pat_n} samples)"
            elif pat_wr < 0.42:      # pattern historically resolves DOWN
                pattern_sell = pat_conf * 2.0
                pattern_reason = f"Pattern memory: {1-pat_wr:.0%} SELL ({pat_n} samples)"
            else:
                pattern_hold = pat_conf * 0.5
                pattern_reason = f"Pattern: mixed ({pat_wr:.0%} win rate, {pat_n} samples)"
            if pattern_reason:
                reasoning.append(pattern_reason)

        # ── Regime-adaptive weights ─────────────────────────────────────────
        # ตลาด trending: เน้น PPO + momentum
        # ตลาด ranging: เน้น oscillators (RSI, Stoch)
        # ตลาด volatile: ระมัดระวัง → เพิ่ม hold score
        if regime == "trending":
            ppo_weight   = 2.5
            kg_weight    = 1.2
            cs_weight    = 0.8
            hold_penalty = 0.0
        elif regime == "ranging":
            ppo_weight   = 1.8
            kg_weight    = 1.5
            cs_weight    = 1.2
            hold_penalty = 0.0
        else:  # volatile
            ppo_weight   = 1.5
            kg_weight    = 0.8
            cs_weight    = 0.5
            hold_penalty = 0.3
            reasoning.append("Volatile market: extra caution")

        # ── Aggregate scores ────────────────────────────────────────────────
        buy_score  = (kg_buy * kg_weight + ep_buy + cs_buy * cs_weight +
                      news_buy + pattern_buy +
                      perf_boost * (1 if ppo_action == 1 else 0))
        sell_score = (kg_sell * kg_weight + ep_sell + cs_sell * cs_weight +
                      news_sell + pattern_sell +
                      perf_boost * (1 if ppo_action == 2 else 0))
        hold_score = kg_hold + news_hold + hold_penalty + pattern_hold

        # PPO agent vote (ใหญ่ที่สุด)
        if ppo_action == 0:
            hold_score += ppo_confidence * ppo_weight
        elif ppo_action == 1:
            buy_score  += ppo_confidence * ppo_weight
        elif ppo_action == 2:
            sell_score += ppo_confidence * ppo_weight

        # ── Strategy signal boost (weight = 2.0) ────────────────────────────
        if strategy_signal is not None:
            strat_action, strat_conf = strategy_signal
            strat_weight = 2.0
            if strat_action == 1:
                buy_score  += strat_conf * strat_weight
                sname = strategy_name or "Strategy"
                reasoning.append(f"Strategy: {sname} → BUY (conf={strat_conf:.2f})")
            elif strat_action == 2:
                sell_score += strat_conf * strat_weight
                sname = strategy_name or "Strategy"
                reasoning.append(f"Strategy: {sname} → SELL (conf={strat_conf:.2f})")
            elif strat_action == 0:
                hold_score += strat_conf * strat_weight * 0.5
                reasoning.append(f"Strategy: {strategy_name or 'Strategy'} → HOLD")

        # ── Contradiction detection ─────────────────────────────────────────
        if buy_score > 0.5 and sell_score > 0.5:
            # มีทั้ง buy และ sell signals ที่แรง → ลด confidence
            reasoning.append("Contradiction: mixed signals → reduce conf")
            buy_score  *= 0.7
            sell_score *= 0.7
            hold_score += 0.3

        # ── Final decision ──────────────────────────────────────────────────
        total    = buy_score + sell_score + hold_score + 1e-9
        buy_p    = buy_score / total
        sell_p   = sell_score / total
        hold_p   = hold_score / total

        if risk_mult < 0.3 or hold_p > max(buy_p, sell_p):
            final_action = 0
            final_conf   = hold_p
        elif buy_p > sell_p:
            final_action = 1
            final_conf   = buy_p
        else:
            final_action = 2
            final_conf   = sell_p

        # ── Signal smoothing (ป้องกัน flip-flop) ───────────────────────────
        self._signal_history.append(final_action)
        if len(self._signal_history) >= 3:
            counts = {0: 0, 1: 0, 2: 0}
            for s in self._signal_history:
                counts[s] += 1
            # ถ้า majority ไม่ตรงกัน → hold
            if counts[final_action] < 2 and final_action != 0:
                reasoning.append("Signal unstable → switching to HOLD")
                final_action = 0
                final_conf   = 0.6

        # ── Confidence calibration ──────────────────────────────────────────
        final_conf = self._calibrate_confidence(final_conf, regime, ppo_confidence)

        signal = BrainSignal(
            action=final_action,
            confidence=round(final_conf, 3),
            risk_multiplier=round(risk_mult, 3),
            reasoning=reasoning,
            regime=regime,
            signal_strength=round(max(buy_p, sell_p, hold_p), 3),
        )
        logger.info("Brain ULTRA: %s | %s", signal, " | ".join(reasoning[:3]))
        return signal

    # ── Post-trade learning ────────────────────────────────────────────────────

    def learn_from_outcome(self, pnl, action_taken, indicator_snapshot, ppo_action):
        won       = pnl > 0
        action_nm = {0:"hold",1:"buy",2:"sell"}.get(action_taken, "unknown")

        # Update reliability tracker
        rsi_val  = indicator_snapshot.get("rsi", 0.5)
        macd_val = indicator_snapshot.get("macd_hist", 0)
        adx_val  = indicator_snapshot.get("adx", 0.25)

        rsi_signal_correct = (
            (rsi_val < 0.35 and action_taken == 1 and won) or  # oversold → buy → win
            (rsi_val > 0.65 and action_taken == 2 and won)      # overbought → sell → win
        )
        self.reliability.record("rsi",  rsi_signal_correct)

        macd_signal_correct = (
            (macd_val > 0 and action_taken == 1 and won) or
            (macd_val < 0 and action_taken == 2 and won)
        )
        self.reliability.record("macd", macd_signal_correct)

        # Reinforce / contradict knowledge graph
        for node_id in self._last_fired_nodes:
            if won:
                self.graph.reinforce(node_id, strength=0.10)
            else:
                self.graph.contradict(node_id, strength=0.07)

        # New experience node
        indicator_str = ", ".join(
            f"{k}={v:.2f}" for k, v in list(indicator_snapshot.items())[:6]
        )
        concept = (
            f"Trade {action_nm.upper()} → {'WIN' if won else 'LOSS'} "
            f"pnl={pnl:+.2f} | {indicator_str}"
        )
        tags = [action_nm, "win" if won else "loss", "experience"]
        tags.extend(self._regime_tags(indicator_snapshot))

        exp_node = KnowledgeNode(
            node_type=NodeType.EXPERIENCE,
            concept=concept,
            data={
                "action": action_nm,
                "pnl": pnl,
                "indicators": {k: round(v, 3) for k, v in indicator_snapshot.items()},
                "won": won,
            },
            confidence=0.60 if won else 0.40,
            evidence_count=1,
            source="trade_result",
            asset=self.asset,
            tags=tags,
        )
        self.graph.add_node(exp_node)

        # Pattern node for winning trades
        if won and action_taken != 0:
            pattern = self._extract_pattern_ultra(indicator_snapshot, action_nm)
            if pattern:
                self.graph.add_node(pattern)

        # Episodic memory
        self.episodic.record(
            asset=self.asset,
            action=action_nm,
            indicators={k: round(v, 3) for k, v in indicator_snapshot.items()},
            pnl=pnl,
            confidence=0.5,
            notes=concept,
            tags=tags,
        )

        self._pnl_history.append(pnl)
        self.graph.tick()
        self.graph.save()

    # ── Internet knowledge ─────────────────────────────────────────────────────

    def set_market_mode(self, mode: str) -> None:
        self._market_mode = mode.upper()

    def absorb_internet_knowledge(self, nodes: List[KnowledgeNode]):
        for node in nodes:
            self.graph.add_node(node)
        if nodes:
            logger.info("Absorbed %d internet knowledge nodes", len(nodes))

    def absorb_news_as_nodes(self):
        try:
            articles = self.news.get_relevant(min_relevance=0.4)
            for article in articles[:5]:  # เพิ่มจาก 3 เป็น 5
                direction = ("bullish" if article.sentiment > 0.2
                             else ("bearish" if article.sentiment < -0.2 else "neutral"))
                node = KnowledgeNode(
                    node_type=NodeType.NEWS_EVENT,
                    concept=f"[News] {article.title[:100]}",
                    data={
                        "sentiment": article.sentiment,
                        "direction": direction,
                        "source": article.source,
                        "is_high_impact": article.is_high_impact,
                    },
                    confidence=0.45,
                    source="internet",
                    asset=self.asset,
                    tags=["news", "internet", self.asset.lower(), direction],
                )
                self.graph.add_node(node)
        except Exception as exc:
            logger.debug("News absorption failed: %s", exc)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _query_graph_ultra(
        self, snapshot: dict, regime: str
    ) -> Tuple[float, float, float, List[str]]:
        buy_score = sell_score = hold_score = 0.0
        fired = []

        for direction, tags in [("buy", ["buy_signal"]), ("sell", ["sell_signal"])]:
            nodes = self.graph.query(
                tags=tags,
                asset=self.asset,
                min_confidence=0.45,   # ลดจาก 0.50 เพื่อ recall ดีขึ้น
                limit=8,
            )
            for node in nodes:
                weight = node.confidence * node.activation
                # ปรับ weight ตาม regime
                if regime == "trending" and "macd" in node.tags:
                    weight *= 1.3
                elif regime == "ranging" and "rsi" in node.tags:
                    weight *= 1.3

                if self._node_matches_state(node, snapshot):
                    if direction == "buy":
                        buy_score  += weight
                    else:
                        sell_score += weight
                    fired.append(node.node_id)
                    node.activation = min(1.0, node.activation + 0.1)

        # Danger signals
        danger_nodes = self.graph.query(tags=["risk", "avoid"], min_confidence=0.5, limit=5)
        for node in danger_nodes:
            hold_score += node.confidence * node.activation * 0.5
            fired.append(node.node_id)

        return buy_score, sell_score, hold_score, fired

    def _recall_episodes_ultra(
        self, snapshot: dict, ppo_action: int
    ) -> Tuple[float, float, str]:
        similar = self.episodic.recall_similar(snapshot, n=8)  # recall มากขึ้น
        if not similar:
            return 0.0, 0.0, ""

        wins = sum(1 for e in similar if e.win)
        total = len(similar)
        win_rate = wins / total

        # แยก buy/sell wins
        buy_wins  = sum(1 for e in similar if e.win and e.action == "buy")
        sell_wins = sum(1 for e in similar if e.win and e.action == "sell")

        ep_buy  = (buy_wins / total) * 0.3
        ep_sell = (sell_wins / total) * 0.3

        reason = ""
        if win_rate > 0.6:
            reason = f"Memory: similar won {wins}/{total} ({win_rate:.0%})"
        elif win_rate < 0.4:
            reason = f"Memory: similar lost {total-wins}/{total} ({1-win_rate:.0%})"

        return ep_buy, ep_sell, reason

    def _candlestick_signals(self, snapshot: dict) -> Tuple[float, float, str]:
        """วิเคราะห์ candlestick patterns จาก indicator vector"""
        buy_score = sell_score = 0.0
        reasons   = []

        hammer      = snapshot.get("is_hammer", 0.0)
        engulfing   = snapshot.get("is_engulfing", 0.0)
        star        = snapshot.get("is_star", 0.0)
        doji        = snapshot.get("is_doji", 0.0)
        divergence  = snapshot.get("rsi_divergence", 0.0)
        sar         = snapshot.get("parabolic_sar", 0.0)

        if hammer > 0.5:
            buy_score += 0.3
            reasons.append("Pattern: Hammer (bullish)")
        elif hammer < -0.5:
            sell_score += 0.3
            reasons.append("Pattern: Shooting Star (bearish)")

        if engulfing > 0.5:
            buy_score += 0.35
            reasons.append("Pattern: Bullish Engulfing")
        elif engulfing < -0.5:
            sell_score += 0.35
            reasons.append("Pattern: Bearish Engulfing")

        if star > 0.5:
            buy_score += 0.4
            reasons.append("Pattern: Morning Star")
        elif star < -0.5:
            sell_score += 0.4
            reasons.append("Pattern: Evening Star")

        if doji > 0.5:
            reasons.append("Pattern: Doji (indecision)")

        if divergence > 0.5:
            buy_score += 0.3
            reasons.append("Divergence: Bullish RSI")
        elif divergence < -0.5:
            sell_score += 0.3
            reasons.append("Divergence: Bearish RSI")

        if sar > 0.5:
            buy_score += 0.15
        elif sar < -0.5:
            sell_score += 0.15

        reason = " | ".join(reasons[:2]) if reasons else ""
        return buy_score, sell_score, reason

    def _news_signals(self) -> Tuple[float, float, float, str]:
        if self._market_mode == "OTC":
            return 0.0, 0.0, 0.0, ""   # irrelevant for OTC synthetic prices
        buy = sell = hold = 0.0
        reason = ""
        try:
            sentiment   = self.news.get_sentiment_for_asset()
            high_impact = self.news.has_high_impact_news()

            if abs(sentiment) > 0.2:
                direction = "bullish" if sentiment > 0 else "bearish"
                reason = f"News: {direction} ({sentiment:+.2f})"
                buy  += max(0, sentiment) * 0.3
                sell += max(0, -sentiment) * 0.3

            if high_impact:
                hold   += 1.5   # strong hold signal — overrides weak directional signals
                reason += " | High-impact news! (avoid trading)"
        except Exception:
            pass
        return buy, sell, hold, reason

    def _calendar_risk(self) -> Tuple[float, str]:
        if self._market_mode == "OTC":
            return 1.0, ""   # no economic events affect OTC prices
        risk_mult = 1.0
        reason    = ""
        try:
            risk_mult = self.calendar.get_risk_multiplier()
            if risk_mult < 1.0:
                upcoming = self.calendar.describe_upcoming()
                reason   = f"Calendar: risk={risk_mult:.1f} {upcoming}"
        except Exception:
            pass
        return risk_mult, reason

    def _performance_signals(self) -> Tuple[float, str]:
        streak   = self.short_term.detect_streak()
        win_rate = self.short_term.win_rate()
        reason   = ""
        boost    = 0.0

        if streak["losses"] >= 3:
            boost  -= 0.2
            reason = f"Caution: {streak['losses']} consecutive losses"
        elif streak["losses"] >= 5:
            boost  -= 0.4
            reason = f"STOP: {streak['losses']} consecutive losses!"

        if win_rate > 0.65:
            boost  += 0.1
            reason = f"Hot streak: {win_rate:.0%} win rate"

        # PnL trend
        if len(self._pnl_history) >= 5:
            recent = list(self._pnl_history)[-5:]
            if all(p > 0 for p in recent):
                boost  += 0.15
                reason += " | 5 wins in a row!"

        return boost, reason

    def _node_matches_state(self, node: KnowledgeNode, snapshot: dict) -> bool:
        data      = node.data
        if not data:
            return True
        indicator = data.get("indicator")
        threshold = data.get("threshold")
        if indicator and threshold is not None:
            val = snapshot.get(indicator)
            if val is None:
                return True
            direction = data.get("direction", "")
            if "sell" in direction or "overbought" in node.tags:
                return float(val) > float(threshold) * 0.85 / 100.0
            if "buy" in direction or "oversold" in node.tags:
                return float(val) < float(threshold) * 1.15 / 100.0
        return True

    def _calibrate_confidence(self, raw_conf: float, regime: str, ppo_conf: float) -> float:
        """ปรับ confidence ให้ calibrate ดีขึ้น"""
        # Blend ระหว่าง raw_conf กับ ppo_conf
        blended = raw_conf * 0.6 + ppo_conf * 0.4

        # regime penalty
        if regime == "volatile":
            blended *= 0.85

        # clip ไม่ให้ over-confident
        return float(np.clip(blended, 0.0, 0.95))

    def _extract_pattern_ultra(self, snapshot: dict, direction: str) -> Optional[KnowledgeNode]:
        rsi       = snapshot.get("rsi", 0.5)
        macd      = snapshot.get("macd_hist", 0)
        adx       = snapshot.get("adx", 0.25)
        bb_pos    = snapshot.get("bb_position", 0.5)
        ichimoku  = snapshot.get("ichimoku_cloud_pos", 0)

        regime     = "trending" if adx > 0.25 else "ranging"
        rsi_state  = "overbought" if rsi > 0.65 else ("oversold" if rsi < 0.35 else "neutral")
        macd_state = "positive" if macd > 0 else "negative"
        cloud      = "above_cloud" if ichimoku > 0 else ("below_cloud" if ichimoku < 0 else "in_cloud")

        concept = (
            f"ULTRA Pattern: {direction.upper()} | RSI={rsi_state} "
            f"MACD={macd_state} ADX={regime} BB_pos={bb_pos:.2f} {cloud}"
        )
        tags = [direction, "pattern", regime, rsi_state, f"macd_{macd_state}", "ultra"]

        return KnowledgeNode(
            node_type=NodeType.PATTERN,
            concept=concept,
            data={
                "direction": direction, "rsi_state": rsi_state,
                "macd_state": macd_state, "regime": regime,
                "bb_position": bb_pos, "cloud": cloud,
            },
            confidence=0.60,
            source="internal",
            asset=self.asset,
            tags=tags + [f"{direction}_signal"],
        )

    @staticmethod
    def _regime_tags(snapshot: dict) -> List[str]:
        tags = []
        rsi  = snapshot.get("rsi", 0.5)
        adx  = snapshot.get("adx", 0.25)
        if rsi > 0.65:
            tags.append("overbought")
        elif rsi < 0.35:
            tags.append("oversold")
        if adx > 0.25:
            tags.append("trending")
        else:
            tags.append("ranging")
        return tags

    @staticmethod
    def _vec_to_snapshot(vec: np.ndarray, names: List[str]) -> dict:
        return {name: float(vec[i]) for i, name in enumerate(names) if i < len(vec)}
