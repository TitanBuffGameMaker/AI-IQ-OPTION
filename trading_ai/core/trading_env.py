"""
Trading Environment — Binary Options Edition

Binary Options rules (IQ Option):
  WIN  (+CALL or +PUT correct direction): +payout% of stake (fixed, e.g. 80-89%)
  LOSS (wrong direction):                 -100% of stake (all gone)
  HOLD (skip this candle):                ±0  (capital preserved — this is free)

Reward scale per step:
  Win  → base_reward ≈ +0.80 to +0.89
  Loss → base_reward = -1.00
  Hold → 0.00  (waiting costs nothing — do NOT penalise)

Break-even win rate: WR > 1/(1 + payout)
  80% payout → need > 55.6% win rate
  89% payout → need > 53.0% win rate
"""
import logging
import time
from collections import deque
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from trading_ai.config import config
from trading_ai.core.iq_connector import IQOptionConnector
from trading_ai.indicators.technical import IndicatorEngine, N_FEATURES
from trading_ai.utils.price_encoder import PriceSequenceEncoder

logger = logging.getLogger(__name__)

N_INDICATORS    = N_FEATURES   # 40 indicators
N_CHART_FEATURES = 256          # CNN features
OBS_SIZE        = N_INDICATORS + N_CHART_FEATURES   # 296


class TradingEnv(gym.Env):
    """
    Gymnasium trading environment พร้อม ULTRA reward shaping:
    - Sharpe ratio reward (รางวัลสำหรับ risk-adjusted return)
    - Anti-overtrading penalty
    - Drawdown protection
    """

    metadata = {"render_modes": []}
    ACTIONS  = {0: "hold", 1: "call", 2: "put"}
    # Binary Options: waiting costs nothing — capital is preserved when you don't trade.
    # Penalising HOLD teaches the AI to "always trade" which destroys the account.
    # Set to 0.0 so the AI learns: only trade when expected value is positive.
    HOLD_PENALTY = 0.0
    FAIL_PENALTY = -0.05   # trade placement failure (API error, market closed)

    def __init__(self, connector: IQOptionConnector):
        super().__init__()
        self.connector       = connector
        self.indicator_engine = IndicatorEngine()
        self.price_encoder   = PriceSequenceEncoder()

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )

        # Active asset for this env — defaults to config.ASSET but can be
        # overridden per cycle (multi-asset trading in the AI loop).
        self._current_asset: str = config.ASSET

        # Optional callback fired right after place_trade succeeds.
        # Signature: fn(order_id, asset, direction, amount, duration_min)
        # Server uses this to register the trade in the live open-orders book.
        self.on_trade_placed = None
        # Optional callback fired when the trade result comes back.
        # Signature: fn(order_id)
        self.on_trade_closed = None

        self._balance_start:      float = 0.0
        self._consecutive_losses: int   = 0
        self._daily_pnl:          float = 0.0
        self._episode_pnl:        float = 0.0
        self._last_obs:           Optional[np.ndarray] = None
        self._last_candles:       Optional[pd.DataFrame] = None
        self._trade_amount:       float = config.TRADE_AMOUNT  # overridden by server for Kelly sizing

        # สำหรับ Sharpe reward
        self._reward_history: deque = deque(maxlen=config.SHARPE_WINDOW)
        # สำหรับ anti-overtrading
        self._recent_trades:  deque = deque(maxlen=10)
        # สำหรับ drawdown protection
        self._peak_balance:   float = 0.0
        self._trade_count_episode: int = 0

    # ── Asset rotation (for multi-asset AI loop) ────────────────────────────

    def set_asset(self, asset: str) -> None:
        """Switch the asset traded on the next step()/observation."""
        if asset and asset != self._current_asset:
            self._current_asset = asset

    # ── Gym API ─────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._balance_start      = self.connector.get_balance()
        self._peak_balance       = self._balance_start
        self._episode_pnl        = 0.0
        self._consecutive_losses = 0
        self._trade_count_episode = 0
        self._reward_history.clear()
        obs = self._get_observation()
        self._last_obs = obs
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action_name = self.ACTIONS[action]
        reward      = 0.0
        info        = {"action": action_name, "pnl": 0.0, "skipped": False}
        terminated  = False

        if action == 0:
            # HOLD: waiting is neutral in binary options (capital preserved)
            reward = self.HOLD_PENALTY   # = 0.0
            info["skipped"] = True
        else:
            if not self.connector.is_market_open(self._current_asset):
                # Tried to trade but market closed → treat as forced HOLD
                reward = self.HOLD_PENALTY   # = 0.0, not a mistake
                info["skipped"] = True
            else:
                # Anti-overtrading: penalise if trading every single candle
                # (binary options needs patience, not spray-and-pray)
                trade_rate = sum(self._recent_trades) / max(len(self._recent_trades), 1)
                if trade_rate > 0.8:
                    reward = self.FAIL_PENALTY
                    info["skipped"] = True
                    logger.debug("Anti-overtrading: trade rate too high (%.1f%%)", trade_rate * 100)
                else:
                    reward, info = self._execute_trade(action_name)
                    self._trade_count_episode += 1

        self._recent_trades.append(1 if not info.get("skipped") else 0)

        # Circuit breakers — REAL account only.
        # PRACTICE: never stop; every trade is a free learning experience.
        # OTC assets are synthetic — losses don't cost real money in practice.
        _is_real = config.IQ_ACCOUNT_TYPE.upper() == "REAL"
        if _is_real:
            if self._daily_pnl <= -config.DAILY_LOSS_LIMIT:
                logger.warning("Daily loss limit hit")
                terminated = True
            if self._consecutive_losses >= config.MAX_CONSECUTIVE_LOSSES:
                logger.warning("Too many consecutive losses")
                terminated = True

            # Drawdown protection (REAL only)
            current_bal = self._balance_start + self._episode_pnl
            if current_bal > self._peak_balance:
                self._peak_balance = current_bal
            drawdown = (self._peak_balance - current_bal) / (self._peak_balance + 1e-9)
            if drawdown > 0.2:
                logger.warning("Max drawdown reached: %.1f%%", drawdown * 100)
                terminated = True

        obs = self._get_observation()
        self._last_obs = obs
        return obs, reward, terminated, False, info

    def render(self):
        pass

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _execute_trade(self, direction: str) -> Tuple[float, dict]:
        asset = self._current_asset
        balance_before = self.connector.get_balance()

        success, order_id = self.connector.place_trade(
            asset=asset,
            direction=direction,
            amount=self._trade_amount,
            duration_minutes=config.TRADE_DURATION,
        )
        if not success or order_id is None:
            return self.FAIL_PENALTY, {"action": direction, "pnl": 0.0, "skipped": True}

        # Notify any external listener (server.py adds to its open-orders book
        # so the dashboard can show a live countdown).
        if self.on_trade_placed:
            try:
                self.on_trade_placed(order_id, asset, direction, self._trade_amount,
                                     config.TRADE_DURATION)
            except Exception as exc:
                logger.debug("on_trade_placed hook error: %s", exc)

        wait_seconds = config.TRADE_DURATION * 60 + 5
        logger.info("Trade %s placed on %s — waiting %ds for expiry…", order_id, asset, wait_seconds)
        time.sleep(wait_seconds)

        # Try official API first (30s), fall back to balance delta if it times out.
        pnl_result = self.connector.get_trade_result(
            order_id, timeout=30, balance_before=balance_before
        )
        if self.on_trade_closed:
            try:
                self.on_trade_closed(order_id)
            except Exception as exc:
                logger.debug("on_trade_closed hook error: %s", exc)
        if pnl_result is None:
            logger.warning("Trade %s: result unavailable — not counted (skipped)", order_id)
            return 0.0, {"action": direction, "pnl": 0.0, "skipped": True}
        pnl = pnl_result
        self._episode_pnl += pnl
        self._daily_pnl   += pnl

        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # Binary Options Reward:
        # base = pnl / stake → +0.80~0.89 (win) or -1.00 (loss)
        # This exact asymmetry teaches the AI: need >55% win rate to profit
        base_reward = pnl / (config.TRADE_AMOUNT + 1e-9)

        # Win-rate consistency bonus: reward sustained winning streaks
        # (replaces Sharpe which is designed for continuous returns, not binary)
        self._reward_history.append(base_reward)
        win_bonus = 0.0
        if len(self._reward_history) >= 5:
            r_arr    = np.array(self._reward_history)
            win_rate = float((r_arr > 0).mean())
            # Bonus only above break-even (~55%); penalise below
            win_bonus = np.clip((win_rate - 0.556) * 0.5, -0.10, 0.10)

        # Streak bonus: reward consistency (3+ consecutive wins)
        if pnl > 0 and self._consecutive_losses == 0 and self._trade_count_episode >= 3:
            streak_bonus = 0.05
        else:
            streak_bonus = 0.0

        reward = base_reward + win_bonus + streak_bonus

        outcome = "WIN" if pnl > 0 else "LOSS"
        logger.info(
            "Binary %s %s: PnL=%.2f reward=%.3f (base=%.3f win_bonus=%.3f streak=%.2f)",
            outcome, direction, pnl, reward, base_reward, win_bonus, streak_bonus,
        )
        return reward, {"action": direction, "pnl": pnl, "skipped": False}

    def _get_observation(self) -> np.ndarray:
        df = self.connector.get_candles(
            asset=self._current_asset,
            timeframe_seconds=config.CANDLE_TIMEFRAME,
            count=config.LOOKBACK_CANDLES + 10,
        )
        self._last_candles = df
        if df is not None and len(df) >= 60:
            ind_vec = self.indicator_engine.compute(df)
        else:
            ind_vec = None

        if ind_vec is None or len(ind_vec) != N_INDICATORS:
            ind_vec = np.zeros(N_INDICATORS, dtype=np.float32)
        ind_vec = ind_vec[:N_INDICATORS]

        chart_vec = self.price_encoder.encode(df)
        if chart_vec is None or len(chart_vec) != N_CHART_FEATURES:
            chart_vec = np.zeros(N_CHART_FEATURES, dtype=np.float32)

        obs = np.concatenate([ind_vec, chart_vec]).astype(np.float32)
        obs = np.clip(obs, -1.0, 1.0)
        return obs
