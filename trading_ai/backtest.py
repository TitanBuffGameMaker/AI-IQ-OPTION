"""
Backtesting engine – train the PPO agent on historical OHLCV data
BEFORE using real money.

Why this is critical:
  - Live trading costs real money for every bad trade during early learning
  - Historical data lets the agent make 10,000+ trades in minutes instead of months
  - You can verify the AI actually improves before going live

Usage:
    python -m trading_ai.backtest --csv data/EURUSD_M1.csv --episodes 50
    python -m trading_ai.backtest --iqoption --asset EURUSD --days 30 --episodes 50
"""
import argparse
import logging
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from trading_ai.config import config
from trading_ai.core.knowledge_base import KnowledgeBase
from trading_ai.core.trading_env import N_INDICATORS, N_CHART_FEATURES, OBS_SIZE
from trading_ai.indicators.technical import IndicatorEngine
from trading_ai.models.ppo_agent import PPOAgent
from trading_ai.utils.logger import setup_logging

logger = logging.getLogger(__name__)

ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}


class BacktestEnv:
    """
    Simulates the trading environment using historical OHLCV data.
    No real money, no IQ Option connection needed.

    Binary option simulation:
      - BUY: win if close[t+duration] > close[t], else lose
      - SELL: win if close[t+duration] < close[t], else lose
      - Payout: +payout% of stake on win, -100% on loss
    """

    def __init__(
        self,
        df: pd.DataFrame,
        stake: float = 1.0,
        duration_candles: int = 1,
        payout_pct: float = 0.85,
        lookback: int = 60,
    ):
        assert all(c in df.columns for c in ["open", "high", "low", "close", "volume"]), \
            "DataFrame must have columns: open, high, low, close, volume"
        self.df = df.reset_index(drop=True)
        self.stake = stake
        self.duration = duration_candles
        self.payout = payout_pct
        self.lookback = lookback
        self.indicator_engine = IndicatorEngine()

        self._cursor: int = lookback
        self._balance: float = 1000.0
        self._start_balance: float = 1000.0
        self._episode_pnl: float = 0.0
        self._n_trades: int = 0
        self._n_wins: int = 0

    # ── Gym-like API ──────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        self._cursor = self.lookback
        self._episode_pnl = 0.0
        self._n_trades = 0
        self._n_wins = 0
        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        reward = 0.0
        pnl = 0.0
        skipped = False

        if action != 0:  # BUY or SELL
            pnl = self._simulate_trade(action)
            self._episode_pnl += pnl
            self._n_trades += 1
            if pnl > 0:
                self._n_wins += 1
            reward = pnl / (self.stake + 1e-9)  # normalized to [-1, +1]
        else:
            reward = -0.01  # small hold penalty
            skipped = True

        self._cursor += 1
        done = self._cursor >= len(self.df) - self.duration - 1

        obs = self._get_obs() if not done else np.zeros(OBS_SIZE, dtype=np.float32)
        info = {
            "pnl": pnl,
            "skipped": skipped,
            "balance": self._balance + self._episode_pnl,
            "cursor": self._cursor,
        }
        return obs, reward, done, info

    def win_rate(self) -> float:
        return self._n_wins / max(self._n_trades, 1)

    def episode_stats(self) -> dict:
        return {
            "pnl": round(self._episode_pnl, 2),
            "trades": self._n_trades,
            "wins": self._n_wins,
            "win_rate": round(self.win_rate(), 3),
            "balance": round(self._balance + self._episode_pnl, 2),
        }

    # ── Private ───────────────────────────────────────────────────────────────

    def _simulate_trade(self, action: int) -> float:
        """Simulate a binary option at cursor, expiring `duration` candles later."""
        entry_price = self.df.loc[self._cursor, "close"]
        exit_idx = min(self._cursor + self.duration, len(self.df) - 1)
        exit_price = self.df.loc[exit_idx, "close"]

        if action == 1:  # BUY (call) – win if price goes up
            win = exit_price > entry_price
        else:            # SELL (put) – win if price goes down
            win = exit_price < entry_price

        return self.stake * self.payout if win else -self.stake

    def _get_obs(self) -> np.ndarray:
        start = max(0, self._cursor - self.lookback)
        window = self.df.iloc[start: self._cursor + 1].copy()

        ind_vec = self.indicator_engine.compute(window)
        if ind_vec is None or len(ind_vec) < N_INDICATORS:
            ind_vec = np.zeros(N_INDICATORS, dtype=np.float32)
        ind_vec = ind_vec[:N_INDICATORS]

        # Chart features not available in backtest – use zeros
        chart_vec = np.zeros(N_CHART_FEATURES, dtype=np.float32)

        obs = np.concatenate([ind_vec, chart_vec]).astype(np.float32)
        return np.clip(obs, -1.0, 1.0)


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    """Load OHLCV data from a CSV file."""
    df = pd.read_csv(path)
    # Normalize column names
    df.columns = [c.lower().strip() for c in df.columns]
    rename = {}
    for col in df.columns:
        if "open" in col:
            rename[col] = "open"
        elif "high" in col:
            rename[col] = "high"
        elif "low" in col:
            rename[col] = "low"
        elif "close" in col:
            rename[col] = "close"
        elif "vol" in col:
            rename[col] = "volume"
    df = df.rename(columns=rename)
    if "volume" not in df.columns:
        df["volume"] = 1.0
    required = ["open", "high", "low", "close", "volume"]
    df = df[required].astype(float).dropna().reset_index(drop=True)
    logger.info("Loaded %d candles from %s", len(df), path)
    return df


def fetch_from_iqoption(asset: str, days: int) -> Optional[pd.DataFrame]:
    """Fetch historical candles from IQ Option API."""
    try:
        from trading_ai.core.iq_connector import IQOptionConnector
        connector = IQOptionConnector(
            email=config.IQ_EMAIL,
            password=config.IQ_PASSWORD,
            account_type="PRACTICE",
        )
        if not connector.connect():
            logger.error("Cannot connect to IQ Option for historical data")
            return None

        candles_needed = days * 24 * 60  # 1-minute candles
        df = connector.get_candles(
            asset=asset,
            timeframe_seconds=60,
            count=min(candles_needed, 5000),
        )
        logger.info("Fetched %d candles from IQ Option", len(df) if df is not None else 0)
        return df
    except Exception as exc:
        logger.error("IQ Option fetch failed: %s", exc)
        return None


# ── Main backtest runner ──────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    n_episodes: int = 50,
    save_dir: str = "./knowledge",
):
    """
    Train the PPO agent on historical data.
    Prints performance stats and saves the trained model.
    """
    logger.info("=" * 60)
    logger.info("  BACKTEST TRAINING")
    logger.info("  Candles: %d | Episodes: %d", len(df), n_episodes)
    logger.info("=" * 60)

    env = BacktestEnv(df)
    agent = PPOAgent(obs_size=OBS_SIZE, n_actions=3)
    knowledge = KnowledgeBase(base_dir=save_dir)

    # Load previous model if exists (continue training)
    knowledge.load_brain(agent)
    logger.info("Starting from step=%d", agent.total_steps)

    best_pnl = float("-inf")
    history = []

    for episode in range(1, n_episodes + 1):
        obs = env.reset()
        done = False
        last_metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        while not done:
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.store(obs, next_obs, action, log_prob, reward, value, done)
            obs = next_obs

            if agent.ready_to_update():
                last_metrics = agent.update(obs)

        stats = env.episode_stats()
        history.append(stats)

        agent.total_episodes = episode
        is_best = knowledge.record_episode(
            episode=episode,
            total_steps=agent.total_steps,
            episode_pnl=stats["pnl"],
            n_trades=stats["trades"],
            win_rate=stats["win_rate"],
            policy_loss=last_metrics["policy_loss"],
            value_loss=last_metrics["value_loss"],
            entropy=last_metrics["entropy"],
        )

        logger.info(
            "Ep %3d/%d | trades=%4d | win=%.1f%% | pnl=$%.2f | "
            "steps=%d | loss_p=%.4f",
            episode, n_episodes,
            stats["trades"], stats["win_rate"] * 100, stats["pnl"],
            agent.total_steps, last_metrics["policy_loss"],
        )

        if stats["pnl"] > best_pnl:
            best_pnl = stats["pnl"]
            knowledge.save_best(agent)

        if episode % 10 == 0:
            knowledge.save_brain(agent)
            _print_learning_curve(history)

    knowledge.save_brain(agent)
    knowledge.print_summary()
    logger.info("Backtest complete. Model saved to %s", save_dir)
    logger.info("Best episode P&L: $%.2f", best_pnl)
    return agent


def _print_learning_curve(history: list):
    """Print a simple ASCII learning curve."""
    if len(history) < 5:
        return
    recent = history[-10:]
    avg_wr = sum(e["win_rate"] for e in recent) / len(recent)
    avg_pnl = sum(e["pnl"] for e in recent) / len(recent)
    bar_len = int(avg_wr * 20)
    bar = "█" * bar_len + "░" * (20 - bar_len)
    logger.info(
        "Learning curve (last %d eps): win_rate=[%s] %.1f%%  avg_pnl=$%.2f",
        len(recent), bar, avg_wr * 100, avg_pnl,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Backtest the trading AI on historical data")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", help="Path to CSV file with OHLCV data")
    source.add_argument("--iqoption", action="store_true", help="Fetch data from IQ Option")
    parser.add_argument("--asset", default=config.ASSET, help="Asset symbol (e.g. EURUSD)")
    parser.add_argument("--days", type=int, default=30, help="Days of historical data (--iqoption)")
    parser.add_argument("--episodes", type=int, default=50, help="Training episodes")
    parser.add_argument("--save-dir", default=config.MODEL_DIR, help="Directory to save model")
    args = parser.parse_args()

    if args.csv:
        df = load_csv(args.csv)
    else:
        df = fetch_from_iqoption(args.asset, args.days)
        if df is None:
            sys.exit(1)

    run_backtest(df, n_episodes=args.episodes, save_dir=args.save_dir)
