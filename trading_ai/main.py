"""
IQ Option RL Trading AI – main entry point.

TWO LEARNING SYSTEMS WORKING TOGETHER
──────────────────────────────────────
1. PPO Agent (neural network):
   - Learns through gradient descent on trading outcomes
   - Weights = accumulated skill (pattern recognition)
   - Gets better the more it trades

2. Brain Core (knowledge graph + internet):
   - Grows like a root system: new knowledge = new branches
   - Searches internet for news, economic events, market analysis
   - Stores episodic memories of significant trades
   - Adjusts the PPO agent's confidence based on external knowledge
   - Never stops growing – knowledge accumulates forever

Both systems reinforce each other: the PPO agent learns market patterns
via gradient descent, while the brain builds explicit symbolic knowledge
from trade outcomes and internet research.
"""
import logging
import os
import sys
import time

import numpy as np

from trading_ai.config import config
from trading_ai.core.iq_connector import IQOptionConnector
from trading_ai.core.trading_env import TradingEnv, OBS_SIZE, N_INDICATORS
from trading_ai.core.knowledge_base import KnowledgeBase
from trading_ai.models.ppo_agent import PPOAgent
from trading_ai.brain.brain_core import BrainCore
from trading_ai.brain.visualizer import BrainVisualizer
from trading_ai.utils.logger import setup_logging

logger = logging.getLogger(__name__)

ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}


def run_trading_session():
    """Main loop: connect → load knowledge → trade → learn → save → repeat."""

    setup_logging(log_dir=config.LOG_DIR)

    logger.info("=" * 65)
    logger.info("  IQ OPTION RL TRADING AI  +  ROOT BRAIN SYSTEM")
    logger.info("  Asset: %s | Amount: $%.2f | Duration: %dmin",
                config.ASSET, config.TRADE_AMOUNT, config.TRADE_DURATION)
    logger.info("  Account: %s", config.IQ_ACCOUNT_TYPE)
    logger.info("=" * 65)

    # ── Initialize PPO knowledge base ────────────────────────────────────────
    knowledge = KnowledgeBase(base_dir=config.MODEL_DIR)
    knowledge.print_summary()

    # ── Connect to IQ Option ─────────────────────────────────────────────────
    connector = IQOptionConnector(
        email=config.IQ_EMAIL,
        password=config.IQ_PASSWORD,
        account_type=config.IQ_ACCOUNT_TYPE,
    )
    if not connector.connect():
        logger.error("Cannot connect to IQ Option. Check credentials in .env")
        sys.exit(1)

    # ── Initialize trading environment ───────────────────────────────────────
    env = TradingEnv(connector)
    agent = PPOAgent(obs_size=OBS_SIZE, n_actions=3)

    # ── Initialize the root-system brain ─────────────────────────────────────
    brain = BrainCore(asset=config.ASSET, base_dir=config.MODEL_DIR)
    visualizer = BrainVisualizer(brain.graph)

    # ── Load previous knowledge ──────────────────────────────────────────────
    loaded = knowledge.load_brain(agent)
    if loaded:
        logger.info("PPO resumed (steps=%d)", agent.total_steps)
    else:
        logger.info("PPO starting fresh")

    logger.info("Brain has %d knowledge nodes", brain.graph.stats()["total_nodes"])

    # Print the current brain tree on startup
    visualizer.print_tree(min_confidence=0.45)

    # ── Main trading loop ────────────────────────────────────────────────────
    episode = agent.total_episodes
    last_metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    try:
        while True:
            episode += 1
            logger.info("── Episode %d ──────────────────────────────────────────", episode)

            obs, _ = env.reset()
            done = False
            episode_trades = 0
            episode_wins = 0
            episode_pnl = 0.0

            while not done:
                # ── Extract indicator portion of observation ──────────────
                indicator_vec = obs[:N_INDICATORS]

                # ── PPO agent's initial decision ──────────────────────────
                ppo_action, ppo_log_prob, ppo_value = agent.select_action(obs)
                _, ppo_confidence = agent.get_confidence(obs)

                # ── Brain synthesizes all knowledge into a signal ─────────
                brain_signal = brain.think(
                    indicator_vec=indicator_vec,
                    ppo_action=ppo_action,
                    ppo_confidence=ppo_confidence,
                )

                # ── Risk gate: brain can block trades ─────────────────────
                if brain_signal.risk_multiplier < 0.2:
                    logger.info(
                        "Brain blocked trade (risk=%.2f) – HOLD forced",
                        brain_signal.risk_multiplier,
                    )
                    final_action = 0
                elif (
                    brain_signal.action != ppo_action
                    and brain_signal.confidence > ppo_confidence + 0.15
                ):
                    # Brain overrides PPO only when significantly more confident
                    logger.info(
                        "Brain overrides PPO: %s→%s (brain_conf=%.2f > ppo_conf=%.2f)",
                        ACTION_NAMES[ppo_action], ACTION_NAMES[brain_signal.action],
                        brain_signal.confidence, ppo_confidence,
                    )
                    final_action = brain_signal.action
                else:
                    # PPO leads; brain's confidence adjusts but doesn't override
                    final_action = ppo_action
                    if brain_signal.confidence < config.MIN_CONFIDENCE and final_action != 0:
                        logger.debug(
                            "Low combined confidence (%.2f) – HOLD",
                            brain_signal.confidence,
                        )
                        final_action = 0

                # ── Execute in environment ────────────────────────────────
                next_obs, reward, terminated, truncated, info = env.step(final_action)
                done = terminated or truncated

                # ── Store in PPO buffer ───────────────────────────────────
                agent.store(obs, ppo_action, ppo_log_prob, reward, ppo_value, done)

                # ── Brain learns from the outcome ─────────────────────────
                if not info.get("skipped", False):
                    pnl = info.get("pnl", 0.0)
                    brain.learn(
                        pnl=pnl,
                        action_taken=final_action,
                        indicator_vec=indicator_vec,
                        ppo_action=ppo_action,
                    )
                    episode_trades += 1
                    if pnl > 0:
                        episode_wins += 1
                    episode_pnl += pnl

                obs = next_obs

                # ── PPO update ─────────────────────────────────────────────
                if agent.ready_to_update():
                    logger.info("PPO: running gradient update …")
                    last_metrics = agent.update(obs)
                    knowledge.save_brain(agent)

            # ── End of episode ───────────────────────────────────────────────
            win_rate = episode_wins / max(episode_trades, 1)
            logger.info(
                "Episode %d | trades=%d wins=%d win_rate=%.1f%% pnl=$%.2f",
                episode, episode_trades, episode_wins, win_rate * 100, episode_pnl,
            )

            agent.total_episodes = episode
            is_best = knowledge.record_episode(
                episode=episode,
                total_steps=agent.total_steps,
                episode_pnl=episode_pnl,
                n_trades=episode_trades,
                win_rate=win_rate,
                policy_loss=last_metrics["policy_loss"],
                value_loss=last_metrics["value_loss"],
                entropy=last_metrics["entropy"],
            )
            if is_best:
                knowledge.save_best(agent)

            if episode % config.CHECKPOINT_EVERY == 0:
                knowledge.save_checkpoint(agent, episode)
                # Export brain graph for visualization
                visualizer.export_json(
                    os.path.join(config.MODEL_DIR, f"brain_graph_ep{episode}.json")
                )

            # Print brain status every 10 episodes
            if episode % 10 == 0:
                brain.print_status()
                visualizer.print_tree(min_confidence=0.50)
                visualizer.print_summary_bar()

            knowledge.print_summary()
            logger.info("Pausing 10s before next episode …")
            time.sleep(10)

    except KeyboardInterrupt:
        logger.info("Interrupted by user – saving and shutting down …")
    finally:
        knowledge.save_brain(agent)
        brain.shutdown()
        visualizer.export_json(os.path.join(config.MODEL_DIR, "brain_graph_final.json"))
        logger.info("All knowledge saved. Goodbye.")


def run_brain_status():
    """Show the current brain tree without connecting to IQ Option."""
    setup_logging()
    brain = BrainCore(asset=config.ASSET, base_dir=config.MODEL_DIR)
    vis = BrainVisualizer(brain.graph)
    brain.print_status()
    vis.print_tree()

    top = vis.most_active_roots(5)
    logger.info("Most active knowledge roots:")
    for i, node in enumerate(top, 1):
        logger.info("  %d. %s", i, node)

    brain.shutdown()


def run_inference_only():
    """Check what the current brain would recommend without placing trades."""
    setup_logging()

    connector = IQOptionConnector(
        email=config.IQ_EMAIL, password=config.IQ_PASSWORD, account_type="PRACTICE"
    )
    if not connector.connect():
        logger.error("Cannot connect"); sys.exit(1)

    env = TradingEnv(connector)
    agent = PPOAgent(obs_size=OBS_SIZE)
    brain = BrainCore(asset=config.ASSET, base_dir=config.MODEL_DIR)

    knowledge = KnowledgeBase(base_dir=config.MODEL_DIR)
    knowledge.load_brain(agent)

    obs, _ = env.reset()
    indicator_vec = obs[:N_INDICATORS]
    ppo_action, ppo_confidence = agent.get_confidence(obs)
    signal = brain.think(indicator_vec, ppo_action, ppo_confidence)

    logger.info("PPO says: %s (conf=%.1f%%)", ACTION_NAMES[ppo_action], ppo_confidence * 100)
    logger.info("Brain says: %s (conf=%.1f%%, risk=%.2f)",
                ACTION_NAMES[signal.action], signal.confidence * 100, signal.risk_multiplier)
    logger.info("Reasoning:\n  " + "\n  ".join(signal.reasoning))

    brain.shutdown()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    {
        "train": run_trading_session,
        "brain": run_brain_status,
        "infer": run_inference_only,
    }.get(mode, run_trading_session)()
