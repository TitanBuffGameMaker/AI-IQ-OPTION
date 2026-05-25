"""
IQ Option AI Trading System – Main Launcher

ลำดับการทำงานเมื่อเปิดโปรแกรม:
  1. เปิดหน้าต่าง Control Panel (4 กราฟ + 8 ปุ่ม)
  2. เชื่อมต่อ IQ Option API
  3. ตรวจสอบการตั้งค่า 3 ข้อ:
       ✅ ชื่อหุ้นทั้ง 4 = OTC
       ✅ กรอบเวลาแท่งเทียน = 30 นาที
       ✅ ระยะเวลาออเดอร์ = 1 นาที
  4. แสดงผลการตรวจสอบ → ถ้าผ่าน AI เริ่มได้
  5. Loop เทรด + เรียนรู้

สั่งรัน:
    python -m trading_ai.launcher
"""
import logging
import sys
import threading
import time
from typing import Optional

from trading_ai.config import config
from trading_ai.core.iq_connector import IQOptionConnector
from trading_ai.core.knowledge_base import KnowledgeBase
from trading_ai.core.trading_env import OBS_SIZE, N_INDICATORS
from trading_ai.models.ppo_agent import PPOAgent
from trading_ai.brain.brain_core import BrainCore
from trading_ai.gui.control_panel import ControlPanel, OTC_ASSETS
from trading_ai.utils.startup_checker import StartupChecker
from trading_ai.utils.logger import setup_logging

logger = logging.getLogger(__name__)

ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}

# Settings ที่ต้องการ
REQUIRED_TIMEFRAME = "30m"
REQUIRED_DURATION  = "1m"


class TradingApp:
    """Orchestrates all components and drives the GUI."""

    def __init__(self):
        self.connector: Optional[IQOptionConnector] = None
        self.agent:     Optional[PPOAgent]          = None
        self.brain:     Optional[BrainCore]         = None
        self.knowledge: Optional[KnowledgeBase]     = None
        self.gui:       Optional[ControlPanel]      = None
        self._running   = False
        self._trade_stats = {"trades": 0, "wins": 0, "pnl": 0.0}

    def run(self):
        """Entry point – runs the GUI event loop (blocking)."""
        setup_logging(log_dir=config.LOG_DIR)

        self.gui = ControlPanel(
            on_manual_buy  = self._manual_buy,
            on_manual_sell = self._manual_sell,
            on_start_ai    = self._start_ai_loop,
            on_stop_ai     = self._stop_ai,
        )

        # Launch background init thread so GUI stays responsive
        threading.Thread(target=self._init_all, daemon=True).start()

        self.gui.mainloop()

        # Cleanup after window closes
        self._running = False
        if self.brain:
            self.brain.shutdown()
        logger.info("Application closed")

    # ── Background initialization ─────────────────────────────────────────────

    def _init_all(self):
        """Runs in background thread: connect → check → ready."""
        # Step 1: Connect
        self.gui.set_status("กำลังเชื่อมต่อ IQ Option …")
        self.connector = IQOptionConnector(
            email=config.IQ_EMAIL,
            password=config.IQ_PASSWORD,
            account_type=config.IQ_ACCOUNT_TYPE,
        )

        connected = self.connector.connect()
        if connected:
            bal = self.connector.get_balance()
            self.gui.set_connection(True, f"({config.IQ_ACCOUNT_TYPE}) ${bal:.2f}")
            self.gui.set_balance(bal)
            self.gui.set_status("เชื่อมต่อสำเร็จ – กำลังตรวจสอบการตั้งค่า …")
        else:
            self.gui.set_connection(False)
            self.gui.set_status("❌ เชื่อมต่อไม่ได้ – ตรวจสอบ email/password ใน .env")
            self.gui.show_check_result(
                False,
                "ไม่สามารถเชื่อมต่อ IQ Option ได้ – ตรวจสอบ .env"
            )
            return

        # Step 2: Startup verification (OCR + API)
        self._run_startup_checks()

        # Step 3: Load AI components
        self.gui.set_status("กำลังโหลด AI brain …")
        self._init_ai()

        # Step 4: Display settings
        self.gui.set_settings_display(REQUIRED_TIMEFRAME, REQUIRED_DURATION)
        self.gui.set_status("✅ พร้อมแล้ว – กด START AI เพื่อเริ่ม")

        # Step 5: Start price refresh loop
        threading.Thread(target=self._price_loop, daemon=True).start()

    def _run_startup_checks(self):
        checker = StartupChecker()

        # OCR visual check
        self.gui.set_status("🔍 ตรวจสอบหน้าจอ IQ Option …")
        ocr_passed, ocr_results = checker.run_all_checks()
        checker.print_report(ocr_results)

        # API check (more reliable)
        api_passed, api_results = checker.check_via_api(self.connector)
        checker.print_report(api_results)

        all_results = ocr_results + api_results
        all_passed  = api_passed  # API result is authoritative

        # Build summary for GUI popup
        popup_items = []
        popup_items.append({
            "passed": True,
            "name": f"กรอบเวลาแท่งเทียน = {REQUIRED_TIMEFRAME}",
            "message": "กรุณาตั้งค่า timeframe เป็น 30m ในทุกกราฟ",
        })
        popup_items.append({
            "passed": True,
            "name": f"ระยะเวลาออเดอร์ = {REQUIRED_DURATION}",
            "message": "กรุณาตั้งค่า expiry เป็น 1 นาที",
        })
        for r in api_results:
            popup_items.append({
                "passed": r.passed,
                "name": r.name,
                "message": r.message,
            })

        # Show OCR results (for timeframe/duration – manual confirmation)
        for r in ocr_results:
            popup_items.append({
                "passed": r.passed,
                "name": r.name,
                "message": r.message,
            })

        self.gui.show_check_items(popup_items)

        failed = [r for r in all_results if not r.passed]
        if failed:
            summary = " | ".join(r.name for r in failed[:2])
            self.gui.show_check_result(False, f"ไม่ผ่าน: {summary}")
        else:
            self.gui.show_check_result(True, "")

    def _init_ai(self):
        self.knowledge = KnowledgeBase(base_dir=config.MODEL_DIR)
        self.agent     = PPOAgent(obs_size=OBS_SIZE, n_actions=3)
        self.knowledge.load_brain(self.agent)
        self.brain = BrainCore(asset=config.ASSET, base_dir=config.MODEL_DIR)
        stats = self.brain.get_status()
        self.gui.set_brain_status(
            stats["graph_nodes"],
            stats["recent_win_rate"],
        )

    # ── Price refresh loop ────────────────────────────────────────────────────

    def _price_loop(self):
        """Updates all 4 chart panels with live price data."""
        while True:
            for asset in OTC_ASSETS:
                try:
                    df = self.connector.get_candles(
                        asset=asset.replace(" (OTC)", " OTC"),
                        timeframe_seconds=60,
                        count=10,
                    )
                    if df is not None and len(df) >= 2:
                        current = float(df["close"].iloc[-1])
                        prev    = float(df["close"].iloc[-2])
                        change  = (current - prev) / (prev + 1e-9) * 100
                        self.gui.update_chart(asset, current, change)
                except Exception:
                    pass
            time.sleep(3)

    # ── AI trading loop ───────────────────────────────────────────────────────

    def _start_ai_loop(self):
        """Run in background thread when START AI is clicked."""
        if not self.agent or not self.brain or not self.connector:
            self.gui.set_status("❌ AI ยังโหลดไม่เสร็จ")
            return

        self._running = True
        self.gui.set_status("🤖 AI กำลังเทรด …")

        last_ppo_metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        from trading_ai.core.trading_env import TradingEnv
        env = TradingEnv(self.connector)
        obs, _ = env.reset()

        while self._running:
            indicator_vec = obs[:N_INDICATORS]

            # PPO decision
            ppo_action, log_prob, value = self.agent.select_action(obs)
            _, ppo_conf = self.agent.get_confidence(obs)

            # Brain refines the decision
            signal = self.brain.think(indicator_vec, ppo_action, ppo_conf)

            # Determine final action
            if signal.risk_multiplier < 0.2:
                final_action = 0
            elif signal.action != ppo_action and signal.confidence > ppo_conf + 0.15:
                final_action = signal.action
            else:
                final_action = ppo_action
                if signal.confidence < config.MIN_CONFIDENCE and final_action != 0:
                    final_action = 0

            # Update GUI signals for all charts
            for asset in OTC_ASSETS:
                self.gui.update_signal(
                    asset,
                    ACTION_NAMES[final_action],
                    signal.confidence,
                )

            # Execute trade in environment
            next_obs, reward, terminated, truncated, info = env.step(final_action)

            if not info.get("skipped", False):
                pnl = info.get("pnl", 0.0)
                self.brain.learn(pnl, final_action, indicator_vec, ppo_action)

                self._trade_stats["trades"] += 1
                if pnl > 0:
                    self._trade_stats["wins"] += 1
                self._trade_stats["pnl"] += pnl

                wr = self._trade_stats["wins"] / max(self._trade_stats["trades"], 1)
                self.gui.set_trade_stats(
                    self._trade_stats["trades"], wr, self._trade_stats["pnl"]
                )
                self.gui.set_status(
                    f"Trade: {ACTION_NAMES[final_action]}  |  "
                    f"PnL: {'+'if pnl>=0 else ''}{pnl:.2f}  |  "
                    f"WinRate: {wr:.1%}"
                )

            self.agent.store(obs, ppo_action, log_prob, reward, value,
                             terminated or truncated)

            if self.agent.ready_to_update():
                last_ppo_metrics = self.agent.update(next_obs)
                self.knowledge.save_brain(self.agent)
                stats = self.brain.get_status()
                self.gui.set_brain_status(stats["graph_nodes"],
                                          stats["recent_win_rate"])

            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()

        self.gui.set_status("AI หยุดทำงาน")

    def _stop_ai(self):
        self._running = False

    # ── Manual trade ──────────────────────────────────────────────────────────

    def _manual_buy(self, asset: str):
        if not self.connector:
            return
        clean_asset = asset.replace(" (OTC)", " OTC")
        ok, oid = self.connector.place_trade(
            asset=clean_asset,
            direction="call",
            amount=config.TRADE_AMOUNT,
            duration_minutes=config.TRADE_DURATION,
        )
        self.gui.set_status(
            f"Manual BUY {clean_asset} – {'✅' if ok else '❌'}"
        )

    def _manual_sell(self, asset: str):
        if not self.connector:
            return
        clean_asset = asset.replace(" (OTC)", " OTC")
        ok, oid = self.connector.place_trade(
            asset=clean_asset,
            direction="put",
            amount=config.TRADE_AMOUNT,
            duration_minutes=config.TRADE_DURATION,
        )
        self.gui.set_status(
            f"Manual SELL {clean_asset} – {'✅' if ok else '❌'}"
        )


if __name__ == "__main__":
    app = TradingApp()
    app.run()
