"""
IQ Option AI Trading Control Panel

หน้าต่างควบคุม:
  - แสดง 4 กราฟ OTC พร้อมสัญญาณ AI แบบ real-time
  - 8 ปุ่ม (4 ขึ้น + 4 ลง) สำหรับแต่ละหุ้น
  - แถบสถานะ brain / PPO / connection
  - Startup verification ก่อนเริ่ม

ใช้ tkinter (built-in Python) ไม่ต้องติดตั้งเพิ่ม
"""
import logging
import queue
import threading
import time
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, ttk
from typing import Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── สีธีม (เลียนแบบ IQ Option) ────────────────────────────────────────────
BG_DARK     = "#1a1d2e"
BG_PANEL    = "#252840"
BG_CHART    = "#1c1f35"
GREEN       = "#26a69a"
RED         = "#ef5350"
BLUE        = "#2962ff"
GOLD        = "#ffc107"
TEXT_WHITE  = "#e8eaf6"
TEXT_GRAY   = "#9fa8da"
TEXT_DIM    = "#5c6bc0"

OTC_ASSETS = ["EUR/USD (OTC)", "GBP/USD (OTC)", "AUD/USD (OTC)", "GBP/JPY (OTC)"]

POSITIONS = [
    ("top_left",     0, 0),
    ("top_right",    0, 1),
    ("bottom_left",  1, 0),
    ("bottom_right", 1, 1),
]


class ChartPanel(tk.Frame):
    """
    ช่องแสดงข้อมูลสำหรับหุ้น 1 ตัว:
      - ชื่อหุ้น + สถานะ
      - ราคาปัจจุบัน
      - สัญญาณ AI (BUY / SELL / HOLD) + ความเชื่อมั่น
      - ปุ่ม BUY ↑ และ SELL ↓
      - Mini chart (candlestick bars)
    """

    BAR_COUNT = 30   # จำนวนแท่งเทียนที่แสดง

    def __init__(self, parent, asset: str, position: str,
                 on_buy: Callable, on_sell: Callable):
        super().__init__(parent, bg=BG_CHART,
                         relief="flat", bd=0,
                         highlightthickness=1,
                         highlightbackground=TEXT_DIM)
        self.asset = asset
        self.position = position
        self._on_buy = on_buy
        self._on_sell = on_sell
        self._price_history: List[float] = []
        self._last_action = "HOLD"
        self._confidence = 0.0
        self._build_ui()

    def _build_ui(self):
        # ── Header ────────────────────────────────────────────────────────
        header = tk.Frame(self, bg=BG_PANEL, pady=4)
        header.pack(fill="x")

        # OTC badge
        tk.Label(header, text="OTC", bg="#ff6d00", fg="white",
                 font=("Consolas", 7, "bold"),
                 padx=4, pady=1).pack(side="left", padx=(8, 0))

        # Asset name
        self.lbl_asset = tk.Label(
            header, text=self.asset.replace(" (OTC)", ""),
            bg=BG_PANEL, fg=TEXT_WHITE,
            font=("Consolas", 10, "bold"),
        )
        self.lbl_asset.pack(side="left", padx=6)

        # Signal indicator
        self.lbl_signal = tk.Label(
            header, text="── HOLD ──",
            bg=BG_PANEL, fg=TEXT_DIM,
            font=("Consolas", 8, "bold"),
        )
        self.lbl_signal.pack(side="right", padx=8)

        # ── Price display ──────────────────────────────────────────────────
        price_frame = tk.Frame(self, bg=BG_CHART)
        price_frame.pack(fill="x", padx=8, pady=(4, 0))

        self.lbl_price = tk.Label(
            price_frame, text="─.─────",
            bg=BG_CHART, fg=TEXT_WHITE,
            font=("Consolas", 16, "bold"),
        )
        self.lbl_price.pack(side="left")

        self.lbl_change = tk.Label(
            price_frame, text="+0.00%",
            bg=BG_CHART, fg=GREEN,
            font=("Consolas", 9),
        )
        self.lbl_change.pack(side="left", padx=8)

        # ── Mini chart canvas ──────────────────────────────────────────────
        self.canvas = tk.Canvas(
            self, bg=BG_CHART, height=90,
            highlightthickness=0,
        )
        self.canvas.pack(fill="x", padx=4, pady=4)

        # ── Confidence bar ────────────────────────────────────────────────
        conf_frame = tk.Frame(self, bg=BG_CHART)
        conf_frame.pack(fill="x", padx=8, pady=(0, 4))

        tk.Label(conf_frame, text="AI confidence",
                 bg=BG_CHART, fg=TEXT_DIM,
                 font=("Consolas", 7)).pack(side="left")

        self.conf_bar_var = tk.DoubleVar(value=0.0)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Conf.Horizontal.TProgressbar",
                        troughcolor=BG_PANEL,
                        background=BLUE, thickness=6)

        self.conf_bar = ttk.Progressbar(
            conf_frame, variable=self.conf_bar_var,
            maximum=100, length=120,
            style="Conf.Horizontal.TProgressbar",
        )
        self.conf_bar.pack(side="left", padx=8)

        self.lbl_conf_pct = tk.Label(
            conf_frame, text="0%",
            bg=BG_CHART, fg=TEXT_GRAY,
            font=("Consolas", 7),
        )
        self.lbl_conf_pct.pack(side="left")

        # ── Buttons ───────────────────────────────────────────────────────
        btn_frame = tk.Frame(self, bg=BG_CHART)
        btn_frame.pack(fill="x", padx=6, pady=(0, 6))

        btn_cfg = dict(
            font=("Consolas", 9, "bold"),
            relief="flat", bd=0,
            cursor="hand2", pady=6,
        )

        self.btn_buy = tk.Button(
            btn_frame,
            text="▲  BUY  ขึ้น",
            bg=GREEN, fg="white",
            activebackground="#1de9b6",
            activeforeground="white",
            command=lambda: self._on_buy(self.asset),
            **btn_cfg,
        )
        self.btn_buy.pack(side="left", fill="x", expand=True, padx=(0, 3))

        self.btn_sell = tk.Button(
            btn_frame,
            text="▼  SELL  ลง",
            bg=RED, fg="white",
            activebackground="#ff8a80",
            activeforeground="white",
            command=lambda: self._on_sell(self.asset),
            **btn_cfg,
        )
        self.btn_sell.pack(side="left", fill="x", expand=True, padx=(3, 0))

    # ── Update methods (called from the data thread) ────────────────────────

    def update_price(self, price: float, change_pct: float):
        self.lbl_price.config(text=f"{price:.5f}")
        color = GREEN if change_pct >= 0 else RED
        sign  = "+" if change_pct >= 0 else ""
        self.lbl_change.config(
            text=f"{sign}{change_pct:.2f}%", fg=color
        )
        self._price_history.append(price)
        if len(self._price_history) > self.BAR_COUNT:
            self._price_history = self._price_history[-self.BAR_COUNT:]
        self._draw_mini_chart()

    def update_signal(self, action: str, confidence: float):
        self._last_action = action
        self._confidence = confidence

        if action == "BUY":
            self.lbl_signal.config(text=f"▲ BUY  {confidence:.0%}", fg=GREEN)
            self.btn_buy.config(bg="#1de9b6")
            self.btn_sell.config(bg=RED)
            style_bg = "#1de9b6"
            self.conf_bar.configure(style="Buy.Horizontal.TProgressbar")
            ttk.Style().configure("Buy.Horizontal.TProgressbar",
                                  background=GREEN)
        elif action == "SELL":
            self.lbl_signal.config(text=f"▼ SELL {confidence:.0%}", fg=RED)
            self.btn_buy.config(bg=GREEN)
            self.btn_sell.config(bg="#ff8a80")
            ttk.Style().configure("Conf.Horizontal.TProgressbar",
                                  background=RED)
        else:
            self.lbl_signal.config(text="── HOLD ──", fg=TEXT_DIM)
            self.btn_buy.config(bg=GREEN)
            self.btn_sell.config(bg=RED)
            ttk.Style().configure("Conf.Horizontal.TProgressbar",
                                  background=TEXT_DIM)

        self.conf_bar_var.set(confidence * 100)
        self.lbl_conf_pct.config(text=f"{confidence:.0%}")

    def set_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        self.btn_buy.config(state=state)
        self.btn_sell.config(state=state)

    def _draw_mini_chart(self):
        """Draw simple bar chart from price history."""
        self.canvas.delete("all")
        prices = self._price_history
        if len(prices) < 2:
            return

        cw = self.canvas.winfo_width() or 300
        ch = self.canvas.winfo_height() or 90

        lo = min(prices)
        hi = max(prices)
        span = hi - lo or 1e-8

        bar_w = max(3, cw // max(len(prices), 1) - 1)

        for i, p in enumerate(prices):
            x1 = i * (bar_w + 1)
            x2 = x1 + bar_w
            y  = ch - int((p - lo) / span * (ch - 4)) - 2

            prev = prices[i - 1] if i > 0 else p
            color = GREEN if p >= prev else RED

            # Body bar
            self.canvas.create_rectangle(
                x1, min(y, ch - 2), x2, ch - 2,
                fill=color, outline="",
            )

        # Current price line
        last_y = ch - int((prices[-1] - lo) / span * (ch - 4)) - 2
        self.canvas.create_line(0, last_y, cw, last_y,
                                fill=GOLD, width=1, dash=(4, 3))


class ControlPanel(tk.Tk):
    """
    Main trading control panel window.
    """

    TITLE = "IQ Option AI Trading System"
    REFRESH_MS = 2000   # price update interval

    def __init__(
        self,
        on_manual_buy:  Optional[Callable] = None,
        on_manual_sell: Optional[Callable] = None,
        on_start_ai:    Optional[Callable] = None,
        on_stop_ai:     Optional[Callable] = None,
    ):
        super().__init__()
        self._on_manual_buy  = on_manual_buy  or (lambda a: None)
        self._on_manual_sell = on_manual_sell or (lambda a: None)
        self._on_start_ai    = on_start_ai    or (lambda: None)
        self._on_stop_ai     = on_stop_ai     or (lambda: None)

        self._chart_panels: Dict[str, ChartPanel] = {}
        self._update_queue: queue.Queue = queue.Queue()
        self._ai_running = False

        self._build_window()
        self._build_ui()
        self._schedule_queue_drain()

    # ── Window setup ──────────────────────────────────────────────────────────

    def _build_window(self):
        self.title(self.TITLE)
        self.configure(bg=BG_DARK)
        self.geometry("1100x720")
        self.minsize(900, 600)
        self.resizable(True, True)
        try:
            self.state("zoomed")   # start maximised on Windows
        except Exception:
            pass

    def _build_ui(self):
        # ── Top bar ───────────────────────────────────────────────────────
        top_bar = tk.Frame(self, bg=BG_PANEL, pady=6)
        top_bar.pack(fill="x")

        # Logo
        tk.Label(top_bar, text=" 🤖 IQ Option AI ", bg=BG_PANEL,
                 fg=TEXT_WHITE, font=("Consolas", 12, "bold")).pack(side="left", padx=10)

        # Status labels
        self.lbl_connection = tk.Label(
            top_bar, text="⬤ Connecting …",
            bg=BG_PANEL, fg=GOLD,
            font=("Consolas", 8),
        )
        self.lbl_connection.pack(side="left", padx=16)

        self.lbl_brain = tk.Label(
            top_bar, text="🧠 Brain: loading …",
            bg=BG_PANEL, fg=TEXT_GRAY,
            font=("Consolas", 8),
        )
        self.lbl_brain.pack(side="left", padx=8)

        self.lbl_checks = tk.Label(
            top_bar, text="🔍 Checking settings …",
            bg=BG_PANEL, fg=GOLD,
            font=("Consolas", 8),
        )
        self.lbl_checks.pack(side="left", padx=8)

        # AI toggle button (top-right)
        self.btn_ai = tk.Button(
            top_bar,
            text="▶  START AI",
            bg=GREEN, fg="white",
            font=("Consolas", 9, "bold"),
            relief="flat", bd=0,
            padx=12, pady=4,
            cursor="hand2",
            command=self._toggle_ai,
        )
        self.btn_ai.pack(side="right", padx=10)

        # Balance display
        self.lbl_balance = tk.Label(
            top_bar, text="$ ─,─── . ──",
            bg=BG_PANEL, fg=GOLD,
            font=("Consolas", 10, "bold"),
        )
        self.lbl_balance.pack(side="right", padx=16)

        # Settings info: timeframe + duration
        self.lbl_settings = tk.Label(
            top_bar, text="TF: ─── | DUR: ─── ",
            bg=BG_PANEL, fg=TEXT_DIM,
            font=("Consolas", 8),
        )
        self.lbl_settings.pack(side="right", padx=8)

        # ── Startup check banner ──────────────────────────────────────────
        self.check_banner = tk.Frame(self, bg="#1a1a2e", pady=3)
        self.check_banner.pack(fill="x")

        self.lbl_banner = tk.Label(
            self.check_banner,
            text="  🔍 กำลังตรวจสอบการตั้งค่า IQ Option …",
            bg="#1a1a2e", fg=GOLD,
            font=("Consolas", 9),
            anchor="w",
        )
        self.lbl_banner.pack(fill="x", padx=10)

        # ── 4-chart grid ──────────────────────────────────────────────────
        charts_frame = tk.Frame(self, bg=BG_DARK)
        charts_frame.pack(fill="both", expand=True, padx=6, pady=4)

        charts_frame.columnconfigure(0, weight=1)
        charts_frame.columnconfigure(1, weight=1)
        charts_frame.rowconfigure(0, weight=1)
        charts_frame.rowconfigure(1, weight=1)

        for asset, (pos, row, col) in zip(OTC_ASSETS, POSITIONS):
            panel = ChartPanel(
                charts_frame, asset, pos,
                on_buy=self._manual_buy,
                on_sell=self._manual_sell,
            )
            panel.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
            self._chart_panels[asset] = panel
            panel.set_enabled(False)  # disabled until checks pass

        # ── Bottom status bar ─────────────────────────────────────────────
        status_bar = tk.Frame(self, bg=BG_PANEL, pady=3)
        status_bar.pack(fill="x", side="bottom")

        self.lbl_status = tk.Label(
            status_bar, text="รอการเชื่อมต่อ …",
            bg=BG_PANEL, fg=TEXT_DIM,
            font=("Consolas", 8), anchor="w",
        )
        self.lbl_status.pack(side="left", padx=10)

        self.lbl_stats = tk.Label(
            status_bar, text="trades: 0  |  win: ─%  |  pnl: $0.00",
            bg=BG_PANEL, fg=TEXT_GRAY,
            font=("Consolas", 8),
        )
        self.lbl_stats.pack(side="right", padx=10)

    # ── Public update API (thread-safe) ──────────────────────────────────────

    def set_connection(self, connected: bool, account: str = ""):
        self._queue(lambda: self.lbl_connection.config(
            text=f"⬤ {'Connected' if connected else 'Disconnected'} {account}",
            fg=GREEN if connected else RED,
        ))

    def set_balance(self, balance: float):
        self._queue(lambda: self.lbl_balance.config(
            text=f"${balance:,.2f}",
        ))

    def set_brain_status(self, nodes: int, win_rate: float):
        self._queue(lambda: self.lbl_brain.config(
            text=f"🧠 Brain: {nodes} nodes | wr {win_rate:.0%}",
            fg=GREEN if win_rate > 0.55 else TEXT_GRAY,
        ))

    def show_check_result(self, all_passed: bool, summary: str):
        def _update():
            if all_passed:
                self.lbl_banner.config(
                    text=f"  ✅ ผ่านการตรวจสอบทั้งหมด – AI พร้อมเทรด",
                    fg=GREEN, bg="#0a1f0a",
                )
                self.check_banner.config(bg="#0a1f0a")
                for panel in self._chart_panels.values():
                    panel.set_enabled(True)
            else:
                self.lbl_banner.config(
                    text=f"  ❌ {summary}",
                    fg=RED, bg="#1f0a0a",
                )
                self.check_banner.config(bg="#1f0a0a")
        self._queue(_update)

    def show_check_items(self, items: List[Dict]):
        """Show individual check results in a popup."""
        def _popup():
            win = tk.Toplevel(self)
            win.title("ผลการตรวจสอบการตั้งค่า")
            win.configure(bg=BG_DARK)
            win.geometry("520x340")
            win.resizable(False, False)
            win.grab_set()

            tk.Label(win, text="🔍 ผลการตรวจสอบการตั้งค่า IQ Option",
                     bg=BG_DARK, fg=TEXT_WHITE,
                     font=("Consolas", 11, "bold")).pack(pady=12)

            for item in items:
                row = tk.Frame(win, bg=BG_PANEL, pady=6, padx=12,
                               relief="flat")
                row.pack(fill="x", padx=16, pady=3)

                icon  = "✅" if item["passed"] else "❌"
                color = GREEN if item["passed"] else RED

                tk.Label(row, text=icon, bg=BG_PANEL, fg=color,
                         font=("Consolas", 11)).pack(side="left")
                tk.Label(row, text=f"  {item['name']}",
                         bg=BG_PANEL, fg=TEXT_WHITE,
                         font=("Consolas", 9)).pack(side="left")
                if not item["passed"]:
                    tk.Label(row, text=f"→ {item.get('message', '')}",
                             bg=BG_PANEL, fg=GOLD,
                             font=("Consolas", 8),
                             wraplength=280, justify="left").pack(
                                 side="left", padx=8)

            all_ok = all(i["passed"] for i in items)
            tk.Button(
                win,
                text="ตกลง" if all_ok else "แก้ไขการตั้งค่าแล้วลองใหม่",
                bg=GREEN if all_ok else GOLD, fg="white",
                font=("Consolas", 9, "bold"),
                relief="flat", pady=6, padx=16,
                command=win.destroy,
            ).pack(pady=12)

        self._queue(_popup)

    def update_chart(self, asset: str, price: float, change_pct: float):
        panel = self._chart_panels.get(asset)
        if panel:
            self._queue(lambda p=panel, pr=price, ch=change_pct:
                        p.update_price(pr, ch))

    def update_signal(self, asset: str, action: str, confidence: float):
        panel = self._chart_panels.get(asset)
        if panel:
            self._queue(lambda p=panel, a=action, c=confidence:
                        p.update_signal(a, c))

    def set_status(self, text: str):
        self._queue(lambda: self.lbl_status.config(text=f"  {text}"))

    def set_trade_stats(self, trades: int, win_rate: float, pnl: float):
        color = GREEN if pnl >= 0 else RED
        self._queue(lambda: self.lbl_stats.config(
            text=f"trades: {trades}  |  win: {win_rate:.0%}  |  pnl: ${pnl:+.2f}",
            fg=color,
        ))

    def set_settings_display(self, timeframe: str, duration: str):
        self._queue(lambda: self.lbl_settings.config(
            text=f"TF: {timeframe} | DUR: {duration}  ",
        ))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _toggle_ai(self):
        if self._ai_running:
            self._ai_running = False
            self.btn_ai.config(text="▶  START AI", bg=GREEN)
            self.set_status("AI หยุดทำงาน")
            self._on_stop_ai()
        else:
            self._ai_running = True
            self.btn_ai.config(text="⏹  STOP AI", bg=RED)
            self.set_status("AI กำลังทำงาน …")
            threading.Thread(target=self._on_start_ai, daemon=True).start()

    def _manual_buy(self, asset: str):
        self.set_status(f"Manual BUY: {asset}")
        threading.Thread(
            target=self._on_manual_buy, args=(asset,), daemon=True
        ).start()

    def _manual_sell(self, asset: str):
        self.set_status(f"Manual SELL: {asset}")
        threading.Thread(
            target=self._on_manual_sell, args=(asset,), daemon=True
        ).start()

    def _queue(self, fn: Callable):
        self._update_queue.put(fn)

    def _schedule_queue_drain(self):
        """Drain the update queue in the main thread (thread-safe UI updates)."""
        try:
            while not self._update_queue.empty():
                fn = self._update_queue.get_nowait()
                fn()
        except Exception:
            pass
        self.after(50, self._schedule_queue_drain)  # drain every 50ms
