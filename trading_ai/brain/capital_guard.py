"""
CapitalGuard — เกราะป้องกันทุน สำหรับ REAL account เท่านั้น

PRACTICE account: ไม่มีการหยุดเลย — ยิ่งเทรดมาก ยิ่งเรียนรู้เร็ว
  - OTC assets เป็นตลาดสังเคราะห์ที่ IQ Option สร้างขึ้น ไม่ใช่ตลาดจริง
  - บัญชีทดลองไม่เสียเงินจริง → ให้ AI เทรดต่อเสมอ ไม่ว่าจะ loss มากแค่ไหน
  - เป้าหมาย: สะสม trade experience ให้มากที่สุดเพื่อเรียนรู้ OTC patterns

REAL account: เปิดใช้งาน 4 ระบบป้องกัน
1. Daily Loss Limit   — หยุดวันนั้นทันทีถ้าขาดทุน > 20% ของทุนต้นวัน
2. Session Profit Target — หยุดเก็บกำไร ไม่โลภ ถ้าถึง +10% ของทุนต้นวัน
3. Kelly Criterion sizing — ปรับขนาด bet ให้เหมาะสมกับ win rate จริง
4. Min Confidence (REAL) — threshold สูงกว่า PRACTICE เพื่อเทรดเฉพาะสัญญาณที่มั่นใจ

หมายเหตุ OTC: ราคา OTC ไม่ตอบสนองต่อข่าวจริง (Fed, ECB ฯลฯ)
  AI ต้องเรียนรู้ pattern เทคนิคของ OTC เอง ไม่ใช่จากข่าวพื้นฐาน
"""
import logging
import time
from typing import Tuple

logger = logging.getLogger(__name__)


class CapitalGuard:
    """
    เกราะป้องกันทุน — ใช้กับ REAL account เท่านั้น
    PRACTICE → ทุก check คืน False (ไม่มีการหยุด, ไม่มีการปรับ bet)
    """

    # ── ค่าตั้งต้น (ผู้ใช้เปลี่ยนได้ผ่าน UI) ─────────────────────────────────
    DAILY_LOSS_LIMIT_PCT  = 0.20   # หยุดถ้าขาดทุน > 20% ของทุนต้นวัน
    PROFIT_TARGET_PCT     = 0.10   # หยุดถ้ากำไร > 10% ของทุนต้นวัน
    KELLY_PAYOUT          = 0.85   # IQ Option จ่าย ~85% เมื่อชนะ
    KELLY_FRACTION        = 0.25   # ใช้ 25% ของ Full Kelly (safety margin)

    # ── Min confidence ตาม account + trade count ─────────────────────────────
    #  PRACTICE: threshold ต่ำ — ต้องการ trade volume สูงเพื่อเรียนรู้ OTC patterns
    #            บัญชีทดลองไม่เสียเงินจริง ยิ่งเทรดมาก ยิ่งเรียนรู้เร็ว
    #  REAL:     threshold สูง — ป้องกันเงินจริง เทรดเฉพาะสัญญาณที่มั่นใจ
    _CONF_PRACTICE = [(0, 0.33), (50, 0.38), (200, 0.43), (500, 0.48)]
    _CONF_REAL     = [(0, 0.58), (10, 0.62), (30, 0.65), (100, 0.68)]

    def __init__(self, account_type: str = "PRACTICE"):
        self.account_type = (account_type or "PRACTICE").upper()

        # ── State ──────────────────────────────────────────────────────────
        self._day_date:          str   = ""       # "YYYY-MM-DD"
        self._day_start_balance: float = 0.0      # ยอด balance ต้นวัน
        self._session_pnl:       float = 0.0      # P&L รวมของวันนี้
        self._stopped:           bool  = False     # เซสชันหยุดแล้ว?
        self._stop_reason:       str   = ""
        self._last_kelly_amount: float = 0.0      # bet ล่าสุดที่ Kelly แนะนำ

    # ── Account switching ─────────────────────────────────────────────────────

    def set_account_type(self, account_type: str) -> None:
        new = (account_type or "PRACTICE").upper()
        if new != self.account_type:
            logger.info("CapitalGuard: account %s → %s", self.account_type, new)
            self.account_type = new
            if new == "PRACTICE":
                self._stopped = False   # PRACTICE ไม่หยุดเลย

    # ── Balance / PnL tracking ────────────────────────────────────────────────

    def update_day_start(self, balance: float) -> None:
        """
        เรียกตอนเริ่ม session และหลัง trade แต่ละไม้เพื่อ reset วันใหม่
        ถ้าวันเปลี่ยน → reset ทุกตัวนับ (new trading day)
        """
        today = time.strftime("%Y-%m-%d")
        if self._day_date != today:
            logger.info(
                "CapitalGuard: new day %s | start_balance=%.2f (prev pnl=%+.2f)",
                today, balance, self._session_pnl,
            )
            self._day_date          = today
            self._day_start_balance = balance
            self._session_pnl       = 0.0
            self._stopped           = False
            self._stop_reason       = ""
        elif self._day_start_balance <= 0:
            self._day_start_balance = balance

    def record_trade_pnl(self, pnl: float) -> None:
        """บันทึก P&L หลังทุก trade (REAL เท่านั้น)"""
        if self.account_type != "REAL":
            return
        self._session_pnl += pnl

    # ── Core decision ─────────────────────────────────────────────────────────

    def should_stop(self) -> Tuple[bool, str]:
        """
        ตรวจว่าควรหยุดเทรดวันนี้หรือไม่
        PRACTICE → (False, "")
        REAL → (True, reason) ถ้าถึง limit/target
        """
        if self.account_type != "REAL":
            return False, ""

        if self._stopped:
            return True, self._stop_reason

        if self._day_start_balance <= 0:
            return False, ""

        # ── Daily loss limit ───────────────────────────────────────────────
        if self._session_pnl < 0:
            loss_pct = abs(self._session_pnl) / self._day_start_balance
            if loss_pct >= self.DAILY_LOSS_LIMIT_PCT:
                reason = (
                    f"⛔ Daily loss limit: ขาดทุน {loss_pct:.1%} "
                    f"(ขีดจำกัด {self.DAILY_LOSS_LIMIT_PCT:.0%}) — หยุดวันนี้"
                )
                self._stopped     = True
                self._stop_reason = reason
                logger.warning("CapitalGuard: %s", reason)
                return True, reason

        # ── Daily profit target ────────────────────────────────────────────
        if self._session_pnl > 0:
            profit_pct = self._session_pnl / self._day_start_balance
            if profit_pct >= self.PROFIT_TARGET_PCT:
                reason = (
                    f"✅ Profit target reached: กำไร {profit_pct:.1%} "
                    f"(เป้า {self.PROFIT_TARGET_PCT:.0%}) — เก็บกำไรวันนี้"
                )
                self._stopped     = True
                self._stop_reason = reason
                logger.info("CapitalGuard: %s", reason)
                return True, reason

        return False, ""

    # ── Kelly Criterion bet sizing ────────────────────────────────────────────

    def kelly_amount(
        self,
        base_amount: float,
        win_rate:    float,
        payout:      float = None,
    ) -> float:
        """
        คำนวณขนาด bet ที่เหมาะสมตาม Kelly Criterion
        ใช้ Quarter Kelly (25%) เพื่อความปลอดภัย

        สูตร Kelly: f* = (p × b − q) / b
          p = win_rate, q = 1 − p, b = payout (e.g. 0.85)

        คืนค่า min(kelly_amount, base_amount) เสมอ
        (ไม่เกินที่ผู้ใช้ตั้งไว้, ไม่ต่ำกว่า $1)

        PRACTICE → คืน base_amount โดยไม่ปรับ
        """
        if self.account_type != "REAL":
            return base_amount

        pay = payout if payout is not None else self.KELLY_PAYOUT
        p   = float(win_rate)
        q   = 1.0 - p

        if p <= 0.0 or p >= 1.0 or pay <= 0:
            return base_amount

        # Kelly fraction
        kelly_f = (p * pay - q) / pay

        if kelly_f <= 0:
            # Negative edge → bet minimum ($1 or 10% of base as floor)
            amount = max(1.0, base_amount * 0.10)
            self._last_kelly_amount = round(amount, 2)
            return self._last_kelly_amount

        # Quarter Kelly
        safe_f = kelly_f * self.KELLY_FRACTION

        # Apply to balance
        balance = max(self._day_start_balance, base_amount * 10)
        kelly_amt = balance * safe_f

        # Cap at user's base amount, floor at $1
        amount = max(1.0, min(kelly_amt, base_amount))
        self._last_kelly_amount = round(amount, 2)
        return self._last_kelly_amount

    # ── Confidence threshold ──────────────────────────────────────────────────

    def min_confidence(self, trades_done: int) -> float:
        """
        Min confidence threshold ตาม account type และจำนวน trades ที่ผ่านมา

        PRACTICE: เริ่ม 0.30 → เพิ่มขึ้นตามประสบการณ์
        REAL:     เริ่ม 0.55 → เพิ่มขึ้นตามประสบการณ์ (สูงกว่า PRACTICE)
        """
        table = (
            self._CONF_REAL if self.account_type == "REAL"
            else self._CONF_PRACTICE
        )
        threshold = table[0][1]
        for min_trades, conf in table:
            if trades_done >= min_trades:
                threshold = conf
        return threshold

    # ── Status for UI ─────────────────────────────────────────────────────────

    def status(self) -> dict:
        bal = max(self._day_start_balance, 1.0)
        loss_pct   = abs(self._session_pnl) / bal if self._session_pnl < 0 else 0.0
        profit_pct = self._session_pnl / bal       if self._session_pnl > 0 else 0.0
        return {
            "account_type":       self.account_type,
            "day_start_balance":  round(self._day_start_balance, 2),
            "session_pnl":        round(self._session_pnl, 2),
            "loss_pct":           round(loss_pct, 4),
            "profit_pct":         round(profit_pct, 4),
            "daily_loss_limit":   self.DAILY_LOSS_LIMIT_PCT,
            "profit_target":      self.PROFIT_TARGET_PCT,
            "stopped":            self._stopped,
            "stop_reason":        self._stop_reason,
            "kelly_amount":       round(self._last_kelly_amount, 2),
            "day_date":           self._day_date,
        }
