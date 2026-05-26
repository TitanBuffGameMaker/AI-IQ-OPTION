"""
IQ Option API connector.
Wraps iqoptionapi to provide clean methods for the trading agent.
"""
import sys
import time
import logging
import threading
import pandas as pd
from typing import Optional, Tuple
from iqoptionapi.stable_api import IQ_Option

logger = logging.getLogger(__name__)

# ── ป้องกันการ connect พร้อมกันหลาย thread ────────────────────────────────────
_connect_lock = threading.Lock()
_last_connect_time: float = 0.0
_MIN_CONNECT_INTERVAL = 10.0   # วินาที — ห้าม connect ถี่กว่านี้
_RATE_LIMIT_WAIT     = 310.0   # วินาที — รอ 5 นาที + buffer เมื่อถูก rate limit

# ── WS shared-buffer race-condition guard ─────────────────────────────────────
# iqoptionapi uses a single global candle buffer.  Two concurrent get_candles()
# calls (from _price_loop + _ai_loop) routinely return data for the wrong asset.
# Serialise every call behind this lock and add a post-request settle delay.
_candle_lock = threading.Lock()
_CANDLE_SETTLE_SECS = 0.7   # time to let iqoptionapi's WS buffer clear after each call

# IQ Option sometimes returns OTC assets under plain Forex names (e.g. "EURUSD"
# instead of "EURUSD-OTC").  These are still OTC instruments and always open.
_ALWAYS_OPEN_NAMES = {"USDAED", "EURUSD", "GBPUSD", "EURJPY", "USDCAD", "USDCHF"}


# ── Silence iqoptionapi internal noise ────────────────────────────────────────
# iqoptionapi 7.x raises KeyError('underlying') from __get_digital_open whenever
# the digital-options payload is incomplete (very common at session start and
# on weekends).  It's harmless but spams the console — swallow it via a
# threading.excepthook so other exceptions still surface.
_prev_excepthook = threading.excepthook

def _filter_iqapi_thread_errors(args: "threading.ExceptHookArgs") -> None:
    exc, val, tb, thread = args.exc_type, args.exc_value, args.exc_traceback, args.thread
    name = getattr(thread, "name", "")
    if exc is KeyError and "underlying" in str(val) and "digital" in name.lower():
        return
    if exc is KeyError and "underlying" in str(val):
        # Match by traceback function name as a fallback
        frame = tb
        while frame is not None:
            if "__get_digital_open" in (frame.tb_frame.f_code.co_name or ""):
                return
            frame = frame.tb_next
    _prev_excepthook(args)

threading.excepthook = _filter_iqapi_thread_errors

# Also silence noisy "logging" output from iqoptionapi's internal modules
for _noisy in (
    "iqoptionapi", "iqoptionapi.api", "iqoptionapi.stable_api",
    "iqoptionapi.ws.client", "iqoptionapi.ws.chanels.candles",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


class IQOptionConnector:
    """Manages connection and trading operations with IQ Option."""

    def __init__(self, email: str, password: str, account_type: str = "PRACTICE"):
        self.email        = email
        self.password     = password
        self.account_type = account_type
        self.api: Optional[IQ_Option] = None
        self._connected      = False
        self._ever_connected = False   # True หลัง connect สำเร็จครั้งแรก
        self._in_2fa         = False
        self._dead           = False   # True เมื่อ connect ล้มเหลว ห้าม auto-retry

    # ── Connection ─────────────────────────────────────────────────────────────

    def connect_with_ssid(self, ssid: str) -> Tuple[bool, str]:
        """
        Connect using SSID session token (from browser cookies).
        ข้ามขั้นตอน HTTP login ทั้งหมด — ไม่โดน rate limit.
        """
        try:
            logger.info("Connecting via SSID token…")
            self.api = IQ_Option(self.email or "user@example.com",
                                 self.password or "placeholder")

            # inject SSID ก่อน connect (iqoptionapi 7.x)
            _injected = False
            inner = getattr(self.api, 'api', None)
            if inner is not None:
                if hasattr(inner, 'token'):
                    inner.token = ssid
                    _injected = True
                elif hasattr(inner, 'ssid'):
                    inner.ssid = ssid
                    _injected = True

            if not _injected and hasattr(self.api, 'token'):
                self.api.token = ssid
                _injected = True

            if _injected:
                # เชื่อม WebSocket โดยตรง (ข้าม HTTP login)
                for _attr in ('start_ws_connect', 'connect_ws', 'connect_websocket'):
                    _fn = getattr(inner or self.api, _attr, None)
                    if callable(_fn):
                        _fn()
                        break

                for _ in range(10):
                    time.sleep(1)
                    try:
                        if self.api.check_connect():
                            self.api.change_balance(self.account_type)
                            self._connected = self._ever_connected = True
                            self._dead = self._in_2fa = False
                            logger.info("Connected via SSID (%s)", self.account_type)
                            return True, "ok"
                    except Exception:
                        pass

            # fallback: ลอง connect ปกติ (SSID inject ไม่ work)
            logger.warning("SSID inject ไม่ work — ลอง normal login…")
            return self.connect()

        except Exception as exc:
            logger.error("SSID connect error: %s", exc)
            return False, str(exc)

    def connect(self) -> Tuple[bool, str]:
        """
        Establish connection to IQ Option.
        Returns (True, "ok") | (False, "2FA") | (False, reason).
        Enforces minimum interval between attempts to avoid rate limiting.
        """
        global _last_connect_time

        with _connect_lock:
            # rate-limit guard
            elapsed = time.time() - _last_connect_time
            if elapsed < _MIN_CONNECT_INTERVAL:
                wait = _MIN_CONNECT_INTERVAL - elapsed
                logger.info("Connection cooldown — waiting %.1fs", wait)
                time.sleep(wait)

            try:
                self.api = IQ_Option(self.email, self.password)
                check, reason = self.api.connect()
                _last_connect_time = time.time()

                if not check:
                    reason_str = str(reason)
                    logger.error("IQ Option connection failed: %s", reason_str)

                    # Rate limit — หยุด retry
                    if "number of requests" in reason_str.lower() or "exceeded" in reason_str.lower():
                        logger.warning("Rate limited — set dead flag (no auto-retry)")
                        self._dead = True

                    return False, reason_str

                self.api.change_balance(self.account_type)
                self._connected      = True
                self._ever_connected = True
                self._in_2fa         = False
                self._dead           = False
                logger.info("Connected to IQ Option (%s account)", self.account_type)
                return True, "ok"

            except Exception as exc:
                _last_connect_time = time.time()
                logger.error("Connection error: %s", exc)
                self._dead = True   # หยุด auto-retry
                return False, str(exc)

    def submit_otp(self, otp: str) -> bool:
        """Submit 5-digit OTP for 2FA. Call after connect() returns (False, '2FA')."""
        try:
            sent = False

            # iqoptionapi 7.x — method อยู่บน IQ_Option โดยตรง
            if hasattr(self.api, 'send_sms_code'):
                self.api.send_sms_code(otp)
                sent = True

            # iqoptionapi 7.x — method อยู่บน internal .api object
            elif hasattr(self.api, 'api') and hasattr(self.api.api, 'send_sms_code'):
                self.api.api.send_sms_code(otp)
                sent = True

            # iqoptionapi บางเวอร์ชัน ใช้ resend_sms
            elif hasattr(self.api, 'resend_sms'):
                self.api.resend_sms(otp)
                sent = True

            if not sent:
                logger.error(
                    "iqoptionapi version ไม่รองรับ 2FA อัตโนมัติ — "
                    "กรุณาปิด 2FA ใน IQ Option settings แล้วลองใหม่"
                )
                return False

            # รอให้ WebSocket ยืนยัน (ไม่ connect() ซ้ำ)
            logger.info("OTP ส่งแล้ว รอยืนยัน…")
            for _ in range(15):
                time.sleep(1)
                try:
                    if self.api.check_connect():
                        self.api.change_balance(self.account_type)
                        self._connected = True
                        self._in_2fa    = False
                        logger.info("Connected via OTP (%s account)", self.account_type)
                        return True
                except Exception:
                    pass

            logger.error("OTP login timeout")
            return False

        except Exception as exc:
            logger.error("OTP error: %s", exc)
            return False

    def ensure_connected(self) -> bool:
        """Reconnect if session dropped. Only retries after a previous successful connection."""
        if self._in_2fa or self._dead:
            return False
        # ถ้ายังไม่เคย connect สำเร็จเลย ห้าม auto-retry (ต้องแก้ .env หรือรอ rate limit)
        if not self._ever_connected:
            return False
        if not self._connected or not self.api:
            ok, _ = self.connect()
            return ok
        try:
            if not self.api.check_connect():
                logger.warning("Connection lost – reconnecting…")
                ok, _ = self.connect()
                return ok
        except Exception:
            ok, _ = self.connect()
            return ok
        return True

    # ── Data ───────────────────────────────────────────────────────────────────

    def get_candles(self, asset: str, timeframe_seconds: int, count: int) -> Optional[pd.DataFrame]:
        if not self.ensure_connected():
            return None
        with _candle_lock:
            try:
                candles = self.api.get_candles(asset, timeframe_seconds, count, time.time())
                # Settle time: let the WS buffer clear before the next caller
                # acquires the lock and issues a request for a different asset.
                time.sleep(_CANDLE_SETTLE_SECS)
                if not candles:
                    return None
                df = pd.DataFrame(candles)
                df = df.rename(columns={"max": "high", "min": "low"})
                df = df[["open", "high", "low", "close", "volume"]].astype(float)
                return df.reset_index(drop=True)
            except Exception as exc:
                logger.error("get_candles error: %s", exc)
                return None

    def get_balance(self) -> float:
        if not self.ensure_connected():
            return 0.0
        try:
            return float(self.api.get_balance())
        except Exception:
            return 0.0

    def place_trade(self, asset: str, direction: str, amount: float, duration_minutes: int) -> Tuple[bool, Optional[int]]:
        if not self.ensure_connected():
            return False, None
        try:
            check, order_id = self.api.buy(amount, asset, direction, duration_minutes)
            if check:
                logger.info("Trade placed: %s %s $%.2f %dmin | id=%s",
                            direction.upper(), asset, amount, duration_minutes, order_id)
                return True, order_id
            logger.warning("Trade rejected by IQ Option: %s %s $%.2f %dmin (reason=%r)",
                           direction.upper(), asset, amount, duration_minutes, order_id)
            return False, None
        except Exception as exc:
            logger.error("place_trade error: %s", exc)
            return False, None

    def get_trade_result(
        self,
        order_id: int,
        timeout: int = 30,
        balance_before: Optional[float] = None,
    ) -> Optional[float]:
        """
        Fetch trade outcome with a hard timeout, plus a balance-delta fallback.

        iqoptionapi's check_win_v3 contains a `while True` busy-wait that blocks
        forever when the result never arrives (very common for OTC binaries on
        weekends).  We:
          1. Run check_win_v3 in a daemon thread with a hard timeout.
          2. If it returns numerically, use that.
          3. Otherwise, if `balance_before` is provided, compute the PnL from
             the current balance minus the pre-trade balance — this is the
             most reliable way to score OTC trades when the official API is
             silent.
        """
        if not self.ensure_connected():
            return None

        holder: dict = {"value": None}

        def _fetch():
            try:
                holder["value"] = self.api.check_win_v3(order_id)
            except Exception as exc:
                logger.debug("check_win_v3 error: %s", exc)

        thread = threading.Thread(target=_fetch, daemon=True, name=f"check_win_{order_id}")
        thread.start()
        thread.join(timeout=timeout)

        if not thread.is_alive():
            result = holder["value"]
            try:
                if result is not None:
                    return float(result)
            except (TypeError, ValueError):
                logger.debug("Trade %s non-numeric result: %r", order_id, result)
        else:
            logger.info("Trade %s: check_win_v3 timeout (%ds) — using balance delta",
                        order_id, timeout)

        # Fallback: balance delta
        if balance_before is not None:
            time.sleep(2.0)   # let server-side balance settle
            try:
                balance_after = float(self.api.get_balance())
                delta = balance_after - balance_before
                if abs(delta) >= 0.01:
                    logger.info("Trade %s: PnL from balance delta = %+.2f", order_id, delta)
                    return delta
                logger.warning("Trade %s: balance unchanged (delta=%.4f) — likely tied/refund",
                               order_id, delta)
                return 0.0   # tie/refund counts as 0, not a loss
            except Exception as exc:
                logger.warning("Trade %s: balance-delta fetch failed: %s", order_id, exc)

        return None

    def get_payout(self, asset: str) -> float:
        """
        Return payout % for an asset.  Tries turbo/binary/digital buckets;
        falls back to 80% for OTC assets (typical IQ Option OTC payout).
        """
        if not self.ensure_connected():
            return 80.0 if "OTC" in asset.upper() else 0.0
        try:
            all_profit = self.api.get_all_profit()
            if asset in all_profit:
                for bucket in ("turbo", "binary", "digital"):
                    val = all_profit[asset].get(bucket)
                    if isinstance(val, dict):
                        p = val.get("profit", 0)
                    else:
                        p = val
                    try:
                        p = float(p or 0)
                    except (TypeError, ValueError):
                        p = 0.0
                    if p > 0:
                        return p * 100
            if "OTC" in asset.upper():
                return 80.0
            return 0.0
        except Exception:
            return 80.0 if "OTC" in asset.upper() else 0.0

    def is_market_open(self, asset: str) -> bool:
        """
        Check if an asset is currently tradeable.

        OTC assets are always open on IQ Option (24/7, including weekends).
        For non-OTC assets, query the standard open-time table.
        """
        if "OTC" in asset.upper():
            return True
        if asset in _ALWAYS_OPEN_NAMES:
            return True   # resolved under plain Forex name but still an OTC instrument
        if not self.ensure_connected():
            return False
        try:
            all_open = self.api.get_all_open_time()
            for category in all_open.values():
                if asset in category:
                    return category[asset].get("open", False)
            return False
        except Exception:
            return False
