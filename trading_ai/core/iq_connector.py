"""
IQ Option API connector.
Wraps iqoptionapi to provide clean methods for the trading agent.
"""
import time
import logging
import pandas as pd
from typing import Optional, Tuple
from iqoptionapi.stable_api import IQ_Option

logger = logging.getLogger(__name__)


class IQOptionConnector:
    """Manages connection and trading operations with IQ Option."""

    def __init__(self, email: str, password: str, account_type: str = "PRACTICE"):
        self.email = email
        self.password = password
        self.account_type = account_type  # "PRACTICE" or "REAL"
        self.api: Optional[IQ_Option] = None
        self._connected = False

    def connect(self) -> bool:
        """Establish connection to IQ Option. Returns True on success."""
        try:
            self.api = IQ_Option(self.email, self.password)
            check, reason = self.api.connect()
            if not check:
                logger.error("IQ Option connection failed: %s", reason)
                return False

            self.api.change_balance(self.account_type)
            self._connected = True
            logger.info("Connected to IQ Option (%s account)", self.account_type)
            return True
        except Exception as exc:
            logger.error("Connection error: %s", exc)
            return False

    def ensure_connected(self) -> bool:
        """Reconnect if session dropped."""
        if not self._connected or not self.api:
            return self.connect()
        try:
            if not self.api.check_connect():
                logger.warning("Connection lost – reconnecting …")
                return self.connect()
        except Exception:
            return self.connect()
        return True

    def get_candles(
        self,
        asset: str,
        timeframe_seconds: int,
        count: int,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch the last `count` candles for `asset`.
        Returns DataFrame[open, high, low, close, volume] sorted oldest→newest,
        or None on failure.
        """
        if not self.ensure_connected():
            return None
        try:
            candles = self.api.get_candles(asset, timeframe_seconds, count, time.time())
            if not candles:
                return None
            df = pd.DataFrame(candles)
            df = df.rename(columns={
                "open": "open",
                "max": "high",
                "min": "low",
                "close": "close",
                "volume": "volume",
            })
            df = df[["open", "high", "low", "close", "volume"]].astype(float)
            return df.reset_index(drop=True)
        except Exception as exc:
            logger.error("get_candles error: %s", exc)
            return None

    def get_balance(self) -> float:
        """Return current account balance."""
        if not self.ensure_connected():
            return 0.0
        try:
            return float(self.api.get_balance())
        except Exception:
            return 0.0

    def place_trade(
        self,
        asset: str,
        direction: str,   # "call" (BUY) or "put" (SELL)
        amount: float,
        duration_minutes: int,
    ) -> Tuple[bool, Optional[int]]:
        """
        Open a binary option trade.
        Returns (success, order_id).
        """
        if not self.ensure_connected():
            return False, None
        try:
            check, order_id = self.api.buy(amount, asset, direction, duration_minutes)
            if check:
                logger.info(
                    "Trade placed: %s %s $%.2f %dmin | id=%s",
                    direction.upper(), asset, amount, duration_minutes, order_id,
                )
                return True, order_id
            logger.warning("Trade rejected by IQ Option")
            return False, None
        except Exception as exc:
            logger.error("place_trade error: %s", exc)
            return False, None

    def get_trade_result(self, order_id: int, timeout: int = 120) -> Optional[float]:
        """
        Wait for a trade to close and return the profit/loss in USD.
        Returns None if the result cannot be retrieved.
        """
        if not self.ensure_connected():
            return None
        try:
            result = self.api.check_win_v3(order_id)
            deadline = time.time() + timeout
            while result is None and time.time() < deadline:
                time.sleep(1)
                result = self.api.check_win_v3(order_id)
            return float(result) if result is not None else None
        except Exception as exc:
            logger.error("get_trade_result error: %s", exc)
            return None

    def get_payout(self, asset: str) -> float:
        """Return current payout percentage for the asset (0‒100)."""
        if not self.ensure_connected():
            return 0.0
        try:
            all_profit = self.api.get_all_profit()
            if asset in all_profit:
                return float(all_profit[asset].get("turbo", {}).get("profit", 0)) * 100
            return 0.0
        except Exception:
            return 0.0

    def is_market_open(self, asset: str) -> bool:
        """Check if the asset is currently tradeable."""
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
