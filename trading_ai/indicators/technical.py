"""
Technical Indicator Engine — ULTRA EDITION (40 indicators)

เพิ่มจาก 18 เป็น 40 indicators รวม:
  - Candlestick Patterns: Doji, Hammer, Engulfing, Morning/Evening Star
  - Ichimoku Cloud: Tenkan, Kijun, Cloud Position
  - Parabolic SAR
  - Fibonacci Retracement Position
  - Support/Resistance Proximity
  - Market Regime Detection
  - RSI Divergence
  - Price Action: HH/LL
  - Volume Acceleration
  - MACD Momentum
  - Volatility Regime
"""
import numpy as np
import pandas as pd
from typing import Optional

N_FEATURES = 40   # จำนวน features ทั้งหมด (sync กับ trading_env.py)


class IndicatorEngine:
    """Compute และ normalize 40 technical indicators"""

    RSI_PERIOD     = 14
    BB_PERIOD      = 20
    BB_STD         = 2
    EMA_PERIODS    = [9, 21, 50, 200]
    MACD_FAST      = 12
    MACD_SLOW      = 26
    MACD_SIGNAL    = 9
    STOCH_K        = 14
    STOCH_D        = 3
    ATR_PERIOD     = 14
    ADX_PERIOD     = 14
    CCI_PERIOD     = 20
    WR_PERIOD      = 14
    MOM_PERIOD     = 10
    OBV_WINDOW     = 50
    ICHIMOKU_CONV  = 9
    ICHIMOKU_BASE  = 26
    ICHIMOKU_SPAN  = 52

    def compute(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Args:
            df: DataFrame [open, high, low, close, volume] oldest→newest, ≥60 rows
        Returns:
            float32 array shape (N_FEATURES,), or None if not enough data
        """
        if len(df) < 60:
            return None

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        open_  = df["open"]
        volume = df.get("volume", pd.Series(np.ones(len(df)), index=df.index))
        price  = float(close.iloc[-1]) or 1.0

        features = []

        # ── 1. RSI ──────────────────────────────────────────────────────────
        rsi = self._rsi(close, self.RSI_PERIOD)
        features.append(self._norm(rsi, 0, 100))

        # ── 2-4. MACD ────────────────────────────────────────────────────────
        macd_line, sig_line, histogram = self._macd(
            close, self.MACD_FAST, self.MACD_SLOW, self.MACD_SIGNAL
        )
        features.append(np.clip(macd_line / price, -0.05, 0.05) / 0.05)
        features.append(np.clip(sig_line / price, -0.05, 0.05) / 0.05)
        features.append(np.clip(histogram / price, -0.05, 0.05) / 0.05)

        # ── 5-6. Bollinger Bands ──────────────────────────────────────────────
        bb_upper, bb_mid, bb_lower = self._bollinger(close, self.BB_PERIOD, self.BB_STD)
        bb_pos = (price - bb_lower) / (bb_upper - bb_lower + 1e-9)
        bb_w   = (bb_upper - bb_lower) / (bb_mid + 1e-9)
        features.append(np.clip(bb_pos, 0, 1))
        features.append(np.clip(bb_w * 10, 0, 1))

        # ── 7-10. EMA distances (9, 21, 50, 200) ─────────────────────────────
        for period in self.EMA_PERIODS:
            ema  = float(close.ewm(span=period, adjust=False).mean().iloc[-1])
            dist = (price - ema) / (ema + 1e-9)
            features.append(np.clip(dist * 50, -1, 1))

        # ── 11-12. Stochastic ─────────────────────────────────────────────────
        stoch_k, stoch_d = self._stochastic(close, high, low, self.STOCH_K, self.STOCH_D)
        features.append(self._norm(stoch_k, 0, 100))
        features.append(self._norm(stoch_d, 0, 100))

        # ── 13. ATR (volatility) ──────────────────────────────────────────────
        atr = self._atr(high, low, close, self.ATR_PERIOD)
        features.append(np.clip(atr / price * 100, 0, 1))

        # ── 14. ADX (trend strength) ──────────────────────────────────────────
        adx = self._adx(high, low, close, self.ADX_PERIOD)
        features.append(self._norm(adx, 0, 100))

        # ── 15. CCI ───────────────────────────────────────────────────────────
        cci = self._cci(high, low, close, self.CCI_PERIOD)
        features.append(np.clip(cci / 200, -1, 1))

        # ── 16. Williams %R ───────────────────────────────────────────────────
        wr = self._williams_r(high, low, close, self.WR_PERIOD)
        features.append(self._norm(wr, -100, 0))

        # ── 17. OBV (normalized) ──────────────────────────────────────────────
        obv   = self._obv(close, volume)
        sl    = obv.iloc[-self.OBV_WINDOW:]
        o_min = float(sl.min())
        o_max = float(sl.max())
        features.append(np.clip((float(obv.iloc[-1]) - o_min) / (o_max - o_min + 1e-9), 0, 1))

        # ── 18. Momentum ──────────────────────────────────────────────────────
        if len(close) >= self.MOM_PERIOD + 1:
            ref = float(close.iloc[-self.MOM_PERIOD - 1])
            mom = (price - ref) / (ref + 1e-9)
            features.append(np.clip(mom * 100, -1, 1))
        else:
            features.append(0.0)

        # ── 19. Volume ratio ──────────────────────────────────────────────────
        vol_ma  = float(volume.rolling(20).mean().iloc[-1]) or 1.0
        features.append(np.clip(float(volume.iloc[-1]) / vol_ma / 3.0, 0, 1))

        # ── 20. Ichimoku Tenkan-sen distance ─────────────────────────────────
        tenkan = (high.rolling(self.ICHIMOKU_CONV).max() + low.rolling(self.ICHIMOKU_CONV).min()) / 2
        features.append(np.clip((price - float(tenkan.iloc[-1])) / price * 50, -1, 1))

        # ── 21. Ichimoku Kijun-sen distance ──────────────────────────────────
        kijun = (high.rolling(self.ICHIMOKU_BASE).max() + low.rolling(self.ICHIMOKU_BASE).min()) / 2
        features.append(np.clip((price - float(kijun.iloc[-1])) / price * 50, -1, 1))

        # ── 22. Ichimoku Cloud position ────────────────────────────────────────
        senkou_a = ((tenkan + kijun) / 2).shift(self.ICHIMOKU_BASE)
        senkou_b = ((high.rolling(self.ICHIMOKU_SPAN).max() + low.rolling(self.ICHIMOKU_SPAN).min()) / 2).shift(self.ICHIMOKU_BASE)
        sa = float(senkou_a.iloc[-1]) if not pd.isna(senkou_a.iloc[-1]) else price
        sb = float(senkou_b.iloc[-1]) if not pd.isna(senkou_b.iloc[-1]) else price
        cloud_top = max(sa, sb)
        cloud_bot = min(sa, sb)
        cloud_pos = 1.0 if price > cloud_top else (-1.0 if price < cloud_bot else 0.0)
        features.append(cloud_pos)

        # ── 23. Parabolic SAR ─────────────────────────────────────────────────
        features.append(self._parabolic_sar(high, low, close))

        # ── 24. RSI Divergence ─────────────────────────────────────────────────
        features.append(self._rsi_divergence(close, self.RSI_PERIOD))

        # ── 25. Support proximity ──────────────────────────────────────────────
        recent_low  = float(low.iloc[-20:].min())
        sup_dist    = (price - recent_low) / (price + 1e-9)
        features.append(np.clip(1.0 - sup_dist * 20, 0, 1))

        # ── 26. Resistance proximity ───────────────────────────────────────────
        recent_high = float(high.iloc[-20:].max())
        res_dist    = (recent_high - price) / (recent_high + 1e-9)
        features.append(np.clip(1.0 - res_dist * 20, 0, 1))

        # ── 27. Fibonacci position ────────────────────────────────────────────
        fib_h   = float(high.iloc[-50:].max())
        fib_l   = float(low.iloc[-50:].min())
        fib_pos = np.clip((price - fib_l) / (fib_h - fib_l + 1e-9), 0, 1)
        features.append(fib_pos)

        # ── 28. Doji pattern ───────────────────────────────────────────────────
        body   = abs(price - float(open_.iloc[-1]))
        range_ = float(high.iloc[-1]) - float(low.iloc[-1]) + 1e-9
        features.append(1.0 if (body / range_) < 0.1 else 0.0)

        # ── 29. Hammer / Shooting Star ────────────────────────────────────────
        features.append(self._is_hammer(open_, high, low, close))

        # ── 30. Engulfing pattern ─────────────────────────────────────────────
        features.append(self._is_engulfing(open_, close))

        # ── 31. Morning / Evening Star ────────────────────────────────────────
        features.append(self._is_star(open_, high, low, close))

        # ── 32. Market regime (trending=1, ranging=0) ─────────────────────────
        features.append(1.0 if adx > 25 else 0.0)

        # ── 33. Higher High ───────────────────────────────────────────────────
        if len(high) >= 6:
            features.append(1.0 if float(high.iloc[-1]) > float(high.iloc[-6:-1].max()) else 0.0)
        else:
            features.append(0.0)

        # ── 34. Lower Low ─────────────────────────────────────────────────────
        if len(low) >= 6:
            features.append(1.0 if float(low.iloc[-1]) < float(low.iloc[-6:-1].min()) else 0.0)
        else:
            features.append(0.0)

        # ── 35. Short-term momentum (3 candles) ───────────────────────────────
        if len(close) >= 4:
            m3 = (price - float(close.iloc[-4])) / (float(close.iloc[-4]) + 1e-9)
            features.append(np.clip(m3 * 100, -1, 1))
        else:
            features.append(0.0)

        # ── 36. Candle body ratio ─────────────────────────────────────────────
        features.append(np.clip(body / range_, 0, 1))

        # ── 37. Volume acceleration ────────────────────────────────────────────
        if len(volume) >= 6:
            v_now  = float(volume.iloc[-3:].mean())
            v_prev = float(volume.iloc[-6:-3].mean()) + 1e-9
            features.append(np.clip((v_now - v_prev) / v_prev, -1, 1))
        else:
            features.append(0.0)

        # ── 38. MACD histogram change ─────────────────────────────────────────
        if len(close) >= 30:
            _, _, h_prev = self._macd(close.iloc[:-1], self.MACD_FAST, self.MACD_SLOW, self.MACD_SIGNAL)
            macd_mom = histogram - h_prev
            features.append(np.clip(macd_mom / price * 1000, -1, 1))
        else:
            features.append(0.0)

        # ── 39. Stochastic crossover ───────────────────────────────────────────
        features.append(np.clip((stoch_k - stoch_d) / 100.0, -1, 1))

        # ── 40. Volatility regime ──────────────────────────────────────────────
        if len(high) >= 50:
            long_atr = self._atr(high.iloc[-50:], low.iloc[-50:], close.iloc[-50:], 20)
        else:
            long_atr = atr
        features.append(np.clip((atr - long_atr) / (long_atr + 1e-9) * 5, -1, 1))

        arr = np.array(features[:N_FEATURES], dtype=np.float32)
        if len(arr) < N_FEATURES:
            arr = np.concatenate([arr, np.zeros(N_FEATURES - len(arr), dtype=np.float32)])
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        arr = np.clip(arr, -1.0, 1.0)
        return arr

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _norm(v: float, lo: float, hi: float) -> float:
        return float(np.clip((v - lo) / (hi - lo + 1e-9), 0, 1))

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> float:
        d    = close.diff()
        gain = d.clip(lower=0).rolling(period).mean()
        loss = (-d.clip(upper=0)).rolling(period).mean()
        rs   = gain / (loss + 1e-9)
        return float((100 - 100 / (1 + rs)).iloc[-1])

    @staticmethod
    def _macd(close: pd.Series, fast: int, slow: int, signal: int):
        ef  = close.ewm(span=fast, adjust=False).mean()
        es  = close.ewm(span=slow, adjust=False).mean()
        ml  = ef - es
        sig = ml.ewm(span=signal, adjust=False).mean()
        return float(ml.iloc[-1]), float(sig.iloc[-1]), float((ml - sig).iloc[-1])

    @staticmethod
    def _bollinger(close: pd.Series, period: int, std_mult: float):
        mid = float(close.rolling(period).mean().iloc[-1])
        std = float(close.rolling(period).std().iloc[-1])
        return mid + std_mult * std, mid, mid - std_mult * std

    @staticmethod
    def _stochastic(close, high, low, k: int, d: int):
        lo = low.rolling(k).min()
        hi = high.rolling(k).max()
        pk = 100 * (close - lo) / (hi - lo + 1e-9)
        pd_ = pk.rolling(d).mean()
        return float(pk.iloc[-1]), float(pd_.iloc[-1])

    @staticmethod
    def _atr(high, low, close, period: int) -> float:
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        v = tr.rolling(period).mean().iloc[-1]
        return float(v) if not pd.isna(v) else 0.0

    @staticmethod
    def _adx(high, low, close, period: int) -> float:
        up  = high.diff()
        dn  = -low.diff()
        pdm = up.where((up > dn) & (up > 0), 0.0)
        ndm = dn.where((dn > up) & (dn > 0), 0.0)
        tr  = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        pdi = 100 * pdm.rolling(period).mean() / (atr + 1e-9)
        ndi = 100 * ndm.rolling(period).mean() / (atr + 1e-9)
        dx  = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-9)
        adx = dx.rolling(period).mean().iloc[-1]
        return float(adx) if not pd.isna(adx) else 0.0

    @staticmethod
    def _cci(high, low, close, period: int) -> float:
        tp  = (high + low + close) / 3
        ma  = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        v   = ((tp - ma) / (0.015 * mad + 1e-9)).iloc[-1]
        return float(v) if not pd.isna(v) else 0.0

    @staticmethod
    def _williams_r(high, low, close, period: int) -> float:
        hm  = high.rolling(period).max()
        lm  = low.rolling(period).min()
        wr  = -100 * (hm - close) / (hm - lm + 1e-9)
        return float(wr.iloc[-1])

    @staticmethod
    def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        return (np.sign(close.diff()).fillna(0) * volume).cumsum()

    @staticmethod
    def _parabolic_sar(high: pd.Series, low: pd.Series, close: pd.Series,
                       af0: float = 0.02, af_max: float = 0.2) -> float:
        n = len(close)
        if n < 3:
            return 0.0
        sar  = float(low.iloc[0])
        bull = True
        ep   = float(high.iloc[0])
        af   = af0
        for i in range(1, n):
            sar = sar + af * (ep - sar)
            if bull:
                if float(low.iloc[i]) < sar:
                    bull = False; sar = ep; ep = float(low.iloc[i]); af = af0
                else:
                    if float(high.iloc[i]) > ep:
                        ep = float(high.iloc[i]); af = min(af + af0, af_max)
                    sar = min(sar, float(low.iloc[max(0, i-1)]), float(low.iloc[max(0, i-2)]))
            else:
                if float(high.iloc[i]) > sar:
                    bull = True; sar = ep; ep = float(high.iloc[i]); af = af0
                else:
                    if float(low.iloc[i]) < ep:
                        ep = float(low.iloc[i]); af = min(af + af0, af_max)
                    sar = max(sar, float(high.iloc[max(0, i-1)]), float(high.iloc[max(0, i-2)]))
        return 1.0 if bull else -1.0

    @staticmethod
    def _rsi_divergence(close: pd.Series, period: int = 14, w: int = 10) -> float:
        if len(close) < period + w + 5:
            return 0.0
        d    = close.diff()
        gain = d.clip(lower=0).rolling(period).mean()
        loss = (-d.clip(upper=0)).rolling(period).mean()
        rsi  = 100 - 100 / (1 + gain / (loss + 1e-9))
        ps   = close.iloc[-w:]
        rs   = rsi.iloc[-w:]
        pc   = float(close.iloc[-1])
        rc   = float(rsi.iloc[-1])
        if pc <= float(ps.min()) * 1.005 and rc > float(rs.min()) * 1.02:
            return 1.0
        if pc >= float(ps.max()) * 0.995 and rc < float(rs.max()) * 0.98:
            return -1.0
        return 0.0

    @staticmethod
    def _is_hammer(open_: pd.Series, high: pd.Series, low: pd.Series,
                   close: pd.Series) -> float:
        body  = abs(float(close.iloc[-1]) - float(open_.iloc[-1]))
        upper = float(high.iloc[-1]) - max(float(close.iloc[-1]), float(open_.iloc[-1]))
        lower = min(float(close.iloc[-1]), float(open_.iloc[-1])) - float(low.iloc[-1])
        rng   = float(high.iloc[-1]) - float(low.iloc[-1]) + 1e-9
        if lower > 2 * body and upper < body and body / rng < 0.4:
            return 1.0
        if upper > 2 * body and lower < body and body / rng < 0.4:
            return -1.0
        return 0.0

    @staticmethod
    def _is_engulfing(open_: pd.Series, close: pd.Series) -> float:
        if len(close) < 2:
            return 0.0
        c0, o0 = float(close.iloc[-1]), float(open_.iloc[-1])
        c1, o1 = float(close.iloc[-2]), float(open_.iloc[-2])
        if c0 > o0 and c1 < o1 and o0 <= c1 and c0 >= o1:
            return 1.0
        if c0 < o0 and c1 > o1 and o0 >= c1 and c0 <= o1:
            return -1.0
        return 0.0

    @staticmethod
    def _is_star(open_: pd.Series, high: pd.Series, low: pd.Series,
                 close: pd.Series) -> float:
        if len(close) < 3:
            return 0.0
        c0, o0 = float(close.iloc[-1]), float(open_.iloc[-1])
        c1, o1 = float(close.iloc[-2]), float(open_.iloc[-2])
        c2, o2 = float(close.iloc[-3]), float(open_.iloc[-3])
        b0 = abs(c0 - o0); b1 = abs(c1 - o1); b2 = abs(c2 - o2)
        if b1 < b2 * 0.3 and c2 < o2 and c0 > o0 and b0 > b2 * 0.5:
            return 1.0
        if b1 < b2 * 0.3 and c2 > o2 and c0 < o0 and b0 > b2 * 0.5:
            return -1.0
        return 0.0
