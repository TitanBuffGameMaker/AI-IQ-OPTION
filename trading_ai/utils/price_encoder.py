"""
PriceSequenceEncoder: replaces CNN chart features (which return zeros in headless mode)
with 256 meaningful statistical features computed from OHLCV candle data.

This is the single highest-impact fix: 256 of 296 observation dimensions were returning
zeros because ChartCapture requires a screen. Now they carry real market information.

Output layout (256 features total):
  [0:240]   = last 48 candles × 5 per-candle features
  [240:248] = 8 multi-scale returns (lags 1,2,3,5,8,13,21,34)
  [248:251] = ATR normalized at windows 5, 10, 20
  [251:254] = EMA deviation at periods 9, 21, 34
  [254:256] = Bollinger band position (upper, lower deviation)
"""
import numpy as np
import pandas as pd
from typing import Optional


class PriceSequenceEncoder:
    """
    Encodes OHLCV candle data into a fixed 256-dim float32 vector.
    Fully stateless: encode(df) can be called repeatedly without side effects.
    """

    OUTPUT_DIM = 256
    N_CANDLES  = 48   # how many recent candles to encode per-candle features

    def encode(self, df: Optional[pd.DataFrame]) -> np.ndarray:
        out = np.zeros(self.OUTPUT_DIM, dtype=np.float32)
        if df is None or len(df) < 5:
            return out

        # Ensure float
        closes = df["close"].values.astype(np.float64)
        opens  = df["open"].values.astype(np.float64)
        highs  = df["high"].values.astype(np.float64)
        lows   = df["low"].values.astype(np.float64)

        n = len(closes)
        last_close = closes[-1] if closes[-1] > 0 else 1.0

        # ── Per-candle features (last 48 candles × 5) ─────────────────────────
        n_enc   = min(self.N_CANDLES, n)
        c_slice = closes[-n_enc:]
        o_slice = opens[-n_enc:]
        h_slice = highs[-n_enc:]
        l_slice = lows[-n_enc:]

        # Normalise all prices to last close so the model sees relative values
        c_n = c_slice / last_close - 1.0   # return vs last close
        o_n = o_slice / last_close - 1.0
        h_n = h_slice / last_close - 1.0
        l_n = l_slice / last_close - 1.0

        ranges  = np.maximum(h_slice - l_slice, 1e-9)

        for i in range(n_enc):
            base = i * 5
            # 1. Return from previous close
            if i == 0 and n_enc < n:
                prev_c = closes[-(n_enc + 1)]
                ret = (c_slice[0] - prev_c) / (prev_c + 1e-9)
            elif i > 0:
                ret = (c_slice[i] - c_slice[i - 1]) / (c_slice[i - 1] + 1e-9)
            else:
                ret = 0.0
            out[base]     = float(np.clip(ret * 50, -3.0, 3.0))

            # 2. Body ratio: (close - open) / range  → [-1, 1]
            out[base + 1] = float(np.clip((c_slice[i] - o_slice[i]) / ranges[i], -1.0, 1.0))

            # 3. Upper wick: (high - max(open,close)) / range → [0, 1]
            upper = h_slice[i] - max(c_slice[i], o_slice[i])
            out[base + 2] = float(np.clip(upper / ranges[i], 0.0, 1.0))

            # 4. Close position in range: (close - low) / range → [-1, 1]
            pos = (c_slice[i] - l_slice[i]) / ranges[i]
            out[base + 3] = float(np.clip(pos * 2.0 - 1.0, -1.0, 1.0))

            # 5. Normalised range: range / last_close (volatility proxy) → [0, 1]
            out[base + 4] = float(np.clip(ranges[i] / last_close * 100, 0.0, 3.0))

        # ── Global features ────────────────────────────────────────────────────
        ptr = self.N_CANDLES * 5   # = 240

        # 8 multi-scale returns (Fibonacci-ish lags)
        for lag in [1, 2, 3, 5, 8, 13, 21, 34]:
            if n > lag:
                r = (closes[-1] - closes[-1 - lag]) / (closes[-1 - lag] + 1e-9)
                out[ptr] = float(np.clip(r * 50, -3.0, 3.0))
            ptr += 1

        # ATR at windows 5, 10, 20 (normalised by last close)
        for win in [5, 10, 20]:
            if n >= win + 1:
                trs = []
                for j in range(1, win + 1):
                    tr = max(
                        highs[-j] - lows[-j],
                        abs(highs[-j] - closes[-j - 1]),
                        abs(lows[-j] - closes[-j - 1]),
                    )
                    trs.append(tr)
                atr = float(np.mean(trs))
                out[ptr] = float(np.clip(atr / last_close * 100, 0.0, 3.0))
            ptr += 1

        # EMA deviation at periods 9, 21, 34 (normalised)
        for period in [9, 21, 34]:
            if n >= period:
                alpha = 2.0 / (period + 1)
                ema   = closes[0]
                for c in closes[1:]:
                    ema = alpha * c + (1 - alpha) * ema
                dev = (closes[-1] - ema) / (ema + 1e-9)
                out[ptr] = float(np.clip(dev * 50, -3.0, 3.0))
            ptr += 1

        # Bollinger band position (2 features: upper/lower sigma deviation)
        win = min(20, n)
        if win >= 5:
            bb_closes = closes[-win:]
            bb_mean   = bb_closes.mean()
            bb_std    = bb_closes.std() + 1e-9
            out[ptr]     = float(np.clip((closes[-1] - bb_mean) / bb_std, -3.0, 3.0))
            out[ptr + 1] = float(np.clip(bb_std / last_close * 100, 0.0, 3.0))
        ptr += 2

        return out
