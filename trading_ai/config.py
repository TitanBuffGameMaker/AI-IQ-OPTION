"""
Central configuration — ULTRA EDITION
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── IQ Option credentials ──────────────────────────────────────────────
    IQ_EMAIL:        str   = os.getenv("IQ_EMAIL", "your@email.com")
    IQ_PASSWORD:     str   = os.getenv("IQ_PASSWORD", "yourpassword")
    IQ_ACCOUNT_TYPE: str   = os.getenv("IQ_ACCOUNT_TYPE", "PRACTICE")

    # ── Trading parameters ─────────────────────────────────────────────────
    ASSET:            str   = os.getenv("ASSET", "EURUSD")
    TRADE_AMOUNT:     float = float(os.getenv("TRADE_AMOUNT", "1.0"))
    TRADE_DURATION:   int   = int(os.getenv("TRADE_DURATION", "1"))
    CANDLE_TIMEFRAME: int   = int(os.getenv("CANDLE_TIMEFRAME", "60"))

    # ── Observation window ─────────────────────────────────────────────────
    LOOKBACK_CANDLES: int = 100        # เพิ่มจาก 50 เป็น 100
    CHART_IMG_SIZE:   int = 84

    # ── PPO hyperparameters (ULTRA) ────────────────────────────────────────
    LEARNING_RATE:    float = 2.5e-4   # ปรับจาก 3e-4
    GAMMA:            float = 0.99
    GAE_LAMBDA:       float = 0.95
    CLIP_EPSILON:     float = 0.2
    ENTROPY_COEF:     float = 0.01
    VALUE_LOSS_COEF:  float = 0.5
    MAX_GRAD_NORM:    float = 0.5
    PPO_EPOCHS:       int   = 10       # เพิ่มจาก 4 เป็น 10
    BATCH_SIZE:       int   = 64
    UPDATE_EVERY:     int   = 1024     # ลดจาก 2048 เพื่อ update บ่อยขึ้น

    # ── Knowledge persistence ──────────────────────────────────────────────
    MODEL_DIR:         str = os.getenv("MODEL_DIR", "./knowledge")
    CHECKPOINT_EVERY:  int = 50        # บันทึก checkpoint บ่อยขึ้น
    LOG_DIR:           str = "./logs"

    # ── Risk management (ULTRA) ────────────────────────────────────────────
    MAX_CONSECUTIVE_LOSSES: int   = 5
    DAILY_LOSS_LIMIT:       float = float(os.getenv("DAILY_LOSS_LIMIT", "20.0"))
    MIN_CONFIDENCE:         float = 0.60
    MAX_POSITION_SIZE:      float = 2.0   # USD สูงสุด
    SHARPE_WINDOW:          int   = 20    # window สำหรับคำนวณ Sharpe reward

    # ── Advanced AI settings ───────────────────────────────────────────────
    USE_ICM:            bool  = True    # Intrinsic Curiosity Module
    ICM_COEF:           float = 0.01
    ENTROPY_START:      float = 0.05
    ENTROPY_END:        float = 0.003
    ENTROPY_DECAY_STEPS: int  = 50_000


config = Config()
