"""
TemporalSequenceMemory – จำ trajectory ของ indicators ข้ามการเทรด 8 ครั้ง

แทนที่จะมองแค่สภาวะ ณ จุดเดียว เราดูว่า indicators เปลี่ยนทิศทางอย่างไร
ในช่วง 8 การเทรดที่ผ่านมา เช่น RSI ขาขึ้น + MACD ลง + BB ทรง → อาจทำนาย
ผลลัพธ์การเทรดถัดไปได้ดีกว่าการดูแค่ค่าปัจจุบัน

Fingerprint key ตัวอย่าง: "rsi:UP macd:DOWN bb:FLAT adx:UP mom:UP stoch:DOWN"
"""
import json
import logging
import os
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Indices ของ 6 key indicators ใน vector 40 มิติ
_INDICATOR_INDICES = {
    "rsi":   0,
    "macd":  3,   # macd_hist
    "bb":    4,   # bb_position
    "adx":   13,
    "mom":   17,  # momentum
    "stoch": 10,  # stoch_k
}

_WINDOW_SIZE  = 8          # จำนวนการเทรดที่ใช้สร้าง fingerprint
_UP_THRESH    = 0.05       # diff > 0.05 → UP
_DOWN_THRESH  = -0.05      # diff < -0.05 → DOWN
_MIN_SAMPLES  = 3          # ต้องการ samples อย่างน้อยกี่ครั้งก่อนเชื่อถือ
_SAVE_EVERY   = 15         # flush ทุกกี่ records


class TemporalSequenceMemory:
    """
    จำ rolling window ของ indicator snapshots 8 ครั้งล่าสุด
    และ fingerprint trajectory เพื่อทำนายผลลัพธ์การเทรดถัดไป

    API:
      record(indicator_vec, outcome) → บันทึก snapshot ใหม่ + outcome
      lookup(indicator_vec)          → (win_rate, confidence, n) หรือ None
    """

    def __init__(self, base_dir: str = "./knowledge"):
        self._path = os.path.join(base_dir, "sequence_memory.json")
        os.makedirs(base_dir, exist_ok=True)

        # Rolling buffer ของ indicator vectors (แต่ละตัวคือ np.ndarray)
        self._window: Deque[np.ndarray] = deque(maxlen=_WINDOW_SIZE)

        # fingerprint → {wins, total, last_seen}
        self._db: Dict[str, dict] = {}

        self._pending = 0
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def record(self, indicator_vec: np.ndarray, outcome: float) -> None:
        """
        เพิ่ม snapshot ลงใน rolling buffer
        เมื่อ buffer ครบ 8 ครั้ง จะ fingerprint trajectory แล้วบันทึก outcome

        Parameters
        ----------
        indicator_vec : np.ndarray ขนาด 40 floats (normalized)
        outcome       : float → บวก = win, ลบ = loss
        """
        if indicator_vec is None or len(indicator_vec) < max(_INDICATOR_INDICES.values()) + 1:
            logger.debug("SequenceMemory.record: indicator_vec สั้นเกินไป ข้ามไป")
            return

        self._window.append(np.array(indicator_vec, dtype=float))

        # ต้องมีครบ 8 snapshots ก่อนถึงจะสร้าง fingerprint ได้
        if len(self._window) < _WINDOW_SIZE:
            return

        key = self._fingerprint(list(self._window))
        won = outcome > 0

        if key not in self._db:
            self._db[key] = {"wins": 0, "total": 0}
        entry = self._db[key]
        entry["total"] += 1
        if won:
            entry["wins"] += 1

        self._pending += 1
        if self._pending >= _SAVE_EVERY:
            self._save()
            self._pending = 0

        logger.debug(
            "SequenceMemory: recorded key=%s win=%s (total=%d)",
            key[:40], won, entry["total"],
        )

    def lookup(self, indicator_vec: np.ndarray) -> Optional[Tuple[float, float, int]]:
        """
        ค้นหา trajectory ปัจจุบัน (window + snapshot ใหม่) ใน database

        Returns
        -------
        (win_rate, confidence, n) ถ้าพบ หรือ None ถ้าไม่พบ / ข้อมูลน้อยเกินไป

        win_rate ใช้ Laplace smoothing: (wins+1)/(total+2)
        confidence เพิ่มขึ้นตาม n แต่ไม่เกิน 0.85
        """
        if indicator_vec is None or len(indicator_vec) < max(_INDICATOR_INDICES.values()) + 1:
            return None

        # สร้าง preview window: window ปัจจุบัน + snapshot ใหม่
        preview = list(self._window) + [np.array(indicator_vec, dtype=float)]
        if len(preview) < _WINDOW_SIZE:
            return None

        # ใช้แค่ 8 ตัวล่าสุด
        preview = preview[-_WINDOW_SIZE:]
        key = self._fingerprint(preview)

        entry = self._db.get(key)
        if entry is None or entry["total"] < _MIN_SAMPLES:
            return None

        wins  = entry["wins"]
        total = entry["total"]

        # Laplace smoothing
        win_rate   = (wins + 1) / (total + 2)
        confidence = min(0.85, 0.45 + total * 0.015)

        return round(win_rate, 4), round(confidence, 4), total

    def stats(self) -> dict:
        """สถิติ summary ของ sequence memory"""
        total    = len(self._db)
        trusted  = sum(1 for e in self._db.values() if e["total"] >= _MIN_SAMPLES)
        tot_wins = sum(e["wins"]  for e in self._db.values())
        tot_all  = sum(e["total"] for e in self._db.values())
        return {
            "total_sequences":   total,
            "trusted_sequences": trusted,
            "total_wins":        tot_wins,
            "total_records":     tot_all,
            "window_fill":       len(self._window),
        }

    def flush(self) -> None:
        """บันทึก force"""
        self._save()
        self._pending = 0

    # ── Fingerprinting ────────────────────────────────────────────────────────

    @staticmethod
    def _fingerprint(snapshots: List[np.ndarray]) -> str:
        """
        แปลง window 8 snapshots → fingerprint string

        สำหรับแต่ละ indicator:
          เปรียบเทียบค่าสุดท้ายกับค่าแรกใน window
          UP   ถ้า diff > 0.05
          DOWN ถ้า diff < -0.05
          FLAT ถ้าอยู่ระหว่างนั้น

        ตัวอย่าง: "rsi:UP macd:DOWN bb:FLAT adx:UP mom:UP stoch:DOWN"
        """
        first = snapshots[0]
        last  = snapshots[-1]

        parts = []
        for name, idx in _INDICATOR_INDICES.items():
            try:
                diff = float(last[idx]) - float(first[idx])
            except (IndexError, TypeError):
                diff = 0.0

            if diff > _UP_THRESH:
                direction = "UP"
            elif diff < _DOWN_THRESH:
                direction = "DOWN"
            else:
                direction = "FLAT"

            parts.append(f"{name}:{direction}")

        return " ".join(parts)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._db, f, separators=(",", ":"))
            os.replace(tmp, self._path)
            logger.debug("SequenceMemory: saved %d sequences", len(self._db))
        except Exception as exc:
            logger.warning("SequenceMemory: save failed – %s", exc)

    def _load(self) -> None:
        if not os.path.exists(self._path):
            logger.info("SequenceMemory: ไม่พบไฟล์ เริ่มต้นใหม่")
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                self._db = json.load(f)
            st = self.stats()
            logger.info(
                "SequenceMemory: loaded %d sequences (%d trusted)",
                st["total_sequences"], st["trusted_sequences"],
            )
        except Exception as exc:
            logger.warning("SequenceMemory: load failed – %s", exc)
            self._db = {}
