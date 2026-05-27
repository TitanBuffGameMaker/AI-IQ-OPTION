"""
Sleep Cycle (Memory Consolidation) — จากประสาทวิทยาศาสตร์
══════════════════════════════════════════════════════════

ระหว่างนอนหลับสมองทำอะไร?
  1. Slow-wave sleep: Hippocampus replay ประสบการณ์วันนี้
  2. สัญญาณส่งต่อไปยัง neocortex ซ้ำๆ
  3. Neocortex ค่อยๆ บันทึก pattern ลง long-term memory
  4. ประสบการณ์ที่มี emotional weight สูง → replay บ่อยกว่า
  5. Synaptic homeostasis: ลด noise, เก็บเฉพาะสิ่งสำคัญ

"Sleep is the price we pay for plasticity"
(Giulio Tononi — Synaptic Homeostasis Hypothesis)

"We are such stuff as dreams are made on"
  → ความฝันคือ hippocampal replay ที่ "รั่ว" เข้าสู่จิตสำนึก

AI Equivalent:
  ทุก 50 trades → trigger "sleep session"
  → replay trades ที่มี RPE สูง (surprising) ก่อน
  → consolidate patterns เข้า CLSMemory
  → distill rules ใหม่ผ่าน RuleDistillationEngine
  → reset short-term noise counters
  → broadcast "brain is sleeping / waking" to UI
"""
import logging
import threading
import time
from typing import Callable, Optional

_consolidation_lock = threading.Lock()

logger = logging.getLogger(__name__)


class SleepCycle:
    """
    Triggers periodic memory consolidation ("sleep") every N trades.
    Consolidation runs in a background thread to not block trading.
    """

    def __init__(self, trigger_every: int = 50, on_sleep_callback: Optional[Callable] = None):
        self._trigger_every = trigger_every
        self._trade_count = 0
        self._sleep_count = 0
        self._is_sleeping = False
        self._last_sleep_ts = 0.0
        self._on_sleep_callback = on_sleep_callback   # (sleeping: bool) → None

    def tick(self, brain) -> bool:
        """
        Call after every trade.
        Returns True if a sleep session was triggered.
        """
        self._trade_count += 1
        if self._trade_count % self._trigger_every != 0:
            return False
        if self._is_sleeping:
            return False

        threading.Thread(
            target=self._sleep_session,
            args=(brain,),
            daemon=True,
        ).start()
        return True

    def _sleep_session(self, brain) -> None:
        """Background consolidation session."""
        self._is_sleeping = True
        self._last_sleep_ts = time.time()
        self._sleep_count += 1
        logger.info(
            "💤 Sleep cycle #%d started — consolidating %d hippocampal episodes",
            self._sleep_count, brain.cls_memory.stats()["hippocampus_size"],
        )

        if self._on_sleep_callback:
            try:
                self._on_sleep_callback(True)
            except Exception:
                pass

        try:
            # CLS consolidation: replay surprising episodes → neocortex (locked)
            with _consolidation_lock:
                n_updated = brain.cls_memory.consolidate(n_replays=30)
            logger.info("💤 CLS: %d neocortical patterns updated", n_updated)

            # Rule distillation: extract new rules from recent trades
            try:
                brain.rule_distiller.distill()
            except Exception as e:
                logger.debug("Sleep rule distill: %s", e)

            time.sleep(0.5)   # brief pause to let consolidation settle

        except Exception as e:
            logger.warning("Sleep session error: %s", e)
        finally:
            self._is_sleeping = False
            elapsed = time.time() - self._last_sleep_ts
            logger.info("☀️ Sleep cycle #%d complete (%.1fs)", self._sleep_count, elapsed)
            if self._on_sleep_callback:
                try:
                    self._on_sleep_callback(False)
                except Exception:
                    pass

    def stats(self) -> dict:
        return {
            "sleep_count":    self._sleep_count,
            "trade_count":    self._trade_count,
            "is_sleeping":    self._is_sleeping,
            "next_sleep_in":  self._trigger_every - (self._trade_count % self._trigger_every),
            "last_sleep":     (
                time.strftime("%H:%M:%S", time.localtime(self._last_sleep_ts))
                if self._last_sleep_ts else "ยังไม่เคย"
            ),
        }
