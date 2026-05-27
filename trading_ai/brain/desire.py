"""
DesireEngine — records AI's wants/needs.
When the brain wants to expand beyond its current scope, it registers a desire
and notifies the creator via UI + email (handled by server.py callback).
"""
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

_HOUR = 3600


class DesireEngine:

    STATUS_PENDING  = "pending"
    STATUS_APPROVED = "approved"
    STATUS_DENIED   = "denied"

    def __init__(self, base_dir: str, notify_callback: Optional[Callable] = None):
        self._path = Path(base_dir) / "knowledge" / "desires.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._desires: List[dict] = self._load()
        self._notify_callback = notify_callback
        self._category_cooldowns: dict = {}  # category → expire_ts

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> List[dict]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def _save(self) -> None:
        try:
            self._path.write_text(
                json.dumps(self._desires, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("DesireEngine save failed: %s", e)

    # ── Public API ────────────────────────────────────────────────────────────

    def register(
        self,
        title: str,
        description: str,
        urgency: int = 5,
        category: str = "learning",
    ) -> Optional[dict]:
        """Register a new desire. Returns desire dict, or None if on cooldown."""
        if time.time() < self._category_cooldowns.get(category, 0):
            return None
        self._category_cooldowns[category] = time.time() + _HOUR

        desire = {
            "id":           str(uuid.uuid4())[:8],
            "title":        title,
            "description":  description,
            "urgency":      min(10, max(1, int(urgency))),
            "category":     category,
            "status":       self.STATUS_PENDING,
            "created_at":   time.strftime("%Y-%m-%d %H:%M:%S"),
            "resolved_at":  None,
            "resolve_note": None,
        }
        self._desires.append(desire)
        self._save()
        logger.info("AI Desire [%s] %s", category, title)

        if self._notify_callback:
            try:
                self._notify_callback(desire)
            except Exception as e:
                logger.warning("Desire notify_callback error: %s", e)

        return desire

    def approve(self, desire_id: str, note: str = "") -> bool:
        for d in self._desires:
            if d["id"] == desire_id and d["status"] == self.STATUS_PENDING:
                d["status"]      = self.STATUS_APPROVED
                d["resolved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                d["resolve_note"] = note
                self._save()
                logger.info("Desire APPROVED %s — %s", desire_id, d["title"])
                return True
        return False

    def deny(self, desire_id: str, reason: str = "") -> bool:
        for d in self._desires:
            if d["id"] == desire_id and d["status"] == self.STATUS_PENDING:
                d["status"]       = self.STATUS_DENIED
                d["resolved_at"]  = time.strftime("%Y-%m-%d %H:%M:%S")
                d["resolve_note"] = reason
                self._save()
                logger.info("Desire DENIED %s — %s", desire_id, d["title"])
                return True
        return False

    def get_all(self) -> List[dict]:
        return list(reversed(self._desires))

    def get_pending(self) -> List[dict]:
        return [d for d in self._desires if d["status"] == self.STATUS_PENDING]

    def stats(self) -> dict:
        return {
            "total":   len(self._desires),
            "pending": sum(1 for d in self._desires if d["status"] == self.STATUS_PENDING),
        }

    def set_notify_callback(self, cb: Callable) -> None:
        self._notify_callback = cb
