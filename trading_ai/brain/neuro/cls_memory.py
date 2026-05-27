"""
Complementary Learning Systems (CLS) — จากประสาทวิทยาศาสตร์
════════════════════════════════════════════════════════════

สมองมนุษย์มีระบบความจำ 2 ระบบที่ทำงานคู่กัน:

  🧠 Hippocampus  — จำประสบการณ์เฉพาะอย่างเร็ว (fast, specific)
                   ถูก overwrite ได้ง่าย แต่ encode ได้ทันที

  🧠 Neocortex    — เรียนรู้ช้า แต่ generalize ได้ดี (slow, general)
                   เก็บ pattern นามธรรม ไม่ใช่ตัวอย่างเฉพาะ

ทั้งสองทำงานร่วมกัน:
  → ระหว่างนอนหลับ hippocampus replay ประสบการณ์
  → neocortex ค่อยๆ ดูดซับและ generalize

"This is why sleep helps you learn better"
(McClelland, McNaughton & O'Reilly, 1995)

AI Equivalent:
  Hippocampus → EpisodicBuffer (fast write, limited capacity)
  Neocortex   → GeneralizedPatterns (slow distillation, unlimited)
  Sleep       → consolidate() method
"""
import time
import numpy as np
from collections import deque
from typing import List, Optional


class CLSMemory:
    """
    Complementary Learning Systems memory.
    Maintains two stores: fast hippocampal + slow neocortical.
    """

    def __init__(self, hippocampus_size: int = 500):
        # Hippocampus: fast, specific, limited capacity
        self._hippocampus: deque = deque(maxlen=hippocampus_size)

        # Neocortex: slow, generalized patterns (distilled from replays)
        self._neocortex: List[dict] = []   # [{"centroid", "win_rate", "n", "action"}]

        self._total_encoded = 0
        self._total_replays = 0
        self._last_consolidation = time.time()

    # ── Hippocampus: rapid encoding ────────────────────────────────────────────

    def encode(self, indicators: np.ndarray, action: int, pnl: float,
               confidence: float = 0.5) -> None:
        """Hippocampal encoding — fast, immediate, specific."""
        episode = {
            "indicators": indicators[:20].tolist() if len(indicators) >= 20 else indicators.tolist(),
            "action":     action,
            "pnl":        float(pnl),
            "win":        pnl > 0,
            "confidence": float(confidence),
            "ts":         time.time(),
        }
        self._hippocampus.append(episode)
        self._total_encoded += 1

    # ── Neocortex: consolidation ───────────────────────────────────────────────

    def consolidate(self, n_replays: int = 20) -> int:
        """
        Sleep-like consolidation: replay hippocampal episodes,
        distill into neocortical generalized patterns.
        Returns number of patterns updated.
        """
        if len(self._hippocampus) < 10:
            return 0

        # Sample episodes weighted by surprise (|pnl| = more surprising)
        episodes = list(self._hippocampus)
        weights = np.array([abs(e["pnl"]) + 0.1 for e in episodes])
        weights /= weights.sum()

        n = min(n_replays, len(episodes))
        idxs = np.random.choice(len(episodes), size=n, replace=False, p=weights)
        replayed = [episodes[i] for i in idxs]
        self._total_replays += n

        # Cluster replayed episodes by indicator similarity
        updated = 0
        for ep in replayed:
            ind = np.array(ep["indicators"][:10])
            matched = self._find_neocortex_pattern(ind)
            if matched is not None:
                # Update existing pattern (running average)
                alpha = 1.0 / (matched["n"] + 1)
                matched["centroid"] = (
                    (1 - alpha) * np.array(matched["centroid"]) + alpha * ind
                ).tolist()
                matched["win_rate"] = (
                    matched["win_rate"] * (1 - alpha) + float(ep["win"]) * alpha
                )
                matched["n"] += 1
            else:
                # New neocortical pattern
                self._neocortex.append({
                    "centroid":  ind.tolist(),
                    "win_rate":  float(ep["win"]),
                    "action":    ep["action"],
                    "n":         1,
                })
            updated += 1

        self._last_consolidation = time.time()
        return updated

    def _find_neocortex_pattern(self, indicators: np.ndarray,
                                threshold: float = 0.3) -> Optional[dict]:
        """Find the closest neocortical pattern by cosine similarity."""
        if not self._neocortex:
            return None
        best_sim, best_pat = -1.0, None
        for pat in self._neocortex:
            c = np.array(pat["centroid"])
            sim = float(np.dot(indicators, c) / (
                np.linalg.norm(indicators) * np.linalg.norm(c) + 1e-8
            ))
            if sim > best_sim:
                best_sim, best_pat = sim, pat
        return best_pat if best_sim > (1.0 - threshold) else None

    # ── Recall: combine both systems ──────────────────────────────────────────

    def recall(self, indicators: np.ndarray) -> dict:
        """
        Combined hippocampal + neocortical recall.
        Hippocampus: recent specific match
        Neocortex: generalized pattern
        """
        ind = np.array(indicators[:10]) if len(indicators) >= 10 else np.array(indicators)

        # Hippocampal recall (most recent similar)
        hip_result = self._hippocampal_recall(ind)

        # Neocortical recall (generalized)
        neo_pat = self._find_neocortex_pattern(ind)
        neo_result = {
            "win_rate": neo_pat["win_rate"] if neo_pat else 0.5,
            "n": neo_pat["n"] if neo_pat else 0,
        } if neo_pat else None

        # Combine: neocortex for general, hippocampus for recent bias
        if hip_result and neo_result:
            combined_wr = 0.4 * hip_result["win_rate"] + 0.6 * neo_result["win_rate"]
        elif neo_result:
            combined_wr = neo_result["win_rate"]
        elif hip_result:
            combined_wr = hip_result["win_rate"]
        else:
            combined_wr = 0.5

        return {
            "combined_win_rate": combined_wr,
            "hippocampal":       hip_result,
            "neocortical":       neo_result,
            "has_memory":        hip_result is not None or neo_result is not None,
        }

    def _hippocampal_recall(self, indicators: np.ndarray,
                            lookback: int = 50) -> Optional[dict]:
        """Recall recent similar episodes from hippocampus."""
        recent = list(self._hippocampus)[-lookback:]
        if not recent:
            return None
        best_sim, best = -1.0, None
        for ep in recent:
            c = np.array(ep["indicators"][:len(indicators)])
            sim = float(np.dot(indicators[:len(c)], c) / (
                np.linalg.norm(indicators[:len(c)]) * np.linalg.norm(c) + 1e-8
            ))
            if sim > best_sim:
                best_sim, best = sim, ep
        if best_sim < 0.7:
            return None
        return {"win_rate": float(best["win"]), "pnl": best["pnl"], "sim": best_sim}

    # ── Stats ────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "hippocampus_size":    len(self._hippocampus),
            "neocortex_patterns":  len(self._neocortex),
            "total_encoded":       self._total_encoded,
            "total_replays":       self._total_replays,
            "last_consolidation":  time.strftime(
                "%H:%M:%S", time.localtime(self._last_consolidation)
            ),
        }
