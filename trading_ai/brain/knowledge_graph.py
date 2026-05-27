"""
KnowledgeGraph – the growing root system of the AI brain.

The graph stores all KnowledgeNodes and their connections.
It grows organically:
  - New experiences → new nodes or strengthen existing ones
  - Related nodes connect automatically (roots intertwine)
  - Stale, contradicted knowledge weakens (roots shrink)
  - Strong knowledge spreads activation to neighbours (root network)

Persisted as a JSON file so the brain survives across sessions.
"""
import json
import logging
import os
import time as _time
import threading
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from trading_ai.brain.knowledge_node import EdgeType, KnowledgeNode, NodeType

logger = logging.getLogger(__name__)

GRAPH_FILE = "brain_graph.json"


class KnowledgeGraph:
    """
    The root-system brain.

    Internally: dict of node_id → KnowledgeNode, plus an inverted index
    by tag and asset for fast lookup.
    """

    PRUNE_THRESHOLD = 0.05        # remove nodes with confidence below this
    MIN_EVIDENCE_TO_PRUNE = 10    # only prune mature nodes

    def __init__(self, base_dir: str = "./knowledge"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self._graph_path = os.path.join(base_dir, GRAPH_FILE)

        self._nodes: Dict[str, KnowledgeNode] = {}
        self._tag_index: Dict[str, List[str]] = defaultdict(list)   # tag → [node_ids]
        self._type_index: Dict[str, List[str]] = defaultdict(list)  # type → [node_ids]
        self._asset_index: Dict[str, List[str]] = defaultdict(list) # asset → [node_ids]
        self._lock = threading.RLock()  # protect concurrent writes from worker + brain threads

        self.load()
        self._seed_root_knowledge()

    # ── Growth: add nodes ─────────────────────────────────────────────────────

    def add_node(self, node: KnowledgeNode) -> str:
        """
        Plant a new knowledge node (or reinforce an existing one if duplicate).
        Returns the node_id that was stored.
        Thread-safe: uses RLock so server brain thread and worker WebSocket
        handler can both call this concurrently without data corruption.
        """
        with self._lock:
            existing = self._find_similar(node.concept, node.node_type, node.asset)
            if existing:
                existing.confirm(0.06)
                for tag in node.tags:
                    if tag not in existing.tags:
                        existing.tags.append(tag)
                logger.debug("Reinforced existing node: %s", existing)
                self._save_incremental()
                return existing.node_id

            self._nodes[node.node_id] = node
            self._index_node(node)
            self._auto_connect(node)

            logger.debug("New knowledge node planted: %s", node)
            self._save_incremental()
            return node.node_id

    def reinforce(self, node_id: str, strength: float = 0.08):
        """A trade result or internet source confirms this node."""
        if node_id in self._nodes:
            node = self._nodes[node_id]
            node.confirm(strength)
            # Propagate activation to strong neighbours (roots share nutrients)
            for edge in node.get_strong_edges(min_strength=0.5):
                if edge.target_id in self._nodes:
                    self._nodes[edge.target_id].activation = min(
                        1.0, self._nodes[edge.target_id].activation + 0.2
                    )
            self._save_incremental()

    def contradict(self, node_id: str, strength: float = 0.06):
        """Evidence contradicts this node – it weakens but doesn't disappear."""
        if node_id in self._nodes:
            self._nodes[node_id].contradict(strength)
            self._save_incremental()

    def connect(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        strength: float = 0.5,
    ):
        """Manually create a branch between two nodes."""
        if source_id in self._nodes and target_id in self._nodes:
            self._nodes[source_id].connect(target_id, edge_type, strength)
            self._save_incremental()

    # ── Query: find relevant knowledge ────────────────────────────────────────

    def query(
        self,
        tags: Optional[List[str]] = None,
        node_type: Optional[NodeType] = None,
        asset: Optional[str] = None,
        min_confidence: float = 0.40,
        limit: int = 10,
    ) -> List[KnowledgeNode]:
        """
        Find the most relevant, reliable nodes.
        Returns a list sorted by (confidence × activation).
        """
        candidates: List[str] = []

        if tags:
            for tag in tags:
                candidates.extend(self._tag_index.get(tag, []))
        if node_type:
            candidates.extend(self._type_index.get(node_type.value, []))
        if asset:
            candidates.extend(self._asset_index.get(asset, []))
            candidates.extend(self._asset_index.get("universal", []))

        if not candidates:
            candidates = list(self._nodes.keys())

        seen = set()
        results = []
        for nid in candidates:
            if nid in seen or nid not in self._nodes:
                continue
            seen.add(nid)
            node = self._nodes[nid]
            if node.confidence >= min_confidence:
                results.append(node)

        results.sort(key=lambda n: n.confidence * n.activation, reverse=True)
        return results[:limit]

    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        return self._nodes.get(node_id)

    def get_neighbourhood(self, node_id: str, depth: int = 2) -> List[KnowledgeNode]:
        """
        BFS from a node through its connections.
        Like following a root system outward.
        """
        visited: set = set()
        queue = [(node_id, 0)]
        result = []
        while queue:
            nid, d = queue.pop(0)
            if nid in visited or nid not in self._nodes:
                continue
            visited.add(nid)
            node = self._nodes[nid]
            result.append(node)
            if d < depth:
                for edge in node.edges:
                    if edge.strength > 0.3:
                        queue.append((edge.target_id, d + 1))
        return result

    def find_by_tags(self, *tags: str) -> List[KnowledgeNode]:
        ids: set = set()
        for tag in tags:
            ids.update(self._tag_index.get(tag, []))
        return [self._nodes[i] for i in ids if i in self._nodes]

    # ── Maintenance: prune and decay ──────────────────────────────────────────

    def tick(self):
        """
        Called periodically (e.g. each episode).
        Decays activation, prunes dead roots.
        """
        to_prune = []
        for node in self._nodes.values():
            node.tick_decay(rate=0.005)
            if (
                node.confidence < self.PRUNE_THRESHOLD
                and node.evidence_count >= self.MIN_EVIDENCE_TO_PRUNE
            ):
                to_prune.append(node.node_id)

        for nid in to_prune:
            self._prune_node(nid)

        if to_prune:
            logger.info("Pruned %d stale knowledge nodes", len(to_prune))

    def stats(self) -> dict:
        by_type = defaultdict(int)
        total_conf = 0.0
        for node in self._nodes.values():
            by_type[node.node_type.value] += 1
            total_conf += node.confidence
        n = len(self._nodes)
        return {
            "total_nodes": n,
            "avg_confidence": round(total_conf / max(n, 1), 3),
            "by_type": dict(by_type),
            "total_edges": sum(len(n.edges) for n in self._nodes.values()),
        }

    def print_stats(self):
        s = self.stats()
        logger.info(
            "── Brain Graph ──────────────────────────────────\n"
            "  Nodes: %d  |  Avg confidence: %.2f\n"
            "  Edges (branches): %d\n"
            "  By type: %s\n"
            "─────────────────────────────────────────────────",
            s["total_nodes"], s["avg_confidence"],
            s["total_edges"], s["by_type"],
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self):
        data = {
            "saved_at": datetime.utcnow().isoformat(),
            "nodes": [n.to_dict() for n in self._nodes.values()],
        }
        tmp = self._graph_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # Windows: os.replace fails with WinError 32 if another thread holds the
        # file open — retry with backoff instead of crashing.
        for _attempt in range(6):
            try:
                os.replace(tmp, self._graph_path)
                break
            except OSError:
                if _attempt < 5:
                    _time.sleep(0.15 * (_attempt + 1))
        logger.debug("Brain graph saved (%d nodes)", len(self._nodes))

    def load(self):
        if not os.path.exists(self._graph_path):
            logger.info("No brain graph found – starting with empty mind")
            return
        try:
            with open(self._graph_path, encoding="utf-8") as f:
                data = json.load(f)
            for nd in data.get("nodes", []):
                node = KnowledgeNode.from_dict(nd)
                self._nodes[node.node_id] = node
                self._index_node(node)
            logger.info(
                "Brain graph loaded: %d nodes from %s",
                len(self._nodes), self._graph_path,
            )
        except Exception as exc:
            logger.error("Failed to load brain graph: %s", exc)

    # ── Private ───────────────────────────────────────────────────────────────

    def _save_incremental(self):
        """Save only if the graph is large enough to be worth it."""
        if len(self._nodes) % 5 == 0:  # every 5 new nodes
            self.save()

    def _index_node(self, node: KnowledgeNode):
        self._type_index[node.node_type.value].append(node.node_id)
        for tag in node.tags:
            self._tag_index[tag].append(node.node_id)
        key = node.asset or "universal"
        self._asset_index[key].append(node.node_id)

    def _find_similar(
        self, concept: str, node_type: NodeType, asset: Optional[str]
    ) -> Optional[KnowledgeNode]:
        """Check if a sufficiently similar node already exists."""
        concept_lower = concept.lower()
        for node in self._nodes.values():
            if node.node_type != node_type:
                continue
            if node.asset != asset:
                continue
            # Simple Jaccard-like word overlap
            existing_words = set(node.concept.lower().split())
            new_words = set(concept_lower.split())
            if not new_words:
                continue
            overlap = len(existing_words & new_words) / len(existing_words | new_words)
            if overlap > 0.70:
                return node
        return None

    def _auto_connect(self, new_node: KnowledgeNode):
        """
        When a new node is planted, automatically find related existing nodes
        and grow branches between them.
        """
        new_words = set(new_node.concept.lower().split())
        new_tags = set(new_node.tags)

        for node in self._nodes.values():
            if node.node_id == new_node.node_id:
                continue

            # Tag overlap → correlates
            tag_overlap = len(set(node.tags) & new_tags)
            if tag_overlap >= 2:
                new_node.connect(node.node_id, EdgeType.CORRELATES, 0.3 + tag_overlap * 0.1)

            # Asset match + type proximity → strong correlation
            if (
                node.asset == new_node.asset
                and node.node_type == new_node.node_type
            ):
                words = set(node.concept.lower().split())
                overlap = len(words & new_words) / (len(words | new_words) + 1)
                if overlap > 0.30:
                    new_node.connect(node.node_id, EdgeType.CORRELATES, overlap)

            # Calendar events precede price moves
            if (
                new_node.node_type == NodeType.CALENDAR_EVENT
                and node.node_type in (NodeType.PATTERN, NodeType.EXPERIENCE)
            ):
                new_node.connect(node.node_id, EdgeType.PRECEDES, 0.4)

    def _prune_node(self, node_id: str):
        del self._nodes[node_id]
        # Remove from indices
        for idx in (self._tag_index, self._type_index, self._asset_index):
            for lst in idx.values():
                if node_id in lst:
                    lst.remove(node_id)
        # Remove edges pointing to this node
        for node in self._nodes.values():
            node.edges = [e for e in node.edges if e.target_id != node_id]

    def _seed_root_knowledge(self):
        """
        Plant the initial core knowledge that every trader knows.
        These are the primary roots. They'll grow stronger (or weaker)
        as the AI gains experience.
        """
        if len(self._nodes) > 0:
            return  # already has knowledge, don't overwrite

        seeds = [
            KnowledgeNode(
                node_type=NodeType.RULE,
                concept="RSI above 70 indicates overbought conditions, price may reverse down",
                confidence=0.65,
                evidence_count=5,
                source="seed",
                tags=["rsi", "overbought", "reversal", "sell_signal"],
                data={"indicator": "rsi", "threshold": 70, "direction": "sell"},
            ),
            KnowledgeNode(
                node_type=NodeType.RULE,
                concept="RSI below 30 indicates oversold conditions, price may reverse up",
                confidence=0.65,
                evidence_count=5,
                source="seed",
                tags=["rsi", "oversold", "reversal", "buy_signal"],
                data={"indicator": "rsi", "threshold": 30, "direction": "buy"},
            ),
            KnowledgeNode(
                node_type=NodeType.RULE,
                concept="MACD line crossing above signal line is a bullish signal",
                confidence=0.60,
                evidence_count=5,
                source="seed",
                tags=["macd", "crossover", "bullish", "buy_signal"],
                data={"indicator": "macd", "event": "bullish_cross"},
            ),
            KnowledgeNode(
                node_type=NodeType.RULE,
                concept="MACD line crossing below signal line is a bearish signal",
                confidence=0.60,
                evidence_count=5,
                source="seed",
                tags=["macd", "crossover", "bearish", "sell_signal"],
                data={"indicator": "macd", "event": "bearish_cross"},
            ),
            KnowledgeNode(
                node_type=NodeType.RULE,
                concept="Price touching upper Bollinger Band with RSI overbought suggests reversal down",
                confidence=0.62,
                evidence_count=4,
                source="seed",
                tags=["bollinger", "rsi", "overbought", "sell_signal", "reversal"],
                data={"indicators": ["bb_upper", "rsi"], "direction": "sell"},
            ),
            KnowledgeNode(
                node_type=NodeType.RULE,
                concept="Price touching lower Bollinger Band with RSI oversold suggests reversal up",
                confidence=0.62,
                evidence_count=4,
                source="seed",
                tags=["bollinger", "rsi", "oversold", "buy_signal", "reversal"],
                data={"indicators": ["bb_lower", "rsi"], "direction": "buy"},
            ),
            KnowledgeNode(
                node_type=NodeType.RULE,
                concept="ADX above 25 confirms strong trend in current direction",
                confidence=0.58,
                evidence_count=3,
                source="seed",
                tags=["adx", "trend", "strength"],
                data={"indicator": "adx", "threshold": 25},
            ),
            KnowledgeNode(
                node_type=NodeType.RULE,
                concept="Narrow Bollinger Bands squeeze indicates upcoming volatility breakout",
                confidence=0.55,
                evidence_count=3,
                source="seed",
                tags=["bollinger", "squeeze", "volatility", "breakout"],
                data={"indicator": "bb_width", "event": "squeeze"},
            ),
            KnowledgeNode(
                node_type=NodeType.MARKET_FACT,
                concept="NFP (Non-Farm Payrolls) release causes high volatility in USD pairs",
                confidence=0.80,
                evidence_count=10,
                source="seed",
                tags=["nfp", "news", "volatility", "usd", "risk"],
                data={"event": "NFP", "effect": "high_volatility"},
            ),
            KnowledgeNode(
                node_type=NodeType.MARKET_FACT,
                concept="Trading during major economic releases increases risk significantly",
                confidence=0.75,
                evidence_count=8,
                source="seed",
                tags=["news", "risk", "economic_calendar", "avoid"],
                data={"recommendation": "reduce_position_or_avoid"},
            ),
        ]

        for seed in seeds:
            self._nodes[seed.node_id] = seed
            self._index_node(seed)

        self._auto_connect_all_seeds()
        self.save()
        logger.info("Planted %d seed knowledge nodes (primary roots)", len(seeds))

    def _auto_connect_all_seeds(self):
        """Connect seed nodes to each other where appropriate."""
        nodes = list(self._nodes.values())
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i + 1:]:
                tags1 = set(n1.tags)
                tags2 = set(n2.tags)
                overlap = len(tags1 & tags2)
                if overlap >= 2:
                    n1.connect(n2.node_id, EdgeType.CORRELATES, 0.3 + overlap * 0.1)

    # ── Inference engine ──────────────────────────────────────────────────────

    def infer_from_node(
        self,
        node_id: str,
        relation_type: str = "causes",
        max_hops: int = 2,
    ) -> List[Tuple[KnowledgeNode, float]]:
        """
        Follow CAUSES/PRECEDES edges outward from a node and collect inferred nodes.

        Each hop reduces confidence by 30 %:
          hop 1: original_conf * 0.70
          hop 2: original_conf * 0.49

        Parameters
        ----------
        node_id       : starting node
        relation_type : edge type to follow ("causes" or "precedes"); case-insensitive
        max_hops      : maximum depth to traverse (default 2)

        Returns
        -------
        List of (KnowledgeNode, inferred_confidence) — does NOT include the start node
        """
        start = self._nodes.get(node_id)
        if start is None:
            logger.debug("infer_from_node: node %s not found", node_id)
            return []

        # Normalise to EdgeType values we want to follow
        target_types: set = set()
        rt_lower = relation_type.lower()
        if rt_lower in ("causes", "all"):
            target_types.add(EdgeType.CAUSES.value)
        if rt_lower in ("precedes", "all"):
            target_types.add(EdgeType.PRECEDES.value)
        if not target_types:
            # Fall back: follow whatever was requested by string
            target_types.add(rt_lower)

        # BFS with confidence decay
        results: List[Tuple[KnowledgeNode, float]] = []
        visited: set = {node_id}
        # queue items: (current_node_id, current_inferred_conf, depth)
        queue = [(node_id, start.confidence, 0)]

        while queue:
            cur_id, cur_conf, depth = queue.pop(0)
            cur_node = self._nodes.get(cur_id)
            if cur_node is None:
                continue

            if depth >= max_hops:
                continue

            for edge in cur_node.edges:
                if edge.edge_type.value not in target_types:
                    continue
                if edge.target_id in visited:
                    continue
                target = self._nodes.get(edge.target_id)
                if target is None:
                    continue

                visited.add(edge.target_id)
                # Confidence decays 30 % per hop
                inferred_conf = round(cur_conf * 0.70, 4)
                results.append((target, inferred_conf))

                logger.debug(
                    "Inferred: %s → %s (conf=%.3f, hop=%d)",
                    cur_id, edge.target_id, inferred_conf, depth + 1,
                )
                queue.append((edge.target_id, inferred_conf, depth + 1))

        return results

    def query_with_inference(
        self,
        tags: Optional[List[str]] = None,
        min_confidence: float = 0.40,
        limit: int = 10,
    ) -> List[KnowledgeNode]:
        """
        Like query() but also follows CAUSES/PRECEDES chains from matched nodes.

        Inferred nodes are added with reduced confidence (minimum threshold 0.30).
        The combined list is deduplicated and sorted by (confidence × activation).

        Parameters
        ----------
        tags           : tag list for initial query
        min_confidence : minimum confidence for directly matched nodes
        limit          : maximum nodes to return (applies after dedup/sort)

        Returns
        -------
        List[KnowledgeNode] sorted by score, up to `limit` entries
        """
        _INFER_MIN_CONF = 0.30  # floor for inferred nodes

        # Direct matches
        direct = self.query(tags=tags, min_confidence=min_confidence, limit=limit)

        seen_ids: set = {n.node_id for n in direct}
        combined: List[KnowledgeNode] = list(direct)

        # Gather inferences from each direct match
        for node in direct:
            for relation in (EdgeType.CAUSES.value, EdgeType.PRECEDES.value):
                inferred_pairs = self.infer_from_node(
                    node.node_id, relation_type=relation, max_hops=2
                )
                for inf_node, inf_conf in inferred_pairs:
                    if inf_node.node_id in seen_ids:
                        continue
                    if inf_conf < _INFER_MIN_CONF:
                        continue
                    seen_ids.add(inf_node.node_id)
                    # Temporarily override activation-weighted score via a shallow copy
                    # We do NOT mutate the stored node; sort key uses inf_conf directly.
                    combined.append(inf_node)

        # Sort by effective score: for inferred nodes we approximate with their
        # current confidence × activation (same metric as query()).
        combined.sort(key=lambda n: n.confidence * n.activation, reverse=True)

        logger.debug(
            "query_with_inference: %d direct + inferred → %d total (limit=%d)",
            len(direct), len(combined), limit,
        )
        return combined[:limit]

    def get_causal_chain(self, node_id: str) -> str:
        """
        Return a human-readable causal chain starting from node_id.

        Example output:
          "RSI Oversold → PRECEDES → Bullish Reversal (conf=0.71) → CAUSES → Price Rally (conf=0.50)"

        Follows the single strongest CAUSES or PRECEDES edge at each hop,
        up to 3 hops, to keep the description concise.
        """
        start = self._nodes.get(node_id)
        if start is None:
            return f"(node {node_id} not found)"

        _CAUSAL_TYPES = {EdgeType.CAUSES.value, EdgeType.PRECEDES.value}
        MAX_HOPS = 3

        parts = [_short_concept(start.concept)]
        current_id = node_id
        visited = {node_id}

        for _ in range(MAX_HOPS):
            current = self._nodes.get(current_id)
            if current is None:
                break

            # Find the strongest CAUSES or PRECEDES edge
            best_edge = None
            for edge in current.edges:
                if edge.edge_type.value not in _CAUSAL_TYPES:
                    continue
                if edge.target_id in visited:
                    continue
                if best_edge is None or edge.strength > best_edge.strength:
                    best_edge = edge

            if best_edge is None:
                break

            target = self._nodes.get(best_edge.target_id)
            if target is None:
                break

            visited.add(best_edge.target_id)
            relation_label = best_edge.edge_type.value.upper()
            inferred_conf  = round(target.confidence * 0.70, 2)

            parts.append(f"→ {relation_label} →")
            parts.append(f"{_short_concept(target.concept)} (conf={inferred_conf:.2f})")

            current_id = best_edge.target_id

        chain = " ".join(parts)
        logger.debug("Causal chain for %s: %s", node_id, chain)
        return chain


def _short_concept(concept: str, max_len: int = 40) -> str:
    """Truncate a concept string for display purposes."""
    return concept if len(concept) <= max_len else concept[:max_len - 1] + "…"
