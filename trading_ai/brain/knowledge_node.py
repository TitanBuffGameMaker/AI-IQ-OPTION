"""
KnowledgeNode – one unit of knowledge in the brain tree.

Like a single root node: it starts small, grows stronger when confirmed,
branches out when connected to new knowledge, and weakens when contradicted.
"""
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class NodeType(str, Enum):
    # What kind of knowledge this node holds
    PATTERN = "pattern"          # technical indicator pattern (e.g. RSI>70 + MACD neg)
    RULE = "rule"                # "when X happens, Y tends to follow"
    MARKET_FACT = "market_fact"  # factual info (e.g. "EUR weakens on ECB hike")
    NEWS_EVENT = "news_event"    # something that happened in the world
    CALENDAR_EVENT = "calendar"  # scheduled economic release
    EXPERIENCE = "experience"    # outcome of a real trade
    SENTIMENT = "sentiment"      # market mood / bias at a point in time
    HYPOTHESIS = "hypothesis"    # AI's own conjecture, unconfirmed
    TECHNIQUE = "technique"      # indicator-based technique (RSI divergence, MACD cross)
    STRATEGY_CONCEPT = "strategy_concept"  # higher-level strategy (mean reversion, breakout)
    RISK_CONCEPT = "risk_concept"  # risk-management idea (position sizing, drawdown)
    PSYCHOLOGY = "psychology"    # trader-psychology insight (fear/greed, discipline)


class EdgeType(str, Enum):
    CAUSES = "causes"
    CONFIRMS = "confirms"
    CONTRADICTS = "contradicts"
    PRECEDES = "precedes"        # A happens before B
    CORRELATES = "correlates"
    SPAWNED = "spawned"          # parent node created this child


@dataclass
class Edge:
    target_id: str
    edge_type: EdgeType
    strength: float = 0.5        # 0.0 (weak) … 1.0 (strong)
    evidence_count: int = 1

    def reinforce(self, amount: float = 0.05):
        self.strength = min(1.0, self.strength + amount)
        self.evidence_count += 1

    def weaken(self, amount: float = 0.05):
        self.strength = max(0.0, self.strength - amount)


@dataclass
class KnowledgeNode:
    """
    One node in the knowledge tree (root).

    confidence: how certain we are this knowledge is valid (0–1)
    evidence_count: total confirmations seen
    activation: recency score – fades with time, spikes when referenced
    """
    node_type: NodeType
    concept: str                     # human-readable description
    data: Dict[str, Any] = field(default_factory=dict)  # structured payload
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    confidence: float = 0.5          # starts neutral
    evidence_count: int = 1
    contradiction_count: int = 0
    activation: float = 1.0          # decays over time

    source: str = "internal"         # "internal" | "internet" | "trade_result"
    asset: Optional[str] = None      # which asset this applies to (None = universal)

    edges: List[Edge] = field(default_factory=list)

    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    tags: List[str] = field(default_factory=list)

    # ── Growth / decay ──────────────────────────────────────────────────────

    def confirm(self, strength: float = 0.08):
        """Evidence confirms this knowledge → root grows stronger."""
        self.confidence = min(1.0, self.confidence + strength * (1 - self.confidence))
        self.evidence_count += 1
        self.activation = 1.0
        self.last_updated = datetime.utcnow().isoformat()

    def contradict(self, strength: float = 0.06):
        """Evidence contradicts this knowledge → root weakens, but survives."""
        self.confidence = max(0.05, self.confidence - strength * self.confidence)
        self.contradiction_count += 1
        self.activation = min(self.activation + 0.3, 1.0)  # contradictions are memorable
        self.last_updated = datetime.utcnow().isoformat()

    def tick_decay(self, rate: float = 0.01):
        """Called each time step – activation fades if not reinforced."""
        self.activation = max(0.1, self.activation - rate)

    def is_reliable(self, threshold: float = 0.60) -> bool:
        return self.confidence >= threshold and self.evidence_count >= 3

    def is_stale(self) -> bool:
        return self.activation < 0.15

    # ── Edges (branches) ────────────────────────────────────────────────────

    def connect(self, target_id: str, edge_type: EdgeType, strength: float = 0.5):
        """Grow a branch to another node."""
        for edge in self.edges:
            if edge.target_id == target_id and edge.edge_type == edge_type:
                edge.reinforce()
                return
        self.edges.append(Edge(target_id=target_id, edge_type=edge_type, strength=strength))

    def get_strong_edges(self, min_strength: float = 0.4) -> List[Edge]:
        return [e for e in self.edges if e.strength >= min_strength]

    # ── Serialization ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "concept": self.concept,
            "data": self.data,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "contradiction_count": self.contradiction_count,
            "activation": self.activation,
            "source": self.source,
            "asset": self.asset,
            "edges": [
                {
                    "target_id": e.target_id,
                    "edge_type": e.edge_type.value,
                    "strength": e.strength,
                    "evidence_count": e.evidence_count,
                }
                for e in self.edges
            ],
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KnowledgeNode":
        node = cls(
            node_type=NodeType(d["node_type"]),
            concept=d["concept"],
            data=d.get("data", {}),
            node_id=d["node_id"],
            confidence=d.get("confidence", 0.5),
            evidence_count=d.get("evidence_count", 1),
            contradiction_count=d.get("contradiction_count", 0),
            activation=d.get("activation", 1.0),
            source=d.get("source", "internal"),
            asset=d.get("asset"),
            created_at=d.get("created_at", datetime.utcnow().isoformat()),
            last_updated=d.get("last_updated", datetime.utcnow().isoformat()),
            tags=d.get("tags", []),
        )
        for ed in d.get("edges", []):
            node.edges.append(Edge(
                target_id=ed["target_id"],
                edge_type=EdgeType(ed["edge_type"]),
                strength=ed.get("strength", 0.5),
                evidence_count=ed.get("evidence_count", 1),
            ))
        return node

    def __repr__(self):
        return (
            f"[{self.node_type.value.upper()}] {self.concept[:60]} "
            f"(conf={self.confidence:.2f}, ev={self.evidence_count})"
        )
