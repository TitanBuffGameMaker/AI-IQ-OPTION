"""
Brain tree visualizer – prints a text-based view of the knowledge root system.

Shows the graph as an ASCII tree so you can watch it grow over time.
Optionally exports to JSON for external visualization tools.
"""
import json
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional

from trading_ai.brain.knowledge_graph import KnowledgeGraph
from trading_ai.brain.knowledge_node import KnowledgeNode, NodeType

logger = logging.getLogger(__name__)

NODE_ICONS = {
    NodeType.PATTERN:       "◆",
    NodeType.RULE:          "⚙",
    NodeType.MARKET_FACT:   "★",
    NodeType.NEWS_EVENT:    "📰",
    NodeType.CALENDAR_EVENT:"📅",
    NodeType.EXPERIENCE:    "◉",
    NodeType.SENTIMENT:     "◐",
    NodeType.HYPOTHESIS:    "?",
}

CONF_BARS = ["░", "▒", "▓", "█"]


def _conf_bar(confidence: float, width: int = 8) -> str:
    """Visual bar showing confidence level."""
    filled = int(confidence * width)
    bar = ""
    for i in range(width):
        if i < filled:
            if confidence > 0.75:
                bar += "█"
            elif confidence > 0.50:
                bar += "▓"
            elif confidence > 0.30:
                bar += "▒"
            else:
                bar += "░"
        else:
            bar += "·"
    return bar


def _activation_marker(activation: float) -> str:
    if activation > 0.8:
        return "◉"  # highly active
    if activation > 0.5:
        return "○"  # active
    return "·"      # dormant


class BrainVisualizer:
    """Renders the knowledge graph as ASCII art and exports to JSON."""

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph

    def print_tree(
        self,
        root_type: Optional[NodeType] = None,
        min_confidence: float = 0.40,
        max_depth: int = 2,
        max_nodes: int = 30,
    ):
        """
        Print the knowledge tree to the console.
        Groups nodes by type; shows children via edges.
        """
        nodes = list(self.graph._nodes.values())
        nodes = [n for n in nodes if n.confidence >= min_confidence]
        nodes.sort(key=lambda n: n.confidence * n.activation, reverse=True)

        if root_type:
            nodes = [n for n in nodes if n.node_type == root_type]

        nodes = nodes[:max_nodes]

        print("\n" + "═" * 65)
        print("  BRAIN KNOWLEDGE TREE")
        print("═" * 65)

        # Group by type
        by_type: Dict[str, List[KnowledgeNode]] = defaultdict(list)
        for node in nodes:
            by_type[node.node_type.value].append(node)

        for type_name, group in sorted(by_type.items()):
            icon = NODE_ICONS.get(NodeType(type_name), "•")
            print(f"\n  {icon} {type_name.upper()} ({len(group)} nodes)")
            print("  " + "─" * 55)

            for node in group[:8]:  # max 8 per type
                act = _activation_marker(node.activation)
                bar = _conf_bar(node.confidence)
                concept_short = node.concept[:50]
                if len(node.concept) > 50:
                    concept_short += "…"
                print(
                    f"  {act} [{bar}] {concept_short}"
                    f"  (ev={node.evidence_count})"
                )

                # Show strong branches
                strong_edges = node.get_strong_edges(min_strength=0.45)
                if strong_edges and max_depth > 0:
                    for i, edge in enumerate(strong_edges[:3]):
                        child = self.graph.get_node(edge.target_id)
                        if child:
                            prefix = "     └─" if i == len(strong_edges) - 1 else "     ├─"
                            edge_bar = _conf_bar(edge.strength, width=4)
                            child_short = child.concept[:35]
                            if len(child.concept) > 35:
                                child_short += "…"
                            print(f"  {prefix} [{edge_bar}] {child_short}")

        print("\n" + "═" * 65)
        stats = self.graph.stats()
        print(
            f"  Roots: {stats['total_nodes']}  |  Branches: {stats['total_edges']}"
            f"  |  Avg confidence: {stats['avg_confidence']:.2f}"
        )
        print("═" * 65 + "\n")

    def print_summary_bar(self):
        """One-line status bar showing brain health."""
        stats = self.graph.stats()
        conf_bar = _conf_bar(stats["avg_confidence"], width=10)
        node_types = stats.get("by_type", {})
        type_str = " ".join(
            f"{t[0].upper()}:{c}" for t, c in sorted(node_types.items())
        )
        logger.info(
            "Brain [%s] conf=%.2f | nodes=%d branches=%d | %s",
            conf_bar, stats["avg_confidence"],
            stats["total_nodes"], stats["total_edges"],
            type_str,
        )

    def export_json(self, output_path: str):
        """Export graph to JSON for external visualization (e.g. D3.js)."""
        nodes_data = []
        edges_data = []

        for node in self.graph._nodes.values():
            nodes_data.append({
                "id": node.node_id,
                "label": node.concept[:60],
                "type": node.node_type.value,
                "confidence": round(node.confidence, 3),
                "activation": round(node.activation, 3),
                "evidence": node.evidence_count,
                "source": node.source,
                "asset": node.asset or "universal",
                "tags": node.tags[:5],
            })
            for edge in node.edges:
                if edge.strength >= 0.3:
                    edges_data.append({
                        "source": node.node_id,
                        "target": edge.target_id,
                        "type": edge.edge_type.value,
                        "strength": round(edge.strength, 3),
                    })

        export = {
            "nodes": nodes_data,
            "edges": edges_data,
            "stats": self.graph.stats(),
        }

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2, ensure_ascii=False)

        logger.info(
            "Brain graph exported → %s (%d nodes, %d edges)",
            output_path, len(nodes_data), len(edges_data),
        )
        return output_path

    def most_active_roots(self, n: int = 5) -> List[KnowledgeNode]:
        """Return the most currently active knowledge nodes."""
        nodes = sorted(
            self.graph._nodes.values(),
            key=lambda node: node.confidence * node.activation,
            reverse=True,
        )
        return nodes[:n]
