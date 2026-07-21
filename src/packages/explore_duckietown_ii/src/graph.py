#!/usr/bin/env python3
"""
Graph-Datenstruktur fuer Challenge 4: bekannte Stadt-Topologie (Knoten = Kreuzungen,
Kanten = Strassenabschnitte), Zuordnung der Tor-AprilTags (5-13) zu Kanten, sowie
die Uebersetzung eines Fahrplans (Liste von Hops) in konkrete Abbiege-Kommandos.

Die Ausfahrt-Nummerierung an jeder Kreuzung ist fix: 1 und 3 liegen sich gegenueber,
2 ist rechts von 1, 4 ist links von 1. Fehlende Ausfahrten (T-Kreuzungen) werden als
None in Node.exits gefuehrt.

Bewusst ohne rospy-Abhaengigkeit: Graph-Aufbau, Planung und Dashboard-Logik sollen
auch ohne ROS-Umgebung lokal testbar sein. Die Anbindung an switch_control_node.py
(IntersectionsDirections) erfolgt ueber gleiche numerische Werte in Direction.
"""

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Direction(Enum):
    """Numerisch identisch mit IntersectionsDirections in switch_control_node.py,
    damit .value direkt auf das bestehende Int32-Topic (switch/intersection_direction)
    passt."""
    Left = 0
    Straight = 1
    Right = 2


@dataclass
class Node:
    id: str
    # Ausfahrt-Nr. (1-4) -> edge_id, oder None wenn diese Ausfahrt an dieser
    # Kreuzung physisch nicht existiert.
    exits: Dict[int, Optional[int]] = field(
        default_factory=lambda: {1: None, 2: None, 3: None, 4: None})


@dataclass
class Edge:
    id: int
    node_a: str
    exit_a: int
    node_b: str
    exit_b: int
    gate_tag: Optional[int] = None  # 5-13, erst nach dem Mapping-Lauf gesetzt
    travel_time: Optional[float] = None  # gemessene Fahrzeit (s), erst nach dem Mapping-Lauf


@dataclass
class Hop:
    """Eine gerichtete Durchfahrt einer Kante: von from_node/from_exit nach
    to_node/to_exit. Eine Plan ist eine Liste solcher Hops."""
    edge_id: int
    from_node: str
    from_exit: int
    to_node: str
    to_exit: int


Plan = List[Hop]


class Graph:
    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[int, Edge] = {}
        self._next_edge_id = 0

    # --- Aufbau (die bekannte Stadt-Topologie wird damit von Hand codiert) ---

    def add_node(self, node_id: str) -> Node:
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(id=node_id)
        return self.nodes[node_id]

    def add_edge(self, node_a: str, exit_a: int, node_b: str, exit_b: int,
                 gate_tag: Optional[int] = None,
                 edge_id: Optional[int] = None) -> Edge:
        """
        Verbindet zwei Kreuzungen ueber die angegebenen Ausfahrten. Legt Knoten
        automatisch an, falls sie noch nicht existieren. Wirft einen Fehler,
        wenn eine Ausfahrt beim Hand-Codieren versehentlich doppelt belegt wird.
        """
        if edge_id is None:
            edge_id = self._next_edge_id
        self._next_edge_id = max(self._next_edge_id, edge_id + 1)

        a = self.add_node(node_a)
        b = self.add_node(node_b)

        if a.exits.get(exit_a) not in (None, edge_id):
            raise ValueError(
                f"Ausfahrt {exit_a} an Knoten {node_a} ist bereits belegt "
                f"(edge {a.exits[exit_a]})")
        if b.exits.get(exit_b) not in (None, edge_id):
            raise ValueError(
                f"Ausfahrt {exit_b} an Knoten {node_b} ist bereits belegt "
                f"(edge {b.exits[exit_b]})")

        edge = Edge(id=edge_id, node_a=node_a, exit_a=exit_a,
                    node_b=node_b, exit_b=exit_b, gate_tag=gate_tag)
        self.edges[edge_id] = edge
        a.exits[exit_a] = edge_id
        b.exits[exit_b] = edge_id
        return edge

    # --- Abfragen ---

    def other_end(self, edge_id: int, from_node: str) -> Tuple[str, int]:
        """Gibt (node_id, exit_nr) am anderen Ende der Kante zurueck."""
        e = self.edges[edge_id]
        if e.node_a == from_node:
            return e.node_b, e.exit_b
        if e.node_b == from_node:
            return e.node_a, e.exit_a
        raise ValueError(f"Edge {edge_id} ist nicht mit Knoten {from_node} verbunden")

    def neighbors(self, node_id: str):
        """Liefert (edge_id, nachbar_node_id) fuer alle vorhandenen Ausfahrten."""
        for exit_nr, edge_id in self.nodes[node_id].exits.items():
            if edge_id is not None:
                neighbor, _ = self.other_end(edge_id, node_id)
                yield edge_id, neighbor

    def edge_by_tag(self, tag: int) -> Optional[Edge]:
        """Jeder Tor-Tag kommt genau einmal vor -> eindeutiger Lookup."""
        return next((e for e in self.edges.values() if e.gate_tag == tag), None)

    def set_gate_tag(self, edge_id: int, tag: int) -> None:
        self.edges[edge_id].gate_tag = tag

    def set_travel_time(self, edge_id: int, seconds: float) -> None:
        self.edges[edge_id].travel_time = seconds

    def mean_travel_time(self) -> Optional[float]:
        """Durchschnitt der gemessenen Kanten-Fahrzeiten -- Fallback-Gewicht fuer
        Kanten ohne Messung, damit sie beim Planen weder bevorzugt noch gemieden
        werden. None, wenn noch gar keine Kante gemessen wurde."""
        times = [e.travel_time for e in self.edges.values() if e.travel_time is not None]
        return sum(times) / len(times) if times else None

    def hop_from_edge(self, edge_id: int, from_node: str) -> Hop:
        """Baut einen orientierten Hop fuer die Fahrt ueber edge_id, gestartet
        an from_node. Wird sowohl fuer den Mapping-Plan (Edge-Coverage) als auch
        fuer den Gate-Plan (BFS-Verkettung zwischen Ziel-Kanten) benutzt."""
        edge = self.edges[edge_id]
        from_exit = edge.exit_a if edge.node_a == from_node else edge.exit_b
        to_node, to_exit = self.other_end(edge_id, from_node)
        return Hop(edge_id=edge_id, from_node=from_node, from_exit=from_exit,
                   to_node=to_node, to_exit=to_exit)

    # --- Persistenz ---

    def to_dict(self) -> dict:
        return {
            "nodes": {nid: asdict(n) for nid, n in self.nodes.items()},
            "edges": {eid: asdict(e) for eid, e in self.edges.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Graph":
        g = cls()
        for nid, ndata in data["nodes"].items():
            g.nodes[nid] = Node(
                id=ndata["id"],
                exits={int(k): v for k, v in ndata["exits"].items()})
        for eid, edata in data["edges"].items():
            edge = Edge(**edata)
            g.edges[int(eid)] = edge
            g._next_edge_id = max(g._next_edge_id, edge.id + 1)
        return g

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Graph":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


def save_plan(plan: Plan, path: str) -> None:
    """Speichert einen fertig berechneten Plan (Mapping- oder Gate-Plan) als JSON,
    damit ein ROS-Node ihn beim Start laden kann, ohne den Planer erneut laufen
    zu lassen."""
    with open(path, "w") as f:
        json.dump([asdict(hop) for hop in plan], f, indent=2)


def load_plan(path: str) -> Plan:
    with open(path, "r") as f:
        data = json.load(f)
    return [Hop(**h) for h in data]


def relative_direction(entry_exit: int, target_exit: int) -> Direction:
    """
    Uebersetzt Einfahrt- und Zielausfahrt (beide im selben 1-4-Schema derselben
    Kreuzung) in ein Abbiege-Kommando. diff==0 (Umkehr) darf in einem gueltigen
    Plan nicht vorkommen.
    """
    diff = (target_exit - entry_exit) % 4
    try:
        return {1: Direction.Right, 2: Direction.Straight, 3: Direction.Left}[diff]
    except KeyError:
        raise ValueError(
            f"Ungueltiger Plan: Einfahrt {entry_exit} == Zielausfahrt {target_exit} (Umkehr)")


class PlanState:
    """
    Laufzeit-Zustand beim Abfahren eines Plans (Mapping- oder Torlauf). `step`
    zeigt auf den Hop, der gerade befahren wird, und ist zugleich die einzige
    "Positions"-Information, die ohne Hardware-Lokalisierung verfuegbar ist --
    genau das, was im Dashboard als "wo denkt der Bot dass er ist" angezeigt wird.
    """

    def __init__(self, plan: Plan):
        self.plan = plan
        self.step = 0

    @property
    def finished(self) -> bool:
        return self.step >= len(self.plan)

    @property
    def current_hop(self) -> Optional[Hop]:
        return None if self.finished else self.plan[self.step]

    @property
    def current_node(self) -> Optional[str]:
        hop = self.current_hop
        return hop.from_node if hop else None

    @property
    def current_edge_id(self) -> Optional[int]:
        hop = self.current_hop
        return hop.edge_id if hop else None

    def next_direction(self) -> Optional[Direction]:
        """
        Richtung fuer die Kreuzung am Ende des aktuellen Hops -- die Richtung,
        die beim naechsten erkannten Kreuzungs-Tag ausgefuehrt werden muss.
        None, wenn kein weiterer Hop mehr folgt (Ziel erreicht).
        """
        if self.finished or self.step + 1 >= len(self.plan):
            return None
        current = self.plan[self.step]
        upcoming = self.plan[self.step + 1]
        return relative_direction(current.to_exit, upcoming.from_exit)

    def advance(self) -> None:
        """Wird aufgerufen, sobald die Kante des aktuellen Hops durchfahren
        (und eine evtl. noetige Abbiegung ausgefuehrt) wurde."""
        self.step += 1
