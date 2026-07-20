#!/usr/bin/env python3
"""
Plant eine Fahrroute, die im bekannten Graphen JEDE Kante mindestens einmal
befaehrt (Mapping-Lauf) -- damit an jeder Kante die dort verbaute Tor-AprilTag-ID
beobachtet werden kann.

Algorithmus (Edge-Coverage per DFS mit Backtracking):
  1. An der aktuellen Kreuzung eine noch unbefahrene Kante waehlen (zufaellig bei
     mehreren Optionen) und darueber fahren.
  2. Gibt es an der aktuellen Kreuzung keine unbefahrene Kante mehr, zum naechsten
     Knoten zurueckfahren, der noch eine unbefahrene Kante besitzt -- ueber den
     kuerzesten bekannten Weg (Kreuzungsanzahl), da die Topologie ja bereits
     bekannt ist. Dabei incidentell mitbefahrene, bisher unbefahrene Kanten
     zaehlen ebenfalls als erledigt.
  3. Wiederholen, bis alle Kanten befahren wurden.

Das ist bewusst kein exaktes Route-Inspection-Problem (kuerzeste GESAMT-Route) --
der Mapping-Lauf ist laut Aufgabenstellung nicht zeitkritisch, es soll nur nicht
unnoetig lang gefahren werden. Bei mehreren gleich guten Optionen wird zufaellig
gewaehlt (siehe Aufgabenstellung).
"""

import random
from collections import deque
from typing import List, Optional

from graph import Graph, Hop, Plan


def _shuffled(items: list, rng: random.Random) -> list:
    items = list(items)
    rng.shuffle(items)
    return items


def shortest_path_edges(graph: Graph, start: str, goal: str,
                         rng: random.Random) -> List[int]:
    """
    Kuerzester Weg (kleinste Anzahl Kreuzungen) von start zu goal, als Liste von
    edge_ids. Leer, wenn start == goal. Zufaellige Reihenfolge der Nachbarn ->
    bei mehreren gleich kurzen Wegen wird zufaellig einer gefunden.
    """
    if start == goal:
        return []

    visited = {start}
    queue = deque([(start, [])])
    while queue:
        node, path = queue.popleft()
        for edge_id, neighbor in _shuffled(list(graph.neighbors(node)), rng):
            if neighbor in visited:
                continue
            new_path = path + [edge_id]
            if neighbor == goal:
                return new_path
            visited.add(neighbor)
            queue.append((neighbor, new_path))

    raise ValueError(f"Kein Weg von {start} nach {goal} gefunden (Graph nicht zusammenhaengend?)")


def _has_unvisited_edge(graph: Graph, node: str, unvisited: set) -> bool:
    return any(edge_id in unvisited for edge_id, _ in graph.neighbors(node))


def find_nearest_frontier(graph: Graph, start: str, unvisited: set,
                           rng: random.Random) -> Optional[str]:
    """
    Naechster Knoten (per Kreuzungsanzahl von start aus), der noch mindestens
    eine unbefahrene Kante besitzt. None, wenn keiner mehr existiert.
    """
    if _has_unvisited_edge(graph, start, unvisited):
        return start

    visited = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for edge_id, neighbor in _shuffled(list(graph.neighbors(node)), rng):
            if neighbor in visited:
                continue
            if _has_unvisited_edge(graph, neighbor, unvisited):
                return neighbor
            visited.add(neighbor)
            queue.append(neighbor)

    return None


def build_mapping_plan(graph: Graph, start_node: str, start_exit: int,
                        seed: Optional[int] = None) -> Plan:
    """
    Baut den Mapping-Plan: Startet an start_node ueber start_exit und deckt
    danach alle Kanten des Graphen ab.
    """
    rng = random.Random(seed)
    unvisited = set(graph.edges.keys())
    plan: Plan = []

    first_edge = graph.nodes[start_node].exits.get(start_exit)
    if first_edge is None:
        raise ValueError(f"Keine Kante an Ausfahrt {start_exit} von Knoten {start_node}")

    hop = graph.hop_from_edge(first_edge, start_node)
    plan.append(hop)
    unvisited.discard(first_edge)
    current = hop.to_node

    while unvisited:
        candidates = [edge_id for edge_id, _ in graph.neighbors(current) if edge_id in unvisited]

        if candidates:
            edge_id = rng.choice(candidates)
            hop = graph.hop_from_edge(edge_id, current)
            plan.append(hop)
            unvisited.discard(edge_id)
            current = hop.to_node
            continue

        # keine unbefahrene Kante hier -> zum naechsten Knoten mit unbefahrener
        # Kante zurueckfahren (rein bekannte Kanten, keine neue Tag-Info zu
        # erwarten, ausser zufaellig unterwegs doch eine unbefahrene Kante
        # mitgenommen wird -- zaehlt dann ebenfalls als erledigt).
        target = find_nearest_frontier(graph, current, unvisited, rng)
        if target is None:
            break  # sollte bei zusammenhaengendem Graphen nie eintreten

        for edge_id in shortest_path_edges(graph, current, target, rng):
            hop = graph.hop_from_edge(edge_id, current)
            plan.append(hop)
            unvisited.discard(edge_id)
            current = hop.to_node

    return plan


if __name__ == "__main__":
    import os

    CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "known_map.json")
    g = Graph.load(CONFIG)

    plan = build_mapping_plan(g, start_node="A", start_exit=1, seed=0)

    print(f"Plan mit {len(plan)} Hops:")
    for h in plan:
        print(" ", h)

    covered = {h.edge_id for h in plan}
    missing = set(g.edges.keys()) - covered
    print(f"\nAbgedeckte Kanten: {len(covered)}/{len(g.edges)}")
    if missing:
        print(f"FEHLEND: {missing}")
    else:
        print("Alle Kanten abgedeckt.")
