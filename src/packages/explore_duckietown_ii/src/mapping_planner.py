#!/usr/bin/env python3
"""
Mapping-Plan: Route, die jede Kante mindestens einmal befaehrt (damit jede Tor-ID
gesehen wird). Edge-Coverage per DFS mit Backtracking; an Sackgassen ueber den
kuerzesten bekannten Weg zum naechsten Knoten mit unbefahrener Kante. Nicht
zeitkritisch, bei Gleichstand zufaellige Wahl.
"""

import random
from collections import deque
import heapq
from typing import List, Optional

from graph import Graph, Hop, Plan


def _shuffled(items: list, rng: random.Random) -> list:
    items = list(items)
    rng.shuffle(items)
    return items


def is_uturn(graph: Graph, node: str, entry_exit: int, edge_id: int) -> bool:
    """Wuerde das Verlassen ueber `edge_id` eine 180-Grad-Wende bedeuten? Die kann der
    Bot nicht (nur links/geradeaus/rechts), muss also ausgeschlossen werden."""
    if entry_exit is None:
        return False
    return graph.nodes[node].exits.get(entry_exit) == edge_id


def path_weight(path: List[int], weights: Optional[dict], default: float = 1.0) -> float:
    """Gesamtkosten eines Wegs. Ohne weights = Anzahl Kanten (Kreuzungsanzahl)."""
    if weights is None:
        return float(len(path))
    return sum(weights.get(eid, default) for eid in path)


def shortest_path_edges(graph: Graph, start: str, goal: str,
                         rng: random.Random, entry_exit: Optional[int] = None,
                         next_edge: Optional[int] = None,
                         weights: Optional[dict] = None) -> List[int]:
    """
    Kuerzester wendefreier Weg start->goal als Liste von edge_ids. Suche laeuft ueber
    Zustaende (Knoten, Einfahrt), weil Erreichbarkeit von der Einfahrt abhaengt.

    entry_exit: Einfahrt an `start`.
    next_edge:  Kante nach dem Weg; der Zielzustand muss sie ohne Wende erlauben.
    weights:    edge_id -> Kosten (Fahrzeit); ohne -> jede Kante 1 (Kreuzungsanzahl).
    Bei Gleichstand zufaellige Wahl.
    """
    def reached(node: str, ex: Optional[int]) -> bool:
        if node != goal:
            return False
        return next_edge is None or not is_uturn(graph, node, ex, next_edge)

    if reached(start, entry_exit):
        return []

    # Dijkstra ueber (Knoten, Einfahrt), Ziel-Pruefung beim Poppen. rng.random() = Tie-Break.
    default = (graph.mean_travel_time() or 1.0) if weights is not None else 1.0
    best = {(start, entry_exit): 0.0}
    heap = [(0.0, rng.random(), start, entry_exit, [])]
    while heap:
        cost, _, node, ex, path = heapq.heappop(heap)
        if reached(node, ex):
            return path
        if cost > best.get((node, ex), float('inf')):
            continue
        for edge_id, _n in _shuffled(list(graph.neighbors(node)), rng):
            if is_uturn(graph, node, ex, edge_id):
                continue
            hop = graph.hop_from_edge(edge_id, node)
            state = (hop.to_node, hop.to_exit)
            w = 1.0 if weights is None else weights.get(edge_id, default)
            new_cost = cost + w
            if new_cost < best.get(state, float('inf')):
                best[state] = new_cost
                heapq.heappush(heap, (new_cost, rng.random(), hop.to_node, hop.to_exit,
                                      path + [edge_id]))

    raise ValueError(
        f"Kein wendefreier Weg von {start} nach {goal} gefunden "
        f"(Einfahrt {entry_exit}, danach Kante {next_edge}).")


def _has_unvisited_edge(graph: Graph, node: str, unvisited: set) -> bool:
    return any(edge_id in unvisited for edge_id, _ in graph.neighbors(node))


def find_nearest_frontier(graph: Graph, start: str, unvisited: set,
                           rng: random.Random) -> Optional[str]:
    """Naechster Knoten (per Kreuzungsanzahl) mit noch unbefahrener Kante. None wenn keiner."""
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
    entry_exit = hop.to_exit   # durch diese Ausfahrt sind wir hereingekommen

    while unvisited:
        # Wende ausschliessen (Bot kann nur links/geradeaus/rechts).
        candidates = [edge_id for edge_id, _ in graph.neighbors(current)
                      if edge_id in unvisited and not is_uturn(graph, current, entry_exit, edge_id)]

        if candidates:
            edge_id = rng.choice(candidates)
            hop = graph.hop_from_edge(edge_id, current)
            plan.append(hop)
            unvisited.discard(edge_id)
            current = hop.to_node
            entry_exit = hop.to_exit
            continue

        # Sackgasse -> zum naechsten Knoten mit unbefahrener Kante zurueckfahren.
        target = find_nearest_frontier(graph, current, unvisited, rng)
        if target is None:
            break  # sollte bei zusammenhaengendem Graphen nie eintreten

        # entry_exit mitgeben, damit auch der Rueckweg keine Wende verlangt.
        for edge_id in shortest_path_edges(graph, current, target, rng, entry_exit):
            hop = graph.hop_from_edge(edge_id, current)
            plan.append(hop)
            unvisited.discard(edge_id)
            current = hop.to_node
            entry_exit = hop.to_exit

    return plan


def validate_plan(graph: Graph, plan: Plan) -> None:
    """Prueft, dass der Plan keine 180-Grad-Wende verlangt (die bleibt auf der Strecke haengen)."""
    for i in range(len(plan) - 1):
        if plan[i].to_exit == plan[i + 1].from_exit:
            raise ValueError(
                f"Plan verlangt eine Wende an Knoten {plan[i].to_node} "
                f"(Schritt {i}->{i+1}): herein ueber Ausfahrt {plan[i].to_exit}, "
                f"hinaus ueber Ausfahrt {plan[i + 1].from_exit}. "
                f"Der Bot kann nur links/geradeaus/rechts.")


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
