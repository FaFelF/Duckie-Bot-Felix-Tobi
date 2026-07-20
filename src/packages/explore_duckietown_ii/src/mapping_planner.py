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


def is_uturn(graph: Graph, node: str, entry_exit: int, edge_id: int) -> bool:
    """
    Wuerde das Verlassen von `node` ueber `edge_id` bedeuten, dass der Bot wieder
    durch die Ausfahrt hinausfaehrt, durch die er hereingekommen ist?

    Der Bot kann an einer Kreuzung nur LINKS, GERADEAUS oder RECHTS -- eine 180-Grad-
    Wende gibt es als Kommando nicht (relative_direction wirft bei diff==0). Ein Plan
    mit so einem Uebergang bleibt auf der Strecke haengen, statt nur schlecht zu sein.
    """
    if entry_exit is None:
        return False
    return graph.nodes[node].exits.get(entry_exit) == edge_id


def shortest_path_edges(graph: Graph, start: str, goal: str,
                         rng: random.Random, entry_exit: Optional[int] = None,
                         next_edge: Optional[int] = None) -> List[int]:
    """
    Kuerzester wendefreier Weg von start zu goal, als Liste von edge_ids.

    BFS laeuft ueber ZUSTAENDE (Knoten, Einfahrt) statt nur ueber Knoten. Das ist
    noetig, weil "erreichbar" hier von der Einfahrt abhaengt: derselbe Knoten kann
    ueber die eine Einfahrt eine Weiterfahrt erlauben und ueber die andere nicht.
    Eine reine Knoten-BFS findet dann den kurzen Weg, der in eine Wende laeuft, und
    uebersieht den etwas laengeren, der fahrbar waere.

    entry_exit: Ausfahrt, durch die der Bot an `start` hereingekommen ist.
    next_edge:  Kante, die NACH dem Weg befahren werden soll -- der Zielzustand muss
                sie ohne Wende erlauben. Damit kann derselbe Aufruf auch einen Umweg
                finden, wenn das Ziel direkt auf der Einfahrtskante liegt.
    Zufaellige Nachbar-Reihenfolge -> bei mehreren gleich kurzen Wegen zufaellige Wahl.
    """
    def reached(node: str, ex: Optional[int]) -> bool:
        if node != goal:
            return False
        return next_edge is None or not is_uturn(graph, node, ex, next_edge)

    if reached(start, entry_exit):
        return []

    visited = {(start, entry_exit)}
    queue = deque([(start, entry_exit, [])])
    while queue:
        node, ex, path = queue.popleft()
        for edge_id, _ in _shuffled(list(graph.neighbors(node)), rng):
            if is_uturn(graph, node, ex, edge_id):
                continue
            hop = graph.hop_from_edge(edge_id, node)
            state = (hop.to_node, hop.to_exit)
            if state in visited:
                continue
            new_path = path + [edge_id]
            if reached(hop.to_node, hop.to_exit):
                return new_path
            visited.add(state)
            queue.append((hop.to_node, hop.to_exit, new_path))

    raise ValueError(
        f"Kein wendefreier Weg von {start} nach {goal} gefunden "
        f"(Einfahrt {entry_exit}, danach Kante {next_edge}).")


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
    entry_exit = hop.to_exit   # durch diese Ausfahrt sind wir hereingekommen

    while unvisited:
        # Wende ausschliessen: der Bot kann an einer Kreuzung nur links/geradeaus/rechts.
        # Ohne diese Pruefung waehlte rng.choice u.U. genau die Kante, ueber die er gerade
        # hereingekommen ist -> Plan mit 180-Grad-Wende -> switch_control bleibt dort haengen.
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

        # keine unbefahrene Kante hier -> zum naechsten Knoten mit unbefahrener
        # Kante zurueckfahren (rein bekannte Kanten, keine neue Tag-Info zu
        # erwarten, ausser zufaellig unterwegs doch eine unbefahrene Kante
        # mitgenommen wird -- zaehlt dann ebenfalls als erledigt).
        target = find_nearest_frontier(graph, current, unvisited, rng)
        if target is None:
            break  # sollte bei zusammenhaengendem Graphen nie eintreten

        # Auch beim Zurueckfahren keine Wende: entry_exit mitgeben, damit der erste
        # Schritt des Rueckwegs nicht durch die Einfahrt zurueckgeht.
        for edge_id in shortest_path_edges(graph, current, target, rng, entry_exit):
            hop = graph.hop_from_edge(edge_id, current)
            plan.append(hop)
            unvisited.discard(edge_id)
            current = hop.to_node
            entry_exit = hop.to_exit

    return plan


def validate_plan(graph: Graph, plan: Plan) -> None:
    """
    Prueft, dass der Plan an keiner Kreuzung eine 180-Grad-Wende verlangt. So ein Plan
    laesst switch_control_node an genau dieser Kreuzung haengen -- besser hier beim
    Erzeugen auffallen als erst auf der Strecke.
    """
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
