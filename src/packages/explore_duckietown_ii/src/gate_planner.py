#!/usr/bin/env python3
"""
Torlauf-Plan: uebersetzt eine vorgegebene Tor-Reihenfolge (Tags 5-13) in eine
Fahrroute durch den gemappten Graphen. Zeitkritisch -> schnellster Weg zu jedem Tor.
Pro Ziel-Tag: Kante per gate_tag suchen, wendefrei zu einem Ende fahren, Kante queren.
"""

import random
from typing import List, Optional, Tuple

from graph import Graph, Hop, Plan
from mapping_planner import shortest_path_edges, path_weight


def build_gate_plan(graph: Graph, tag_sequence: List[int], start_node: str,
                     start_exit: int, seed: Optional[int] = None
                     ) -> Tuple[Plan, List[int]]:
    """
    Start ueber start_exit an start_node, dann die Tore in tag_sequence der Reihe nach.
    Gibt (plan, target_hop_indices) zurueck: die Indizes der BEABSICHTIGTEN Tor-Durchfahrten
    (eine Transitfahrt kann zufaellig ueber ein anderes Tor fuehren, das zaehlt nicht).
    """
    rng = random.Random(seed)
    plan: Plan = []
    target_hop_indices: List[int] = []

    # Gewichte = gemessene Fahrzeiten -> schnellster Weg. Ohne Messung None (Kreuzungsanzahl).
    weights = {e.id: e.travel_time for e in graph.edges.values() if e.travel_time is not None}
    weights = weights or None
    default_w = (graph.mean_travel_time() or 1.0) if weights is not None else 1.0

    first_edge = graph.nodes[start_node].exits.get(start_exit)
    if first_edge is None:
        raise ValueError(f"Keine Kante an Ausfahrt {start_exit} von Knoten {start_node}")

    hop = graph.hop_from_edge(first_edge, start_node)
    plan.append(hop)
    current = hop.to_node
    # Einfahrt am aktuellen Knoten, damit kein Wegabschnitt eine Wende verlangt.
    entry_exit = hop.to_exit

    # Startkante kann schon das erste Ziel-Tor tragen -> dann gilt es als bereits durchfahren
    # (sonst faehrt der Bot eine unnoetige Schleife, um dieselbe Kante nochmal zu nehmen).
    pending = list(tag_sequence)
    if pending and graph.edges[first_edge].gate_tag == pending[0]:
        target_hop_indices.append(0)
        pending.pop(0)

    for tag in pending:
        edge = graph.edge_by_tag(tag)
        if edge is None:
            raise ValueError(f"Kein Tor mit Tag {tag} im Graphen -- Mapping unvollstaendig?")

        # Zu einem Kanten-Ende fahren, von dem das Tor wendefrei befahrbar ist (next_edge).
        routes = []
        for endpoint in (edge.node_a, edge.node_b):
            try:
                routes.append(shortest_path_edges(graph, current, endpoint, rng,
                                                  entry_exit, next_edge=edge.id,
                                                  weights=weights))
            except ValueError:
                continue
        if not routes:
            raise ValueError(
                f"Tor {tag} (Kante {edge.id}) von {current} aus nicht wendefrei erreichbar.")
        # Schnellster (nicht kuerzester) Weg zum Tor; bei Gleichstand zufaellig.
        best_cost = min(path_weight(r, weights, default_w) for r in routes)
        route = rng.choice([r for r in routes
                            if abs(path_weight(r, weights, default_w) - best_cost) < 1e-9])

        for edge_id in route:
            hop = graph.hop_from_edge(edge_id, current)
            plan.append(hop)
            current = hop.to_node
            entry_exit = hop.to_exit

        hop = graph.hop_from_edge(edge.id, current)
        plan.append(hop)
        target_hop_indices.append(len(plan) - 1)
        current = hop.to_node
        entry_exit = hop.to_exit

    return plan, target_hop_indices




def incidental_gate_crossings(plan: Plan, target_hop_indices: List[int],
                               graph: Graph) -> List[Tuple[int, int]]:
    """Hops, die zufaellig ein Tor queren, ohne beabsichtigte Ziel-Durchfahrt zu sein.
    Gibt (hop_index, gate_tag) zurueck."""
    target_set = set(target_hop_indices)
    result = []
    for i, hop in enumerate(plan):
        if i in target_set:
            continue
        tag = graph.edges[hop.edge_id].gate_tag
        if tag is not None:
            result.append((i, tag))
    return result


if __name__ == "__main__":
    import os

    CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "known_map.json")
    g = Graph.load(CONFIG)

    # Mapping-Ergebnis simulieren (im echten Ablauf kommt das aus dem Mapping-Lauf)
    g.set_gate_tag(4, 7)   # Kante B<->C
    g.set_gate_tag(2, 12)  # Kante A<->C (direkt)
    g.set_gate_tag(0, 5)   # Kante A<->B (langer Bogen)

    tag_sequence = [12, 7, 5]

    plan, target_hop_indices = build_gate_plan(g, tag_sequence, start_node="A", start_exit=1, seed=0)

    print(f"Plan mit {len(plan)} Hops:")
    for i, h in enumerate(plan):
        tag = g.edges[h.edge_id].gate_tag
        if i in target_hop_indices:
            marker = f"  <-- Tor {tag} (Ziel Nr. {target_hop_indices.index(i) + 1})"
        elif tag is not None:
            marker = f"  (durchfahren, Tor {tag} nicht an der Reihe)"
        else:
            marker = ""
        print(f"  {h}{marker}")

    order_found = [g.edges[plan[i].edge_id].gate_tag for i in target_hop_indices]
    print(f"\nTor-Reihenfolge der BEABSICHTIGTEN Durchfahrten: {order_found}")
    assert order_found == tag_sequence, "Reihenfolge stimmt nicht mit Vorgabe ueberein!"
    print("Reihenfolge stimmt mit Vorgabe ueberein.")

    incidental = incidental_gate_crossings(plan, target_hop_indices, g)
    if incidental:
        print(f"\nHinweis: {len(incidental)} unbeabsichtigte Tor-Durchquerung(en) waehrend "
              f"Durchgangsfahrten (kein Fehler, aber ggf. relevant fuer die Wertung):")
        for i, tag in incidental:
            print(f"  Hop {i}: {plan[i]} quert zufaellig Tor {tag}")
    else:
        print("\nKeine unbeabsichtigten Tor-Durchquerungen.")
