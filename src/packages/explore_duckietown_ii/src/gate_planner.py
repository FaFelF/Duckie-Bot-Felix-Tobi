#!/usr/bin/env python3
"""
Plant den Torlauf: eine vorgegebene Reihenfolge von Tor-AprilTag-IDs (5-13) wird
ueber den (nach dem Mapping-Lauf vollstaendigen) Graphen in eine Fahrroute
uebersetzt. Diese Phase ist laut Aufgabenstellung zeitkritisch -- der Pfad muss
algorithmisch aus dem Graphen entstehen (nicht hard-gecodet), soll aber nicht
zwingend global optimal sein: "kuerzeste Anzahl Kreuzungen zur naechsten
Graphkante", bei mehreren gleich kurzen Optionen zufaellig gewaehlt.

Pro Ziel-Tag:
  1. Kante mit diesem gate_tag im Graphen nachschlagen (eindeutig, da jeder Tag
     nur einmal vorkommt).
  2. Von der aktuellen Position zum naeheren der beiden Kanten-Enden fahren
     (kuerzeste Kreuzungsanzahl, Zufallsentscheid bei Gleichstand).
  3. Die Ziel-Kante selbst durchqueren (das eigentliche Tor).
"""

import random
from typing import List, Optional, Tuple

from graph import Graph, Hop, Plan
from mapping_planner import shortest_path_edges


def build_gate_plan(graph: Graph, tag_sequence: List[int], start_node: str,
                     start_exit: int, seed: Optional[int] = None
                     ) -> Tuple[Plan, List[int]]:
    """
    Baut den Torlauf-Plan: Start ueber start_exit an start_node, danach die
    Tore in tag_sequence in genau dieser Reihenfolge.

    Gibt (plan, target_hop_indices) zurueck: target_hop_indices sind die
    Indizes in plan, die die BEABSICHTIGTEN Tor-Durchfahrten markieren (in
    Reihenfolge von tag_sequence). Das ist noetig, weil eine Durchgangsfahrt
    zwischen zwei Toren zufaellig ueber die Kante eines noch nicht faelligen
    (oder schon erledigten) Tors fuehren kann -- das zaehlt NICHT als
    Tor-Durchfahrt, auch wenn die Kante denselben gate_tag traegt.
    """
    rng = random.Random(seed)
    plan: Plan = []
    target_hop_indices: List[int] = []

    first_edge = graph.nodes[start_node].exits.get(start_exit)
    if first_edge is None:
        raise ValueError(f"Keine Kante an Ausfahrt {start_exit} von Knoten {start_node}")

    hop = graph.hop_from_edge(first_edge, start_node)
    plan.append(hop)
    current = hop.to_node
    # Ausfahrt, durch die wir am aktuellen Knoten hereingekommen sind. Noetig, damit kein
    # Wege-Abschnitt eine 180-Grad-Wende verlangt -- die kann der Bot nicht fahren
    # (relative_direction kennt nur links/geradeaus/rechts).
    entry_exit = hop.to_exit

    # Die Startkante kann selbst schon das erste Ziel-Tor tragen (z.B. Start an C ueber
    # Ausfahrt 4 = Kante mit Tor 9, erstes Ziel ist Tor 9). Dann ist es bereits durchfahren.
    # Ohne diese Pruefung galt es weiter als offen, und weil ein direktes Zurueck eine
    # Wende waere, fuhr der Bot erst eine komplette Schleife, um dieselbe Kante nochmal
    # zu nehmen -- drei ueberfluessige Hops, bevor der Lauf ueberhaupt begann.
    pending = list(tag_sequence)
    if pending and graph.edges[first_edge].gate_tag == pending[0]:
        target_hop_indices.append(0)
        pending.pop(0)

    for tag in pending:
        edge = graph.edge_by_tag(tag)
        if edge is None:
            raise ValueError(f"Kein Tor mit Tag {tag} im Graphen -- Mapping unvollstaendig?")

        # Zu einem Kanten-Ende fahren, von dem aus das Tor OHNE Wende befahrbar ist.
        # next_edge=edge.id erledigt beides in einem: die Anfahrt und (falls noetig)
        # einen Umweg, wenn wir gerade ueber genau diese Kante hereingekommen sind.
        routes = []
        for endpoint in (edge.node_a, edge.node_b):
            try:
                routes.append(shortest_path_edges(graph, current, endpoint, rng,
                                                  entry_exit, next_edge=edge.id))
            except ValueError:
                continue
        if not routes:
            raise ValueError(
                f"Tor {tag} (Kante {edge.id}) von {current} aus nicht wendefrei erreichbar.")
        shortest = min(len(r) for r in routes)
        route = rng.choice([r for r in routes if len(r) == shortest])

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
    """
    Findet Hops, die ein Tor (gate_tag gesetzt) durchqueren, OHNE als
    beabsichtigte Ziel-Durchfahrt geplant zu sein -- z.B. weil die kuerzeste
    Verbindung zwischen zwei Toren zufaellig ueber ein drittes Tor fuehrt.
    Gibt (hop_index, gate_tag) Paare zurueck. Kein Fehler an sich, aber
    relevant, falls die Streckenwertung jede Tor-Durchquerung einzeln zaehlt.
    """
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
