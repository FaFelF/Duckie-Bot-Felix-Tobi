#!/usr/bin/env python3
"""
Kodiert die bekannte Stadt-Topologie aus dem Beispielgraphen (Folie "Mapping &
Path Finding") als Graph und speichert sie als JSON fuer die spaetere Nutzung
durch Mapping-Lauf und Torlauf.

Ausfahrt-Konvention (fix, global, unabhaengig von der Anfahrtsrichtung):
    1 = Osten, 2 = Norden, 3 = Westen, 4 = Sueden

Referenz-Dictionary von der Folie (aequivalent zu den add_edge-Aufrufen unten,
jede Kante kommt dort einmal pro Endpunkt vor -> hier einmal pro Strasse):
    A: {1:(B,1), 2:(C,2), 3:(C,1), 4:(B,2)}
    B: {1:(A,1), 2:(A,4), 3:(C,4)}
    C: {1:(A,3), 2:(A,2), 4:(B,3)}

gate_tag bleibt bei allen Kanten auf None -- die Zuordnung der Tor-AprilTags
(5-13) zu den Kanten passiert erst im Mapping-Lauf ueber graph.set_gate_tag().
"""

import os

from graph import Graph

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'known_map.json')


def build_known_map() -> Graph:
    g = Graph()

    # A ist eine Vierweg-Kreuzung (alle 4 Ausfahrten belegt).
    # A und B sowie A und C sind jeweils ueber ZWEI Strassen verbunden
    # (direkte Verbindung + langer Bogen) -> bewusst als Parallelkanten,
    # kein Hilfsknoten noetig.

    g.add_edge(node_a="A", exit_a=1, node_b="B", exit_b=1)  # A-Ost <-> B-Ost: langer Bogen rechts rum
    g.add_edge(node_a="A", exit_a=2, node_b="C", exit_b=2)  # A-Nord <-> C-Nord: Bogen oben rum
    g.add_edge(node_a="A", exit_a=3, node_b="C", exit_b=1)  # A-West <-> C-Ost: direkte kurze Verbindung
    g.add_edge(node_a="A", exit_a=4, node_b="B", exit_b=2)  # A-Sued <-> B-Nord: direkte kurze Verbindung

    # B fehlt die Sued-Ausfahrt (T-Kreuzung), C fehlt die West-Ausfahrt (T-Kreuzung).
    g.add_edge(node_a="B", exit_a=3, node_b="C", exit_b=4)  # B-West <-> C-Sued: Bogen unten/links rum

    return g


if __name__ == "__main__":
    graph = build_known_map()
    graph.save(CONFIG_PATH)
    print(f"Bekannte Karte gespeichert: {CONFIG_PATH}")
    print(f"Knoten: {list(graph.nodes.keys())}")
    print(f"Kanten: {len(graph.edges)}")
    for edge in graph.edges.values():
        print(f"  {edge}")
