#!/usr/bin/env python3
"""
Kodiert die (alte 3-Knoten-)Beispieltopologie von Hand und speichert sie als
known_map.json. Ausfahrt-Konvention: 1=Ost, 2=Nord, 3=West, 4=Sued.
gate_tags bleiben None (werden erst im Mapping-Lauf gesetzt).
"""

import os

from graph import Graph

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'known_map.json')


def build_known_map() -> Graph:
    g = Graph()

    # A = Vierweg-Kreuzung; A-B und A-C jeweils ueber zwei parallele Strassen.
    g.add_edge(node_a="A", exit_a=1, node_b="B", exit_b=1)
    g.add_edge(node_a="A", exit_a=2, node_b="C", exit_b=2)
    g.add_edge(node_a="A", exit_a=3, node_b="C", exit_b=1)
    g.add_edge(node_a="A", exit_a=4, node_b="B", exit_b=2)

    # B und C sind T-Kreuzungen (B ohne Sued, C ohne West).
    g.add_edge(node_a="B", exit_a=3, node_b="C", exit_b=4)

    return g


if __name__ == "__main__":
    graph = build_known_map()
    graph.save(CONFIG_PATH)
    print(f"Bekannte Karte gespeichert: {CONFIG_PATH}")
    print(f"Knoten: {list(graph.nodes.keys())}")
    print(f"Kanten: {len(graph.edges)}")
    for edge in graph.edges.values():
        print(f"  {edge}")
