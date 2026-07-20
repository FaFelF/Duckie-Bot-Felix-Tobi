#!/usr/bin/env python3
"""
Berechnet eine schematische 2D-Positionierung der Graph-Knoten fuers Dashboard
(Fruchterman-Reingold-artiges Force-Directed-Layout). Bewusst KEINE geografische
Genauigkeit -- die echten Strassen sind teils kurvig/ueber Eck (siehe Parallel-
kanten wie der lange Bogen zwischen zwei Kreuzungen), ein kompass-basiertes
Layout wuerde bei solchen Schleifen inkonsistent. Ein schematisches Layout
(wie ein Metroplan) reicht fuer die Anforderung "Karte anzeigen" voellig aus.

ROS-unabhaengig, nur numpy -- lokal testbar.
"""

from typing import Dict, Tuple

import numpy as np

from graph import Graph


def compute_layout(graph: Graph, iterations: int = 300, seed: int = 0
                    ) -> Dict[str, Tuple[float, float]]:
    """Gibt {node_id: (x, y)} zurueck, Koordinaten normiert auf ca. [-1, 1]."""
    node_ids = list(graph.nodes.keys())
    n = len(node_ids)
    if n == 0:
        return {}
    if n == 1:
        return {node_ids[0]: (0.0, 0.0)}

    rng = np.random.RandomState(seed)
    idx = {nid: i for i, nid in enumerate(node_ids)}
    pos = rng.uniform(-1, 1, size=(n, 2))
    edges = [(idx[e.node_a], idx[e.node_b]) for e in graph.edges.values()]

    k = 1.0 / np.sqrt(n)  # ideale Kantenlaenge (Fruchterman-Reingold-Heuristik)

    for it in range(iterations):
        disp = np.zeros((n, 2))

        # Abstossung zwischen allen Knotenpaaren, damit sich nichts ueberlappt
        for i in range(n):
            delta = pos[i] - pos
            dist = np.linalg.norm(delta, axis=1)
            dist[i] = 1e-6
            dist = np.maximum(dist, 1e-6)
            force = (k * k) / dist
            disp[i] += np.sum((delta.T * (force / dist)).T, axis=0)

        # Anziehung entlang der Kanten (Feder-Modell)
        for a, b in edges:
            delta = pos[a] - pos[b]
            dist = max(float(np.linalg.norm(delta)), 1e-6)
            force = (dist * dist) / k
            direction = delta / dist
            disp[a] -= direction * force
            disp[b] += direction * force

        # Abkuehlung ueber die Iterationen -> konvergiert stabil statt zu oszillieren
        temperature = 0.1 * (1 - it / iterations)
        disp_len = np.linalg.norm(disp, axis=1)
        disp_len = np.maximum(disp_len, 1e-6)
        scale = np.minimum(disp_len, temperature) / disp_len
        pos += disp * scale[:, None]

    pos -= pos.mean(axis=0)
    max_abs = np.max(np.abs(pos))
    if max_abs > 1e-9:
        pos /= max_abs

    return {nid: (float(pos[idx[nid]][0]), float(pos[idx[nid]][1])) for nid in node_ids}


if __name__ == "__main__":
    import os

    CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "known_map.json")
    g = Graph.load(CONFIG)
    layout = compute_layout(g, seed=0)
    print("Layout:")
    for nid, (x, y) in layout.items():
        print(f"  {nid}: ({x:+.3f}, {y:+.3f})")
