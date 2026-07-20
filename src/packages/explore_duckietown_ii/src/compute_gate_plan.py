#!/usr/bin/env python3
"""
Berechnet den Torlauf-Plan aus dem (nach dem Mapping-Lauf vollstaendigen)
Graphen und der vorgegebenen Tor-Reihenfolge, speichert ihn als plan.json.

Laeuft VOR dem eigentlichen Torlauf (kein ROS noetig, reine Planung).

Aufruf:
    python3 compute_gate_plan.py --map mapped_map.json --tags 12,7,5 \\
        --start-node A --start-exit 1 --output plan.json
"""

import argparse

from graph import Graph, save_plan
from gate_planner import build_gate_plan, incidental_gate_crossings


def parse_tags(value: str):
    return [int(t) for t in value.split(',') if t.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', required=True, help='Pfad zum gemappten Graphen (JSON, mit gesetzten gate_tags)')
    parser.add_argument('--tags', required=True, type=parse_tags, help='Vorgegebene Tor-Reihenfolge, z.B. 12,7,5')
    parser.add_argument('--start-node', required=True)
    parser.add_argument('--start-exit', required=True, type=int)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    graph = Graph.load(args.map)

    missing_tags = [t for t in args.tags if graph.edge_by_tag(t) is None]
    if missing_tags:
        raise SystemExit(f"FEHLER: Tags {missing_tags} sind im Graphen '{args.map}' nicht zugeordnet "
                          f"-- Mapping-Lauf unvollstaendig oder falsche Datei?")

    plan, target_hop_indices = build_gate_plan(
        graph, args.tags, start_node=args.start_node, start_exit=args.start_exit, seed=args.seed)
    save_plan(plan, args.output)

    print(f"Torlauf-Plan mit {len(plan)} Hops gespeichert: {args.output}")
    print(f"Beabsichtigte Tor-Reihenfolge: {args.tags} (Hop-Indizes: {target_hop_indices})")

    incidental = incidental_gate_crossings(plan, target_hop_indices, graph)
    if incidental:
        print(f"Hinweis: {len(incidental)} unbeabsichtigte Nebenkreuzung(en) anderer Tore waehrend Transitfahrten.")


if __name__ == '__main__':
    main()
