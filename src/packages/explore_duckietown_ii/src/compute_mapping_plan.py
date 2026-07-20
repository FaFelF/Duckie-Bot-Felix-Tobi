#!/usr/bin/env python3
"""
Berechnet den Mapping-Plan aus der bekannten Stadt-Topologie und speichert ihn
als plan.json, damit switch_control_node.py ihn beim Start laden kann.

Laeuft VOR dem eigentlichen Mapping-Lauf (kein ROS noetig, reine Planung).

Aufruf:
    python3 compute_mapping_plan.py --known-map known_map.json \\
        --start-node A --start-exit 1 --output plan.json
"""

import argparse

from graph import Graph, save_plan
from mapping_planner import build_mapping_plan, validate_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--known-map', required=True, help='Pfad zur bekannten Stadt-Topologie (JSON)')
    parser.add_argument('--start-node', required=True, help='Knoten-ID, an der die Fahrt beginnt')
    parser.add_argument('--start-exit', required=True, type=int, help='Ausfahrt (1-4), in die zuerst gefahren wird')
    parser.add_argument('--output', required=True, help='Zielpfad fuer den berechneten Plan (JSON)')
    parser.add_argument('--seed', type=int, default=None, help='Zufalls-Seed (optional, fuer reproduzierbare Plaene)')
    args = parser.parse_args()

    graph = Graph.load(args.known_map)
    plan = build_mapping_plan(graph, start_node=args.start_node, start_exit=args.start_exit, seed=args.seed)
    # Lieber hier abbrechen als einen unfahrbaren Plan speichern (Wende -> der Bot
    # bleibt an der betroffenen Kreuzung haengen).
    validate_plan(graph, plan)
    save_plan(plan, args.output)

    covered = {h.edge_id for h in plan}
    print(f"Mapping-Plan mit {len(plan)} Hops gespeichert: {args.output}")
    print(f"Kantenabdeckung: {len(covered)}/{len(graph.edges)}")
    missing = set(graph.edges.keys()) - covered
    if missing:
        print(f"WARNUNG: nicht abgedeckte Kanten: {missing}")


if __name__ == '__main__':
    main()
