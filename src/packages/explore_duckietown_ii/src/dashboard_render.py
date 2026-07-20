#!/usr/bin/env python3
"""
Reine Rendering-Logik fuers Dashboard (siehe dashboard_node.py). Getrennt von
der ROS-Glue, damit sie ohne rospy lokal testbar ist.

Zeigt in einem Bild:
  a) die Karte (alle Kanten/Knoten des Graphen),
  b) den gewaehlten Pfad (Kanten, die im Plan vorkommen, hervorgehoben),
  c) die aktuelle Position (aktueller Hop, falls step gegeben).
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from graph import Graph, Hop

MAP_SIZE = 700
PANEL_WIDTH = 300
MARGIN = 0.82
PARALLEL_EDGE_STEP = 22  # Pixelabstand zwischen parallelen Kanten (z.B. direkte Verbindung + Bogen)


def to_pixel(pos: Tuple[float, float]) -> Tuple[int, int]:
    x, y = pos
    half = MAP_SIZE / 2
    px = half + x * half * MARGIN
    py = half - y * half * MARGIN
    return int(round(px)), int(round(py))


def bezier_points(p1, p2, control, num=16):
    ts = np.linspace(0, 1, num)
    pts = [((1 - t) ** 2 * p1[0] + 2 * (1 - t) * t * control[0] + t ** 2 * p2[0],
            (1 - t) ** 2 * p1[1] + 2 * (1 - t) * t * control[1] + t ** 2 * p2[1]) for t in ts]
    return np.array(pts, dtype=np.int32)


def _group_parallel_edges(graph: Graph) -> Dict[frozenset, List[int]]:
    groups: Dict[frozenset, List[int]] = {}
    for edge in graph.edges.values():
        key = frozenset((edge.node_a, edge.node_b))
        groups.setdefault(key, []).append(edge.id)
    return groups


def _edge_offset(edge_id: int, edge_groups: Dict[frozenset, List[int]]) -> float:
    for ids in edge_groups.values():
        if edge_id in ids:
            count = len(ids)
            i = ids.index(edge_id)
            return (i - (count - 1) / 2.0) * PARALLEL_EDGE_STEP
    return 0.0


def _draw_edge(canvas, edge, layout, edge_groups, color, thickness):
    p1 = to_pixel(layout[edge.node_a])
    p2 = to_pixel(layout[edge.node_b])
    offset = _edge_offset(edge.id, edge_groups)
    if abs(offset) < 1e-6:
        cv2.line(canvas, p1, p2, color, thickness)
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    else:
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = max((dx ** 2 + dy ** 2) ** 0.5, 1e-6)
        perp = (-dy / length, dx / length)
        mid_x, mid_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        control = (mid_x + perp[0] * offset, mid_y + perp[1] * offset)
        pts = bezier_points(p1, p2, control)
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=thickness)
        mid = tuple(int(c) for c in pts[len(pts) // 2])
    return mid


def build_dashboard_image(graph: Graph, plan: List[Hop], step: Optional[int],
                           layout: Dict[str, Tuple[float, float]], run_label: str = "") -> np.ndarray:
    """
    step: roher Plan-Index (PlanState.step) oder None, wenn noch keine Nachricht
    empfangen wurde (Lauf startet gerade erst). step >= len(plan) heisst "Ziel
    erreicht" und wird explizit so angezeigt statt einen IndexError zu werfen.
    """
    path_edge_ids = {hop.edge_id for hop in plan}
    edge_groups = _group_parallel_edges(graph)

    current_hop = plan[step] if (step is not None and 0 <= step < len(plan)) else None
    current_edge_id = current_hop.edge_id if current_hop else None

    canvas = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)

    for edge in graph.edges.values():
        if edge.id == current_edge_id:
            color, thickness = (0, 220, 255), 5
        elif edge.id in path_edge_ids:
            color, thickness = (255, 140, 0), 3
        else:
            color, thickness = (110, 110, 110), 2
        mid = _draw_edge(canvas, edge, layout, edge_groups, color, thickness)

        if edge.gate_tag is not None:
            cv2.putText(canvas, f"T{edge.gate_tag}", (mid[0] + 4, mid[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    for node_id, pos in layout.items():
        p = to_pixel(pos)
        is_current_node = current_hop is not None and current_hop.from_node == node_id
        color = (0, 220, 255) if is_current_node else (200, 200, 200)
        radius = 22 if is_current_node else 16
        cv2.circle(canvas, p, radius, color, -1)
        cv2.circle(canvas, p, radius, (0, 0, 0), 2)
        cv2.putText(canvas, str(node_id), (p[0] - 8, p[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    panel = np.zeros((MAP_SIZE, PANEL_WIDTH, 3), dtype=np.uint8)
    y = 35
    if run_label:
        cv2.putText(panel, f"Lauf: {run_label}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 35

    cv2.putText(panel, "Aktueller Knoten:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y += 30
    if current_hop is not None:
        node_text = current_hop.from_node
    elif step is not None:
        node_text = "Ziel erreicht"
    else:
        node_text = "--"
    cv2.putText(panel, str(node_text), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
    y += 45

    cv2.putText(panel, "Aktuelle Kante:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y += 30
    if current_edge_id is not None:
        tag = graph.edges[current_edge_id].gate_tag
        edge_text = f"{current_edge_id}" + (f" (Tor {tag})" if tag is not None else "")
    else:
        edge_text = "--"
    cv2.putText(panel, edge_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
    y += 50

    cv2.line(panel, (10, y), (PANEL_WIDTH - 10, y), (80, 80, 80), 1)
    y += 35
    cv2.putText(panel, "Plan-Fortschritt:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y += 30
    if plan:
        shown_step = min(step, len(plan)) if step is not None else 0
        step_text = f"{shown_step} / {len(plan)}"
    else:
        step_text = "--"
    cv2.putText(panel, step_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    y += 45

    cv2.putText(panel, f"Kanten gesamt: {len(graph.edges)}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    y += 25
    cv2.putText(panel, f"Kanten im Pfad: {len(path_edge_ids)}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    return np.hstack([canvas, panel])
