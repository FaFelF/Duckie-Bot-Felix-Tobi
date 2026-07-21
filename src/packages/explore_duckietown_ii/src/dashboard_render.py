#!/usr/bin/env python3
"""
Reine Rendering-Logik fuers Dashboard (siehe dashboard_node.py). Getrennt von
der ROS-Glue, damit sie ohne rospy lokal testbar ist.

Zeigt in einem Bild:
  a) die Karte (alle Kanten/Knoten des Graphen),
  b) den gewaehlten Pfad (Kanten, die im Plan vorkommen, hervorgehoben),
  c) die aktuelle Position (aktueller Hop, falls step gegeben).
"""

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from graph import Direction, Graph, Hop, relative_direction

MAP_SIZE = 700
PANEL_WIDTH = 300
MARGIN = 0.82
PARALLEL_EDGE_STEP = 64  # Pixelabstand zwischen parallelen Kanten (z.B. direkte Verbindung + Bogen)


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


# Farbrollen (BGR). Bewusst wenige, feste Rollen statt bunter Kanten: die Farbe sagt
# WAS die Kante gerade ist, nicht welche Nummer sie hat.
COL_SURFACE      = (32, 30, 28)
COL_EDGE_OTHER   = (95, 92, 88)     # nicht im Plan -> zuruecknehmen
COL_EDGE_PATH    = (190, 130, 55)   # im Plan
COL_EDGE_CURRENT = (60, 200, 255)   # gerade befahren
COL_NODE         = (70, 66, 62)
COL_NODE_CURRENT = (60, 200, 255)
COL_TEXT         = (238, 238, 238)
COL_TEXT_MUTED   = (155, 152, 148)
COL_EXIT         = (150, 210, 120)  # Ausgangsnummern
COL_GATE         = (120, 255, 255)  # Tor-Tags
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _chip(canvas, text, center, fg, scale=0.42, pad=4):
    """Text mit dunklem Kaestchen dahinter -- sonst ist er auf einer Kante unlesbar."""
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    x, y = int(center[0] - tw / 2), int(center[1] + th / 2)
    cv2.rectangle(canvas, (x - pad, y - th - pad), (x + tw + pad, y + pad), COL_SURFACE, -1)
    cv2.putText(canvas, text, (x, y), FONT, scale, fg, 1, cv2.LINE_AA)


def _edge_points(edge, layout, edge_groups):
    """Polylinie der Kante von node_a nach node_b (gerade oder Bogen bei Parallelkanten)."""
    p1 = to_pixel(layout[edge.node_a])
    p2 = to_pixel(layout[edge.node_b])
    offset = _edge_offset(edge.id, edge_groups)
    if abs(offset) < 1e-6:
        return np.array([p1, p2], dtype=np.int32)
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = max((dx ** 2 + dy ** 2) ** 0.5, 1e-6)
    perp = (-dy / length, dx / length)
    control = ((p1[0] + p2[0]) / 2 + perp[0] * offset, (p1[1] + p2[1]) / 2 + perp[1] * offset)
    return bezier_points(p1, p2, control, num=32)


def _draw_edge(canvas, edge, layout, edge_groups, color, thickness):
    pts = _edge_points(edge, layout, edge_groups)
    cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=thickness,
                  lineType=cv2.LINE_AA)
    return tuple(int(c) for c in pts[len(pts) // 2])


def _point_along(pts, dist):
    """Punkt in `dist` Pixeln Bogenlaenge entlang der Polylinie -- mit Interpolation
    INNERHALB der Segmente. Nur die Stuetzpunkte abzulaufen reicht nicht: eine gerade
    Kante hat nur zwei davon, da landet man sofort am anderen Ende."""
    acc = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        seg = float(np.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1])))
        if seg < 1e-6:
            continue
        if acc + seg >= dist:
            t = (dist - acc) / seg
            return int(a[0] + t * (b[0] - a[0])), int(a[1] + t * (b[1] - a[1]))
        acc += seg
    return int(pts[-1][0]), int(pts[-1][1])


def _departure_point(edge, node_id, layout, edge_groups, dist=52):
    """
    Punkt kurz hinter dem Knoten IN Fahrtrichtung dieser Kante -- dorthin kommt die
    Ausgangsnummer. Ueber die echte Polylinie, damit die Nummer bei gebogenen
    Parallelkanten auch wirklich am richtigen Ast steht.
    """
    pts = _edge_points(edge, layout, edge_groups).astype(float)
    if node_id == edge.node_b:
        pts = pts[::-1]
    return _point_along(pts, dist)


def _label_point(edge, layout, edge_groups, shift=16):
    """Mittelpunkt der Kante, SENKRECHT nach aussen versetzt. Die Richtung folgt dem
    Bogen-Offset der Kante -- sonst wandern die Labels zweier Parallelkanten beide in
    dieselbe Richtung und ruecken wieder zusammen, statt sich zu trennen."""
    pts = _edge_points(edge, layout, edge_groups).astype(float)
    total = float(sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts[:-1], pts[1:])))
    mx, my = _point_along(pts, total / 2)
    ax, ay = _point_along(pts, max(0.0, total / 2 - 12))
    bx, by = _point_along(pts, min(total, total / 2 + 12))
    dx, dy = bx - ax, by - ay
    n = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    offset = _edge_offset(edge.id, edge_groups)
    s = shift if offset == 0 else math.copysign(shift, offset)
    return int(mx - dy / n * s), int(my + dx / n * s)


def _exit_label_positions(graph, node_id, layout, edge_groups,
                           radius=64, min_sep_deg=36):
    """
    Positionen der Ausgangsnummern rund um eine Kreuzung.

    Nicht einfach ein Punkt auf der Kante: Parallelkanten starten am Knoten praktisch
    gleich und trennen sich erst in der Mitte -- die Nummern lagen dadurch uebereinander.
    Stattdessen der WINKEL der Abfahrtsrichtung, auf einem Kreis um den Knoten, und
    Winkel, die zu dicht beieinander liegen, werden auseinandergeschoben.
    """
    p0 = to_pixel(layout[node_id])
    items = []
    for exit_no, edge_id in sorted(graph.nodes[node_id].exits.items()):
        if edge_id is None:
            continue
        dp = _departure_point(graph.edges[edge_id], node_id, layout, edge_groups, dist=80)
        items.append([exit_no, math.degrees(math.atan2(dp[1] - p0[1], dp[0] - p0[0]))])
    if not items:
        return {}

    items.sort(key=lambda t: t[1])
    n = len(items)
    if n > 1 and n * min_sep_deg < 360:
        for _ in range(80):
            moved = False
            for i in range(n):
                j = (i + 1) % n
                gap = (items[j][1] - items[i][1]) % 360
                if gap < min_sep_deg:
                    push = (min_sep_deg - gap) / 2.0
                    items[i][1] -= push
                    items[j][1] += push
                    moved = True
            if not moved:
                break

    return {ex: (int(p0[0] + math.cos(math.radians(a)) * radius),
                 int(p0[1] + math.sin(math.radians(a)) * radius))
            for ex, a in items}


def build_dashboard_image(graph: Graph, plan: List[Hop], step: Optional[int],
                           layout: Dict[str, Tuple[float, float]], run_label: str = "",
                           gate_targets: Optional[dict] = None) -> np.ndarray:
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
    canvas[:] = COL_SURFACE

    # Kanten: erst alle "stillen", dann Pfad, zuletzt die aktuelle -> Wichtiges liegt oben
    order = sorted(graph.edges.values(),
                   key=lambda e: (e.id == current_edge_id, e.id in path_edge_ids))
    mids = {}
    for edge in order:
        if edge.id == current_edge_id:
            color, thickness = COL_EDGE_CURRENT, 6
        elif edge.id in path_edge_ids:
            color, thickness = COL_EDGE_PATH, 3
        else:
            color, thickness = COL_EDGE_OTHER, 2
        mids[edge.id] = _draw_edge(canvas, edge, layout, edge_groups, color, thickness)

    # Ausgangsnummern an den Kreuzungen: zeigen, ueber welche Ausfahrt welche Kante geht.
    # Ohne die laesst sich der Plan ("Ausgang 3") nicht auf die Karte uebertragen.
    for node_id in graph.nodes:
        for exit_no, p in _exit_label_positions(graph, node_id, layout, edge_groups).items():
            _chip(canvas, str(exit_no), p, COL_EXIT, scale=0.45)

    # Kanten-Beschriftung mittig: Kanten-ID (passend zur Panel-Liste) + ggf. Tor-Tag
    for edge in graph.edges.values():
        label = f"K{edge.id}"
        fg = COL_TEXT if edge.id in path_edge_ids else COL_TEXT_MUTED
        if edge.gate_tag is not None:
            label += f" | T{edge.gate_tag}"
            fg = COL_GATE
        # Gemessene Fahrzeit (aus dem Mapping-Lauf) direkt an der Kante -- danach bewertet
        # der gate_planner, welcher Weg schneller ist.
        if edge.travel_time is not None:
            label += f" | {edge.travel_time:.1f}s"
        _chip(canvas, label, _label_point(edge, layout, edge_groups), fg, scale=0.45)

    for node_id, pos in layout.items():
        p = to_pixel(pos)
        is_current = current_hop is not None and current_hop.from_node == node_id
        radius = 26 if is_current else 20
        cv2.circle(canvas, p, radius + 3, COL_SURFACE, -1, cv2.LINE_AA)
        cv2.circle(canvas, p, radius, COL_NODE_CURRENT if is_current else COL_NODE, -1, cv2.LINE_AA)
        cv2.circle(canvas, p, radius, COL_TEXT if is_current else COL_TEXT_MUTED, 2, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(str(node_id), FONT, 0.7, 2)
        cv2.putText(canvas, str(node_id), (p[0] - tw // 2, p[1] + th // 2),
                    FONT, 0.7, COL_SURFACE if is_current else COL_TEXT, 2, cv2.LINE_AA)

    # kleine Legende, damit die Farben ohne Nachfragen lesbar sind
    ly = MAP_SIZE - 58
    for col, txt in ((COL_EDGE_CURRENT, "aktuelle Kante"),
                     (COL_EDGE_PATH, "im Plan"),
                     (COL_EDGE_OTHER, "nicht im Plan")):
        cv2.line(canvas, (16, ly), (44, ly), col, 3, cv2.LINE_AA)
        cv2.putText(canvas, txt, (52, ly + 4), FONT, 0.4, COL_TEXT_MUTED, 1, cv2.LINE_AA)
        ly += 20
    cv2.putText(canvas, "K = Kante   T = Tor   s = Fahrzeit   gruen = Ausgang", (16, MAP_SIZE - 96),
                FONT, 0.4, COL_TEXT_MUTED, 1, cv2.LINE_AA)

    panel = np.zeros((MAP_SIZE, PANEL_WIDTH, 3), dtype=np.uint8)
    y = 35
    if run_label:
        cv2.putText(panel, f"Lauf: {run_label}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 35

    # 1. Wo der Bot laut Plan gerade ist: Knoten UND ueber welchen Ausgang er ihn
    #    verlassen hat -- der Ausgang ist noetig, um die Abbiegerichtung nachzuvollziehen.
    # Vorgegebene Tor-Reihenfolge mit Fortschritt: erledigt (gedimmt), gerade dran
    # (hervorgehoben), noch offen. Nur beim Torlauf vorhanden.
    if gate_targets and gate_targets.get('tags'):
        tags = gate_targets['tags']
        hops = gate_targets.get('hops', [])
        cv2.putText(panel, "Vorgegebene Tor-Reihenfolge:", (10, y), FONT, 0.5, (200, 200, 200), 1)
        y += 30
        x = 12
        for i, tag in enumerate(tags):
            hop_i = hops[i] if i < len(hops) else None
            if step is None or hop_i is None:
                col = COL_TEXT_MUTED
            elif step > hop_i:
                col = (110, 160, 110)          # erledigt
            elif step == hop_i:
                col = COL_EDGE_CURRENT         # gerade dran
            else:
                col = COL_TEXT                 # noch offen
            txt = str(tag)
            (tw, _), _ = cv2.getTextSize(txt, FONT, 0.6, 2)
            cv2.putText(panel, txt, (x, y), FONT, 0.6, col, 2, cv2.LINE_AA)
            x += tw + 5
            if i < len(tags) - 1:
                cv2.putText(panel, ">", (x, y), FONT, 0.45, COL_TEXT_MUTED, 1, cv2.LINE_AA)
                x += 15
        y += 36

    cv2.putText(panel, "Aktueller Knoten / Ausgang:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y += 30
    if current_hop is not None:
        node_text = f"{current_hop.from_node}  (Ausgang {current_hop.from_exit})"
    elif step is not None:
        node_text = "Ziel erreicht"
    else:
        node_text = "--"
    cv2.putText(panel, str(node_text), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2)
    y += 45

    # 3. Zu welchem Knoten der aktuelle Hop fuehrt (ueber welchen Eingang).
    cv2.putText(panel, "Naechster Knoten:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y += 30
    if current_hop is not None:
        next_text = f"{current_hop.to_node}  (Eingang {current_hop.to_exit})"
    else:
        next_text = "--"
    cv2.putText(panel, next_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 140), 2)
    y += 45

    # 2. Richtung, die an der KOMMENDEN Kreuzung gefahren werden muss. Gleiche Rechnung
    #    wie in PlanState.next_direction(): (Zielausfahrt - Einfahrt) mod 4.
    cv2.putText(panel, "Abbiegen an naechster Kreuzung:", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y += 32
    dir_text, dir_color = "--", (150, 150, 150)
    if current_hop is not None and step is not None and step + 1 < len(plan):
        try:
            d = relative_direction(current_hop.to_exit, plan[step + 1].from_exit)
            dir_text = {Direction.Left: "<< LINKS",
                        Direction.Straight: "^ GERADEAUS",
                        Direction.Right: "RECHTS >>"}[d]
            dir_color = (0, 165, 255)
        except ValueError:
            dir_text, dir_color = "PLAN FEHLERHAFT", (0, 0, 255)
    elif current_hop is not None:
        dir_text = "letzter Hop - Ziel"
    cv2.putText(panel, dir_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, dir_color, 2)
    y += 45

    cv2.putText(panel, "Aktuelle Kante:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y += 30
    if current_edge_id is not None:
        cur_edge = graph.edges[current_edge_id]
        edge_text = f"{current_edge_id}"
        if cur_edge.gate_tag is not None:
            edge_text += f" (Tor {cur_edge.gate_tag})"
        if cur_edge.travel_time is not None:
            edge_text += f" {cur_edge.travel_time:.1f}s"
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
    y += 40

    # 4. Bisher gemappte Tore: welche Kante hat schon welches Tor-Tag bekommen
    #    (mapping_recorder_node -> graph.set_gate_tag). Zeigt live den Mapping-Fortschritt.
    cv2.line(panel, (10, y - 20), (PANEL_WIDTH - 10, y - 20), (80, 80, 80), 1)
    mapped = sorted((e.id, e.gate_tag) for e in graph.edges.values() if e.gate_tag is not None)
    cv2.putText(panel, f"Gemappte Tore: {len(mapped)}/{len(graph.edges)}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y += 26
    if not mapped:
        cv2.putText(panel, "noch keine", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
    for edge_id, tag in mapped:
        if y > MAP_SIZE - 12:      # Panel voll -> Rest weglassen statt rauszuzeichnen
            cv2.putText(panel, "...", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            break
        cv2.putText(panel, f"Kante {edge_id}  ->  Tor {tag}", (14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y += 24

    return np.hstack([canvas, panel])
