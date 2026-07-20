#!/usr/bin/env python3
"""
Reine Entscheidungslogik: aus mehreren AprilTag-Sichtungen auf einer Kante den
zuverlaessigsten Tor-Tag waehlen. Getrennt von mapping_recorder_node.py, damit
diese Logik ohne ROS testbar ist.
"""

from collections import Counter
from typing import Optional


def pick_confident_tag(counts: Counter, min_sightings: int) -> Optional[int]:
    """
    Gibt den haeufigst gesehenen Tag zurueck, wenn er mindestens min_sightings
    mal gesehen wurde. None, wenn gar keine Sichtungen vorliegen (z.B. Kante
    ohne Tor) oder die haeufigste Sichtung unter der Schwelle bleibt (Rauschen).
    """
    if not counts:
        return None
    tag, count = counts.most_common(1)[0]
    if count < min_sightings:
        return None
    return tag
