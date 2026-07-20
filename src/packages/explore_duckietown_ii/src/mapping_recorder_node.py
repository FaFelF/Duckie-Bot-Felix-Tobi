#!/usr/bin/env python3
"""
Beobachtet waehrend des Mapping-Laufs die erkannten Tor-AprilTags (5-13) und
ordnet sie der aktuell befahrenen Kante zu (graph.set_gate_tag). Ergaenzt
switch_control_node.py, das die aktuelle Kante ueber .../switch/current_edge
veroeffentlicht (siehe _publish_current_edge dort).

Ablauf:
  - Waehrend Modus Lane (=1): jede erkannte Tag-ID im Bereich 5-13 wird fuer
    die aktuell befahrene Kante gezaehlt (Hysterese gegen einzelne
    Fehl-Detektionen -- siehe gate_tag_voting.pick_confident_tag).
  - Wechselt die aktuelle Kante (naechster Hop) oder ist der Plan fertig
    (Modus Stop=3), wird die haeufigste Sichtung der VORHERIGEN Kante als
    deren gate_tag uebernommen.
  - Bei Planende (Stop) wird der so vervollstaendigte Graph gespeichert.

Nur fuer den Mapping-Lauf gedacht -- beim Torlauf sind die Tags ja schon
bekannt, dieser Node muss dort nicht mitlaufen.
"""

import os
from collections import Counter, defaultdict

import rospy
from std_msgs.msg import Int32, String

from graph import Graph
from gate_tag_voting import pick_confident_tag

GATE_TAG_MIN = 5
GATE_TAG_MAX = 13
CONTROL_MODE_LANE = 1
CONTROL_MODE_STOP = 3


class MappingRecorderNode:
    def __init__(self, node_name):
        rospy.init_node(node_name)
        self._vehicle_name = os.environ['VEHICLE_NAME']

        known_map_path = rospy.get_param(
            '~known_map_path',
            os.path.join(os.path.dirname(__file__), '..', 'config', 'known_map.json'))
        self._output_path = rospy.get_param(
            '~output_path',
            os.path.join(os.path.dirname(__file__), '..', 'config', 'mapped_map.json'))
        self._min_sightings = rospy.get_param('~min_sightings', 3)

        self.graph = Graph.load(known_map_path)

        self._current_edge = None
        self._control_mode = None
        self._sightings = defaultdict(Counter)  # edge_id -> Counter[tag_id]
        self._finalized = set()
        self._saved = False

        v = self._vehicle_name
        # Zugeordnete Tore live melden ("<edge_id>:<tag>"), damit das Dashboard den
        # Mapping-Fortschritt waehrend der Fahrt zeigen kann. Ohne das saehe es nur die
        # known_map.json vom Start, in der noch kein Tor eingetragen ist -> immer 0.
        # latch=True: ein spaeter startendes Dashboard bekommt den letzten Wert noch.
        self.pub_gate_mapped = rospy.Publisher(f'/{v}/mapping/gate_mapped', String,
                                               queue_size=10, latch=True)
        rospy.Subscriber(f'/{v}/detect/apriltag', Int32, self.cb_apriltag, queue_size=1)
        rospy.Subscriber(f'/{v}/switch/current_edge', Int32, self.cb_current_edge, queue_size=1)
        rospy.Subscriber(f'/{v}/switch/control', Int32, self.cb_control, queue_size=1)

    def cb_control(self, msg):
        self._control_mode = msg.data
        if self._control_mode == CONTROL_MODE_STOP and not self._saved:
            self._finalize_edge(self._current_edge)
            self.graph.save(self._output_path)
            self._saved = True
            rospy.loginfo(f"Mapping abgeschlossen, Graph gespeichert: {self._output_path}")

    def cb_current_edge(self, msg):
        new_edge = None if msg.data < 0 else msg.data
        if new_edge != self._current_edge:
            self._finalize_edge(self._current_edge)
            self._current_edge = new_edge

    def cb_apriltag(self, msg):
        tag = msg.data
        if self._control_mode != CONTROL_MODE_LANE:
            return
        if self._current_edge is None or not (GATE_TAG_MIN <= tag <= GATE_TAG_MAX):
            return
        self._sightings[self._current_edge][tag] += 1

    def _finalize_edge(self, edge_id):
        if edge_id is None or edge_id in self._finalized:
            return
        self._finalized.add(edge_id)
        tag = pick_confident_tag(self._sightings.get(edge_id, Counter()), self._min_sightings)
        if tag is None:
            rospy.loginfo(f"Kante {edge_id}: kein sicheres Tor erkannt (0 Sichtungen oder zu wenige).")
            return
        self.graph.set_gate_tag(edge_id, tag)
        self.pub_gate_mapped.publish(String(data=f"{edge_id}:{tag}"))
        rospy.loginfo(f"Kante {edge_id} -> Tor {tag}")


if __name__ == '__main__':
    node = MappingRecorderNode('mapping_recorder_node')
    rospy.spin()
