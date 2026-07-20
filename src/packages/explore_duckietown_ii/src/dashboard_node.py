#!/usr/bin/env python3
"""
Dashboard-Node: zeigt live
  a) die erstellte Karte,
  b) den gewaehlten Pfad,
  c) wo der Bot denkt, dass er ist
(Anforderungen aus der Aufgabenstellung). Reine ROS-Glue -- die eigentliche
Zeichenlogik steckt in dashboard_render.py (ROS-unabhaengig, lokal testbar).

Positionen der Knoten kommen aus graph_layout.compute_layout (schematisches
Force-Directed-Layout -- keine geografische Genauigkeit, aber einmalig beim
Start berechnet und danach fix gehalten, damit das Bild nicht bei jedem Frame
"zittert").

Die aktuelle Position kommt ueber .../switch/current_step (roher Plan-Index,
von switch_control_node.py veroeffentlicht) -- NICHT ueber current_edge, weil
dieselbe Kante mehrfach im Plan vorkommen kann (Mapping-Backtracking,
Nebenkreuzungen beim Torlauf) und der Schritt daraus sonst mehrdeutig waere.
"""

import os

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32, String

from graph import Graph, load_plan
from graph_layout import compute_layout
from dashboard_render import build_dashboard_image


class DashboardNode:
    def __init__(self, node_name):
        rospy.init_node(node_name)
        self._vehicle_name = os.environ['VEHICLE_NAME']

        map_path = rospy.get_param(
            '~map_path', os.path.join(os.path.dirname(__file__), '..', 'config', 'known_map.json'))
        plan_path = rospy.get_param(
            '~plan_path', os.path.join(os.path.dirname(__file__), '..', 'config', 'plan.json'))
        self._run_label = rospy.get_param('~run_label', '')

        self.graph = Graph.load(map_path)
        self.plan = load_plan(plan_path)
        self.layout = compute_layout(self.graph, seed=0)

        self._step = None  # None = noch keine Nachricht empfangen (Lauf startet gerade erst)

        v = self._vehicle_name
        rospy.Subscriber(f'/{v}/switch/current_step', Int32, self.cb_current_step, queue_size=1)
        # Live-Meldungen des mapping_recorder_node. Ohne die wuerde das Dashboard nur die
        # beim Start geladene Karte kennen (im Mapping-Lauf known_map.json, dort steht noch
        # kein Tor drin) und die gemappten Tore immer als 0 anzeigen -- der Recorder haelt
        # seinen Graphen im Speicher und schreibt ihn erst am Ende in mapped_map.json.
        rospy.Subscriber(f'/{v}/mapping/gate_mapped', String, self.cb_gate_mapped, queue_size=10)
        self.pub_dashboard = rospy.Publisher(f'/{v}/debug/dashboard', CompressedImage, queue_size=1)

    def cb_current_step(self, msg):
        self._step = msg.data

    def cb_gate_mapped(self, msg):
        try:
            edge_id, tag = (int(x) for x in msg.data.split(':'))
        except (ValueError, AttributeError):
            rospy.logwarn_throttle(5, f"gate_mapped unlesbar: {msg.data!r}")
            return
        if edge_id in self.graph.edges:
            self.graph.set_gate_tag(edge_id, tag)

    def run(self):
        rate = rospy.Rate(2)
        # Fenster immer oeffnen. Vorher hing es hinter DUCKIE_GUI=1 -- das ist wegen der
        # X11-Probleme aber dauerhaft aus, dadurch war das Dashboard faktisch unsichtbar
        # (es lief nur als Topic /debug/dashboard).
        while not rospy.is_shutdown():
            try:
                img = build_dashboard_image(self.graph, self.plan, self._step, self.layout, self._run_label)
                msg = CompressedImage()
                msg.header.stamp = rospy.Time.now()
                msg.format = "jpeg"
                msg.data = np.array(cv2.imencode('.jpg', img)[1]).tobytes()
                self.pub_dashboard.publish(msg)
                cv2.imshow('Dashboard', img)
                cv2.waitKey(1)
            except Exception as e:
                rospy.logwarn_throttle(5, f"dashboard_node failed: {e}")
            rate.sleep()


if __name__ == '__main__':
    node = DashboardNode('dashboard_node')
    node.run()
