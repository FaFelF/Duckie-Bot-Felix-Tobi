#!/usr/bin/env python3
"""
ROS-Glue fuers Dashboard (Zeichenlogik in dashboard_render.py). Zeigt Karte, Pfad
und aktuelle Position. Layout einmalig beim Start (compute_layout), danach fix.
Position kommt als Plan-Index (current_step), nicht current_edge, weil eine Kante
mehrfach im Plan vorkommen kann.
"""

import json
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

        # Vorgegebene Tor-Reihenfolge (nur beim Torlauf, neben plan.json abgelegt).
        targets_path = rospy.get_param(
            '~gate_targets_path',
            os.path.join(os.path.dirname(os.path.abspath(plan_path)), 'gate_targets.json'))
        self.gate_targets = None
        try:
            with open(targets_path) as fh:
                self.gate_targets = json.load(fh)
        except (IOError, OSError, ValueError):
            pass

        self._step = None  # None = noch keine Nachricht empfangen (Lauf startet gerade erst)

        v = self._vehicle_name
        rospy.Subscriber(f'/{v}/switch/current_step', Int32, self.cb_current_step, queue_size=1)
        # Live gemappte Tore vom mapping_recorder (sonst zeigt das Dashboard bis zum Speichern 0).
        rospy.Subscriber(f'/{v}/mapping/gate_mapped', String, self.cb_gate_mapped, queue_size=10)
        # Live gemittelte Fahrzeiten pro Kante (nur im Mapping-Lauf).
        rospy.Subscriber(f'/{v}/mapping/edge_time', String, self.cb_edge_time, queue_size=10)
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

    def cb_edge_time(self, msg):
        try:
            edge_str, secs_str = msg.data.split(':')
            edge_id, seconds = int(edge_str), float(secs_str)
        except (ValueError, AttributeError):
            rospy.logwarn_throttle(5, f"edge_time unlesbar: {msg.data!r}")
            return
        if edge_id in self.graph.edges:
            self.graph.set_travel_time(edge_id, seconds)

    def run(self):
        rate = rospy.Rate(2)
        # Fenster immer oeffnen (Anzeige + Topic /debug/dashboard).
        while not rospy.is_shutdown():
            try:
                img = build_dashboard_image(self.graph, self.plan, self._step, self.layout,
                                            self._run_label, self.gate_targets)
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
