#!/usr/bin/env python3


import rospy
from std_msgs.msg import Float64, Int32, Bool, String
from enum import Enum

import os

from graph import PlanState, load_plan

class ControlType(Enum):
    Lane = 1
    Obstacle = 2
    Stop = 3
    Intersection = 4
    Align = 5

class IntersectionsDirections(Enum):
    Left = 0
    Straight = 1
    Right = 2

class SwitchControlNode:
    def __init__(self,node_name):
        rospy.init_node(node_name)
        #super(SwitchControlNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self.sub_duckie = rospy.Subscriber(f"/{self._vehicle_name}/detect/duckie_control_active", Bool, self.cbDuckieDetected, queue_size = 1)
        self.sub_lane = rospy.Subscriber(f"/{self._vehicle_name}/detect/lane", Float64, self.cbLaneDetected, queue_size = 1)
        self.pub_control = rospy.Publisher(f"/{self._vehicle_name}/switch/control", Int32, queue_size = 1)

        self.pub_chosen_direction = rospy.Publisher(f"/{self._vehicle_name}/switch/intersection_direction", Int32, queue_size = 1)

        self.sub_intersection = rospy.Subscriber(f'/{self._vehicle_name}/detect/intersection', Bool, self.cbIntersection, queue_size=1)

        # Plan (Mapping- oder Gate-Lauf, siehe mapping_planner.py/gate_planner.py) legt die
        # Abbiege-Richtung an jeder Kreuzung vorab fest -- ersetzt die vormals zufaellige Wahl
        # anhand des Kreuzungstyp-Tags.
        # latch=True: Subscriber, die erst nach dem Start verbinden (z.B. mapping_recorder_node,
        # dashboard_node), bekommen den zuletzt veroeffentlichten Wert trotzdem sofort.
        # current_edge: Kanten-ID (fuer mapping_recorder_node -- Tag-Zuordnung braucht nur
        #   "welche Kante JETZT", keine Mehrdeutigkeit in Echtzeit).
        # current_step: roher Plan-Index (fuer's Dashboard -- current_edge allein reicht dort
        #   nicht, weil dieselbe Kante mehrfach im Plan vorkommen kann, z.B. beim Mapping-
        #   Backtracking, und "welcher Schritt" dann aus der Kanten-ID nicht eindeutig waere).
        self.pub_current_edge = rospy.Publisher(f'/{self._vehicle_name}/switch/current_edge', Int32, queue_size=1, latch=True)
        self.pub_current_step = rospy.Publisher(f'/{self._vehicle_name}/switch/current_step', Int32, queue_size=1, latch=True)
        # Fahrzeit pro Kante messen: Zeit vom Losfahren auf einer Kante bis zur naechsten
        # Kreuzung ("edge_id:sekunden"). Der mapping_recorder mittelt das und schreibt es
        # in die Karte -> gate_planner bevorzugt spaeter die schnellere Route.
        self.pub_edge_time = rospy.Publisher(f'/{self._vehicle_name}/switch/edge_time', String, queue_size=10)
        # Start der Fahrt auf der aktuellen Kante. Wird beim Losfahren gesetzt (hier fuer die
        # Startkante, danach nach jeder Abbiegung) und bei Ankunft an der naechsten Kreuzung
        # zur Fahrzeit-Messung ausgewertet.
        self._edge_start_time = rospy.Time.now()
        plan_path = rospy.get_param('~plan_path', os.path.join(os.path.dirname(__file__), '..', 'config', 'plan.json'))
        self._plan_state = PlanState(load_plan(plan_path))
        self._publish_plan_progress()

        self.intersection_finished = False
        self.sub_intersection_finished = rospy.Subscriber(f'/{self._vehicle_name}/switch/intersection_finished', Bool, self.cbIntersectionFinished, queue_size=1)

        # Ausricht-Status: erst abbiegen, wenn ausgerichtet (oder Sicherheits-Timeout)
        self._aligned = False
        self.sub_align_finished = rospy.Subscriber(f'/{self._vehicle_name}/switch/align_finished', Bool, self.cbAlignFinished, queue_size=1)

        self.duckie_detected = 0

        self._control_mode = ControlType.Lane

        self.intersection_running = False
        self._intersection_cooldown_start = None
        self.intersection_cooldown_time = 2  # seconds, ab Abbiege-Ende: nur Wegfahren von der Linie
        self.is_intersection_free = True
        self._waiting_on_intersection = False
        self._state_start_time = None
        self.intersection_wait_time = 3  # seconds, Mindest-Standzeit
        self.intersection_align_max_time = 6  # seconds, Sicherheits-Obergrenze fürs Ausrichten



    def cbDuckieDetected(self, msg):
        if msg.data > 0:
            self._control_mode = ControlType.Obstacle
            print('Duckies detected!')
        else:
            if self._control_mode == ControlType.Obstacle:
                self._control_mode = ControlType.Lane

            


    def cbLaneDetected(self, msg):
        print('received message')
        # Write your own code her

    def _publish_plan_progress(self):
        hop = self._plan_state.current_hop
        self.pub_current_edge.publish(Int32(data=hop.edge_id if hop is not None else -1))
        self.pub_current_step.publish(Int32(data=self._plan_state.step))

    def cbIntersectionFinished(self, msg):
        self.intersection_finished = msg.data
        if self.intersection_finished:
            self._control_mode = ControlType.Lane
            self.intersection_finished = False
            # Abbiegung laut Plan ausgefuehrt -> einen Schritt weiter im Plan.
            self._plan_state.advance()
            self._publish_plan_progress()
            # Jetzt erst den Cooldown starten: ab hier fährt er von der Kreuzung weg.
            # Der Timer muss nur noch das Wegfahren von der roten Linie abdecken.
            self._intersection_cooldown_start = rospy.Time.now()
            # Ab hier faehrt er auf der naechsten Kante los -> Fahrzeit-Messung neu starten.
            self._edge_start_time = rospy.Time.now()

    def cbIntersection(self, msg):
        """
        ROS-Callback, wird aufgerufen wenn detect_intersection_node einen neuen Wert publiziert.
        Wechselt in den Stop-Modus wenn eine Kreuzung erkannt wird, wartet und fährt dann weiter.
        Die Abbiege-Richtung kommt nicht mehr zufaellig vom Kreuzungstyp-Tag, sondern aus dem
        vorab berechneten Plan (siehe mapping_planner.py/gate_planner.py).

        Autor: Felix Faass

        Args:
            msg (Bool): True wenn roter Streifen erkannt, sonst False
        """
        if self.intersection_running:
            return

        if not (msg.data and self._control_mode == ControlType.Lane):
            return

        self.intersection_running = True
        # Cooldown startet NICHT hier, sondern erst nach dem Abbiegen (cbIntersectionFinished),
        # damit Ausrichten/Abbiegen beliebig lang dauern dürfen, ohne dass er erneut triggert.
        self._intersection_cooldown_start = None

        # Kante zu Ende gefahren (rote Linie an der naechsten Kreuzung erkannt) -> Fahrzeit
        # dieser Kante melden. Plan ist hier noch NICHT weitergeschaltet, current_edge_id ist
        # also die gerade befahrene Kante. Der mapping_recorder mittelt das pro Kante.
        edge_id = self._plan_state.current_edge_id
        if edge_id is not None and self._edge_start_time is not None:
            elapsed = (rospy.Time.now() - self._edge_start_time).to_sec()
            self.pub_edge_time.publish(String(data=f"{edge_id}:{elapsed:.3f}"))

        # Letzter Streckenabschnitt des Plans beendet -> Ziel erreicht, keine weitere
        # Abbiegung vorgesehen. Einfach stoppen statt in Align/Intersection weiterzumachen.
        if self._plan_state.step >= len(self._plan_state.plan) - 1:
            rospy.loginfo("Plan abgeschlossen -- Ziel erreicht, stoppe.")
            self._control_mode = ControlType.Stop
            self._plan_state.advance()
            self._publish_plan_progress()
            return

        # Während der ohnehin vorhandenen Stop-Wartezeit gleich ausrichten,
        # statt nur stillzustehen. Kostet keine zusätzliche Zeit.
        self._control_mode = ControlType.Align
        self._aligned = False
        direction = self._plan_state.next_direction()
        rospy.loginfo(f"Intersection detected! Plan-Richtung: {direction}. "
                      f"Aligning (min {self.intersection_wait_time}s, max {self.intersection_align_max_time}s)...")
        self._state_start_time = rospy.Time.now()
        self._waiting_on_intersection = True
        self.pub_chosen_direction.publish(direction.value)

    def cbAlignFinished(self, msg):
        """
        Wird gesetzt, wenn control_lane_node das Ausrichten abgeschlossen hat.
        Erst dann (und nach Ablauf der Mindest-Standzeit) wird abgebogen.

        Autor: Felix Faass
        """
        if msg.data:
            self._aligned = True

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():

            if self.intersection_running and self._intersection_cooldown_start is not None:
                if (rospy.Time.now() - self._intersection_cooldown_start).to_sec() >= self.intersection_cooldown_time:
                    self.intersection_running = False

            if self._waiting_on_intersection:
                elapsed = (rospy.Time.now() - self._state_start_time).to_sec()
                min_passed = elapsed >= self.intersection_wait_time
                max_passed = elapsed >= self.intersection_align_max_time
                # Abbiegen, sobald ausgerichtet UND Mindest-Standzeit vorbei.
                # Braucht das Ausrichten länger, bekommt es die Zeit (bis zum Sicherheits-Timeout).
                if (min_passed and self._aligned) or max_passed:
                    if max_passed and not self._aligned:
                        rospy.logwarn("Align: Sicherheits-Timeout, biege trotz Schieflage ab")
                    self._waiting_on_intersection = False
                    self._control_mode = ControlType.Intersection

            self.pub_control.publish(self._control_mode.value)
            rate.sleep()

if __name__ == '__main__':
    # create the node
    node = SwitchControlNode(node_name='switch_control_node')
    node.run()
    # keep the process from terminating
    rospy.spin()