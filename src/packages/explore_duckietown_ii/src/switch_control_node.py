#!/usr/bin/env python3


import rospy
import random
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

        # current_edge (Kanten-ID fuer mapping_recorder) + current_step (Plan-Index fuers
        # Dashboard, eindeutig auch bei mehrfach vorkommender Kante). latch fuer spaete Subscriber.
        self.pub_current_edge = rospy.Publisher(f'/{self._vehicle_name}/switch/current_edge', Int32, queue_size=1, latch=True)
        self.pub_current_step = rospy.Publisher(f'/{self._vehicle_name}/switch/current_step', Int32, queue_size=1, latch=True)
        # Fahrzeit pro Kante ("edge_id:sekunden") -> mapping_recorder mittelt fuer die Route-Wahl.
        self.pub_edge_time = rospy.Publisher(f'/{self._vehicle_name}/switch/edge_time', String, queue_size=10)
        # Start der Fahrt auf der aktuellen Kante (fuer die Fahrzeit-Messung).
        self._edge_start_time = rospy.Time.now()

        # Fahr-Modus: 'plan' = Richtung aus plan.json (Challenge 4), 'random_tag' = zufaellig
        # nach Kreuzungstyp-Tag (Challenge 2). Default 'plan'.
        self._direction_mode = rospy.get_param('~direction_mode', 'plan')
        if self._direction_mode == 'plan':
            plan_path = rospy.get_param('~plan_path', os.path.join(os.path.dirname(__file__), '..', 'config', 'plan.json'))
            self._plan_state = PlanState(load_plan(plan_path))
            self._publish_plan_progress()
        else:
            # Challenge 2: kein Plan, Richtung kommt zufaellig aus saved_apriltag.
            self._plan_state = None
            self._chosen_direction = IntersectionsDirections.Straight
            self.sub_saved_apriltag = rospy.Subscriber(
                f'/{self._vehicle_name}/detect/saved_apriltag', Int32, self.cbChooseDirection, queue_size=1)

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
        self.intersection_wait_time = 0.2  # seconds, Mindest-Standzeit
        self.intersection_align_max_time = 6  # seconds, Sicherheits-Obergrenze fürs Ausrichten



    def cbDuckieDetected(self, msg):
        if msg.data > 0:
            self._control_mode = ControlType.Obstacle
        else:
            if self._control_mode == ControlType.Obstacle:
                self._control_mode = ControlType.Lane

            


    def cbLaneDetected(self, msg):
        pass

    def cbChooseDirection(self, msg):
        """
        Challenge 2: zufaellige Richtung aus den vom Kreuzungstyp-Tag erlaubten waehlen.
        Tag: 1=┼ alle, 2=┴ links/rechts, 3=┤ links/geradeaus, 4=├ geradeaus/rechts.
        """
        tag_id = msg.data
        if tag_id == 1:      # ┼
            self._chosen_direction = IntersectionsDirections(random.randint(0, 2))
        elif tag_id == 3:    # ┤
            self._chosen_direction = IntersectionsDirections(random.randint(0, 1))
        elif tag_id == 4:    # ├
            self._chosen_direction = IntersectionsDirections(random.randint(1, 2))
        elif tag_id == 2:    # ┴
            self._chosen_direction = IntersectionsDirections(random.choice([0, 2]))

    def _publish_plan_progress(self):
        hop = self._plan_state.current_hop
        self.pub_current_edge.publish(Int32(data=hop.edge_id if hop is not None else -1))
        self.pub_current_step.publish(Int32(data=self._plan_state.step))

    def cbIntersectionFinished(self, msg):
        self.intersection_finished = msg.data
        if self.intersection_finished:
            self._control_mode = ControlType.Lane
            self.intersection_finished = False
            # Abbiegung ausgefuehrt -> im Plan-Modus einen Schritt weiter (random_tag hat keinen Plan).
            if self._plan_state is not None:
                self._plan_state.advance()
                self._publish_plan_progress()
            # Jetzt erst den Cooldown starten: ab hier fährt er von der Kreuzung weg.
            # Der Timer muss nur noch das Wegfahren von der roten Linie abdecken.
            self._intersection_cooldown_start = rospy.Time.now()
            # Ab hier faehrt er auf der naechsten Kante los -> Fahrzeit-Messung neu starten.
            self._edge_start_time = rospy.Time.now()

    def cbIntersection(self, msg):
        """
        Bei erkannter Kreuzung ausrichten und abbiegen. Richtung je nach Modus aus dem Plan
        (Challenge 4) oder zufaellig aus dem Kreuzungstyp-Tag (Challenge 2).

        Autor: Felix Faass
        """
        if self.intersection_running:
            return

        if not (msg.data and self._control_mode == ControlType.Lane):
            return

        self.intersection_running = True
        # Cooldown startet NICHT hier, sondern erst nach dem Abbiegen (cbIntersectionFinished),
        # damit Ausrichten/Abbiegen beliebig lang dauern dürfen, ohne dass er erneut triggert.
        self._intersection_cooldown_start = None

        if self._plan_state is not None:
            # Kante zu Ende gefahren -> Fahrzeit melden (Plan noch nicht weitergeschaltet).
            edge_id = self._plan_state.current_edge_id
            if edge_id is not None and self._edge_start_time is not None:
                elapsed = (rospy.Time.now() - self._edge_start_time).to_sec()
                self.pub_edge_time.publish(String(data=f"{edge_id}:{elapsed:.3f}"))

            # Letzter Hop beendet -> Ziel erreicht, stoppen.
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
        # Richtung: aus dem Plan oder (random_tag) die zuvor zufaellig gewaehlte.
        if self._plan_state is not None:
            direction = self._plan_state.next_direction()
        else:
            direction = self._chosen_direction
        rospy.loginfo(f"Intersection detected! Richtung: {direction}. "
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