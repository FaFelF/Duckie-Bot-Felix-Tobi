#!/usr/bin/env python3


import rospy
import random
from std_msgs.msg import Float64, Int32, Bool
from enum import Enum

import os

class ControlType(Enum):
    Lane = 1
    Obstacle = 2
    Stop = 3
    Intersection = 4

class IntersectionsDirections(Enum):
    Left = 0
    Straight = 1
    Right = 2

class SwitchControlNode:
    def __init__(self,node_name):
        rospy.init_node(node_name)
        #super(SwitchControlNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self.sub_duckie = rospy.Subscriber(f"/{self._vehicle_name}/detect/duckies", Bool, self.cbDuckieDetected, queue_size = 1)
        self.sub_lane = rospy.Subscriber(f"/{self._vehicle_name}/detect/lane", Float64, self.cbLaneDetected, queue_size = 1)
        self.pub_control = rospy.Publisher(f"/{self._vehicle_name}/switch/control", Int32, queue_size = 1)

        self.pub_chosen_direction = rospy.Publisher(f"/{self._vehicle_name}/switch/intersection_direction", Int32, queue_size = 1)

        self.sub_intersection = rospy.Subscriber(f'/{self._vehicle_name}/detect/intersection', Bool, self.cbIntersection, queue_size=1)

        self.current_apriltag_id = -1
        self.sub_Apriltag = rospy.Subscriber(f'/{self._vehicle_name}/detect/apriltag', Int32, self.cbApriltag, queue_size=1)

        self.intersection_finished = False
        self.sub_intersection_finished = rospy.Subscriber(f'/{self._vehicle_name}/switch/intersection_finished', Bool, self.cbIntersectionFinished, queue_size=1)

        self.duckie_detected = 0

        self._control_mode = ControlType.Lane

        self.is_intersection_free = True
        self._waiting_on_intersection = False
        self._state_start_time = None
        self.intersection_wait_time = 3  # seconds

        self._chosen_direction = IntersectionsDirections.Straight



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

    def cbIntersectionFinished(self, msg):
        self.intersection_finished = msg.data
        if self.intersection_finished:
            self._control_mode = ControlType.Lane   
            self.intersection_finished = False

    def cbIntersection(self, msg):
        """
        ROS-Callback, wird aufgerufen wenn detect_intersection_node einen neuen Wert publiziert.
        Wechselt in den Stop-Modus wenn eine Kreuzung erkannt wird, wartet und fährt dann weiter.

        Autor: Felix Faass

        Args:
            msg (Bool): True wenn roter Streifen erkannt, sonst False
        """

        if msg.data == True and self._control_mode == ControlType.Lane:
            
            self._chosen_direction = self.fnChooseDirection()

            self._control_mode = ControlType.Stop
            rospy.loginfo(f"Intersection detected! Waiting for {self.intersection_wait_time} seconds...")
            self._state_start_time = rospy.Time.now()
            self._waiting_on_intersection = True


    def fnChooseDirection(self):
        tag_id = self.current_apriltag_id
        chosen_direction= IntersectionsDirections.Straight
        if tag_id == 0:       #┼
            chosen_direction = IntersectionsDirections(random.randint(0, 2))
            
        elif tag_id == 1:     #┤
            chosen_direction = IntersectionsDirections(random.randint(0, 1))
            
        elif tag_id == 2:     #├
            chosen_direction = IntersectionsDirections(random.randint(1, 2))
            
        elif tag_id == 3:     #┴
            chosen_direction = IntersectionsDirections(random.choice([0, 2]))

        return chosen_direction

    def cbApriltag(self, msg):
        self.current_apriltag_id = msg.data

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():

            if self._waiting_on_intersection:
                if (rospy.Time.now() - self._state_start_time).to_sec() >= self.intersection_wait_time:
                    self._waiting_on_intersection = False
                    self._control_mode = ControlType.Intersection
                    self.pub_chosen_direction.publish(self._chosen_direction.value)

            self.pub_control.publish(self._control_mode.value)
            rate.sleep()

if __name__ == '__main__':
    # create the node
    node = SwitchControlNode(node_name='switch_control_node')
    node.run()
    # keep the process from terminating
    rospy.spin()