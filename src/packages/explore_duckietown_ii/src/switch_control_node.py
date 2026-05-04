#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64, Int32, Bool
from enum import Enum

import os

class ControlType(Enum):
    Lane = 1
    Obstacle = 2
    Stop = 3

class SwitchControlNode:
    def __init__(self,node_name):
        rospy.init_node(node_name)
        #super(SwitchControlNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self.sub_duckie = rospy.Subscriber(f"/{self._vehicle_name}/detect/duckie", Float64, self.cbDuckieDetected, queue_size = 1)
        self.sub_lane = rospy.Subscriber(f"/{self._vehicle_name}/detect/lane", Float64, self.cbLaneDetected, queue_size = 1)
        self.pub_control = rospy.Publisher(f"/{self._vehicle_name}/switch/control", Int32, queue_size = 1)
        self.sub_intersection = rospy.Subscriber(f'/{self._vehicle_name}/detect/intersection', Bool, self.cbIntersection, queue_size=1)

        self._control_mode = ControlType.Lane

        self.is_intersection_free = True

        self.intersection_wait_time = 3  # seconds

    def cbDuckieDetected(self, msg):
        print('received message')
        # Write your own code her

    def cbLaneDetected(self, msg):
        print('received message')
        # Write your own code her


    def cbIntersection(self, msg):
        """
        ROS-Callback, wird aufgerufen wenn detect_intersection_node einen neuen Wert publiziert.
        Wechselt in den Stop-Modus wenn eine Kreuzung erkannt wird, wartet und fährt dann weiter.

        Autor: Felix Faass

        Args:
            msg (Bool): True wenn roter Streifen erkannt, sonst False
        """
        #Abfragen, ob Kreuzung erkannt wurde
        if msg.data == True:
            #Wechsel in Stop-Modus
            self._control_mode = ControlType.Stop
            rospy.loginfo(f"Intersection detected! Waiting for {self.intersection_wait_time} seconds...")
            rospy.sleep(self.intersection_wait_time)
            #Nach Wartezeit Kreuzung als frei markieren und zurück zum Lane-Modus wechseln
            if self.is_intersection_free:
                #Kreuzung als frei markieren und zurück zum Lane-Modus wechseln
                #self._control_mode = ControlType.Lane
                rospy.sleep(5)

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():

            msg_control = Int32()
            msg_control.data = self._control_mode.value
            self.pub_control.publish(msg_control)
            rate.sleep()

if __name__ == '__main__':
    # create the node
    node = SwitchControlNode(node_name='switch_control_node')
    node.run()
    # keep the process from terminating
    rospy.spin()