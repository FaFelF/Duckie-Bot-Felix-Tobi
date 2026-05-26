#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64, Int32, String, Bool

from duckietown_msgs.msg import Twist2DStamped
import os
from switch_control_node import ControlType, IntersectionsDirections
import yaml
import util

class ControlLaneNode:
    def __init__(self,node_name):
        rospy.init_node(node_name)
        self._control_mode = ControlType.Lane

        self._vehicle_name = os.environ['VEHICLE_NAME']
        util.init_parameters(node_name, self.cbUpdateParameters)

        self.lastError = 0
        self.integral = 0
        self.v = 0
        self.a = 0
        self.pending_direction = None

        twist_topic = f"/{self._vehicle_name}/car_cmd_switch_node/cmd"
        self.pub_cmd_vel = rospy.Publisher(twist_topic, Twist2DStamped, queue_size = 1)
        self.pub_debug_v     = rospy.Publisher(f'/{self._vehicle_name}/debug/control_v',     Float64, queue_size=1)
        self.pub_debug_omega = rospy.Publisher(f'/{self._vehicle_name}/debug/control_omega', Float64, queue_size=1)

        detect_lane_topic = f"/{self._vehicle_name}/detect/lane"
        self.sub_lane = rospy.Subscriber(detect_lane_topic, Float64, self.cbFollowLane, queue_size = 1)

        control_change_topic = f"/{self._vehicle_name}/switch/control"
        self.sub_control = rospy.Subscriber(control_change_topic, Int32, self.cbControl , queue_size = 1)

        intersection_direction_topic = f"/{self._vehicle_name}/switch/intersection_direction"
        self.sub_intersection_direction = rospy.Subscriber(intersection_direction_topic, Int32, self.cbIntersectionControl, queue_size = 1)

        self.pub_intersection_finished = rospy.Publisher(f'/{self._vehicle_name}/switch/intersection_finished', Bool, queue_size=1)

        self.sub_duckie_error = rospy.Subscriber(f'/{self._vehicle_name}/detect/duckie_error', Float64, self.cbAvoidDuckie, queue_size=1)



      


        rospy.on_shutdown(self.fnShutDown)

    def cbControl(self,msg):
        self._control_mode = ControlType(msg.data)
        


    def cbUpdateParameters(self,parameters):
        self.kp = parameters["pid"]["p"]["default"]
        self.ki = parameters["pid"]["i"]["default"]
        self.kd = parameters["pid"]["d"]["default"]
        self.MAX_VEL = parameters["pid"]["max_vel"]["default"]
        self.speed_curve_factor = parameters["pid"]["speed_curve_factor"]["default"]

        self.kp_duckie = parameters["pid_duckie"]["p"]["default"]
        self.ki_duckie = parameters["pid_duckie"]["i"]["default"]
        self.kd_duckie = parameters["pid_duckie"]["d"]["default"]
        self.MAX_VEL_duckie = parameters["pid_duckie"]["max_vel"]["default"]

        self.sleep_time_left     = parameters["intersection"]["sleep_time_left"]["default"]
        self.sleep_time_straight = parameters["intersection"]["sleep_time_straight"]["default"]
        self.sleep_time_right    = parameters["intersection"]["sleep_time_right"]["default"]

    # error between 1 and -1
    def cbFollowLane(self, error):
        if self._control_mode != ControlType.Lane:
            return
        
        print(f'received message. enabled : {self._control_mode == ControlType.Lane}')
        error = error.data

        #PID-Regler eingefügt
        self.integral += error
        self.integral = max(-1.0, min(1.0, self.integral))  # ← Clamp
        p = self.kp * error
        i = self.ki * self.integral
        d = self.kd * ((error - self.lastError)/0.1) #soll wert gewichtung des D-Teils erhöhen, da die Funktion 10 mal pro Sekunde aufgerufen wird
        self.lastError = error

        self.v = self.MAX_VEL * max(0.5, 1 - abs(error) * self.speed_curve_factor)
        self.a = p + i + d
        
    def cbIntersectionControl(self, msg):
        self.pending_direction = IntersectionsDirections(msg.data)
        rospy.loginfo(f"Direction stored: {self.pending_direction}")

    def cbGetStopError(self, msg):
        pass

    def cbAvoidDuckie(self, msg):
        if self._control_mode != ControlType.Obstacle:
            return
        
        error = msg.data

        self.integral += error
        self.integral = max(-1.0, min(1.0, self.integral))  # ← Clamp
        p = self.kp_duckie * error
        i = self.ki_duckie * self.integral
        d = self.kd_duckie * ((error - self.lastError)/0.1) #soll wert gewichtung des D-Teils erhöhen, da die Funktion 10 mal pro Sekunde aufgerufen wird
        self.lastError = error
        self.v = self.MAX_VEL_duckie * max(0.5, 1 - abs(error) * self.speed_curve_factor)
        self.a = p + i + d  




    def fnShutDown(self):
        rospy.loginfo("Shutting down. cmd_vel will be 0")

        twist = Twist2DStamped(v=0.0, omega=0.0)
        self.pub_cmd_vel.publish(twist) 

    def run(self):
        rospy.loginfo("control_lane_node: waiting 5s before start...")
        rospy.sleep(5)
        rospy.loginfo("control_lane_node: starting")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            twist = Twist2DStamped()
            twist.header.stamp = rospy.Time.now()
            if self._control_mode == ControlType.Intersection and self.pending_direction is not None:
                direction = self.pending_direction
                self.pending_direction = None
                if direction == IntersectionsDirections.Left:
                    twist.v = 0.2
                    twist.omega = 1.0
                    self.pub_cmd_vel.publish(twist)
                    rospy.sleep(self.sleep_time_left)
                elif direction == IntersectionsDirections.Straight:
                    twist.v = 0.2
                    twist.omega = 0.0
                    self.pub_cmd_vel.publish(twist)
                    rospy.sleep(self.sleep_time_straight)
                elif direction == IntersectionsDirections.Right:
                    twist.v = 0.2
                    twist.omega = -1.0
                    self.pub_cmd_vel.publish(twist)
                    rospy.sleep(self.sleep_time_right)
                self.pub_intersection_finished.publish(Bool(data=True))
                rate.sleep()
                continue
            elif self._control_mode != ControlType.Stop:
                twist.v = self.v
                twist.omega = self.a
            else:
                twist.v = 0.0
                twist.omega = 0.0

            print(f'publishing {twist}')
            self.pub_cmd_vel.publish(twist)
            self.pub_debug_v.publish(Float64(data=twist.v))
            self.pub_debug_omega.publish(Float64(data=twist.omega))

            rate.sleep()

if __name__ == '__main__':
    # create the node
    node = ControlLaneNode('control_lane_node')
    node.run()
    rospy.spin()
