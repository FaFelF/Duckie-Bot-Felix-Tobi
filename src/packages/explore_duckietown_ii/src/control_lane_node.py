#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64, Int32, String, Bool

from duckietown_msgs.msg import Twist2DStamped
import os
import math
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
        self.duckie_distance_factor = 0.0

        # Ausrichten an der Kreuzung
        self.intersection_angle = float('nan')
        self._align_start_time = None
        self._align_stable = 0
        self._align_done = False

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

        self.pub_align_finished = rospy.Publisher(f'/{self._vehicle_name}/switch/align_finished', Bool, queue_size=1)
        self.sub_intersection_angle = rospy.Subscriber(f'/{self._vehicle_name}/detect/intersection_angle', Float64, self.cbIntersectionAngle, queue_size=1)

        self.sub_duckie_error = rospy.Subscriber(f'/{self._vehicle_name}/detect/duckie_error', Float64, self.cbAvoidDuckie, queue_size=1)
        self.sub_duckie_factor = rospy.Subscriber(f'/{self._vehicle_name}/detect/duckie_distance_factor', Float64, self.cbDuckieDistanceFactor, queue_size=1)



      


        rospy.on_shutdown(self.fnShutDown)

    def cbControl(self,msg):
        new_mode = ControlType(msg.data)
        # Beim Eintritt in den Align-Zustand das Ausricht-Manöver frisch starten
        if new_mode == ControlType.Align and self._control_mode != ControlType.Align:
            self._align_start_time = None
            self._align_stable = 0
            self._align_done = False
        self._control_mode = new_mode

    def cbIntersectionAngle(self, msg):
        self.intersection_angle = msg.data



    def cbUpdateParameters(self,parameters):
        self.kp = parameters["pid"]["p"]["default"]
        self.ki = parameters["pid"]["i"]["default"]
        self.kd = parameters["pid"]["d"]["default"]
        self.MAX_VEL = parameters["pid"]["max_vel"]["default"]
        self.speed_curve_factor = parameters["pid"]["speed_curve_factor"]["default"]
        # Untergrenze der Kurven-Drosselung (A): wie weit darf v in scharfen Kurven
        # runter. 0.5 = max. halbe Geschwindigkeit, kleiner = langsamer/mehr Reaktionszeit.
        self.min_speed_factor = parameters["pid"]["min_speed_factor"]["default"]

        self.kp_duckie = parameters["pid_duckie"]["p"]["default"]
        self.ki_duckie = parameters["pid_duckie"]["i"]["default"]
        self.kd_duckie = parameters["pid_duckie"]["d"]["default"]
        self.MAX_VEL_duckie = parameters["pid_duckie"]["max_vel"]["default"]
        # Harter Stopp (B): ab diesem distance_factor (Naehe, 1.0 = ganz nah) v=0,
        # statt in die Ente zu kriechen. 1.0 = praktisch nie stoppen.
        self.duckie_stop_factor = parameters["pid_duckie"]["stop_factor"]["default"]

        self.sleep_time_left     = parameters["intersection"]["sleep_time_left"]["default"]
        self.sleep_time_straight = parameters["intersection"]["sleep_time_straight"]["default"]
        self.sleep_time_right    = parameters["intersection"]["sleep_time_right"]["default"]

        # Ausrichten an der Kreuzung
        self.align_kp           = parameters["align"]["kp"]["default"]
        self.align_v            = parameters["align"]["v"]["default"]
        self.align_v_period     = parameters["align"]["v_period"]["default"]
        self.align_omega_min    = parameters["align"]["omega_min"]["default"]
        self.align_max_omega    = parameters["align"]["max_omega"]["default"]
        self.align_tolerance    = parameters["align"]["tolerance_deg"]["default"]
        self.align_timeout      = parameters["align"]["timeout"]["default"]
        self.align_stable_count = parameters["align"]["stable_count"]["default"]

    # error between 1 and -1
    def cbFollowLane(self, error):
        if self._control_mode != ControlType.Lane:
            return

        error = error.data

        #PID-Regler eingefügt
        self.integral += error
        self.integral = max(-1.0, min(1.0, self.integral))  # ← Clamp
        p = self.kp * error
        i = self.ki * self.integral
        d = self.kd * ((error - self.lastError)/0.1) #soll wert gewichtung des D-Teils erhöhen, da die Funktion 10 mal pro Sekunde aufgerufen wird
        self.lastError = error

        self.v = self.MAX_VEL * max(self.min_speed_factor, 1 - abs(error) * self.speed_curve_factor)
        self.a = p + i + d
        
    def cbIntersectionControl(self, msg):
        self.pending_direction = IntersectionsDirections(msg.data)
        rospy.loginfo(f"Direction stored: {self.pending_direction}")

    def cbGetStopError(self, msg):
        pass

    def cbDuckieDistanceFactor(self, msg):
        self.duckie_distance_factor = msg.data

    def cbAvoidDuckie(self, msg):
        if self._control_mode != ControlType.Obstacle:
            return

        #error = (msg.data / 650) * self.duckie_distance_factor
        error = (msg.data / 750)

        self.integral += error
        self.integral = max(-1.0, min(1.0, self.integral))
        p = self.kp_duckie * error
        i = self.ki_duckie * self.integral
        d = self.kd_duckie * ((error - self.lastError) / 0.1)
        self.lastError = error

        danger = self.duckie_distance_factor * abs(error)
        self.v = self.MAX_VEL_duckie * max(0.0, 1 - danger * 2)

        # Harter Stopp, wenn die Ente sehr nah ist (B): sonst kroch der Bot in die
        # Ente, weil die Formel oben nie ganz auf 0 kommt. stop_factor=1.0 -> aus.
        if self.duckie_distance_factor >= self.duckie_stop_factor:
            self.v = 0.0

        self.a = p + i + d




    def fnAlign(self, twist):
        """
        Dreht den Bot auf der Stelle, bis der rote Streifen (gemessener Winkel)
        waagerecht ist -> Bot steht orthogonal zur Kreuzung.

        Bewusst KEIN exakter Null-Abgleich: ein Toleranzband (align_tolerance)
        verhindert Schwingen/Pendeln. Innerhalb der Toleranz wird gestoppt,
        sobald der Wert über mehrere Zyklen stabil ist. Ein Timeout sorgt dafür,
        dass das Manöver auch bei fehlender/verrauschter Linie nicht hängen bleibt.

        Autor: Felix Faass
        """
        twist.v = 0.0
        twist.omega = 0.0
        if self._align_done:
            rospy.loginfo_throttle(1.0, "Align: ausgerichtet, halte Position (done=True)")
            return

        now = rospy.Time.now()
        if self._align_start_time is None:
            self._align_start_time = now
        elapsed = (now - self._align_start_time).to_sec()
        angle = self.intersection_angle

        done = False
        if elapsed >= self.align_timeout:
            rospy.logwarn("Align: Timeout erreicht, fahre ohne perfekte Ausrichtung fort")
            done = True
        elif angle is None or math.isnan(angle):
            # keine gültige Linie sichtbar -> kurz warten, Timeout greift notfalls
            pass
        elif abs(angle) <= self.align_tolerance:
            # innerhalb der Toleranz -> stillstehen und Stabilität zählen
            self._align_stable += 1
            if self._align_stable >= self.align_stable_count:
                done = True
        else:
            self._align_stable = 0
            # Pendel-Bogen: Fahrtrichtung periodisch wechseln. Die Räder bleiben am Rollen
            # (keine Haftreibung, kein Stall), aber der Versatz hebt sich netto auf, weil
            # omega durchgehend gleich dreht -> "auf der Stelle drehen", nur rollend.
            # Großes align_v_period = reiner Rückwärtsbogen (Pendel praktisch aus).
            half = int(elapsed / self.align_v_period)
            v_dir = -1.0 if (half % 2 == 0) else 1.0
            twist.v = self.align_v * v_dir
            omega = self.align_kp * angle
            if abs(omega) < self.align_omega_min:
                omega = math.copysign(self.align_omega_min, omega)
            twist.omega = max(-self.align_max_omega, min(self.align_max_omega, omega))

        if done:
            self._align_done = True
            twist.omega = 0.0
            self.pub_align_finished.publish(Bool(data=True))
            rospy.loginfo("Align: ausgerichtet")

        rospy.loginfo(f"Align: angle={angle:.2f} elapsed={elapsed:.2f} stable={self._align_stable} v={twist.v:.2f} omega={twist.omega:.2f} done={self._align_done}")

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
            if self._control_mode == ControlType.Align:
                self.fnAlign(twist)
                self.pub_cmd_vel.publish(twist)
                self.pub_debug_v.publish(Float64(data=twist.v))
                self.pub_debug_omega.publish(Float64(data=twist.omega))
                rate.sleep()
                continue
            if self._control_mode == ControlType.Intersection and self.pending_direction is not None:
                direction = self.pending_direction
                self.pending_direction = None
                if direction == IntersectionsDirections.Left:
                    ##Trick: 
                    # twist.v = 0.2
                    # twist.omega = 1.0

                    ## Tick
                    twist.v = 0.2
                    twist.omega = 1.2

                    #Gundel:
                    # twist.v = 0.2
                    # twist.omega = 1.8
                    self.pub_cmd_vel.publish(twist)
                    rospy.sleep(self.sleep_time_left)
                elif direction == IntersectionsDirections.Straight:
                    twist.v = 0.2
                    twist.omega = 0.0
                    self.pub_cmd_vel.publish(twist)
                    rospy.sleep(self.sleep_time_straight)
                elif direction == IntersectionsDirections.Right:
                    twist.v = 0.10
                    twist.omega = 0
                    rospy.sleep(0.2)
                    twist.v = 0.15
                    twist.omega = -2.5
                    self.pub_cmd_vel.publish(twist)
                    rospy.sleep(self.sleep_time_right)
                self.pub_intersection_finished.publish(Bool(data=True))
                rate.sleep()
                continue
            elif self._control_mode != ControlType.Stop:
                twist.v = self.v
                twist.omega = self.a
                # twist.v = 0.0
                # twist.omega = 0.0
            else:
                twist.v = 0.0
                twist.omega = 0.0

            self.pub_cmd_vel.publish(twist)
            self.pub_debug_v.publish(Float64(data=twist.v))
            self.pub_debug_omega.publish(Float64(data=twist.omega))

            rate.sleep()

if __name__ == '__main__':
    # create the node
    node = ControlLaneNode('control_lane_node')
    node.run()
    rospy.spin()
