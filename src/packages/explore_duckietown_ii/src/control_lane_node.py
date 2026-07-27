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
        # blocked = detect_duckies meldet: keine Luecke, durch die der Bot passt -> Pivot
        self.duckie_blocked = False
        # Pivot-Drehrichtung von detect_duckies: +1 = links, -1 = rechts
        self.duckie_pivot_dir = 1
        # Seite der zuletzt gesehenen NAHEN Ente (+1 links / -1 rechts). Danach steht sie
        # noch seitlich neben uns -> nicht dorthin lenken, bis wir weit genug weiter sind.
        self._last_duck_side = 0
        # ist die nahe Ente gerade noch sichtbar? Nur wenn NICHT, greift die Daempfung
        # (dann ist sie im toten Winkel neben uns).
        self._duck_visible = False
        # seit der letzten nahen Ente zurueckgelegte Strecke, aus v*dt integriert
        # (Koppelnavigation aus dem Sollwert, keine Encoder).
        self._pass_dist = 0.0
        self._loop_dt = 0.1   # passend zur rospy.Rate(10) in run()
        # Wackel-Pivot: Richtung (+1/-1) und Zeitpunkt des letzten Umschaltens
        self._wiggle_dir = 1.0
        self._wiggle_time = None

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
        self.sub_duckie_blocked = rospy.Subscriber(f'/{self._vehicle_name}/detect/duckie_blocked', Bool, self.cbDuckieBlocked, queue_size=1)
        self.sub_duckie_pivot_dir = rospy.Subscriber(f'/{self._vehicle_name}/detect/duckie_pivot_dir', Int32, self.cbDuckiePivotDir, queue_size=1)
        self.sub_duckie_side = rospy.Subscriber(f'/{self._vehicle_name}/detect/duckie_side', Int32, self.cbDuckieSide, queue_size=1)



      


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
        # Kriech-Untergrenze beim Ausweichen: solange eine passierbare Luecke da ist,
        # NICHT auf 0 bremsen (sonst kann der Differentialantrieb nicht herumlenken).
        # Angehalten wird nur, wenn detect_duckies "blockiert" meldet (keine Luecke).
        self.min_vel_duckie = parameters["pid_duckie"]["min_vel"]["default"]
        if self.min_vel_duckie > self.MAX_VEL_duckie:
            # Sonst gewinnt der Floor immer -> v haengt konstant auf min_vel, max_vel und
            # brake_gain sind wirkungslos (und man dreht an Reglern ohne jeden Effekt).
            rospy.logwarn("[control_lane] pid_duckie.min_vel (%.3f) > max_vel (%.3f) -> "
                          "auf max_vel begrenzt. Bitte Config pruefen.",
                          self.min_vel_duckie, self.MAX_VEL_duckie)
            self.min_vel_duckie = self.MAX_VEL_duckie
        # Horizontales Bremsen: wie stark |error| (seitlicher Ausschlag) das Tempo drueckt.
        self.duckie_brake_gain = parameters["pid_duckie"]["brake_gain"]["default"]
        # Pivot (kein Durchkommen): Drehgeschwindigkeit auf der Stelle. Muss gross genug
        # sein, um die Haftreibung bei v=0 zu ueberwinden (vgl. align.omega_min).
        self.pivot_omega = parameters["pid_duckie"]["pivot_omega"]["default"]
        # Wackel-Pivot: Schubstaerke und Umschaltintervall des Vor-/Zurueck-Ruckelns,
        # das die Haftreibung bricht, damit er sich auf der Stelle ueberhaupt dreht.
        self.wiggle_v = parameters["pid_duckie"]["wiggle_v"]["default"]
        self.wiggle_period = parameters["pid_duckie"]["wiggle_period"]["default"]
        # Wie WEIT (in m) der Bot nach der letzten nahen Ente fahren muss, bevor er wieder
        # zu ihrer Seite lenken darf (sie steht so lange noch neben ihm). Strecke statt
        # Zeit, damit es unabhaengig vom Tempo ist. Zu gross -> Kurven dorthin gehen nicht.
        self.pass_lock_distance = parameters["pid_duckie"]["pass_lock_distance"]["default"]
        # Wie stark im blinden Fenster noch zur Enten-Seite gelenkt werden darf (nicht 0 =
        # kein Verbot, nur Daempfung). Hoch = kaum Daempfung, 0 = hartes Verbot (alt).
        self.pass_lock_omega = parameters["pid_duckie"]["pass_lock_omega"]["default"]

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

    def cbDuckieBlocked(self, msg):
        self.duckie_blocked = msg.data

    def cbDuckiePivotDir(self, msg):
        # +1 = links, -1 = rechts (von detect_duckies: weg von der naechsten Linie)
        self.duckie_pivot_dir = 1 if msg.data >= 0 else -1

    def cbDuckieSide(self, msg):
        # Solange eine nahe Ente sichtbar ist (data != 0), Strecken-Zaehler zuruecksetzen
        # -> die Daempfung laeuft erst ab der LETZTEN Sichtung los. Und: solange sie
        # sichtbar ist, regelt die Ausweich-Logik selbst -> dann KEINE Daempfung.
        if msg.data != 0:
            self._last_duck_side = msg.data
            self._pass_dist = 0.0
            self._duck_visible = True
        else:
            self._duck_visible = False

    def fnApplyPassLock(self, omega):
        """Direkt nach dem Vorbeifahren steht die Ente noch SEITLICH neben dem Bot - sie ist
        nur aus dem Kamerabild verschwunden, nicht aus der Welt (toter Winkel). In diesem
        kurzen blinden Fenster wird das Lenken zu ihrer Seite GEDAEMPFT, damit der Bot sie
        nicht mit der Flanke streift.

        Bewusst nur daempfen, nicht verbieten: ein hartes Verbot hat ihn in der Kurve aus
        der Strecke getragen, weil er nicht mehr zurueklenken durfte. Er darf also weiter
        in ihre Richtung, nur nicht reissen.

        Greift NUR, wenn die Ente nicht mehr sichtbar ist - solange man sie sieht, regelt
        die Ausweich-Logik ohnehin um sie herum. Und ueber die zurueckgelegte STRECKE
        (v*dt) statt ueber Zeit: nur echtes Vorwaertsfahren bringt uns vorbei. Beim Pivot
        (v=0) zaehlt nichts hoch - richtig, Drehen auf der Stelle kommt an keiner Ente vorbei.
        """
        if self.duckie_blocked:
            # Pivot ist ein bewusstes Manoever und die einzige Option, wenn nichts mehr
            # durchpasst -> NICHT daempfen, sonst friert der Bot genau wieder ein.
            return omega
        if self._last_duck_side == 0 or self._duck_visible:
            return omega
        if self._pass_dist >= self.pass_lock_distance:
            return omega
        lim = self.pass_lock_omega
        if self._last_duck_side > 0:
            return min(omega, lim)    # Ente war LINKS -> nach links nur noch gedaempft
        return max(omega, -lim)       # Ente war RECHTS -> nach rechts nur noch gedaempft

    def cbAvoidDuckie(self, msg):
        if self._control_mode != ControlType.Obstacle:
            return

        # Ente zu nah/zu eng zum Drumherumfahren -> PIVOT statt einfrieren: auf der
        # Stelle drehen, WEG von der naechsten Linie (Richtung kommt von detect_duckies).
        # Gedreht wird, bis wieder eine passierbare Luecke da ist (dann faellt
        # duckie_blocked von selbst weg).
        if self.duckie_blocked:
            # WACKEL-PIVOT: bei v=0 stallen die Motoren auf dieser HW (Haftreibung +
            # Stuetzkugel), er dreht sich dann gar nicht. Trick: v staendig zwischen vor
            # und zurueck umschalten -> die Raeder rollen (nur kinetische Reibung), der
            # Versatz hebt sich netto auf, und omega dreht durch. Gleiches Prinzip wie der
            # Pendel-Bogen in fnAlign, nur schneller getaktet.
            now = rospy.Time.now()
            if self._wiggle_time is None or \
                    (now - self._wiggle_time).to_sec() >= self.wiggle_period:
                self._wiggle_dir *= -1.0
                self._wiggle_time = now
            self.v = self.wiggle_v * self._wiggle_dir
            # Drehrichtung kommt als +1/-1 von detect_duckies (_pivot_direction) und meint
            # immer WEG von dem, was im Weg ist. Vorzeichen wie beim Abbiegen an der
            # Kreuzung (links = omega +1.2, rechts = omega -2.5):
            #   duckie_pivot_dir = +1 -> omega positiv -> dreht nach LINKS
            #                            (weisse Linie oder Ente rechts im Weg)
            #   duckie_pivot_dir = -1 -> omega negativ -> dreht nach RECHTS
            #                            (gelbe Linie oder Ente links im Weg)
            self.a = self.pivot_omega * self.duckie_pivot_dir
            self.integral = 0.0
            self.lastError = 0.0
            return
        self._wiggle_time = None   # Pivot vorbei -> Takt fuer den naechsten neu starten

        # duckie_error kommt bereits auf [-1, 1] normiert (halbe Bildbreite), also in der
        # GLEICHEN Konvention wie /detect/lane -> pid_duckie ist direkt mit pid vergleichbar.
        # Frueher: rohe Pixel /750 -> Bereich nur ~+-0.43, damit hatte der Enten-Modus rund
        # ein Drittel der Lenkautoritaet des Spur-Reglers und trug in Kurven raus.
        error = msg.data

        self.integral += error
        self.integral = max(-1.0, min(1.0, self.integral))
        p = self.kp_duckie * error
        i = self.ki_duckie * self.integral
        d = self.kd_duckie * ((error - self.lastError) / 0.1)
        self.lastError = error

        # Horizontal bremsen: je groesser der seitliche Ausweich-Ausschlag (|error|),
        # desto langsamer -> mehr Zeit fuer scharfe Ausweichboegen. NICHT mehr ueber die
        # vertikale Distanz. min_vel_duckie haelt Vortritt (Differentialantrieb muss
        # rollen, um herumzulenken); echt gestoppt wird nur oben (blocked).
        self.v = max(self.min_vel_duckie,
                     self.MAX_VEL_duckie * max(0.0, 1 - abs(error) * self.duckie_brake_gain))

        self.a = p + i + d




    def fnAlign(self, twist):
        """
        Dreht den Bot auf der Stelle, bis der rote Streifen (gemessener Winkel)
        waagerecht ist -> Bot steht orthogonal zur Kreuzung.

        Bewusst KEIN exakter Null-Abgleich: ein Toleranzband (align_tolerance)
        verhindert Schwingen/Pendeln. Innerhalb der Toleranz wird gestoppt,
        sobald der Wert über mehrere Zyklen stabil ist. Ein Timeout sorgt dafür,
        dass das Manöver auch bei fehlender/verrauschter Linie nicht hängen bleibt.
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
                # Strecke seit der letzten nahen Ente mitzaehlen (Koppelnavigation aus
                # dem kommandierten v) -> die Sperre unten laeuft ueber Strecke, nicht Zeit.
                # Beim Wackel-Pivot NICHT zaehlen: er ruckelt vor/zurueck, kommt netto aber
                # nicht vom Fleck - sonst liefe die Pass-Daempfung ab, ohne dass er an der
                # Ente vorbei ist.
                if not self.duckie_blocked:
                    self._pass_dist += abs(twist.v) * self._loop_dt
                # Gerade an einer Ente vorbei? Dann nicht in ihre Richtung zurueklenken,
                # sie steht noch seitlich neben uns. Gilt fuer Lane UND Obstacle, aber
                # NICHT fuer Align/Intersection (bewusste Manoever, oben abgehandelt).
                twist.omega = self.fnApplyPassLock(self.a)
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
