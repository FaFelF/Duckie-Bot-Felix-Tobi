#!/usr/bin/env python3
import os
import rospy
import numpy as np
import cv2
from duckietown_msgs.msg import AprilTagDetection, AprilTagDetectionArray
from std_msgs.msg import Int32, Bool
from sensor_msgs.msg import CompressedImage
from pupil_apriltags import Detector
import util


class DetectAprilTagNode:
    def __init__(self, node_name):
        # initialize the ROS node
        rospy.init_node(node_name)

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._detector = None
        util.init_parameters(node_name, self.cbUpdateParameters)

        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"

        # Kameratopic abonnieren, jedes neue Bild wird automatisch an cbFindAprilTag weitergegeben
        self.sub_image_original = rospy.Subscriber(self._camera_topic, CompressedImage, self.cbFindAprilTag, queue_size=1)

        self.pub_apriltag = rospy.Publisher(f'/{self._vehicle_name}/detect/apriltag', Int32, queue_size=1)

        self.is_running = False
        self.counter = 0

        # Groesstes sicher erkanntes Schild der aktuellen Anfahrt (bleibt bis zur naechsten Kreuzung).
        self._peak_area = 0.0
        self._peak_tag_id = -1
        self._approaching = False
        self.pub_saved_apriltag = rospy.Publisher(f'/{self._vehicle_name}/detect/saved_apriltag', Int32, queue_size=1)
        self.sub_intersection_approaching = rospy.Subscriber(f'/{self._vehicle_name}/detect/intersection_approaching', Bool, self.cbIntersectionApproaching, queue_size=1)
        self.sub_intersection_finished = rospy.Subscriber(f'/{self._vehicle_name}/switch/intersection_finished', Bool, self.cbIntersectionFinished, queue_size=1)

        self.pub_debug_locations_apriltag = rospy.Publisher(f'/{self._vehicle_name}/debug/apriltag_locations', AprilTagDetectionArray, queue_size=1)

        # Watchdog meldet Kamera-Ausfall (der Callback feuert dann nicht mehr).
        self._cam_last = None
        rospy.Timer(rospy.Duration(1.0), self._cam_watchdog)

    def cbUpdateParameters(self, parameters):
        """
        Laedt die Parameter aus der Konfigurationsdatei und aktualisiert die Instanzvariablen.

        Args:
            parameters (dict): Dictionary mit allen Parametern aus der JSON-Konfigurationsdatei
        """
        self.apriltag_family            = parameters["apriltag"]["family"]["default"]
        self.apriltag_nthreads          = parameters["apriltag"]["nthreads"]["default"]
        self.apriltag_quad_decimate     = parameters["apriltag"]["quad_decimate"]["default"]
        self.apriltag_quad_sigma        = parameters["apriltag"]["quad_sigma"]["default"]
        self.apriltag_refine_edges      = parameters["apriltag"]["refine_edges"]["default"]
        self.apriltag_decode_sharpening = parameters["apriltag"]["decode_sharpening"]["default"]
        self.apriltag_debug             = parameters["apriltag"]["debug"]["default"]
        self.roi_width                  = parameters["roi"]["width"]["default"]
        # Mindest-Schildflaeche (px^2): verwirft zu weit entfernte Schilder. 0 = Filter aus.
        self.min_tag_area               = parameters["detection"]["min_tag_area"]["default"]

        self._detector = Detector(
            families=self.apriltag_family,
            nthreads=self.apriltag_nthreads,
            quad_decimate=self.apriltag_quad_decimate,
            quad_sigma=self.apriltag_quad_sigma,
            refine_edges=self.apriltag_refine_edges,
            decode_sharpening=self.apriltag_decode_sharpening,
            debug=self.apriltag_debug
        )

    def cbIntersectionApproaching(self, msg):
        self._approaching = msg.data
        if msg.data:
            # Den waehrend der Anfahrt groessten Tag festhalten.
            self.pub_saved_apriltag.publish(Int32(data=self._peak_tag_id))

    def cbIntersectionFinished(self, msg):
        # Peak fuer die naechste Kreuzung zuruecksetzen.
        if msg.data:
            self._peak_area = 0.0
            self._peak_tag_id = -1

    def _cam_watchdog(self, _event):
        """Meldet, wenn laenger als 1 s kein Kamerabild kam."""
        if self._cam_last is None:
            return
        age_ms = (rospy.get_time() - self._cam_last) * 1000
        if age_ms > 1000:
            rospy.logwarn_throttle(2.0, "[apriltag] Kamera ausgefallen: seit %.0f ms kein Bild", age_ms)

    def cbFindAprilTag(self, msg):
        """
        Verarbeitet jedes empfangene Kamerabild, um AprilTags zu erkennen.

        Args:
            msg (CompressedImage): Das empfangene Bild im komprimierten Format
        """
        try:
            # Bild nach groesserer Luecke -> Kamera war kurz weg, einmal melden.
            now = rospy.get_time()
            if self._cam_last is not None:
                gap_ms = (now - self._cam_last) * 1000
                if gap_ms > 500:
                    rospy.logwarn("[apriltag] Kamera wieder da nach %.0f ms Aussetzer", gap_ms)
            self._cam_last = now

            if self._detector is None:
                rospy.logwarn("detector is None")
                return

            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            width = image.shape[1]
            roi = image[:, int(width * self.roi_width):]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            tags = self._detector.detect(gray, estimate_tag_pose=False)

            if tags:
                tag_detections = AprilTagDetectionArray()

                for tag in tags:
                    detection = AprilTagDetection()
                    corners_offset = tag.corners.astype(float) + [int(width * self.roi_width), 0]
                    detection.corners = corners_offset.flatten().tolist()
                    detection.center = [corners_offset[:, 0].min(), corners_offset[:, 1].max() + 20]
                    detection.tag_id = tag.tag_id
                    tag_detections.detections.append(detection)

                def tag_area(t):
                    return (t.corners[:,0].max() - t.corners[:,0].min()) * \
                           (t.corners[:,1].max() - t.corners[:,1].min())

                # Nur sicher dekodierte (hamming==0, sonst verwechselte IDs) und nah genug
                # (Flaeche >= min_tag_area) gelten.
                confident = [t for t in tags if t.hamming == 0 and tag_area(t) >= self.min_tag_area]
                if confident:
                    largest = max(confident, key=tag_area)
                    self.pub_apriltag.publish(largest.tag_id)
                else:
                    self.pub_apriltag.publish(-1)

                # Peak aktualisieren, falls dieses Schild groesser als das bisher groesste ist.
                if confident:
                    best = max(confident, key=tag_area)
                    best_area = tag_area(best)
                    if best_area > self._peak_area:
                        self._peak_area = best_area
                        self._peak_tag_id = best.tag_id

                if self.pub_debug_locations_apriltag.get_num_connections() > 0:
                    self.pub_debug_locations_apriltag.publish(tag_detections)
            else:
                self.pub_apriltag.publish(-1)
                if self.pub_debug_locations_apriltag.get_num_connections() > 0:
                    self.pub_debug_locations_apriltag.publish(AprilTagDetectionArray())

        except Exception as e:
            rospy.logerr(f"cbFindAprilTag error: {e}")


if __name__ == '__main__':
    node = DetectAprilTagNode('detect_apriltag_node')
    rospy.spin()
