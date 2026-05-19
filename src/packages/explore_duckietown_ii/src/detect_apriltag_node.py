#!/usr/bin/env python3
import os
import rospy
import numpy as np
import cv2
from duckietown_msgs.msg import AprilTagDetection, AprilTagDetectionArray
from std_msgs.msg import Int32
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

        self.pub_debug_locations_apriltag = rospy.Publisher(f'/{self._vehicle_name}/debug/apriltag_locations', AprilTagDetectionArray, queue_size=1)

    def cbUpdateParameters(self, parameters):
        """
        Lädt Parameter aus der Konfigurationsdatei und aktualisiert die Instanzvariablen.
        Wird automatisch aufgerufen wenn Parameter über configuration_node geändert werden.

        Autor: Felix Faass

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

        self._detector = Detector(
            families=self.apriltag_family,
            nthreads=self.apriltag_nthreads,
            quad_decimate=self.apriltag_quad_decimate,
            quad_sigma=self.apriltag_quad_sigma,
            refine_edges=self.apriltag_refine_edges,
            decode_sharpening=self.apriltag_decode_sharpening,
            debug=self.apriltag_debug
        )

    def cbFindAprilTag(self, msg):
        """
        Callback-Funktion für das Abonnieren des Kameratopics. Verarbeitet jedes empfangene Bild, um AprilTags zu erkennen.

        Autor: Felix Faass

        Args:
            msg (CompressedImage): Das empfangene Bild im komprimierten Format
        """
        try:
            if self._detector is None:
                rospy.logwarn("detector is None")
                return

            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            width = image.shape[1]
            roi = image[:, int(width * self.roi_width):]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            tags = self._detector.detect(gray, estimate_tag_pose=False)

            rospy.loginfo_throttle(2, f"AprilTag detect: {len(tags)} tags found")

            if tags:
                tag_detections = AprilTagDetectionArray()

                for tag in tags:
                    detection = AprilTagDetection()
                    corners_offset = tag.corners.astype(float) + [int(width * self.roi_width), 0]
                    detection.corners = corners_offset.flatten().tolist()
                    detection.center = [corners_offset[:, 0].min(), corners_offset[:, 1].max() + 20]
                    detection.tag_id = tag.tag_id
                    tag_detections.detections.append(detection)

                largest = max(tags, key=lambda t: (t.corners[:,0].max() - t.corners[:,0].min()) *
                                                   (t.corners[:,1].max() - t.corners[:,1].min()))
                self.pub_apriltag.publish(largest.tag_id)

                if self.pub_debug_locations_apriltag.get_num_connections() > 0:
                    self.pub_debug_locations_apriltag.publish(tag_detections)
            else:
                self.pub_apriltag.publish(-1)

        except Exception as e:
            rospy.logerr(f"cbFindAprilTag error: {e}")


if __name__ == '__main__':
    node = DetectAprilTagNode('detect_apriltag_node')
    rospy.spin()
