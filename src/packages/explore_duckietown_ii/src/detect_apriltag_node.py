import os
import rospy
import numpy as np
import cv2
from std_msgs.msg import Int32
from sensor_msgs.msg import CompressedImage
from pupil_apriltags import Detector
import util


class DetectAprilTagNode:
    def __init__(self, node_name):
        # initialize the ROS node
        rospy.init_node(node_name)

        self._vehicle_name = os.environ['VEHICLE_NAME']
        util.init_parameters(node_name, self.cbUpdateParameters)

        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"

        # Kameratopic abonnieren, jedes neue Bild wird automatisch an cbFindAprilTag weitergegeben
        self.sub_image_original = rospy.Subscriber(self._camera_topic, CompressedImage, self.cbFindAprilTag, queue_size=1)

        self.pub_apriltag = rospy.Publisher(f'/{self._vehicle_name}/detect/apriltag', Int32, queue_size=1)

        self.is_running = False
        self.counter = 0
        self._detector = None

        self.pub_debug_location_apriltag = rospy.Publisher(f'/{self._vehicle_name}/debug/apriltag_location', CompressedImage, queue_size=1)

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
        if self._detector is None:
            return

        image = util.compressed_img_to_cv2(msg)
        width  = image.shape[1]
        roi = image[:, int(width * self.roi_width):]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        tags = self._detector.detect(gray, estimate_tag_pose=False)

        if tags:
            self.pub_apriltag.publish(tags[0].tag_id)
            apriltag_location = tags[0].corners.astype(int)
            corners = apriltag_location + [int(width * self.roi_width), 0]

            if self.pub_debug_location_apriltag.get_num_connections() > 0:
                debug_img = image.copy()
                cv2.polylines(debug_img, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
                text_pos = (corners[:, 0].min(), corners[:, 1].max() + 20)
                cv2.putText(debug_img, str(tags[0].tag_id), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                debug_msg = CompressedImage()
                debug_msg.header.stamp = rospy.Time.now()
                debug_msg.format = "jpeg"
                debug_msg.data = np.array(cv2.imencode('.jpg', debug_img)[1]).tobytes()
                self.pub_debug_location_apriltag.publish(debug_msg)
        else:
            self.pub_apriltag.publish(-1)


if __name__ == '__main__':
    node = DetectAprilTagNode('detect_apriltag_node')
    rospy.spin()
