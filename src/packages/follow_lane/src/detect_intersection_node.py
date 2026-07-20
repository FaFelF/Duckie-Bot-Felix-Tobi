#!/usr/bin/env python3

import os
import rospy
import numpy as np
import cv2
from std_msgs.msg import Float64, String, Bool
from sensor_msgs.msg import CompressedImage
import util

class DetectIntersectionNode:
    def __init__(self, node_name):
        # initialize the ROS node
        rospy.init_node(node_name)

        #setzen von DuckieBot Namen, benötigt für das Abonnieren und Veröffentlichen von Topics
        self._vehicle_name = os.environ['VEHICLE_NAME']
        util.init_parameters(node_name,self.cbUpdateParameters)

        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"

        # Kameratopic abonnieren, jedes neue Bild wird automatisch an cbFindIntersection weitergegeben
        self.sub_image_original = rospy.Subscriber(self._camera_topic, CompressedImage, self.cbFindIntersection, queue_size = 1)

        #Publisher für das Ergebnis der Kreuzungserkennung
        self.pub_intersection = rospy.Publisher(f'/{self._vehicle_name}/detect/intersection', Bool, queue_size = 1)

        self._crop_im_size = 400
        self.is_running = False
        self.counter = 0

        self.pub_debug_red = rospy.Publisher(f'/{self._vehicle_name}/debug/lane_red', CompressedImage, queue_size=1)
        self.pub_debug_raw = rospy.Publisher(f'/{self._vehicle_name}/debug/raw', CompressedImage, queue_size=1)
        self.debug_img = None
        self.raw_img = None


    def cbUpdateParameters(self, parameters):
        """
        Lädt Parameter aus der Konfigurationsdatei und aktualisiert die Instanzvariablen.
        Wird automatisch aufgerufen wenn Parameter über configuration_node geändert werden.

        Autor: Felix Faass

        Args:
            parameters (dict): Dictionary mit allen Parametern aus der JSON-Konfigurationsdatei
        """
        # Update perspective transform points
        self.top_left_x = parameters["crop_image"]["top_left_x"]["default"]
        self.top_left_y = parameters["crop_image"]["top_left_y"]["default"]
        self.top_right_x = parameters["crop_image"]["top_right_x"]["default"]
        self.top_right_y = parameters["crop_image"]["top_right_y"]["default"]
        self.bottom_left_x = parameters["crop_image"]["bottom_left_x"]["default"]
        self.bottom_left_y = parameters["crop_image"]["bottom_left_y"]["default"]
        self.bottom_right_x = parameters["crop_image"]["bottom_right_x"]["default"]
        self.bottom_right_y = parameters["crop_image"]["bottom_right_y"]["default"]

        # Update red 1 line parameters
        self.hue_red1_l = parameters["red1"]["hl"]["default"]
        self.hue_red1_h = parameters["red1"]["hh"]["default"]
        self.saturation_red1_l = parameters["red1"]["sl"]["default"]
        self.saturation_red1_h = parameters["red1"]["sh"]["default"]
        self.lightness_red1_l = parameters["red1"]["vl"]["default"]
        self.lightness_red1_h = parameters["red1"]["vh"]["default"]

        # Update red 2 line parameters
        self.hue_red2_l = parameters["red2"]["hl"]["default"]
        self.hue_red2_h = parameters["red2"]["hh"]["default"]
        self.saturation_red2_l = parameters["red2"]["sl"]["default"]
        self.saturation_red2_h = parameters["red2"]["sh"]["default"]
        self.lightness_red2_l = parameters["red2"]["vl"]["default"]
        self.lightness_red2_h = parameters["red2"]["vh"]["default"]

        self.thresh_red_pixels = parameters["detection"]["thresh"]["default"]

    def crop_img(self, img):
        """
        Wendet eine perspektivische Transformation auf das Bild an und schneidet es auf crop_im_size x crop_im_size zu.

        Autor: Felix Faass

        Args:
            img: BGR-Kamerabild als numpy Array

        Returns:
            numpy Array: transformiertes und zugeschnittenes Bild
        """
        img = img.copy()

        pts1 = np.float32([
            [self.top_left_x,     self.top_left_y],
            [self.top_right_x,    self.top_right_y],
            [self.bottom_right_x, self.bottom_right_y],
            [self.bottom_left_x,  self.bottom_left_y],])

        pts2 = np.float32([[0,0],[self._crop_im_size,0],[0,self._crop_im_size],[self._crop_im_size,self._crop_im_size]])

        M = cv2.getPerspectiveTransform(pts1,pts2)
        return cv2.warpPerspective(img,M,(self._crop_im_size,self._crop_im_size))


    def cbFindIntersection(self, image_msg):
        """
        ROS-Callback, wird automatisch aufgerufen wenn ein neues Kamerabild ankommt.
        Dekodiert das Bild und übergibt es an detect_intersection.

        Autor: Felix Faass

        Args:
            image_msg (CompressedImage): komprimiertes Kamerabild von ROS
        """
        # Erste 3 Frames überspringen, Kamera liefert beim Start unsaubere Bilder
        if self.counter <= 3:
            self.counter += 1
            return

        # Verhindert dass ein neues Bild verarbeitet wird bevor das vorherige fertig ist
        if self.is_running:
            return

        self.is_running = True
        self.counter = 0

        # Convert the compressed image message to a numpy array
        np_arr = np.frombuffer(image_msg.data, np.uint8)

        # Decode the image from the numpy array
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.raw_img = cv_image

        intersection_detected = self.detect_intersection(cv_image)

        # Publish the result as a Bool message
        self.pub_intersection.publish(Bool(data=intersection_detected))

        self.is_running = False

    def detect_intersection(self, image):
        """
        Analysiert ein Kamerabild und erkennt ob ein roter Streifen (Kreuzung) sichtbar ist.
        Zählt rote Pixel im unteren Bilddrittel und vergleicht mit dem Schwellwert.

        Autor: Felix Faass

        Args:
            image: BGR-Kamerabild als numpy Array

        Returns:
            bool: True wenn roter Streifen erkannt (Kreuzung vorhanden), sonst False
        """
        # Nur das untere Drittel des Bildes analysieren, roter Streifen liegt auf dem Boden direkt vor dem Bot
        height = image.shape[0]
        roi = image[int(height * 0.66):, :]
        # crop_img wird hier nicht benötigt, ROI-Slice reicht für die Farberkennung
        #img = self.crop_img(roi)

        # Convert the ROI to HSV color space
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Create masks for the red lines using the specified HSV ranges
        mask_red1 = cv2.inRange(hsv,
                           (self.hue_red1_l,self.saturation_red1_l, self.lightness_red1_l),
                           (self.hue_red1_h,self.saturation_red1_h, self.lightness_red1_h),)

        mask_red2 = cv2.inRange(hsv,
                           (self.hue_red2_l,self.saturation_red2_l, self.lightness_red2_l),
                           (self.hue_red2_h,self.saturation_red2_h, self.lightness_red2_h),)

        # Combine the two red masks
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # Count the number of red pixels in the combined mask
        num_red_pixels = cv2.countNonZero(mask_red)

        self.debug_img = roi.copy()
        self.debug_img[mask_red > 0] = (0, 0, 255)
        cv2.putText(self.debug_img, f"Red pixels: {num_red_pixels}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Schwellwert wird aus der Konfigurationsdatei geladen und kann dort angepasst werden
        if num_red_pixels > self.thresh_red_pixels:
            return True

        return False

    def run_debug(self):
        """
        Hauptloop der Node. Publiziert das Debug-Bild mit markierten roten Pixeln
        auf dem Debug-Topic, solange Subscriber vorhanden sind.

        Autor: Felix Faass
        """
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.debug_img is not None and self.pub_debug_red.get_num_connections() > 0:
                debug_msg = CompressedImage()
                debug_msg.header.stamp = rospy.Time.now()
                debug_msg.format = "jpeg"
                debug_msg.data = np.array(cv2.imencode('.jpg', self.debug_img)[1]).tobytes()
                self.pub_debug_red.publish(debug_msg)
            if self.raw_img is not None and self.pub_debug_raw.get_num_connections() > 0:
                raw_msg = CompressedImage()
                raw_msg.header.stamp = rospy.Time.now()
                raw_msg.format = "jpeg"
                raw_msg.data = np.array(cv2.imencode('.jpg', self.raw_img)[1]).tobytes()
                self.pub_debug_raw.publish(raw_msg)
            rate.sleep()


if __name__ == '__main__':
    node = DetectIntersectionNode('detect_intersection_node')
    node.run_debug()
