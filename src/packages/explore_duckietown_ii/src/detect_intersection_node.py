#!/usr/bin/env python3

import os
import rospy
import numpy as np
import cv2
from std_msgs.msg import Float64, String, Bool, Int32
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

        self._crop_im_size = 400
        self.is_running = False
        self.counter = 0
        self.debug_img = None
        self.raw_img = None

        #Publisher für das Ergebnis der Kreuzungserkennung
        self.pub_intersection = rospy.Publisher(f'/{self._vehicle_name}/detect/intersection', Bool, queue_size = 1)
        self.pub_intersection_approaching = rospy.Publisher(f'/{self._vehicle_name}/detect/intersection_approaching', Bool, queue_size=1)
        self.pub_debug_red = rospy.Publisher(f'/{self._vehicle_name}/debug/lane_red', CompressedImage, queue_size=1)
        self.pub_red_count = rospy.Publisher(f'/{self._vehicle_name}/debug/red_pixel_count', Int32, queue_size=1)
        self.pub_debug_raw = rospy.Publisher(f'/{self._vehicle_name}/debug/raw', CompressedImage, queue_size=1)

        # Subscriber zuletzt registrieren, damit alle Publisher bereit sind
        self.sub_image_original = rospy.Subscriber(self._camera_topic, CompressedImage, self.cbFindIntersection, queue_size = 1)


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
        self.thresh_red_pixels_apriltag = parameters["detection"]["thresh_apriltag"]["default"]

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

        if rospy.is_shutdown():
            return

        intersection_detected = self.detect_intersection(cv_image)

        self.pub_intersection.publish(Bool(data=intersection_detected))

        self.is_running = False

    def fnGetRedMask(self, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_red1 = cv2.inRange(hsv,
            (self.hue_red1_l, self.saturation_red1_l, self.lightness_red1_l),
            (self.hue_red1_h, self.saturation_red1_h, self.lightness_red1_h))
        mask_red2 = cv2.inRange(hsv,
            (self.hue_red2_l, self.saturation_red2_l, self.lightness_red2_l),
            (self.hue_red2_h, self.saturation_red2_h, self.lightness_red2_h))
        return cv2.bitwise_or(mask_red1, mask_red2)

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
        height = image.shape[0]
        width  = image.shape[1]
        roi = image[int(height * 0.66):, int(width * 0.3):]

        roi_detect_Apriltag = image[int(height * 0.4):int(height * 0.5), int(width * 0.3):int(width * 0.7)]
        mask_detect_Apriltag = self.fnGetRedMask(roi_detect_Apriltag)
        num_red_pixels_detect_Apriltag = cv2.countNonZero(mask_detect_Apriltag)
        self.pub_intersection_approaching.publish(Bool(data=num_red_pixels_detect_Apriltag > self.thresh_red_pixels_apriltag))




        mask_red = self.fnGetRedMask(roi)
        num_red_pixels = cv2.countNonZero(mask_red)

        self.debug_img = roi.copy()
        self.debug_img[mask_red > 0] = (0, 0, 255)
        cv2.putText(self.debug_img, f"Red pixels: {num_red_pixels}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        self.pub_red_count.publish(Int32(data=num_red_pixels))

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
