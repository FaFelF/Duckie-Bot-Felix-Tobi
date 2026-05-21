#!/usr/bin/env python3

import os
import rospy
import numpy as np
import cv2
from std_msgs.msg import Float64, String, Bool, Int32
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import Rect
from explore_duckietown_ii.msg import DuckieDetection, DuckieDetectionArray
import util

class DetectDuckiesNode:
    def __init__(self, node_name):
        # initialize the ROS node
        rospy.init_node(node_name)

        #setzen von DuckieBot Namen, benötigt für das Abonnieren und Veröffentlichen von Topics
        self._vehicle_name = os.environ['VEHICLE_NAME']
        util.init_parameters(node_name,self.cbUpdateParameters)

        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"

        # Kameratopic abonnieren, jedes neue Bild wird automatisch an cbFindDuckies weitergegeben
        self.sub_image_original = rospy.Subscriber(self._camera_topic, CompressedImage, self.cbFindDuckies, queue_size = 1)

        #Publisher für das Ergebnis der Kreuzungserkennung
        self.pub_duckies = rospy.Publisher(f'/{self._vehicle_name}/detect/duckies', Bool, queue_size = 1)

        self.pub_debug_duckies = rospy.Publisher(f'/{self._vehicle_name}/debug/duckie_detection', DuckieDetectionArray, queue_size=1)
        self.pub_debug_contours = rospy.Publisher(f'/{self._vehicle_name}/debug/duckie_contours', CompressedImage, queue_size=1)

        self._crop_im_size = 400
        self.counter = 0
        self.is_running = False
        self.final_duckies = []

    def cbUpdateParameters(self, parameters):
        """
        Lädt Parameter aus der Konfigurationsdatei und aktualisiert die Instanzvariablen.
        Wird automatisch aufgerufen wenn Parameter über configuration_node geändert werden.

        Autor: Felix Faass

        Args:
            parameters (dict): Dictionary mit allen Parametern aus der JSON-Konfigurationsdatei
        """
        self.top_left_x = parameters["crop_image"]["top_left_x"]["default"]
        self.top_left_y = parameters["crop_image"]["top_left_y"]["default"]
        self.top_right_x = parameters["crop_image"]["top_right_x"]["default"]
        self.top_right_y = parameters["crop_image"]["top_right_y"]["default"]
        self.bottom_left_x = parameters["crop_image"]["bottom_left_x"]["default"]
        self.bottom_left_y = parameters["crop_image"]["bottom_left_y"]["default"]
        self.bottom_right_x = parameters["crop_image"]["bottom_right_x"]["default"]
        self.bottom_right_y = parameters["crop_image"]["bottom_right_y"]["default"]

        self.hue_yellow_l = parameters["yellow"]["hl"]["default"]
        self.hue_yellow_h = parameters["yellow"]["hh"]["default"]
        self.saturation_yellow_l = parameters["yellow"]["sl"]["default"]
        self.saturation_yellow_h = parameters["yellow"]["sh"]["default"]
        self.value_yellow_l = parameters["yellow"]["vl"]["default"]
        self.value_yellow_h = parameters["yellow"]["vh"]["default"]

        self.threshold_Duckie = parameters["detection"]["threshold"]["default"]
        self.min_circularity  = parameters["detection"]["circularity"]["default"]
        self.wh_min           = parameters["detection"]["wh_min"]["default"]
        self.wh_max           = parameters["detection"]["wh_max"]["default"]
    

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

    def cbFindDuckies(self, image_msg):
        rospy.loginfo_throttle(2, "cbFindDuckies called")
        if self.counter <= 3:
            self.counter += 1
            return

        # Verhindert dass ein neues Bild verarbeitet wird bevor das vorherige fertig ist
        if self.is_running:
            return

        self.is_running = True
        self.counter = 0

        np_arr = np.frombuffer(image_msg.data, np.uint8)
        cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.raw_img = cv_img

        try:
            self.fnDetectDuckies(cv_img)
        except Exception as e:
            rospy.logerr(f"fnDetectDuckies error: {e}")
        finally:
            self.is_running = False

    def fnDetectDuckies(self, image):
        """
        Hier soll der Code zur Erkennung von Duckies implementiert werden. Das Ergebnis (True wenn Duckie erkannt, sonst False)
        wird auf dem Topic /{vehicle_name}/detect/duckies veröffentlicht.

        Autor: Felix Faass

        Args:
            img: BGR-Kamerabild als numpy Array
        """


        self.final_duckies = []

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask_yellow = cv2.inRange(hsv, (self.hue_yellow_l, self.saturation_yellow_l, self.value_yellow_l), (self.hue_yellow_h, self.saturation_yellow_h, self.value_yellow_h))

        iterations = 1
        kernel = np.ones((5,5), np.uint8)
        opened_mask = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel, iterations=iterations)
        closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)

        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if True:
            debug_img = image.copy()
            for c in contours:
                area = cv2.contourArea(c)
                if area < 100:
                    continue
                perimeter = cv2.arcLength(c, True)
                circ = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 165, 255), 1)
                cv2.putText(debug_img, f"c:{circ:.2f} w/h:{w/h:.2f}", (x, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
            dbg_msg = CompressedImage()
            dbg_msg.header.stamp = rospy.Time.now()
            dbg_msg.format = "jpeg"
            dbg_msg.data = np.array(cv2.imencode('.jpg', debug_img)[1]).tobytes()
            self.pub_debug_contours.publish(dbg_msg)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.threshold_Duckie:
                continue
            perimeter = cv2.arcLength(contour, True)
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity < self.min_circularity:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if self.wh_min < (w / h) < self.wh_max:
                self.final_duckies.append((contour, circularity, x, y, w, h))

        self.pub_duckies.publish(Bool(data=len(self.final_duckies) > 0))


        #cv2.matchTemplate(contours, self.duckie_template, cv2.TM_CCOEFF_NORMED)


        


    def run_debug(self):
        """
        Hauptloop der Node. Publiziert das Debug-Bild mit markierten roten Pixeln
        auf dem Debug-Topic, solange Subscriber vorhanden sind.

        Autor: Felix Faass
        """
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.final_duckies is not None and self.pub_debug_duckies.get_num_connections() > 0:
                msg = DuckieDetectionArray()
                for contour, circularity, x, y, w, h in self.final_duckies:
                    d = DuckieDetection()
                    d.bounding_box = Rect(x=x, y=y, w=w, h=h)
                    d.circularity = circularity
                    msg.detections.append(d)
                self.pub_debug_duckies.publish(msg)
                

if __name__ == '__main__':
    node = DetectDuckiesNode('detect_duckies_node')
    node.run_debug()