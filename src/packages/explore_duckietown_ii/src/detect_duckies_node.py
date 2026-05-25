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

import onnxruntime as ort

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

        self.model = '/root/DuckieRace/src/packages/explore_duckietown_ii/models/duckie.onnx'

        
        self.session = ort.InferenceSession(self.model)
        self.input_name = self.session.get_inputs()[0].name

        self.conf_threshold = 0.3
        self.duckie_distance_activation_threshhold = 400
        self.image_middle = 320
        self.center_white = None
        self.center_yellow = None

        self.pub_duckie_error = rospy.Publisher(f'/{self._vehicle_name}/detect/duckie_error', Float64, queue_size=1)

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

        self.threshold_Duckie = parameters["detection"]["threshold"]["default"]

        self.hue_white_l = parameters["white"]["hl"]["default"]
        self.hue_white_h = parameters["white"]["hh"]["default"]
        self.saturation_white_l = parameters["white"]["sl"]["default"]
        self.saturation_white_h = parameters["white"]["sh"]["default"]
        self.lightness_white_l = parameters["white"]["vl"]["default"]
        self.lightness_white_h = parameters["white"]["vh"]["default"]

        self.hue_yellow_l = parameters["yellow"]["hl"]["default"]
        self.hue_yellow_h = parameters["yellow"]["hh"]["default"]
        self.saturation_yellow_l = parameters["yellow"]["sl"]["default"]
        self.saturation_yellow_h = parameters["yellow"]["sh"]["default"]
        self.lightness_yellow_l = parameters["yellow"]["vl"]["default"]
        self.lightness_yellow_h = parameters["yellow"]["vh"]["default"]

        self.conf_threshold = parameters["detection"]["conf_threshold"]["default"]
    

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
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (640,480))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img,0)  

        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv_image = cv2.resize(cv_image, (640,480))
        self.image_middle = cv_image.shape[1] // 2

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        self.fnDetectDuckies(img)

        duckie_mask = np.zeros(cv_image.shape[:2], dtype=np.uint8)
        for x1, y1, x2, y2, conf in self.final_duckies:
            duckie_mask[int(y1):int(y2), int(x1):int(x2)] = 255


        lowest_duckie = None
        for duckie in self.final_duckies:
            if lowest_duckie is None or duckie[3] > lowest_duckie[3]:
                lowest_duckie = duckie
        
        if lowest_duckie is not None and lowest_duckie[3] <= self.duckie_distance_activation_threshhold:
            #Enten-Bereiche im HSV-Bild schwärzen
            hsv[duckie_mask == 255] = 0

            #Fahrspur erkennen
            self.fnDetectLane(hsv, lowest_duckie[3])

            #Entenfehler berechnen (distanz der Ente zu den Fahrspurlinien)
            duckie_error = self.fnGetLaneDuckieError(lowest_duckie)
            if duckie_error is not None:
                self.pub_duckie_error.publish(Float64(data=duckie_error))       
    
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

        raw_output = self.session.run(None, {self.input_name:image})
        detections = raw_output[0][0] 

        for  box in detections:
            x1, y1, x2, y2, conf, class_id = box
            if conf >= self.conf_threshold:
                self.final_duckies.append((x1, y1, x2, y2, conf))
                

        self.pub_duckies.publish(Bool(data=len(self.final_duckies) > 0))


        #cv2.matchTemplate(contours, self.duckie_template, cv2.TM_CCOEFF_NORMED)
        
    def get_x_for_driving(self, mask, distance, no_lane_value, left_line):
        grad = cv2.Sobel(mask, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

        th1 = []

        if left_line:
            _,th1 = cv2.threshold(-grad,127,255,cv2.THRESH_BINARY)
        else:
            _,th1 = cv2.threshold(grad,127,255,cv2.THRESH_BINARY)


        a = []
        if left_line:
            for row in range(max(0, distance-50), min(len(mask), distance+50)):
                for x in range(self.image_middle, 0,-1):
                    if th1[row][x] == 255:
                        a.append(x)
                        break
        else:
            for row in range(max(0, distance-50), min(len(mask), distance+50)):
                for x in range(self.image_middle, len(mask[0])):
                    if th1[row][x] == 255:
                        a.append(x)
                        break
                
                

        if len(a) > 10:
            return np.median(a)
        else:
            return no_lane_value

    def fnDetectLane(self, hsv, distance):

        mask_yellow = cv2.inRange(hsv,
                               (self.hue_yellow_l,self.saturation_yellow_l, self.lightness_yellow_l),
                               (self.hue_yellow_h,self.saturation_yellow_h, self.lightness_yellow_h),)

        mask_white = cv2.inRange(hsv,
                               (self.hue_white_l,self.saturation_white_l, self.lightness_white_l),
                               (self.hue_white_h,self.saturation_white_h, self.lightness_white_h),)

        #Werte Müssen noch getestet werden
        white_alternative = int(len(hsv[0]) * 0.95)
        yellow_alternative = int(len(hsv[0]) * 0.05)


        self.center_white = self.get_x_for_driving(mask_white, distance, white_alternative, left_line=False)
        self.center_yellow = self.get_x_for_driving(mask_yellow, distance, yellow_alternative, left_line=True)

    def fnGetLaneDuckieError(self, lowest_duckie):
        if self.center_white is not None and self.center_yellow is not None:
            Distance_Duckie_to_white = abs(lowest_duckie[2] - self.center_white)
            Distance_Duckie_to_yellow = abs(self.center_yellow - lowest_duckie[0])
            if Distance_Duckie_to_white < Distance_Duckie_to_yellow:
                largest_gap = self.center_yellow + Distance_Duckie_to_yellow/2
            else:
                largest_gap = self.center_white - Distance_Duckie_to_white/2

            error = self.image_middle - largest_gap
            return error

        else:
            return None

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
                for x1, y1, x2, y2, conf in self.final_duckies:
                    d = DuckieDetection()
                    d.bounding_box = Rect(x=x1, y=y1, w=x2-x1, h=y2-y1)
                    d.confidence = conf
                    msg.detections.append(d)
                self.pub_debug_duckies.publish(msg)
                

if __name__ == '__main__':
    node = DetectDuckiesNode('detect_duckies_node')
    node.run_debug()