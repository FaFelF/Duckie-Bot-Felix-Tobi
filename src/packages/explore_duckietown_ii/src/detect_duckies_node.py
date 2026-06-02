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
import time
import threading

class DetectDuckiesNode:
    def __init__(self, node_name):
        # initialize the ROS node
        rospy.init_node(node_name)

        #setzen von DuckieBot Namen, benötigt für das Abonnieren und Veröffentlichen von Topics
        self._vehicle_name = os.environ['VEHICLE_NAME']
        util.init_parameters(node_name,self.cbUpdateParameters)

        self._crop_im_size = 400
        self.is_running = False
        self.final_duckies = []
        self._last_inference_time = None
        self.min_inference_interval = 0.5
        self.image_middle = 320
        self.center_white = None
        self.center_yellow = None
        self.debug_cv_image = None
        self.debug_largest_gap = None
        self.debug_duckie_error = None
        self.debug_lowest_duckie = None
        self.debug_close_to_white = False

        self._pending_image = None
        self._pending_lock = threading.Lock()

        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"

        # Kameratopic abonnieren, jedes neue Bild wird automatisch an cbFindDuckies weitergegeben
        self.sub_image_original = rospy.Subscriber(self._camera_topic, CompressedImage, self.cbFindDuckies, queue_size = 1)

        #Publisher für das Ergebnis der Kreuzungserkennung

        self.pub_debug_duckies = rospy.Publisher(f'/{self._vehicle_name}/debug/duckie_detection', DuckieDetectionArray, queue_size=1)
        self.pub_debug_contours = rospy.Publisher(f'/{self._vehicle_name}/debug/duckie_contours', CompressedImage, queue_size=1)
        self.pub_debug_duckie_view = rospy.Publisher(f'/{self._vehicle_name}/debug/duckie_view', CompressedImage, queue_size=1)

        self.model = '/root/DuckieRace/src/packages/explore_duckietown_ii/models/duckie_v19.onnx'

        
        self.session = ort.InferenceSession(self.model)
        self.input_name = self.session.get_inputs()[0].name

        self.conf_threshold = 0.3

        self.pub_duckie_control_active = rospy.Publisher(f'/{self._vehicle_name}/detect/duckie_control_active', Bool, queue_size=1)
        self.pub_duckie_error = rospy.Publisher(f'/{self._vehicle_name}/detect/duckie_error', Float64, queue_size=1)
        self.pub_duckie_distance_factor = rospy.Publisher(f'/{self._vehicle_name}/detect/duckie_distance_factor', Float64, queue_size=1)
        self.pub_duckies = rospy.Publisher(f'/{self._vehicle_name}/detect/duckies', Bool, queue_size=1)

        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()

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
        self.duckie_distance_min = parameters["detection"]["distance_min"]["default"]
        self.duckie_distance_max = parameters["detection"]["distance_max"]["default"]
    

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
        np_arr = np.frombuffer(image_msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv_image = cv2.resize(cv_image, (640, 480))
        self.image_middle = cv_image.shape[1] // 2
        self.debug_cv_image = cv_image.copy()
        with self._pending_lock:
            self._pending_image = cv_image

    def _inference_loop(self):
        while not rospy.is_shutdown():
            with self._pending_lock:
                cv_image = self._pending_image
                self._pending_image = None

            if cv_image is None:
                time.sleep(0.01)
                continue

            try:
                img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))
                img = np.expand_dims(img, 0)

                hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

                self.fnDetectDuckies(img)

                duckie_mask = np.zeros(cv_image.shape[:2], dtype=np.uint8)
                for x1, y1, x2, y2, conf in self.final_duckies:
                    duckie_mask[int(y1):int(y2), int(x1):int(x2)] = 255

                lowest_duckie = None
                for duckie in self.final_duckies:
                    if lowest_duckie is None or duckie[3] > lowest_duckie[3]:
                        lowest_duckie = duckie

                if lowest_duckie is not None and self.duckie_distance_min <= lowest_duckie[3] <= self.duckie_distance_max:
                    range_size = max(1, self.duckie_distance_max - self.duckie_distance_min)
                    distance_factor = (lowest_duckie[3] - self.duckie_distance_min) / range_size
                    distance_factor = max(0.0, min(1.0, distance_factor))
                    self.pub_duckie_distance_factor.publish(Float64(data=distance_factor))

                    hsv[duckie_mask == 255] = 0
                    self.fnDetectLane(hsv, lowest_duckie[3])
                    duckie_center_x = (lowest_duckie[0] + lowest_duckie[2]) / 2
                    if self.center_yellow is not None and duckie_center_x < self.center_yellow:
                        self.pub_duckie_control_active.publish(Bool(data=False))
                        self.debug_duckie_error = None
                    else:
                        duckie_error = self.fnGetLaneDuckieError(lowest_duckie)
                        self.debug_duckie_error = duckie_error
                        self.debug_lowest_duckie = lowest_duckie
                        if duckie_error is not None:
                            self.pub_duckie_error.publish(Float64(data=duckie_error))
                            self.pub_duckie_control_active.publish(Bool(data=True))
                else:
                    self.debug_duckie_error = None
                    self.debug_lowest_duckie = None
                    self.debug_largest_gap = None
                    self.pub_duckie_distance_factor.publish(Float64(data=0.0))
                    self.pub_duckie_control_active.publish(Bool(data=False))
            except Exception as e:
                rospy.logerr(f"_inference_loop failed: {e}")

    def fnDetectDuckies(self, image):
        """
        Hier soll der Code zur Erkennung von Duckies implementiert werden. Das Ergebnis (True wenn Duckie erkannt, sonst False)
        wird auf dem Topic /{vehicle_name}/detect/duckies veröffentlicht.

        Autor: Felix Faass

        Args:
            img: BGR-Kamerabild als numpy Array
        """


        self.final_duckies = []

        now = time.time()
        gap_ms = (now - self._last_inference_time) * 1000 if self._last_inference_time else 0
        self._last_inference_time = now

        raw_output = self.session.run(None, {self.input_name:image})
        inference_ms = (time.time() - now) * 1000
        detections = raw_output[0][0]

        max_conf = max((box[4] for box in detections), default=0)
        rospy.loginfo_throttle(2, f"Duckie detect: max_conf={max_conf:.3f}, threshold={self.conf_threshold}, inference={inference_ms:.0f}ms, gap={gap_ms:.0f}ms")

        for box in detections:
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
                for x in range(self.image_middle+50, 0,-1):
                    if th1[row][x] == 255:
                        a.append(x)
                        break
        else:
            for row in range(max(0, distance-50), min(len(mask), distance+50)):
                for x in range(self.image_middle-50, len(mask[0])):
                    if th1[row][x] == 255:
                        a.append(x)
                        break
                
                

        if len(a) > 10:
            return np.median(a)
        else:
            return no_lane_value

    def fnDetectLane(self, hsv, distance):
        distance = int(distance)

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
                self.debug_close_to_white = False
            else:
                largest_gap = self.center_white - Distance_Duckie_to_white/2
                self.debug_close_to_white = True

            self.debug_largest_gap = largest_gap
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
            if self.pub_debug_duckies.get_num_connections() > 0:
                msg = DuckieDetectionArray()
                for x1, y1, x2, y2, conf in self.final_duckies:
                    d = DuckieDetection()
                    d.bounding_box = Rect(x=int(x1), y=int(y1), w=int(x2-x1), h=int(y2-y1))
                    d.confidence = conf
                    msg.detections.append(d)
                self.pub_debug_duckies.publish(msg)

            if self.debug_cv_image is not None:
                img = self.debug_cv_image.copy()
                h, w = img.shape[:2]

                # Alle Duckie BBs — nicht-ausgewählte in gedämpftem Grau
                selected = self.debug_lowest_duckie
                for x1, y1, x2, y2, conf in self.final_duckies:
                    is_selected = (selected is not None and
                                   int(x1) == int(selected[0]) and int(y2) == int(selected[3]))
                    color     = (0, 255, 255) if is_selected else (80, 80, 80)
                    thickness = 3            if is_selected else 1
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                    cv2.putText(img, f"{conf:.2f}", (int(x1), int(y1) - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                # Mittellinie (vertikal)
                cv2.line(img, (self.image_middle, 0), (self.image_middle, h), (255, 255, 255), 1)

                # Distanz-Range-Linien (PID aktiv zwischen min und max)
                min_y = int(self.duckie_distance_min)
                max_y = int(self.duckie_distance_max)
                cv2.line(img, (0, min_y), (w, min_y), (0, 200, 80), 2)
                cv2.putText(img, f"min: {min_y}", (5, min_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 80), 1)
                cv2.line(img, (0, max_y), (w, max_y), (0, 80, 255), 2)
                cv2.putText(img, f"max: {max_y}", (5, max_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1)

                if selected is not None:
                    x1, y1, x2, y2, _ = selected
                    det_y = int(y2)

                    # Gelbe und weiße Linienpunkte an der Detection-Row
                    if self.center_yellow is not None:
                        cv2.circle(img, (int(self.center_yellow), det_y), 7, (0, 200, 255), -1)
                        cv2.putText(img, "Y", (int(self.center_yellow) + 9, det_y + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                    if self.center_white is not None:
                        cv2.circle(img, (int(self.center_white), det_y), 7, (255, 255, 255), -1)
                        cv2.putText(img, "W", (int(self.center_white) + 9, det_y + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    if self.debug_largest_gap is not None:
                        gap_x = int(self.debug_largest_gap)

                        # Linie: Duckie ↔ nächste Spurlinie
                        if self.debug_close_to_white and self.center_white is not None:
                            cv2.line(img, (int(x2), det_y), (int(self.center_white), det_y), (0, 255, 0), 2)
                        elif not self.debug_close_to_white and self.center_yellow is not None:
                            cv2.line(img, (int(x1), det_y), (int(self.center_yellow), det_y), (0, 200, 255), 2)

                        # Mittelpunkt der Lücke
                        cv2.circle(img, (gap_x, det_y), 6, (0, 0, 255), -1)

                        # Fehler-Linie: Bildmitte → Lückenmittelpunkt
                        cv2.line(img, (self.image_middle, det_y - 15), (gap_x, det_y - 15), (0, 0, 255), 2)

                    if self.debug_duckie_error is not None:
                        cv2.putText(img, f"Error: {self.debug_duckie_error:.3f}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.imshow('Duckie View', img)
                cv2.waitKey(1)
                if self.pub_debug_duckie_view.get_num_connections() > 0:
                    debug_msg = CompressedImage()
                    debug_msg.header.stamp = rospy.Time.now()
                    debug_msg.format = "jpeg"
                    debug_msg.data = np.array(cv2.imencode('.jpg', img)[1]).tobytes()
                    self.pub_debug_duckie_view.publish(debug_msg)

            rate.sleep()

if __name__ == '__main__':
    node = DetectDuckiesNode('detect_duckies_node')
    node.run_debug()