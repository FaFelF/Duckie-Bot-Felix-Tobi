#!/usr/bin/env python3

import os
import rospy
import numpy as np
import cv2
from std_msgs.msg import Float64, String, Int32
from sensor_msgs.msg import CompressedImage
from enum import Enum
import yaml
import util

#from duckietown.dtros import DTROS, NodeType

class DetectLaneNode:
    def __init__(self, node_name):
        # initialize the ROS node
        rospy.init_node(node_name)
        
        
        self._vehicle_name = os.environ['VEHICLE_NAME']
        util.init_parameters(node_name,self.cbUpdateParameters)
                
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self.sub_image_original = rospy.Subscriber(self._camera_topic, CompressedImage, self.cbFindLane, queue_size = 1)
        self.pub_lane = rospy.Publisher(f'/{self._vehicle_name}/detect/lane', Float64, queue_size = 1)

        self._crop_im_size = 400
        self.is_running = False
        self.counter = 0

        # init debug channels
        self.pub_debug_lane = rospy.Publisher(f'/{self._vehicle_name}/debug/lane_croped',CompressedImage,queue_size=1)
        self.pub_debug_white = rospy.Publisher(f'/{self._vehicle_name}/debug/lane_white',CompressedImage,queue_size=1)
        self.pub_debug_yellow = rospy.Publisher(f'/{self._vehicle_name}/debug/lane_yellow',CompressedImage,queue_size=1)
        self.pub_debug_full = rospy.Publisher(f'/{self._vehicle_name}/debug/full_view', CompressedImage, queue_size=1)

        self.sub_red_count = rospy.Subscriber(f'/{self._vehicle_name}/debug/red_pixel_count', Int32, self.cbRedCount, queue_size=1)

        self.raw_img = None
        self.M = None
        self.img = np.zeros((400, 400, 3), dtype=np.uint8)
        self.lane_center = 200
        self.white_alternative = 380
        self.yellow_alternative = 20
        self.center_white = 380
        self.center_yellow = 20
        self.debug_img_white = np.zeros((400, 400), dtype=np.uint8)
        self.debug_img_yellow = np.zeros((400, 400), dtype=np.uint8)
        self.min_lane_width = 80
        self.lane_error = 0.0
        self.used_detection_row = int(self._crop_im_size * self.detection_row_factor)
        self.used_detection_row_white  = self.used_detection_row
        self.used_detection_row_yellow = self.used_detection_row
        self.red_pixel_count = 0

    def cbUpdateParameters(self, parameters):
        self.detection_row_factor = parameters["detection_row_factor"]["default"]
        self.min_lane_width = parameters["min_lane_width"]["default"]

        # Update white line parameters
        self.hue_white_l = parameters["white"]["hl"]["default"]
        self.hue_white_h = parameters["white"]["hh"]["default"]
        self.saturation_white_l = parameters["white"]["sl"]["default"]
        self.saturation_white_h = parameters["white"]["sh"]["default"]
        self.lightness_white_l = parameters["white"]["vl"]["default"]
        self.lightness_white_h = parameters["white"]["vh"]["default"]
        
        # Update yellow line parameters
        self.hue_yellow_l = parameters["yellow"]["hl"]["default"]
        self.hue_yellow_h = parameters["yellow"]["hh"]["default"]
        self.saturation_yellow_l = parameters["yellow"]["sl"]["default"]
        self.saturation_yellow_h = parameters["yellow"]["sh"]["default"]
        self.lightness_yellow_l = parameters["yellow"]["vl"]["default"]
        self.lightness_yellow_h = parameters["yellow"]["vh"]["default"]
        
        # Update perspective transform points
        self.top_left_x = parameters["crop_image"]["top_left_x"]["default"]
        self.top_left_y = parameters["crop_image"]["top_left_y"]["default"]
        self.top_right_x = parameters["crop_image"]["top_right_x"]["default"]
        self.top_right_y = parameters["crop_image"]["top_right_y"]["default"]
        self.bottom_left_x = parameters["crop_image"]["bottom_left_x"]["default"]
        self.bottom_left_y = parameters["crop_image"]["bottom_left_y"]["default"]
        self.bottom_right_x = parameters["crop_image"]["bottom_right_x"]["default"]
        self.bottom_right_y = parameters["crop_image"]["bottom_right_y"]["default"]
  
    def crop_img(self,img):
        img = img.copy()
        
        pts1 = np.float32([
            [self.top_left_x,     self.top_left_y],
            [self.top_right_x,    self.top_right_y],
            [self.bottom_right_x, self.bottom_right_y],
            [self.bottom_left_x,  self.bottom_left_y],])
        
        pts2 = np.float32([[0,0],[self._crop_im_size,0],[0,self._crop_im_size],[self._crop_im_size,self._crop_im_size]])

        M = cv2.getPerspectiveTransform(pts1,pts2)
        self.M = M
        return cv2.warpPerspective(img,M,(self._crop_im_size,self._crop_im_size))


    def get_x_for_driving(self, mask, distance, no_lane_value, left_line):
        grad = cv2.Sobel(mask, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        abs_grad = np.absolute(grad)
        uint8_grad = np.uint8(abs_grad)
        
        _,th1 = cv2.threshold(grad,127,255,cv2.THRESH_BINARY)

        a = []
        for row in range(max(0, distance-50), min(len(mask), distance+50)):
            if np.where(th1[row] == 255)[0].size == 0:
                continue
            else:
                if left_line:
                    a.append(np.where(th1[row] == 255)[0][-1])
                else:
                    a.append(np.where(th1[row] == 255)[0][0])

        if len(a) > 10:
            return np.median(a)
        else:
            return no_lane_value
        

    def cbFindLane(self, image_msg):
        detection_row_factor = self.detection_row_factor

        
        if self.counter  <= 3:
            self.counter += 1   
            return

        if self.is_running:
            return
        
        self.is_running = True
        self.conunter = 0   #Wieso? nur jedes 4. Bild gewünscht?
        try:
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.raw_img = cv_image
            img = self.crop_img(cv_image)

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            mask_yellow = cv2.inRange(hsv,
                               (self.hue_yellow_l,self.saturation_yellow_l, self.lightness_yellow_l),
                               (self.hue_yellow_h,self.saturation_yellow_h, self.lightness_yellow_h),)

            mask_white = cv2.inRange(hsv,
                               (self.hue_white_l,self.saturation_white_l, self.lightness_white_l),
                               (self.hue_white_h,self.saturation_white_h, self.lightness_white_h),)

            white_alternative = int(len(img[0]) * 0.95)
            yellow_alternative = int(len(img[0]) * 0.05)

            # Erst gelb finden
            center_yellow = self.get_x_for_driving(mask_yellow, int(len(img)*detection_row_factor), yellow_alternative, left_line=True)

            # Weiße Maske: alles links von gelb + min_lane_width ausblenden
            mask_white_filtered = mask_white.copy()
            corridor_start = max(0, int(center_yellow) + self.min_lane_width)
            mask_white_filtered[:, :corridor_start] = 0

            center_white = self.get_x_for_driving(mask_white_filtered, int(len(img)*detection_row_factor), white_alternative, left_line=False)

            while center_white == white_alternative and detection_row_factor <= 0.95:
                detection_row_factor += 0.05
                center_yellow = self.get_x_for_driving(mask_yellow, int(len(img)*detection_row_factor), yellow_alternative, left_line=True)
                corridor_start = max(0, int(center_yellow) + self.min_lane_width)
                mask_white_filtered[:, :corridor_start] = 0
                center_white = self.get_x_for_driving(mask_white_filtered, int(len(img)*detection_row_factor), white_alternative, left_line=False)

            self.used_detection_row_white  = int(len(img) * detection_row_factor)
            self.used_detection_row_yellow = int(len(img) * detection_row_factor)
            self.used_detection_row = self.used_detection_row_white

            if center_white <= center_yellow:
                if center_white > int(len(img[0]) * 0.4):
                    center_yellow = yellow_alternative
                else:
                    center_white = white_alternative

            lane_center = (center_white + center_yellow) / 2

            msg_error = Float64()
            msg_error.data = 1-(lane_center / len(img) * 2)

            self.pub_lane.publish(msg_error)
            self.lane_error = msg_error.data
            print(f"Lane error: {msg_error.data} range [-1,1]")

            # saving for debug
            self.img = img
            self.lane_center = lane_center
            self.white_alternative = white_alternative
            self.yellow_alternative = yellow_alternative
            self.center_white = center_white
            self.center_yellow = center_yellow

            self.debug_img_white = mask_white
            self.debug_img_yellow = mask_yellow

            image = cv2.circle(img,(int(lane_center),int(len(img) / 2)),3,(255,0,0))
            image = cv2.line(image, (white_alternative , 0), (white_alternative , self._crop_im_size) ,color=(255,255,255))
            image = cv2.line(image, (yellow_alternative , 0), (yellow_alternative , self._crop_im_size) ,color=(255,255,0))
            image = cv2.line(image, (0, self.used_detection_row + 50), (len(img[0]), self.used_detection_row + 50), color=(255,255,255))
            image = cv2.line(image, (0, self.used_detection_row - 50), (len(img[0]), self.used_detection_row - 50), color=(255,255,255))

            image = cv2.line(image,(int(len(img[0])/2),0),(int(len(img[0])/2),len(image)),(0,255,0))
            image = cv2.circle(image, (int(center_white),  self.used_detection_row_white),  5,(255,255,255))
            image = cv2.circle(image, (int(center_yellow), self.used_detection_row_yellow), 5,(0,255,255))

            #auskommentieren, wenn über vs code
            cv2.imshow('lane detection', image)
            #cv2.imshow('white', mask_white)
            #cv2.imshow('yellow', mask_yellow)
            cv2.waitKey(1)
        except Exception as e:
            rospy.logerr(f"cbFindLane failed: {e}")
        finally:
            self.is_running = False
            
    def cbRedCount(self, msg):
        self.red_pixel_count = msg.data

    def build_full_debug_img(self):
        if self.raw_img is None or self.M is None:
            return None

        raw = self.raw_img.copy()
        raw_h, raw_w = raw.shape[:2]
        M_inv = np.linalg.inv(self.M)

        def proj(x, y):
            p = np.array([[[float(x), float(y)]]], dtype=np.float32)
            r = cv2.perspectiveTransform(p, M_inv)
            return (int(r[0][0][0]), int(r[0][0][1]))

        # Spur-Crop Viereck einzeichnen
        quad = np.array([
            [self.top_left_x,     self.top_left_y],
            [self.top_right_x,    self.top_right_y],
            [self.bottom_left_x,  self.bottom_left_y],
            [self.bottom_right_x, self.bottom_right_y],
        ], dtype=np.int32)
        cv2.polylines(raw, [quad], isClosed=True, color=(0, 255, 255), thickness=2)

        # Rote-Linie ROI Rechteck (unteres Drittel, ab 30% von links)
        roi_y = int(raw_h * 0.66)
        roi_x = int(raw_w * 0.3)
        cv2.rectangle(raw, (roi_x, roi_y), (raw_w - 1, raw_h - 1), (0, 0, 255), 2)

        # Erkennungsband (detection row) zurückprojizieren
        det_row = self.used_detection_row
        for dy in [det_row - 50, det_row + 50]:
            if 0 <= dy <= self._crop_im_size:
                cv2.line(raw, proj(0, dy), proj(self._crop_im_size, dy), (180, 180, 180), 1)

        # Erkannte Linienpositionen zurückprojizieren
        cv2.circle(raw, proj(int(self.center_white),  det_row), 6, (255, 255, 255), -1)
        cv2.circle(raw, proj(int(self.center_yellow), det_row), 6, (0, 255, 255),   -1)
        cv2.circle(raw, proj(int(self.lane_center), int(self._crop_im_size / 2)), 5, (255, 0, 0), -1)

        # Schwarzes Info-Panel rechts
        panel_w = 250
        panel = np.zeros((raw_h, panel_w, 3), dtype=np.uint8)
        cv2.putText(panel, "Lane Error:",  (10,  40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(panel, f"{self.lane_error:.3f}", (10,  75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(panel, "Red Pixels:",  (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(panel, f"{self.red_pixel_count}", (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 80, 255), 2)
        cv2.putText(panel, "Det. Row:",    (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(panel, f"{det_row}px", (10, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        return np.hstack([raw, panel])

    def run_debug(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():

            # add debug info to lane image
            if self.pub_debug_lane.get_num_connections() > 0:
                debug_img = self.img.copy()
                debug_img = cv2.circle(debug_img,(int(self.lane_center),int(len(debug_img) / 2)),3,(255,0,0))

                debug_img = cv2.line(debug_img, (self.white_alternative , 0), (self.white_alternative , 1000) ,color=(255,255,255)) 
                debug_img = cv2.line(debug_img, (self.yellow_alternative , 0), (self.yellow_alternative , 1000) ,color=(255,255,0))
                debug_img = cv2.line(debug_img, (0, self.used_detection_row + 50), (len(debug_img[0]), self.used_detection_row + 50), color=(255,255,255))
                debug_img = cv2.line(debug_img, (0, self.used_detection_row - 50), (len(debug_img[0]), self.used_detection_row - 50), color=(255,255,255))

                debug_img = cv2.line(debug_img,(int(len(debug_img[0])/2),0),(int(len(debug_img[0])/2),len(debug_img)),(0,255,0))
                debug_img = cv2.circle(debug_img, (int(self.center_white),  self.used_detection_row_white),  5,(255,255,255))
                debug_img = cv2.circle(debug_img, (int(self.center_yellow), self.used_detection_row_yellow), 5,(0,255,255))

                debug_msg = CompressedImage()
                debug_msg.header.stamp = rospy.Time.now()
                debug_msg.format = "jpeg"
                debug_msg.data = np.array(cv2.imencode('.jpg', debug_img)[1]).tobytes()
                self.pub_debug_lane.publish(debug_msg)

            if self.pub_debug_white.get_num_connections() > 0:
                debug_msg = CompressedImage()
                debug_msg.header.stamp = rospy.Time.now()
                debug_msg.format = "jpeg"
                debug_msg.data = np.array(cv2.imencode('.jpg', self.debug_img_white)[1]).tobytes()
                self.pub_debug_white.publish(debug_msg)

            if self.pub_debug_yellow.get_num_connections() > 0:
                debug_msg = CompressedImage()
                debug_msg.header.stamp = rospy.Time.now()
                debug_msg.format = "jpeg"
                debug_msg.data = np.array(cv2.imencode('.jpg', self.debug_img_yellow)[1]).tobytes()
                self.pub_debug_yellow.publish(debug_msg)

            if self.pub_debug_full.get_num_connections() > 0:
                try:
                    full_img = self.build_full_debug_img()
                    if full_img is not None:
                        debug_msg = CompressedImage()
                        debug_msg.header.stamp = rospy.Time.now()
                        debug_msg.format = "jpeg"
                        debug_msg.data = np.array(cv2.imencode('.jpg', full_img)[1]).tobytes()
                        self.pub_debug_full.publish(debug_msg)
                except Exception as e:
                    rospy.logwarn_throttle(5, f"full_view debug failed: {e}")

            rate.sleep()
        
if __name__ == '__main__':
    node = DetectLaneNode('detect_lane_node')
    node.run_debug()
    rospy.spin()