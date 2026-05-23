#!/usr/bin/env python3

import os
import rospy
import numpy as np
import cv2
from std_msgs.msg import Float64, Int32
from sensor_msgs.msg import CompressedImage
import util


class DebugViewNode:
    def __init__(self, node_name):
        rospy.init_node(node_name)
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._crop_im_size = 400
        self.M = None

        util.init_parameters('detect_lane_node', self.cbUpdateParameters)

        self.raw_img = None
        self.center_white       = 380.0
        self.center_yellow      = 20.0
        self.lane_center        = 200.0
        self.used_detection_row = int(self._crop_im_size * 0.75)
        self.lane_error         = 0.0
        self.red_pixel_count    = 0
        self.control_v          = 0.0

        v = self._vehicle_name
        rospy.Subscriber(f'/{v}/camera_node/image/compressed', CompressedImage, self.cbImage,        queue_size=1)
        rospy.Subscriber(f'/{v}/debug/det_center_white',       Float64,         self.cbCenterWhite,  queue_size=1)
        rospy.Subscriber(f'/{v}/debug/det_center_yellow',      Float64,         self.cbCenterYellow, queue_size=1)
        rospy.Subscriber(f'/{v}/debug/det_lane_center',        Float64,         self.cbLaneCenter,   queue_size=1)
        rospy.Subscriber(f'/{v}/debug/det_row',                Int32,           self.cbDetRow,       queue_size=1)
        rospy.Subscriber(f'/{v}/detect/lane',                  Float64,         self.cbLaneError,    queue_size=1)
        rospy.Subscriber(f'/{v}/debug/red_pixel_count',        Int32,           self.cbRedCount,     queue_size=1)
        rospy.Subscriber(f'/{v}/debug/control_v',              Float64,         self.cbControlV,     queue_size=1)

        self.pub_full = rospy.Publisher(f'/{v}/debug/full_view', CompressedImage, queue_size=1)

    def cbUpdateParameters(self, parameters):
        c = parameters["crop_image"]
        self.top_left_x     = c["top_left_x"]["default"]
        self.top_left_y     = c["top_left_y"]["default"]
        self.top_right_x    = c["top_right_x"]["default"]
        self.top_right_y    = c["top_right_y"]["default"]
        self.bottom_left_x  = c["bottom_left_x"]["default"]
        self.bottom_left_y  = c["bottom_left_y"]["default"]
        self.bottom_right_x = c["bottom_right_x"]["default"]
        self.bottom_right_y = c["bottom_right_y"]["default"]
        pts1 = np.float32([
            [self.top_left_x,     self.top_left_y],
            [self.top_right_x,    self.top_right_y],
            [self.bottom_right_x, self.bottom_right_y],
            [self.bottom_left_x,  self.bottom_left_y],
        ])
        pts2 = np.float32([
            [0, 0],
            [self._crop_im_size, 0],
            [0, self._crop_im_size],
            [self._crop_im_size, self._crop_im_size],
        ])
        self.M = cv2.getPerspectiveTransform(pts1, pts2)

    def cbImage(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.raw_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def cbCenterWhite(self,  msg): self.center_white        = msg.data
    def cbCenterYellow(self, msg): self.center_yellow       = msg.data
    def cbLaneCenter(self,   msg): self.lane_center         = msg.data
    def cbDetRow(self,       msg): self.used_detection_row  = msg.data
    def cbLaneError(self,    msg): self.lane_error          = msg.data
    def cbRedCount(self,     msg): self.red_pixel_count     = msg.data
    def cbControlV(self,     msg): self.control_v           = msg.data

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

        quad = np.array([
            [self.top_left_x,     self.top_left_y],
            [self.top_right_x,    self.top_right_y],
            [self.bottom_left_x,  self.bottom_left_y],
            [self.bottom_right_x, self.bottom_right_y],
        ], dtype=np.int32)
        cv2.polylines(raw, [quad], isClosed=True, color=(0, 255, 255), thickness=2)

        roi_y = int(raw_h * 0.66)
        roi_x = int(raw_w * 0.3)
        cv2.rectangle(raw, (roi_x, roi_y), (raw_w - 1, raw_h - 1), (0, 0, 255), 2)

        det_row = self.used_detection_row
        for dy in [det_row - 50, det_row + 50]:
            if 0 <= dy <= self._crop_im_size:
                cv2.line(raw, proj(0, dy), proj(self._crop_im_size, dy), (180, 180, 180), 1)

        cv2.circle(raw, proj(int(self.center_white),  det_row), 6, (255, 255, 255), -1)
        cv2.circle(raw, proj(int(self.center_yellow), det_row), 6, (0, 255, 255),   -1)
        cv2.circle(raw, proj(int(self.lane_center), self._crop_im_size // 2), 5, (255, 0, 0), -1)

        panel_w = 250
        panel = np.zeros((raw_h, panel_w, 3), dtype=np.uint8)
        cv2.putText(panel, "Lane Error:",         (10,  40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(panel, f"{self.lane_error:.3f}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(panel, "Red Pixels:",         (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(panel, f"{self.red_pixel_count}", (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 80, 255), 2)
        cv2.putText(panel, "Det. Row:",           (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(panel, f"{det_row}px",        (10, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(panel, "Speed (v):",          (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(panel, f"{self.control_v:.3f} m/s", (10, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        return np.hstack([raw, panel])

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.pub_full.get_num_connections() > 0:
                try:
                    img = self.build_full_debug_img()
                    if img is not None:
                        msg = CompressedImage()
                        msg.header.stamp = rospy.Time.now()
                        msg.format = "jpeg"
                        msg.data = np.array(cv2.imencode('.jpg', img)[1]).tobytes()
                        self.pub_full.publish(msg)
                except Exception as e:
                    rospy.logwarn_throttle(5, f"debug_view_node failed: {e}")
            rate.sleep()


if __name__ == '__main__':
    node = DebugViewNode('debug_view_node')
    node.run()
    rospy.spin()
