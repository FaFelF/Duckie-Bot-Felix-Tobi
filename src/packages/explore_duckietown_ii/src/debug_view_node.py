#!/usr/bin/env python3

from curses import raw
import os
import rospy
import numpy as np
import cv2
from std_msgs.msg import Float64, Int32, String
from duckietown_msgs.msg import AprilTagDetection, AprilTagDetectionArray
from duckietown_msgs.msg import Rect
from explore_duckietown_ii.msg import DuckieDetection, DuckieDetectionArray
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
        self.apriltag_locations = []
        self.duckie_detections  = []
        self.control_mode       = -1
        self.saved_apriltag     = -1
        self.intersection_dir   = -1

        v = self._vehicle_name
        rospy.Subscriber(f'/{v}/camera_node/image/compressed', CompressedImage,       self.cbImage,             queue_size=1)
        rospy.Subscriber(f'/{v}/debug/det_center_white',       Float64,               self.cbCenterWhite,       queue_size=1)
        rospy.Subscriber(f'/{v}/debug/det_center_yellow',      Float64,               self.cbCenterYellow,      queue_size=1)
        rospy.Subscriber(f'/{v}/debug/det_lane_center',        Float64,               self.cbLaneCenter,        queue_size=1)
        rospy.Subscriber(f'/{v}/debug/det_row',                Int32,                 self.cbDetRow,            queue_size=1)
        rospy.Subscriber(f'/{v}/detect/lane',                  Float64,               self.cbLaneError,         queue_size=1)
        rospy.Subscriber(f'/{v}/debug/red_pixel_count',        Int32,                 self.cbRedCount,          queue_size=1)
        rospy.Subscriber(f'/{v}/debug/control_v',              Float64,               self.cbControlV,          queue_size=1)
        rospy.Subscriber(f'/{v}/debug/duckie_detection',       DuckieDetectionArray,  self.cbDuckieDetection,   queue_size=1)
        rospy.Subscriber(f'/{v}/debug/apriltag_locations',     AprilTagDetectionArray,self.cbapriltagInfo,      queue_size=1)
        rospy.Subscriber(f'/{v}/switch/control',               Int32,                 self.cbControlMode,       queue_size=1)
        rospy.Subscriber(f'/{v}/detect/saved_apriltag',        Int32,                 self.cbSavedApriltag,     queue_size=1)
        rospy.Subscriber(f'/{v}/switch/intersection_direction',Int32,                 self.cbIntersectionDir,   queue_size=1)
        rospy.Subscriber(f'/{v}/debug/selected_image_topic',   String,                self.cbSelectedTopic,     queue_size=1)

        self.pub_full = rospy.Publisher(f'/{v}/debug/full_view', CompressedImage, queue_size=1)
        self.secondary_image = None
        self.secondary_subscriber = None

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

    def cbapriltagInfo(self,    msg): self.apriltag_locations = msg.detections
    def cbDuckieDetection(self, msg): self.duckie_detections  = msg.detections
    def cbControlMode(self,     msg): self.control_mode       = msg.data
    def cbSavedApriltag(self,   msg): self.saved_apriltag     = msg.data
    def cbIntersectionDir(self, msg): self.intersection_dir   = msg.data

    def cbSelectedTopic(self, msg):
        topic = msg.data
        if self.secondary_subscriber:
            self.secondary_subscriber.unregister()
        self.secondary_image = None
        if topic and topic != f'/{self._vehicle_name}/debug/full_view':
            self.secondary_subscriber = rospy.Subscriber(topic, CompressedImage, self.cbSecondaryImage, queue_size=1)

    def cbSecondaryImage(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.secondary_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

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

        for apriltag_location in self.apriltag_locations:
            corners = apriltag_location.corners
            corners = np.array(corners).reshape((4, 2)).astype(int)
            cv2.rectangle(raw, tuple(corners[0]), tuple(corners[2]), (0, 255, 0), 2)
            cv2.putText(raw, str(apriltag_location.tag_id), (int(apriltag_location.center[0]), int(apriltag_location.center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        for duckie in self.duckie_detections:
            cv2.rectangle(raw, (duckie.bounding_box.x, duckie.bounding_box.y), (duckie.bounding_box.x + duckie.bounding_box.w, duckie.bounding_box.y + duckie.bounding_box.h), (0, 255, 255), 2)
            cv2.putText(raw, f"Conf: {duckie.confidence:.2f}", (duckie.bounding_box.x, duckie.bounding_box.y + duckie.bounding_box.h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.circle(raw, proj(int(self.center_white),  det_row), 6, (255, 255, 255), -1)
        cv2.circle(raw, proj(int(self.center_yellow), det_row), 6, (0, 255, 255),   -1)
        cv2.circle(raw, proj(int(self.lane_center), self._crop_im_size // 2), 5, (255, 0, 0), -1)

        
        

        MODE_NAMES  = {1: "LANE", 2: "OBSTACLE", 3: "STOP", 4: "INTERSECT", 5: "ALIGN"}
        MODE_COLORS = {1: (0, 255, 0), 2: (0, 165, 255), 3: (0, 0, 255), 4: (255, 200, 0), 5: (255, 0, 255)}
        DIR_NAMES   = {0: "< LEFT", 1: "^ STRAIGHT", 2: "> RIGHT"}

        mode_color = MODE_COLORS.get(self.control_mode, (200, 200, 200))

        # --- Panel links: allgemeine Infos ---
        pw = 230
        panel = np.zeros((raw_h, pw, 3), dtype=np.uint8)
        cv2.putText(panel, "Lane Error:",             (10,  35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(panel, f"{self.lane_error:.3f}",  (10,  65), cv2.FONT_HERSHEY_SIMPLEX, 1.0,  (0, 255, 0),    2)
        cv2.putText(panel, "Speed:",                  (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(panel, f"{self.control_v:.3f} m/s", (10, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        cv2.putText(panel, "Red Pixels:",             (10, 163), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(panel, f"{self.red_pixel_count}", (10, 191), cv2.FONT_HERSHEY_SIMPLEX, 0.9,  (0, 80, 255),   2)
        cv2.putText(panel, "Det. Row:",               (10, 226), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(panel, f"{det_row}px",            (10, 252), cv2.FONT_HERSHEY_SIMPLEX, 0.8,  (255, 255, 0),  2)
        cv2.line(panel, (5, 268), (pw - 5, 268), (60, 60, 60), 1)
        cv2.putText(panel, f"Tags:  {len(self.apriltag_locations)}", (10, 298), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(panel, f"Ducks: {len(self.duckie_detections)}",  (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),   2)

        # --- Panel rechts: Mode + mode-spezifische Infos ---
        pw2 = 230
        panel2 = np.zeros((raw_h, pw2, 3), dtype=np.uint8)
        cv2.putText(panel2, "Mode:",                           (10,  35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(panel2, MODE_NAMES.get(self.control_mode, "???"), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, mode_color, 2)
        cv2.line(panel2, (5, 85), (pw2 - 5, 85), (60, 60, 60), 1)

        if self.control_mode in (3, 4, 5):  # Stop, Intersection oder Align — gleiche Infos
            tag_text = f"ID {self.saved_apriltag}" if self.saved_apriltag >= 0 else "---"
            dir_text = DIR_NAMES.get(self.intersection_dir, "???")
            cv2.putText(panel2, "Saved Tag:",  (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.putText(panel2, tag_text,      (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 1.0,  (0, 255, 255),   2)
            cv2.putText(panel2, "Direction:",  (10, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.putText(panel2, dir_text,      (10, 228), cv2.FONT_HERSHEY_SIMPLEX, 0.9,  (255, 200, 0),   2)
        elif self.control_mode == 2:  # Obstacle
            cv2.putText(panel2, "Duckies:",                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.putText(panel2, f"{len(self.duckie_detections)}",  (10, 158), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255),   2)
        elif self.control_mode == 1:  # Lane — zuletzt gespeicherten Apriltag anzeigen
            tag_text = f"ID {self.saved_apriltag}" if self.saved_apriltag >= 0 else "---"
            cv2.putText(panel2, "Saved Tag:",  (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.putText(panel2, tag_text,      (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 1.0,  (0, 255, 255),   2)

        

        return np.hstack([raw, panel, panel2])

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            try:
                img = self.build_full_debug_img()
                if img is not None:
                    if self.pub_full.get_num_connections() > 0:
                        msg = CompressedImage()
                        msg.header.stamp = rospy.Time.now()
                        msg.format = "jpeg"
                        msg.data = np.array(cv2.imencode('.jpg', img)[1]).tobytes()
                        self.pub_full.publish(msg)
                    cv2.imshow('Debug View', img)
                    cv2.waitKey(1)
            except Exception as e:
                rospy.logwarn_throttle(5, f"debug_view_node failed: {e}")
            rate.sleep()


if __name__ == '__main__':
    node = DebugViewNode('debug_view_node')
    node.run()
    rospy.spin()
