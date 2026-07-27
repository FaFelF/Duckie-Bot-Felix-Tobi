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
        # Detection-Hysterese: letzte Erkennung max. _hold_frames Frames halten (gegen Aussetzer).
        self._last_duckies = []
        self._miss_count = 0
        # Linienglaettung: EMA + letzte gute Position halten (gegen Springen/Flackern).
        self._line_smooth_alpha = 0.4   # 0..1: hoeher = reaktiver, niedriger = glatter
        self._line_hold_frames = 5      # so viele Miss-Frames die letzte Position halten
        self._line_state = {'white': {'val': None, 'miss': 0},
                            'yellow': {'val': None, 'miss': 0}}
        self._last_inference_time = None
        self.min_inference_interval = 0.5
        self.image_middle = 320
        # Statische Spur-Referenz (aus Referenzbild): Innenkanten als Geraden x = a*y + b.
        # Innenabstand = Bot-Breite -> Maßstab "passt der Bot durch?" pro Zeile.
        self._ref_aL, self._ref_bL = -0.595, 395.3   # linke Innenkante
        self._ref_aR, self._ref_bR = 0.582, 242.8    # rechte Innenkante
        self._ref_y0, self._ref_y1 = 215, 460        # gueltiger Zeilenbereich
        # lane_px = ueberstrichene Bot-Breite inkl. Reifen -> 1:1 verwenden, kein Zusatzfaktor.
        self.center_white = None
        self.center_yellow = None
        # valid=True: Linie echt gesehen; False: nur Rand-Fallback (nicht als Grenze nutzen).
        self.white_valid = False
        self.yellow_valid = False
        # Streckenbreite / Spurbreite -> Fahrlinie aus EINER echten Linie rekonstruierbar.
        self._track_width_ratio = None
        self.debug_cv_image = None
        self.debug_largest_gap = None
        self.debug_duckie_error = None
        self.debug_lowest_duckie = None
        self.debug_close_to_white = False
        self.debug_blocked = False
        self.debug_pivot_dir = 0   # +1 = links, -1 = rechts (fuer die Anzeige)
        # Masken-Debugfenster: die in fnDetectLane wirklich benutzten Masken + Suchzeile
        self.debug_mask_yellow = None
        self.debug_mask_white = None
        self.debug_lane_row = None

        self._pending_image = None
        self._cam_last = None   # Zeit des letzten empfangenen Kamerabilds (Aussetzer-Diagnose)
        self._pending_lock = threading.Lock()

        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"

        # Kameratopic abonnieren, jedes neue Bild wird automatisch an cbFindDuckies weitergegeben
        self.sub_image_original = rospy.Subscriber(self._camera_topic, CompressedImage, self.cbFindDuckies, queue_size = 1)

        #Publisher für das Ergebnis der Kreuzungserkennung

        self.pub_debug_duckies = rospy.Publisher(f'/{self._vehicle_name}/debug/duckie_detection', DuckieDetectionArray, queue_size=1)
        self.pub_debug_contours = rospy.Publisher(f'/{self._vehicle_name}/debug/duckie_contours', CompressedImage, queue_size=1)
        self.pub_debug_duckie_view = rospy.Publisher(f'/{self._vehicle_name}/debug/duckie_view', CompressedImage, queue_size=1)

        self.model = '/root/DuckieRace/src/packages/explore_duckietown_ii/models/duckie_v19.onnx'

        
        # GPU mit CPU-Fallback; HEURISTIC vermeidet die ~30s Kernel-Suche beim ersten Lauf.
        self.session = ort.InferenceSession(
            self.model,
            providers=[
                ('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'HEURISTIC'}),
                'CPUExecutionProvider',
            ],
        )
        active_providers = self.session.get_providers()
        if 'CUDAExecutionProvider' in active_providers:
            rospy.loginfo("[detect_duckies] ONNX laeuft auf GPU (CUDAExecutionProvider).")
        else:
            rospy.logwarn(
                "[detect_duckies] ONNX faellt auf CPU zurueck! Aktive Provider: %s. "
                "Verfuegbar: %s", active_providers, ort.get_available_providers())
        self.input_name = self.session.get_inputs()[0].name

        # GPU-Warmup: erste Inferenz mit Dummy-Bild, damit die echte sofort schnell ist.
        try:
            rospy.loginfo("[detect_duckies] GPU warmup...")
            _t0 = time.time()
            self.session.run(None, {self.input_name: np.zeros((1, 3, 480, 640), dtype=np.float32)})
            rospy.loginfo("[detect_duckies] GPU warmup fertig (%.1fs).", time.time() - _t0)
        except Exception as e:
            rospy.logwarn("[detect_duckies] GPU warmup fehlgeschlagen: %s", e)

        self.conf_threshold = 0.3

        self.pub_duckie_control_active = rospy.Publisher(f'/{self._vehicle_name}/detect/duckie_control_active', Bool, queue_size=1)
        self.pub_duckie_error = rospy.Publisher(f'/{self._vehicle_name}/detect/duckie_error', Float64, queue_size=1)
        # blocked = keine Luecke, durch die der Bot passt -> control_lane pivotiert
        self.pub_duckie_blocked = rospy.Publisher(f'/{self._vehicle_name}/detect/duckie_blocked', Bool, queue_size=1)
        # Drehrichtung fuer den Pivot: +1 = links, -1 = rechts (weg von der naechsten Linie)
        self.pub_duckie_pivot_dir = rospy.Publisher(f'/{self._vehicle_name}/detect/duckie_pivot_dir', Int32, queue_size=1)
        # Seite einer nahen Ente (+1 links, -1 rechts, 0 keine): control_lane sperrt kurz das Lenken dorthin.
        self.pub_duckie_side = rospy.Publisher(f'/{self._vehicle_name}/detect/duckie_side', Int32, queue_size=1)
        self.pub_duckie_distance_factor = rospy.Publisher(f'/{self._vehicle_name}/detect/duckie_distance_factor', Float64, queue_size=1)
        self.pub_duckies = rospy.Publisher(f'/{self._vehicle_name}/detect/duckies', Bool, queue_size=1)

        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()

    def cbUpdateParameters(self, parameters):
        """
        Lädt Parameter aus der Konfigurationsdatei und aktualisiert die Instanzvariablen.

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

        self.duckie_enabled = bool(parameters["detection"]["enabled"]["default"])
        # duckie_only: NUR Enten-Modus (durchgehend Ziel, control_active=True, keine Kreuzungen).
        self.duckie_only = bool(parameters["detection"]["duckie_only"]["default"])
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
        # Hysterese: Miss-Frames halten + Padding fuers Maskieren.
        self._hold_frames = int(parameters["detection"]["hold_frames"]["default"])
        self._mask_padding = int(parameters["detection"]["mask_padding"]["default"])
        # Luecken-Suche: Mindestbreite (px) und vertikale Toleranz fuer blockierende Enten.
        self._gap_row_band = int(parameters["detection"]["gap_row_band"]["default"])
        # Ab dieser Naehe gilt eine Ente als "seitlich neben mir" -> Seite melden.
        self._pass_side_min_factor = parameters["detection"]["pass_side_min_factor"]["default"]
        # Horizont-Zeile: alles darueber (Waende = "weiss") aus den Masken schneiden.
        self._lane_roi_top = int(parameters["detection"]["lane_roi_top"]["default"])
        # _avoid_gain: wie stark auf die Luecke zugesteuert wird (1.0 = voll, 0.5 = halb).
        self._avoid_gain = parameters["detection"]["avoid_gain"]["default"]
        # Sperrzone: ab dieser Zeile (y2) ist die Ente zu nah fuers PID-Ausweichen -> Pivot.
        self._blocked_zone_top = int(parameters["detection"]["blocked_zone_top"]["default"])
        # Ab so vielen Linien-Pixeln im Fahrband gilt eine Linie als "quer im Weg".
        self._line_min_px = int(parameters["detection"]["line_min_px"]["default"])
        # Wie weit von so einer Linie weggelenkt wird, als Anteil der Spurbreite.
        self._line_push = parameters["detection"]["line_push"]["default"]
    

    def crop_img(self, img):
        """
        Wendet eine perspektivische Transformation auf das Bild an und schneidet es auf crop_im_size x crop_im_size zu.

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
        # detection.enabled aus -> nichts verarbeiten, Bot faehrt normal Spur.
        if not self.duckie_enabled:
            self.pub_duckie_control_active.publish(Bool(data=False))
            self.pub_duckies.publish(Bool(data=False))
            self.pub_duckie_blocked.publish(Bool(data=False))   # sonst bliebe ein Pivot haengen
            self.pub_duckie_side.publish(Int32(data=0))         # sonst bliebe die Pass-Sperre aktiv
            # Debug-Reste loeschen, sonst zeigt die Duckie View alte Boxen/Ziele
            self.final_duckies = []
            self._last_duckies = []
            self.debug_duckie_error = None
            self.debug_lowest_duckie = None
            self.debug_largest_gap = None
            self.debug_blocked = False
            return
        # Zeit des letzten Kamerabilds (fuer die cam_age-Ausfalldiagnose im Inferenz-Log).
        self._cam_last = time.time()
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

                # Detection-Hysterese: kurze Aussetzer mit der letzten Erkennung ueberbruecken.
                if self.final_duckies:
                    self._last_duckies = self.final_duckies
                    self._miss_count = 0
                    active_duckies = self.final_duckies
                elif self._last_duckies and self._miss_count < self._hold_frames:
                    self._miss_count += 1
                    active_duckies = self._last_duckies
                else:
                    self._last_duckies = []
                    active_duckies = []

                h_img, w_img = cv_image.shape[:2]
                pad = self._mask_padding
                duckie_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                for x1, y1, x2, y2, conf in active_duckies:
                    duckie_mask[max(0, int(y1) - pad):min(h_img, int(y2) + pad),
                                max(0, int(x1) - pad):min(w_img, int(x2) + pad)] = 255

                lowest_duckie = None
                for duckie in active_duckies:
                    if lowest_duckie is None or duckie[3] > lowest_duckie[3]:
                        lowest_duckie = duckie

                # Seite einer nahen Ente (ueber ALLE Enten, nicht nur im Korridor) -> Lenk-Sperre.
                side = 0
                if lowest_duckie is not None:
                    rng = max(1, self.duckie_distance_max - self.duckie_distance_min)
                    df_all = (lowest_duckie[3] - self.duckie_distance_min) / rng
                    df_all = max(0.0, min(1.0, df_all))
                    if df_all >= self._pass_side_min_factor:
                        duck_cx = (lowest_duckie[0] + lowest_duckie[2]) / 2.0
                        side = 1 if duck_cx < self.image_middle else -1
                self.pub_duckie_side.publish(Int32(data=side))

                # Nur die untere Schranke filtert (zu weit weg); oben nicht mehr, sonst kroch
                # der Bot in die naechste Ente.
                if lowest_duckie is not None and lowest_duckie[3] >= self.duckie_distance_min:
                    # Spurlinien an der Zeile der naechsten Ente bestimmen (center_yellow zum Filtern).
                    hsv[duckie_mask == 255] = 0
                    self.fnDetectLane(hsv, lowest_duckie[3])

                    # Enten links der gelben Linie (x2 < center_yellow) sind ausserhalb der Spur -> raus.
                    if self.center_yellow is not None:
                        relevant = [d for d in active_duckies if max(d[0], d[2]) >= self.center_yellow]
                    else:
                        relevant = active_duckies

                    # Referenz-Ente = naechste Ente IM Fahrweg (nicht die naechste ueberhaupt),
                    # jede an ihrer eigenen Zeile geprueft.
                    threats = [d for d in relevant if self._duck_in_path(d)]
                    lowest_relevant = None
                    for duckie in (threats or relevant):
                        if lowest_relevant is None or duckie[3] > lowest_relevant[3]:
                            lowest_relevant = duckie

                    if lowest_relevant is not None and lowest_relevant[3] >= self.duckie_distance_min:
                        range_size = max(1, self.duckie_distance_max - self.duckie_distance_min)
                        distance_factor = (lowest_relevant[3] - self.duckie_distance_min) / range_size
                        distance_factor = max(0.0, min(1.0, distance_factor))
                        self.pub_duckie_distance_factor.publish(Float64(data=distance_factor))

                        # blocked=True nur, wenn der Korridor komplett zu ist (keine Luecke).
                        duckie_error, blocked = self.fnGetLaneDuckieError(relevant, int(lowest_relevant[3]))
                        self.debug_duckie_error = duckie_error
                        self.debug_lowest_duckie = lowest_relevant
                        self.debug_blocked = blocked

                        # Diagnose bei naher Ente (gedrosselt): wo faellt eine Ente aus der Rechnung?
                        if distance_factor >= 0.5:
                            lx1, _, lx2, ly2, lconf = lowest_relevant
                            bw = self._lane_px(ly2)
                            pc = self._path_center()
                            rospy.loginfo_throttle(
                                0.4,
                                "[duckie] naeh: y2=%.0f x=[%.0f,%.0f] conf=%.2f df=%.2f | "
                                "band(mitte=%.0f)=[%.0f,%.0f] inpath=%s | Y=%.0f(v=%s) W=%.0f(v=%s) "
                                "spurmitte=%.0f | alle=%d relevant=%d threats=%d | err=%s blocked=%s",
                                ly2, lx1, lx2, lconf, distance_factor,
                                pc, pc - bw / 2.0, pc + bw / 2.0,
                                self._duck_in_path(lowest_relevant),
                                self.center_yellow if self.center_yellow is not None else -1, self.yellow_valid,
                                self.center_white if self.center_white is not None else -1, self.white_valid,
                                ((self.center_yellow + self.center_white) / 2.0)
                                if (self.center_yellow is not None and self.center_white is not None) else -1,
                                len(active_duckies), len(relevant), len(threats),
                                ("%.3f" % duckie_error) if duckie_error is not None else "None", blocked)
                        if duckie_error is not None:
                            # Ente im Weg: Control aktiv lassen, damit der Bot nicht auf Lane zurueckfaellt.
                            self.pub_duckie_error.publish(Float64(data=duckie_error))
                            self.pub_duckie_blocked.publish(Bool(data=blocked))
                            if blocked:
                                # kein Durchkommen -> Pivot-Richtung: weg von der Quer-Linie, sonst von der Ente.
                                ld, lclose = self._line_in_path()
                                pdir = (ld if (lclose and ld != 0)
                                        else self._pivot_direction(threats))
                                self.debug_pivot_dir = pdir
                                self.pub_duckie_pivot_dir.publish(Int32(data=pdir))
                            self.pub_duckie_control_active.publish(Bool(data=True))
                        elif self.duckie_only:
                            # freie Bahn, aber wir bleiben im Enten-Modus -> selbst Spur fahren
                            self._publish_lane_follow()
                        else:
                            # freie Bahn (keine Ente im Korridor) -> kein Ausweichen noetig
                            self.pub_duckie_blocked.publish(Bool(data=False))
                            self.pub_duckie_control_active.publish(Bool(data=False))
                    else:
                        # nur Enten links der gelben Linie (ausserhalb der Spur) -> ignorieren
                        self.debug_lowest_duckie = None
                        self.pub_duckie_distance_factor.publish(Float64(data=0.0))
                        if self.duckie_only:
                            self._publish_lane_follow()
                        else:
                            self.debug_duckie_error = None
                            self.debug_largest_gap = None
                            self.debug_blocked = False
                            self.pub_duckie_blocked.publish(Bool(data=False))
                            self.pub_duckie_control_active.publish(Bool(data=False))
                else:
                    # gar keine (nahe) Ente
                    self.debug_lowest_duckie = None
                    self.pub_duckie_distance_factor.publish(Float64(data=0.0))
                    if self.duckie_only:
                        # Enten-Modus ohne Ente -> Spur selbst fahren (fnDetectLane an fester Zeile).
                        hsv[duckie_mask == 255] = 0
                        self.fnDetectLane(hsv, int((self._ref_y0 + self._ref_y1) / 2))
                        self._publish_lane_follow()
                    else:
                        self.debug_duckie_error = None
                        self.debug_largest_gap = None
                        self.debug_blocked = False
                        self.pub_duckie_blocked.publish(Bool(data=False))
                        self.pub_duckie_control_active.publish(Bool(data=False))
            except Exception as e:
                rospy.logerr(f"_inference_loop failed: {e}")

    def fnDetectDuckies(self, image):
        """
        Erkennt Duckies im Kamerabild (YOLO) und legt die Boxen in self.final_duckies ab.

        Args:
            img: BGR-Kamerabild als numpy Array
        """


        # Ergebnis lokal sammeln und erst am Ende ATOMAR zuweisen (sonst liest der Debug-Thread
        # waehrend der Inferenz eine leere Liste).
        found = []

        now = time.time()
        gap_ms = (now - self._last_inference_time) * 1000 if self._last_inference_time else 0
        self._last_inference_time = now

        raw_output = self.session.run(None, {self.input_name:image})
        inference_ms = (time.time() - now) * 1000
        detections = raw_output[0][0]

        max_conf = max((box[4] for box in detections), default=0)
        # Inferenz-Timing: Rechenzeit + Abstand zum letzten Lauf (= Rate), gedrosselt auf 1/s.
        hz = 1000.0 / gap_ms if gap_ms > 0 else 0.0
        cam_age_ms = (now - self._cam_last) * 1000 if self._cam_last else -1
        rospy.loginfo_throttle(
            1.0,
            f"Duckie detect: inference={inference_ms:4.0f}ms | gap={gap_ms:5.0f}ms "
            f"({hz:4.1f} Hz) | cam_age={cam_age_ms:4.0f}ms | max_conf={max_conf:.2f} thr={self.conf_threshold}")
        # Grosser gap trotz frischer Bilder -> Loop war blockiert (X11/GIL), nicht die Kamera.
        if gap_ms > 500:
            rospy.logwarn("[duckie] AUSSETZER: gap=%.0fms, cam_age=%.0fms -> %s",
                          gap_ms, cam_age_ms,
                          "Loop blockiert (Kamera lieferte, X11/GIL?)" if 0 <= cam_age_ms < 300
                          else "Kamera/WLAN lieferte nichts")

        for box in detections:
            x1, y1, x2, y2, conf, class_id = box
            if conf >= self.conf_threshold:
                found.append((x1, y1, x2, y2, conf))

        self.final_duckies = found
                

        self.pub_duckies.publish(Bool(data=len(self.final_duckies) > 0))


        #cv2.matchTemplate(contours, self.duckie_template, cv2.TM_CCOEFF_NORMED)
        
    def get_x_for_driving(self, mask, distance, left_line):
        # None, wenn keine Linie gefunden -> Caller (fnDetectLane) glaettet/haelt,
        # statt auf den Bildrand zu springen.
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
            return None

    def _smooth_line(self, name, raw, fallback):
        """Glaettet eine Linienposition (EMA) und haelt sie ueber ein paar Miss-Frames.
        Rueckgabe: (wert, valide); valide=False = nur Rand-Fallback."""
        s = self._line_state[name]
        if raw is not None:
            s['val'] = float(raw) if s['val'] is None else (
                self._line_smooth_alpha * float(raw) + (1.0 - self._line_smooth_alpha) * s['val'])
            s['miss'] = 0
            return s['val'], True
        s['miss'] += 1
        if s['val'] is not None and s['miss'] <= self._line_hold_frames:
            return s['val'], True
        s['val'] = None
        return fallback, False

    def fnDetectLane(self, hsv, distance):
        distance = int(distance)

        mask_yellow = cv2.inRange(hsv,
                               (self.hue_yellow_l,self.saturation_yellow_l, self.lightness_yellow_l),
                               (self.hue_yellow_h,self.saturation_yellow_h, self.lightness_yellow_h),)

        mask_white = cv2.inRange(hsv,
                               (self.hue_white_l,self.saturation_white_l, self.lightness_white_l),
                               (self.hue_white_h,self.saturation_white_h, self.lightness_white_h),)

        # ROI: oberhalb der Horizont-Zeile wegschneiden (sonst nimmt get_x_for_driving eine Wand als Rand).
        top = max(0, int(self._lane_roi_top))
        if top > 0:
            mask_yellow[:top, :] = 0
            mask_white[:top, :] = 0

        #Werte Müssen noch getestet werden
        white_alternative = int(len(hsv[0]) * 0.95)
        yellow_alternative = int(len(hsv[0]) * 0.05)

        # Fuers Debugfenster: genau die hier benutzten Masken merken.
        self.debug_mask_yellow = mask_yellow
        self.debug_mask_white = mask_white
        self.debug_lane_row = distance

        raw_white = self.get_x_for_driving(mask_white, distance, left_line=False)
        raw_yellow = self.get_x_for_driving(mask_yellow, distance, left_line=True)
        # Glaetten + valid-Flag (echt gesehen vs. Fallback).
        self.center_white, self.white_valid = self._smooth_line('white', raw_white, white_alternative)
        self.center_yellow, self.yellow_valid = self._smooth_line('yellow', raw_yellow, yellow_alternative)

    def _count_band(self, mask, y_from, y_to, step=4):
        """Zaehlt Maskenpixel innerhalb des Fahrbands zwischen zwei Bildzeilen."""
        if mask is None:
            return 0
        h, w = mask.shape[:2]
        pc = self._path_center()
        n = 0
        for y in range(max(0, int(y_from)), min(h, int(y_to)), step):
            half = self._lane_px(y) / 2.0
            a = max(0, int(pc - half))
            b = min(w, int(pc + half))
            if b > a:
                n += int(np.count_nonzero(mask[y, a:b]))
        return n

    def _line_in_path(self):
        """Liegt Linienfarbe im Fahrband? Faengt quer zulaufende Linien ab, die der
        Sobel-Gradient (nur senkrechte Kanten) uebersieht.

        Rueckgabe (richtung, nah): richtung +1=links / -1=rechts / 0=frei (weg von der Linie);
        nah=True wenn die Linie in der Sperrzone liegt -> Pivot.
        """
        my, mw = self.debug_mask_yellow, self.debug_mask_white
        if my is None or mw is None:
            return 0, False
        y0, zt, y1 = int(self._ref_y0), int(self._blocked_zone_top), int(self._ref_y1)

        # nah (Sperrzone) -> Pivot
        yc, wc = self._count_band(my, zt, y1), self._count_band(mw, zt, y1)
        if max(yc, wc) >= self._line_min_px:
            return (-1 if yc >= wc else 1), True
        # weiter vorne -> nur gegenlenken
        yf, wf = self._count_band(my, y0, zt), self._count_band(mw, y0, zt)
        if max(yf, wf) >= self._line_min_px:
            return (-1 if yf >= wf else 1), False
        return 0, False

    def _push_from_line(self, target, row):
        """Schiebt ein Fahrziel von einer quer im Fahrband liegenden Linie weg."""
        d, _ = self._line_in_path()
        if d == 0:
            return target
        return target - d * self._line_push * self._lane_px(row)

    def _lane_center(self, row):
        """Fahrlinie (Bild-x) an Zeile row. Wo beide Linien echt sind, wird die Streckenbreite
        als Vielfaches der Spurbreite gelernt (_track_width_ratio), um die Mitte aus EINER
        echten Linie zu rekonstruieren, statt auf einen Rand-Fallback hereinzufallen.

        Rueckfall-Kette (nie "geradeaus"):
          1. beide echt     -> (gelb+weiss)/2 (+ Verhaeltnis lernen)
          2. eine + gelernt -> aus der echten Linie rekonstruiert
          3. sonst          -> (gelb+weiss)/2 aus dem, was da ist (verzerrt, aber reagiert)
          4. keine Linien   -> None
        """
        lane = self._lane_px(row)
        if self.center_yellow is not None and self.center_white is not None \
                and self.yellow_valid and self.white_valid:
            ratio = (self.center_white - self.center_yellow) / lane
            if 0.5 <= ratio <= 6.0:   # plausibel -> Verhaeltnis lernen (geglaettet)
                self._track_width_ratio = ratio if self._track_width_ratio is None else (
                    0.2 * ratio + 0.8 * self._track_width_ratio)
            return (self.center_yellow + self.center_white) / 2.0

        # Rekonstruktion aus EINER Linie + gelerntem Verhaeltnis: verworfen (haelt bei
        # schraegem Bot/Kurve nicht). _track_width_ratio wird nur im Debug angezeigt.

        # Sonst: (ggf. verzerrte) Mitte aus dem, was da ist - besser als gar kein Lenkziel.
        if self.center_yellow is not None and self.center_white is not None:
            return (self.center_yellow + self.center_white) / 2.0
        return None

    def _norm_err(self, target):
        """Lenkfehler zu einem Bild-x-Ziel, normiert auf [-1, 1] (0 = geradeaus, positiv =
        Ziel links). GLEICHE Konvention wie detect_lane -> pid_duckie ist damit direkt mit
        pid vergleichbar. Ersetzt die alte Magie-Konstante /750 im Regler, die den
        Fehlerbereich auf ~+-0.43 stauchte und dem Enten-Modus rund ein Drittel der
        Lenkautoritaet des Spur-Reglers liess (Bot trug in Kurven raus).
        """
        return (self.image_middle - target) / float(max(1, self.image_middle))

    def _publish_lane_follow(self):
        """duckie_only ohne Ente: selbst der Spur folgen und control_active aktiv lassen,
        damit switch_control im Enten-Modus bleibt. Setzt voraus, dass fnDetectLane lief.
        """
        # An der Zeile rechnen, an der fnDetectLane die Linien gemessen hat (nicht fest).
        row = self.debug_lane_row
        if row is None:
            row = int((self._ref_y0 + self._ref_y1) / 2)
        # Linie quer im Fahrband und zu nah -> Pivot, auch ohne Ente im Bild.
        line_dir, line_close = self._line_in_path()
        if line_close:
            self.pub_duckie_error.publish(Float64(data=0.0))
            self.pub_duckie_blocked.publish(Bool(data=True))
            self.pub_duckie_pivot_dir.publish(Int32(data=line_dir))
            self.pub_duckie_control_active.publish(Bool(data=True))
            self.debug_duckie_error = 0.0
            self.debug_largest_gap = None
            self.debug_blocked = True
            self.debug_pivot_dir = line_dir
            return

        lane_center = self._lane_center(int(row))
        if lane_center is not None:
            # quer liegende Linie im Fahrband -> von ihr wegschieben
            lane_center = self._push_from_line(lane_center, int(row))
        err = 0.0 if lane_center is None else self._norm_err(lane_center)
        self.pub_duckie_error.publish(Float64(data=err))
        self.pub_duckie_blocked.publish(Bool(data=False))
        self.pub_duckie_control_active.publish(Bool(data=True))
        self.debug_duckie_error = err
        self.debug_largest_gap = None
        self.debug_blocked = False

    def _path_center(self):
        """Fahrweg-Mitte = Spurmitte (folgt der Kurve), sonst Bildmitte als Rueckfall.
        """
        if self.center_yellow is None or self.center_white is None:
            return float(self.image_middle)
        if not (self.white_valid or self.yellow_valid):
            return float(self.image_middle)
        return (self.center_yellow + self.center_white) / 2.0

    def _duck_in_path(self, duck):
        """Liegt die Ente im Fahrband? Geprueft an IHRER Zeile (y2); Band = Spurbreite(y2)
        um die Spurmitte (folgt der Kurve).
        """
        x1, y1, x2, y2, conf = duck
        bot_w = self._lane_px(y2)
        center = self._path_center()
        path_l = center - bot_w / 2.0
        path_r = center + bot_w / 2.0
        return max(x1, x2) > path_l and min(x1, x2) < path_r

    def _target_hits_duck(self, target, row, ducks):
        """Wuerde dieses Ziel eine Ente in ANDERER Tiefe treffen? Der Seitenversatz in
        Spurbreiten (k) wird in jede Zeile projiziert und gegen die dortigen Enten geprueft.
        """
        k = (target - self.image_middle) / self._lane_px(row)
        for x1, y1, x2, y2, conf in ducks:
            if abs(y2 - row) <= self._gap_row_band:
                continue
            bot_w = self._lane_px(y2)
            cx = self.image_middle + k * bot_w
            if max(x1, x2) > cx - bot_w / 2.0 and min(x1, x2) < cx + bot_w / 2.0:
                return True
        return False

    def _pivot_direction(self, ducks_in_path=None):
        """Drehrichtung fuer den Pivot: immer WEG von dem, was im Weg ist.
        Vorzeichen: +1 = omega positiv = LINKS, -1 = RECHTS (wie beim Abbiegen).
        Reihenfolge: 1. weg von der naeheren echten Linie, 2. weg von der Ente im Weg,
        3. Strecken-Prior (kurvt immer links -> +1).
        """
        # Nur valide Linien duerfen entscheiden (eine gelbe Ente sonst als "gelbe Linie").
        yv = self.yellow_valid and self.center_yellow is not None
        wv = self.white_valid and self.center_white is not None
        near = None
        if yv and wv:
            near = (self.center_white
                    if abs(self.image_middle - self.center_white)
                       < abs(self.image_middle - self.center_yellow)
                    else self.center_yellow)
        elif wv:
            near = self.center_white
        elif yv:
            near = self.center_yellow
        if near is not None:
            # WEG von der (naeheren) Linie: Linie rechts der Mitte -> +1 = LINKS,
            # Linie links der Mitte -> -1 = RECHTS.
            return 1 if near > self.image_middle else -1

        if ducks_in_path:
            # naechste (unterste) Ente im Weg -> von ihr wegdrehen:
            # Ente links vom Fahrweg -> -1 = RECHTS, Ente rechts -> +1 = LINKS
            nearest = max(ducks_in_path, key=lambda d: d[3])
            duck_cx = (nearest[0] + nearest[2]) / 2.0
            return -1 if duck_cx < self._path_center() else 1

        return 1   # Strecken-Prior: kurvt immer links -> +1 = LINKS

    def _lane_px(self, y):
        """Spur-Innenbreite (px) an Zeile y aus der statischen Referenz = Massstab dieser
        Zeile (1 Spurbreite ~ Bot-Breite), perspektivkonsistent ohne Kalibrierung.
        """
        y = max(self._ref_y0, min(self._ref_y1, float(y)))
        width = (self._ref_aR - self._ref_aL) * y + (self._ref_bR - self._ref_bL)
        return max(1.0, width)

    def fnGetLaneDuckieError(self, active_duckies, row):
        """
        Sucht die beste Fahr-Luecke: alle Enten der Zeile als blockierte Intervalle, freie
        Luecken zwischen gelb, Enten und weiss bilden, breiteste waehlen und mittig zielen.
        Rueckgabe (error, stop):
          - keine Linien / keine Ente im Korridor -> (None, False)
          - Ente(n) im Weg -> (error, False): mittig durch die breiteste Luecke
          - Ente im Fahrband UND in der Sperrzone -> (0.0, True): Pivot
        Pivot loest NUR die Sperrzone aus; der Fit-Check waehlt nur die Luecke aus.
        """
        # Zuerst: Ente im Fahrband? Braucht keine Linien (Band aus der statischen Referenz),
        # muss vor der Linien-Rechnung kommen.
        in_path_ducks = [d for d in active_duckies if self._duck_in_path(d)]
        in_path = bool(in_path_ducks)

        # Sperrzone = einziger Pivot-Ausloeser: Ente im Fahrweg UND nah (y2 >= _blocked_zone_top)
        # -> auf der Stelle drehen statt PID-Ausweichbogen.
        if any(d[3] >= self._blocked_zone_top for d in in_path_ducks):
            self.debug_largest_gap = None
            return 0.0, True

        # Quer liegende Linie zu nah -> ebenfalls Pivot.
        if self._line_in_path()[1]:
            self.debug_largest_gap = None
            return 0.0, True

        # Sind die Linien verlaesslich genug fuer eine Luecken-Rechnung?
        lines_ok = (self.center_white is not None and self.center_yellow is not None
                    and (self.white_valid or self.yellow_valid))

        if not lines_ok:
            # Ohne echte Linien keine Luecken-Rechnung; Ente ist hier noch weit weg -> weiterfahren.
            self.debug_largest_gap = None
            return None, False

        # Fahrkorridor: gelb (links) .. weiss (rechts). Defensiv sortieren.
        left = min(self.center_yellow, self.center_white)
        right = max(self.center_yellow, self.center_white)

        # Enten, die diese Zeile (mit vertikaler Toleranz) schneiden, auf den
        # Korridor geclippt als Hindernis-Intervalle sammeln.
        obstacles = []
        for x1, y1, x2, y2, conf in active_duckies:
            if y1 - self._gap_row_band <= row <= y2 + self._gap_row_band:
                ox1 = max(left, min(x1, x2))
                ox2 = min(right, max(x1, x2))
                if ox2 > ox1:
                    obstacles.append((ox1, ox2))

        if not obstacles:
            # keine Ente im Fahrkorridor -> freie Bahn, kein Ausweichen
            self.debug_largest_gap = None
            return None, False

        # Ente(n) im Korridor, aber keine im FAHRWEG (in_path oben ueber alle Enten an
        # IHRER jeweils eigenen Zeile geprueft) -> nicht ausweichen.
        if not in_path:
            # Enten im Bild, aber nicht im Weg -> normal Spur halten (ueber _lane_center).
            lane_center = self._lane_center(row)
            if lane_center is None:
                lane_center = (left + right) / 2.0   # gar keine Linien -> Korridormitte
            lane_center = self._push_from_line(lane_center, row)
            self.debug_largest_gap = lane_center
            self.debug_close_to_white = False
            return self._norm_err(lane_center), False

        # freie Luecken zwischen den (sortierten) Hindernissen bilden
        obstacles.sort()
        gaps = []
        cursor = left
        for ox1, ox2 in obstacles:
            if ox1 > cursor:
                gaps.append((cursor, ox1))
            cursor = max(cursor, ox2)
        if right > cursor:
            gaps.append((cursor, right))

        if not gaps:
            # Korridor an dieser Zeile zu, aber Ente noch nicht in der Sperrzone -> auf die
            # Korridormitte zielen und weiterfahren.
            target = (left + right) / 2.0
            self.debug_largest_gap = target
            self.debug_close_to_white = False
            return self._norm_err(target), False

        # Fit-Check: durch welche Luecken passt der Bot? (Luecke vs. Spurbreite derselben
        # Zeile). Nur Luecken-Auswahl, loest keinen Pivot aus.
        need = self._lane_px(row)
        passable = [g for g in gaps if (g[1] - g[0]) >= need]
        pivot_needed = False

        # Ziel = Mitte der breitesten passierbaren Luecke, deren Ziel keine Ente in anderer
        # Tiefe trifft (Tiefen-Veto).
        ordered = sorted(passable or gaps, key=lambda g: g[1] - g[0], reverse=True)
        chosen = next((g for g in ordered
                       if not self._target_hits_duck((g[0] + g[1]) / 2.0, row, active_duckies)),
                      ordered[0])
        target = (chosen[0] + chosen[1]) / 2.0

        # Ausweich-Daempfung: nur den Anteil Spurmitte->Lueckenziel mit _avoid_gain skalieren
        # (gleiche pid_duckie-Gains wie beim Linienfahren waeren beim Ausweichen zu heftig).
        base = self._lane_center(row)
        if base is None:
            base = (left + right) / 2.0
        target_damped = base + (target - base) * self._avoid_gain
        # In die gewaehlte Luecke klemmen, damit das gedaempfte Ziel nie auf einer Ente landet.
        target_damped = max(chosen[0], min(chosen[1], target_damped))

        self.debug_largest_gap = target_damped
        self.debug_close_to_white = target_damped > (left + right) / 2.0
        # Auf halbe Bildbreite normieren, NICHT durch lane_px teilen (die perspektivische
        # Daempfung ferner Versaetze ist gewollt).
        return self._norm_err(target_damped), pivot_needed

    def fnDrawFitCheck(self, img):
        """Debug-only: zeichnet, wo der Bot durchpasst (gruen) und wo nicht (rot).
        """
        try:
            xl = lambda y: self._ref_aL * y + self._ref_bL
            xr = lambda y: self._ref_aR * y + self._ref_bR
            y0, y1 = int(self._ref_y0), int(self._ref_y1)

            # Referenz-Trapez in grau
            corners = np.array([[xl(y0), y0], [xr(y0), y0], [xr(y1), y1], [xl(y1), y1]], np.int32)
            cv2.polylines(img, [corners], True, (200, 200, 200), 1)

            # Fahrband (cyan): bot-breites Band um die Spurmitte (folgt der Kurve).
            pc = self._path_center()
            band = np.array([[pc - self._lane_px(y0) / 2, y0], [pc + self._lane_px(y0) / 2, y0],
                             [pc + self._lane_px(y1) / 2, y1], [pc - self._lane_px(y1) / 2, y1]], np.int32)
            cv2.polylines(img, [band], True, (255, 255, 0), 2)

            # Sperrzone (magenta): Ente hier drin + im Fahrband -> Pivot statt PID.
            zt = int(self._blocked_zone_top)
            cv2.line(img, (0, zt), (img.shape[1], zt), (255, 0, 255), 2)
            cv2.putText(img, "SPERRZONE", (8, zt + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            # Linie quer im Fahrband? (zeigt Auslöser + Richtung)
            ld, lclose = self._line_in_path()
            if ld != 0:
                cv2.putText(img, "LINIE QUER -> %s%s" % ("LINKS" if ld > 0 else "RECHTS",
                                                         " (PIVOT)" if lclose else ""),
                            (8, zt - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            for y in range(y0 + 5, y1, 12):
                left, right = xl(y), xr(y)
                bot_w = right - left                 # Bot-Breite(y) = Spur-Innenbreite
                need = bot_w                         # Spurbreite = Bot-Breite (1:1)

                # Enten dieser Zeile auf den Korridor geclippt
                blocks = []
                for dx1, dy1, dx2, dy2, conf in self.final_duckies:
                    if dy1 <= y <= dy2:
                        a = max(left, min(dx1, dx2))
                        b = min(right, max(dx1, dx2))
                        if b > a:
                            blocks.append((a, b))
                blocks.sort()

                # freie Luecken bilden
                gaps = []
                cur = left
                for a, b in blocks:
                    if a > cur:
                        gaps.append((cur, a))
                    cur = max(cur, b)
                if right > cur:
                    gaps.append((cur, right))

                # jede Luecke gruen (Bot passt) / rot (zu schmal) einzeichnen
                for a, b in gaps:
                    col = (0, 200, 0) if (b - a) >= need else (0, 0, 255)
                    cv2.line(img, (int(a), y), (int(b), y), col, 2)
        except Exception:
            pass

    def fnDrawLineMasks(self):
        """Debug-only: zeigt die Gelb-/Weiss-Masken (inkl. ausmaskierter Enten) mit Suchzeile
        und erkannter Position (gefuellt = echt, hohl = Fallback).
        """
        my, mw = self.debug_mask_yellow, self.debug_mask_white
        if my is None or mw is None:
            return None
        h, w = my.shape[:2]
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[my > 0] = (0, 255, 255)      # gelb
        img[mw > 0] = (255, 255, 255)    # weiss
        img[(my > 0) & (mw > 0)] = (120, 120, 120)   # Ueberlappung -> verdaechtig

        row = self.debug_lane_row
        if row is not None:
            row = int(row)
            # Suchband: get_x_for_driving schaut row-50 .. row+50
            cv2.line(img, (0, max(0, row - 50)), (w, max(0, row - 50)), (0, 140, 255), 1)
            cv2.line(img, (0, min(h - 1, row + 50)), (w, min(h - 1, row + 50)), (0, 140, 255), 1)
            cv2.line(img, (0, row), (w, row), (0, 0, 255), 1)
            cv2.putText(img, f"row {row}", (5, max(12, row - 55)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1)
        cv2.line(img, (self.image_middle, 0), (self.image_middle, h), (80, 80, 80), 1)

        # erkannte Linien: gefuellt = valide, hohl = Fallback
        yr = row if row is not None else h // 2
        if self.center_yellow is not None:
            cv2.circle(img, (int(self.center_yellow), yr), 7, (0, 200, 255),
                       -1 if self.yellow_valid else 2)
        if self.center_white is not None:
            cv2.circle(img, (int(self.center_white), yr), 7, (255, 255, 255),
                       -1 if self.white_valid else 2)
        cv2.putText(img, f"Y {self.center_yellow if self.center_yellow is None else int(self.center_yellow)}"
                         f" {'OK' if self.yellow_valid else 'FALLBACK'}",
                    (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 200, 255) if self.yellow_valid else (0, 0, 255), 1)
        cv2.putText(img, f"W {self.center_white if self.center_white is None else int(self.center_white)}"
                         f" {'OK' if self.white_valid else 'FALLBACK'}",
                    (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255) if self.white_valid else (0, 0, 255), 1)
        # gelerntes Streckenbreiten-Verhaeltnis + rekonstruierte Fahrlinie.
        r = self._track_width_ratio
        cv2.putText(img, f"track {'--' if r is None else '%.2f' % r} lane-widths",
                    (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 0) if r is not None else (0, 0, 255), 1)
        if row is not None:
            lc = self._lane_center(row)
            if lc is not None:
                cv2.drawMarker(img, (int(lc), yr), (255, 0, 255), cv2.MARKER_TRIANGLE_UP, 14, 2)
        return img

    def run_debug(self):
        """
        Hauptloop: zeichnet und publiziert das Duckie-View-Debugbild.
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
                    if self.debug_blocked:
                        # Pivot-Richtung anzeigen (+1 links, -1 rechts).
                        d = self.debug_pivot_dir
                        arrow = "<< LINKS" if d > 0 else (">> RECHTS" if d < 0 else "?")
                        cv2.putText(img, f"PIVOT {arrow}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # Fit-Check einzeichnen (gruen = passt, rot = zu schmal).
                self.fnDrawFitCheck(img)

                cv2.imshow('Duckie View', img)
                mask_img = self.fnDrawLineMasks()
                if mask_img is not None:
                    cv2.imshow('Line Masks', mask_img)
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