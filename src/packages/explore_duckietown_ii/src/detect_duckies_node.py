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
        # Mask-/Detection-Hysterese gegen Aussetzer (bimodale conf): letzte
        # Erkennung max. _hold_frames Frames weiterverwenden -> gelbe Ente bleibt
        # maskiert (sonst als Spurlinie erkannt), Ausweichen flackert nicht und
        # laeuft kurz nach, wenn die Ente seitlich aus dem Bild faehrt.
        # _hold_frames/_mask_padding/_gap_row_band -> cbUpdateParameters
        # (wird in util.init_parameters synchron aufgerufen, bevor der Inferenz-Thread startet).
        self._last_duckies = []
        self._miss_count = 0
        # Glaettung der Linienpositionen gegen Springen/Verschwinden (verrauschte
        # center_yellow/white -> zappelnder Korridor -> Flackern): EMA + letzte gute
        # Position ueber ein paar Miss-Frames halten, statt auf den Bildrand zu springen.
        self._line_smooth_alpha = 0.4   # 0..1: hoeher = reaktiver, niedriger = glatter
        self._line_hold_frames = 5      # so viele Miss-Frames die letzte Position halten
        self._line_state = {'white': {'val': None, 'miss': 0},
                            'yellow': {'val': None, 'miss': 0}}
        self._last_inference_time = None
        self.min_inference_interval = 0.5
        self.image_middle = 320
        # STATISCHE Spur-Referenz (einmal aus dem Referenzbild gemessen): die Innenkanten
        # der gelben Linien als zwei Geraden x = a*y + b. Der Innenabstand = Bot-Breite +
        # Toleranz -> Maßstab "passt der Bot durch?" pro Zeile, ohne Kalibrierung.
        self._ref_aL, self._ref_bL = -0.595, 395.3   # linke Innenkante
        self._ref_aR, self._ref_bR = 0.582, 242.8    # rechte Innenkante
        self._ref_y0, self._ref_y1 = 215, 460        # gueltiger Zeilenbereich
        # WICHTIG (Kalibrierungsbild): die gelben Baender kleben knapp AUSSERHALB der
        # Reifen - lane_px IST also bereits die ueberstrichene Bot-Breite inkl. Reifen
        # (+ ~0.5 cm je Seite). Deshalb 1:1 verwenden, KEIN zusaetzlicher Faktor. Ein
        # frueherer Faktor 0.85 machte das Fahrband ~15% zu schmal -> Reifen traf Enten.
        self.center_white = None
        self.center_yellow = None
        # valid=True: Linie wurde wirklich gesehen (oder kurz gehalten); False: nur
        # Rand-Fallback -> nicht als echte Grenze fuer die Fehler-/Luecken-Rechnung nehmen.
        self.white_valid = False
        self.yellow_valid = False
        # Streckenbreite als Vielfaches der Spurbreite, gemessen wo beide Linien echt sind.
        # Damit laesst sich die Fahrlinie aus EINER echten Linie rekonstruieren, statt auf
        # einen Rand-Fallback hereinzufallen.
        self._track_width_ratio = None
        self.debug_cv_image = None
        self.debug_largest_gap = None
        self.debug_duckie_error = None
        self.debug_lowest_duckie = None
        self.debug_close_to_white = False
        self.debug_blocked = False
        # Masken-Debugfenster: die in fnDetectLane wirklich benutzten Masken + Suchzeile
        self.debug_mask_yellow = None
        self.debug_mask_white = None
        self.debug_lane_row = None

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

        
        # GPU bevorzugen, CPU als Fallback. cudnn_conv_algo_search=HEURISTIC verhindert
        # die langsame, erschoepfende Kernel-Suche beim ERSTEN Lauf (sonst dauert die
        # erste Inferenz teils ~30s -> Boxen erst, nachdem der Bot schon faehrt).
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

        # GPU-Warmup beim Start: erste CUDA-Inferenz richtet Kernel ein. Einmal mit
        # Dummy-Bild durchlaufen, damit die erste ECHTE Erkennung sofort schnell ist
        # und die Boxen nicht erst nach Sekunden (nach dem Losfahren) kommen.
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
        # Seite, auf der eine NAHE Ente steht: +1 = links, -1 = rechts, 0 = keine.
        # Verschwindet sie, steht sie noch seitlich neben dem Bot -> control_lane sperrt
        # kurz das Lenken in diese Richtung (sonst streift er sie).
        self.pub_duckie_side = rospy.Publisher(f'/{self._vehicle_name}/detect/duckie_side', Int32, queue_size=1)
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

        self.duckie_enabled = bool(parameters["detection"]["enabled"]["default"])
        # duckie_only: NUR Enten-Modus. detect_duckies liefert dann DURCHGEHEND ein Ziel und
        # haelt control_active=True -> switch_control bleibt dauerhaft auf Obstacle, faellt
        # nie auf Lane zurueck, und Kreuzungen triggern nicht (die brauchen den Lane-Modus).
        # Fuer die isolierte Ausweich-Challenge. 0 = normales Umschalten wie bisher.
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
        # Hysterese: wie viele Miss-Frames die letzte Erkennung gehalten wird (Nachlauf
        # bei seitlich verschwindender Ente) + Padding fuers Maskieren.
        self._hold_frames = int(parameters["detection"]["hold_frames"]["default"])
        self._mask_padding = int(parameters["detection"]["mask_padding"]["default"])
        # Mehr-Enten-Luckensuche: Mindestbreite einer Luecke in px (~Bot-Breite) und
        # vertikale Toleranz, welche Enten die Detektions-Zeile blockieren.
        self._gap_row_band = int(parameters["detection"]["gap_row_band"]["default"])
        # Ab dieser Naehe gilt eine Ente als "gleich seitlich neben mir" -> ihre Seite
        # wird gemeldet, damit control_lane nach dem Vorbeifahren nicht dorthin lenkt.
        self._pass_side_min_factor = parameters["detection"]["pass_side_min_factor"]["default"]
        # Horizont-Zeile fuer die Linien-Suche: alles darueber (Waende/Moebel = fast alles
        # "weiss") wird aus den Masken geschnitten. Default = distance_min-Bereich.
        self._lane_roi_top = int(parameters["detection"]["lane_roi_top"]["default"])
        # Wie stark auf die Luecke zugesteuert wird, relativ zur Spurmitte. Trennt die
        # Ausweich-Schaerfe vom Linienfahren, das durch dieselben pid_duckie-Gains laeuft.
        # 1.0 = voll auf die Lueckenmitte (altes Verhalten), 0.5 = halb so weit raus.
        self._avoid_gain = parameters["detection"]["avoid_gain"]["default"]
        # Sperrzone: ab dieser Bildzeile (y2 der Ente) ist sie zu nah fuers PID-Ausweichen
        # -> Pivot. Kleiner = Zone beginnt weiter oben = frueher pivotieren.
        self._blocked_zone_top = int(parameters["detection"]["blocked_zone_top"]["default"])
    

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
        # Schalter (detection.enabled, live umschaltbar): Duckie-Modus aus -> nichts
        # verarbeiten, Obstacle-Modus freigeben -> switch_control bleibt auf Lane und der
        # Bot faehrt normal Spur. Fuer die isolierte "Challenge startet jetzt"-Aufgabe.
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

                # Detection-Hysterese: verliert das Modell die Ente fuer wenige
                # Frames, halten wir die letzte Erkennung weiter (siehe __init__).
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

                # Seite einer NAHEN Ente melden - bewusst ueber ALLE Enten, nicht nur die
                # im Fahrkorridor: eine nahe Ente neben dem Bot ist physisch neben ihm,
                # egal ob sie in der Spur steht. Verschwindet sie gleich aus dem Bild
                # (weil sie neben dem Bot ist), sperrt control_lane kurz das Lenken dorthin.
                side = 0
                if lowest_duckie is not None:
                    rng = max(1, self.duckie_distance_max - self.duckie_distance_min)
                    df_all = (lowest_duckie[3] - self.duckie_distance_min) / rng
                    df_all = max(0.0, min(1.0, df_all))
                    if df_all >= self._pass_side_min_factor:
                        duck_cx = (lowest_duckie[0] + lowest_duckie[2]) / 2.0
                        side = 1 if duck_cx < self.image_middle else -1
                self.pub_duckie_side.publish(Int32(data=side))

                # Untere Schranke (zu weit weg) deaktiviert weiterhin. Oben NICHT mehr
                # deaktivieren: ist die Ente naeher als distance_max, ist sie maximal
                # gefaehrlich -> Control aktiv lassen, distance_factor unten auf 1.0 geclampt.
                # (Vorher fiel die Ente bei y2>distance_max aus dem Bereich -> Bremse/Lenkung
                # aus -> Bot kroch in die Ente.)
                if lowest_duckie is not None and lowest_duckie[3] >= self.duckie_distance_min:
                    # Spurlinien an der Zeile der (vorlaeufig) naechsten Ente bestimmen,
                    # damit wir center_yellow zum Filtern kennen.
                    hsv[duckie_mask == 255] = 0
                    self.fnDetectLane(hsv, lowest_duckie[3])

                    # Enten LINKS der gelben Linie ignorieren: liegt die rechte Kante (x2)
                    # links von center_yellow, ist die Ente ausserhalb der Spur -> raus,
                    # auch fuer die Wahl der naechsten Ente / distance_factor (nicht nur in
                    # der Luecken-Rechnung). Steht sie AUF der Linie (x2 >= center_yellow),
                    # bleibt sie drin. (Bei ungueltiger gelber Linie -> spaeter ueber das
                    # Gueltigkeits-Flag, Stufe C; hier greift der aktuelle center_yellow.)
                    if self.center_yellow is not None:
                        relevant = [d for d in active_duckies if max(d[0], d[2]) >= self.center_yellow]
                    else:
                        relevant = active_duckies

                    # Referenz-Ente = die naechste Ente IM FAHRWEG, nicht die naechste
                    # ueberhaupt. Sonst bestimmt eine Ente, die nur seitlich danebensteht,
                    # die Referenzzeile - und die echte Gefahr in anderer Tiefe faellt aus
                    # der Luecken-Rechnung raus (genau der "um die eine rum, die andere
                    # umgefahren"-Fall). Jede Ente wird an IHRER eigenen Zeile geprueft.
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

                        # stop=True nur, wenn der Korridor KOMPLETT zu ist (gar keine
                        # Luecke). Rein horizontal, kein distance_factor mehr -> ferne
                        # Enten am Horizont loesen keinen Stopp mehr aus (die filtert
                        # distance_min raus, wenn noetig).
                        duckie_error, blocked = self.fnGetLaneDuckieError(relevant, int(lowest_relevant[3]))
                        self.debug_duckie_error = duckie_error
                        self.debug_lowest_duckie = lowest_relevant
                        self.debug_blocked = blocked

                        # DIAGNOSE (nur bei naher Ente, gedrosselt): zeigt genau, an welchem
                        # Tor eine Ente rausfaellt - Erkennung / distance_min / relevant /
                        # Fahrband. Bei "ueberfahren ohne Reaktion" ist inpath=False der
                        # Verdaechtige (Fahrband ist geradeaus, Bot faehrt aber Kurve).
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
                            # Ente im Weg (ausweichen ODER kein Durchkommen): Control aktiv
                            # lassen, damit der Bot NICHT auf Lane zurueckfaellt und reinfaehrt.
                            self.pub_duckie_error.publish(Float64(data=duckie_error))
                            self.pub_duckie_blocked.publish(Bool(data=blocked))
                            if blocked:
                                # kein Durchkommen -> Drehrichtung mitgeben. threats mitgeben,
                                # damit ohne gueltige Linien wenigstens VON DER ENTE weg
                                # gedreht wird statt blind in den Prior.
                                self.pub_duckie_pivot_dir.publish(
                                    Int32(data=self._pivot_direction(threats)))
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
                        # Enten-Modus bleibt -> Spur selbst fahren. fnDetectLane lief hier
                        # noch nicht, also einmal an einer festen Zeile mitten im gueltigen
                        # Referenzbereich nachziehen.
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
        Hier soll der Code zur Erkennung von Duckies implementiert werden. Das Ergebnis (True wenn Duckie erkannt, sonst False)
        wird auf dem Topic /{vehicle_name}/detect/duckies veröffentlicht.

        Autor: Felix Faass

        Args:
            img: BGR-Kamerabild als numpy Array
        """


        # WICHTIG: Ergebnis lokal sammeln und erst am Ende ATOMAR zuweisen. Vorher wurde
        # self.final_duckies gleich zu Beginn geleert und erst nach der Inferenz wieder
        # gefuellt -> die Liste war waehrend der ~15 ms Inferenz LEER. run_debug liest sie
        # aus einem anderen Thread und malte dann keine Box, obwohl die Ente sauber erkannt
        # war (bei 30 fps Kamera grob jedes zweite Debug-Bild). Reines Anzeige-Artefakt, die
        # Regelung war nie betroffen (liest im selben Thread nach der Inferenz + Hysterese).
        found = []

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
        """Glaettet eine Linienposition (EMA) und haelt die letzte gute Position fuer
        ein paar Miss-Frames, statt bei Nicht-Erkennung auf den Bildrand zu springen.
        Rueckgabe: (wert, valide). valide=False = nur Rand-Fallback (nicht gesehen)."""
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

        # ROI: alles OBERHALB der Horizont-Zeile wegschneiden. Ueber dem Boden sind Waende,
        # Moebel und Fenster fast komplett "weiss" -> ohne Crop kann get_x_for_driving eine
        # Wand als Fahrbahnrand nehmen (es sucht in row +- 50, reicht also nach oben).
        top = max(0, int(self._lane_roi_top))
        if top > 0:
            mask_yellow[:top, :] = 0
            mask_white[:top, :] = 0

        #Werte Müssen noch getestet werden
        white_alternative = int(len(hsv[0]) * 0.95)
        yellow_alternative = int(len(hsv[0]) * 0.05)

        # Fuers Masken-Debugfenster merken: GENAU die Masken, mit denen hier gesucht wird
        # (inkl. ausmaskierter Enten), nicht neu gerechnete.
        self.debug_mask_yellow = mask_yellow
        self.debug_mask_white = mask_white
        self.debug_lane_row = distance

        raw_white = self.get_x_for_driving(mask_white, distance, left_line=False)
        raw_yellow = self.get_x_for_driving(mask_yellow, distance, left_line=True)
        # Glaetten + letzte gute Position halten -> stabile Linien (ruhiger Korridor/PID).
        # Zusaetzlich merken, ob die Linie echt gesehen wurde (valid) oder nur Fallback.
        self.center_white, self.white_valid = self._smooth_line('white', raw_white, white_alternative)
        self.center_yellow, self.yellow_valid = self._smooth_line('yellow', raw_yellow, yellow_alternative)

    def _lane_center(self, row):
        """Fahrlinie (Bild-x) an Zeile row, ohne sich auf einen Fallback zu verlassen.

        (gelb+weiss)/2 ist nur brauchbar, wenn BEIDE Linien echt gesehen wurden. Faellt
        Gelb auf seinen Rand-Fallback (bei uns staendig), zieht das die "Mitte" nach
        aussen und der Bot verlaesst die Strecke.

        Stattdessen: wo beide Linien echt sind, wird die Streckenbreite als VIELFACHES der
        Spurbreite gemessen (_track_width_ratio = (weiss-gelb)/lane_px). Dieses Verhaeltnis
        ist perspektivfrei (beides skaliert gleich mit der Zeile) und physisch konstant ->
        faellt eine Linie weg, laesst sich die Mitte aus der verbliebenen ECHTEN Linie
        rekonstruieren. Das ist die white_offset-Idee, nur mit gemessener statt geratener
        Breite.

        WICHTIG - Rueckfall-Kette, NIE "geradeaus":
          1. beide Linien echt      -> (gelb+weiss)/2  (+ Verhaeltnis lernen)
          2. eine echt + gelernt    -> aus der echten Linie rekonstruiert
          3. sonst, Linien bekannt  -> (gelb+weiss)/2 aus dem, was da ist (verzerrt, wenn
             eine ein Fallback ist - aber er reagiert wenigstens, statt stur geradeaus)
          4. gar keine Linien       -> None
        Punkt 3 ist der Grund, warum das hier ueberhaupt existiert: ohne ihn lieferte die
        Funktion None, der Aufrufer machte daraus err=0.0 und der Bot ignorierte die weisse
        Linie komplett. Ein verzerrtes Ziel ist besser als gar keins.

        Autor: Felix Faass
        """
        lane = self._lane_px(row)
        if self.center_yellow is not None and self.center_white is not None \
                and self.yellow_valid and self.white_valid:
            ratio = (self.center_white - self.center_yellow) / lane
            if 0.5 <= ratio <= 6.0:   # plausibel -> Verhaeltnis lernen (geglaettet)
                self._track_width_ratio = ratio if self._track_width_ratio is None else (
                    0.2 * ratio + 0.8 * self._track_width_ratio)
            return (self.center_yellow + self.center_white) / 2.0

        # Rekonstruktion aus EINER Linie + gelerntem Verhaeltnis: VERWORFEN. Die Annahme
        # "Verhaeltnis ist perspektivfrei" haelt nicht - bei schraegem Bot oder in der
        # Kurve skaliert die Streckenbreite im Bild nicht wie lane_px. Isolierter Test:
        # ratio bei row=250 gelernt (3.63), bei row=337 angewandt -> lane_center=13, also
        # Vollausschlag an den Bildrand. Schlimmer als der Fallback, den es ersetzen sollte.
        # _track_width_ratio wird weiterhin gelernt und im Debug angezeigt (interessant),
        # aber NICHT zum Fahren benutzt.

        # Sonst: (ggf. verzerrte) Mitte aus dem, was da ist. Verzerrt (Fallback-Gelb) ist
        # immer noch besser als gar kein Lenkziel - ohne das ignorierte der Bot die weisse
        # Linie komplett (err=0.0 = stur geradeaus).
        if self.center_yellow is not None and self.center_white is not None:
            return (self.center_yellow + self.center_white) / 2.0
        return None

    def _norm_err(self, target):
        """Lenkfehler zu einem Bild-x-Ziel, normiert auf [-1, 1] (0 = geradeaus, positiv =
        Ziel links). GLEICHE Konvention wie detect_lane -> pid_duckie ist damit direkt mit
        pid vergleichbar. Ersetzt die alte Magie-Konstante /750 im Regler, die den
        Fehlerbereich auf ~+-0.43 stauchte und dem Enten-Modus rund ein Drittel der
        Lenkautoritaet des Spur-Reglers liess (Bot trug in Kurven raus).

        Autor: Felix Faass
        """
        return (self.image_middle - target) / float(max(1, self.image_middle))

    def _publish_lane_follow(self):
        """duckie_only: kein Ausweichen noetig -> trotzdem SELBST der Spur folgen und
        control_active aktiv lassen. So faellt switch_control nie aus dem Enten-Modus raus
        (kein Lane/Kreuzung/Align) und es gibt kein Lane<->Obstacle-Flackern.
        Setzt voraus, dass fnDetectLane vorher lief (center_yellow/white gesetzt).

        Autor: Felix Faass
        """
        # An DER Zeile rechnen, an der fnDetectLane die Linien wirklich gemessen hat.
        # Eine feste Zeile waere falsch: bei vorhandener Ente misst fnDetectLane an DEREN
        # Zeile - dann wuerden Linienpositionen aus Zeile A mit lane_px(B) verrechnet.
        row = self.debug_lane_row
        if row is None:
            row = int((self._ref_y0 + self._ref_y1) / 2)
        lane_center = self._lane_center(int(row))
        err = 0.0 if lane_center is None else self._norm_err(lane_center)
        self.pub_duckie_error.publish(Float64(data=err))
        self.pub_duckie_blocked.publish(Bool(data=False))
        self.pub_duckie_control_active.publish(Bool(data=True))
        self.debug_duckie_error = err
        self.debug_largest_gap = None
        self.debug_blocked = False

    def _path_center(self):
        """Wo faehrt der Bot wirklich hin? Auf der Geraden ist das die Bildmitte, in der
        KURVE aber nicht - dort liegt der Fahrweg seitlich versetzt. Die Spurmitte sagt
        das bereits (sie IST die Kurve), und fnDetectLane hat sie ohnehin schon berechnet
        -> kostenlos. Sind die Linien unbrauchbar, bleibt die Bildmitte als Rueckfall.

        Autor: Felix Faass
        """
        if self.center_yellow is None or self.center_white is None:
            return float(self.image_middle)
        if not (self.white_valid or self.yellow_valid):
            return float(self.image_middle)
        return (self.center_yellow + self.center_white) / 2.0

    def _duck_in_path(self, duck):
        """Liegt diese Ente im Fahrband des Bots? Geprueft an IHRER EIGENEN Bildzeile (y2,
        also ihrem Bodenpunkt) - NICHT an der Zeile irgendeiner anderen Ente. Sonst faellt
        eine Ente in anderer Tiefe aus der Betrachtung und wird ueberfahren.
        Fahrband = Spurbreite(y2) um die SPURMITTE (1:1 = Bot inkl. Reifen).
        Frueher lag das Band um die Bildmitte = stur geradeaus -> in der Kurve sass die
        Ente seitlich daneben, fiel aus dem Band und wurde ueberfahren.

        Autor: Felix Faass
        """
        x1, y1, x2, y2, conf = duck
        bot_w = self._lane_px(y2)
        center = self._path_center()
        path_l = center - bot_w / 2.0
        path_r = center + bot_w / 2.0
        return max(x1, x2) > path_l and min(x1, x2) < path_r

    def _target_hits_duck(self, target, row, ducks):
        """Wuerde dieses Ziel eine Ente in einer ANDEREN Tiefe treffen?

        Trick ueber den Spurbreiten-Massstab: der Seitenversatz des Ziels ist
        k = (target - Bildmitte) / lane_px(row)  -> Versatz in Spurbreiten. Derselbe reale
        Versatz liegt in Zeile y2 bei  Bildmitte + k * lane_px(y2). Damit laesst sich das
        Ziel in JEDE Tiefe projizieren und gegen die dortigen Enten pruefen - genau der
        Fall "ich weiche der einen aus und fahre die andere um".
        Enten nahe der Referenzzeile werden uebersprungen, die deckt die Luecken-Rechnung
        selbst ab.

        Autor: Felix Faass
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
        """Drehrichtung fuer den Pivot (+1 = links, -1 = rechts), in dieser Reihenfolge:

        1. Linien verlaesslich -> WEG von der naeheren Linie (dort ist mehr Platz).
        2. Sonst, wenn Enten im Weg bekannt -> WEG von der Ente. Ohne Linien wissen wir
           zwar nicht, wo die Strasse ist, aber YOLO sagt uns, wo die ENTE ist - und in
           die zu drehen ist immer falsch. (Vorher: blind Prior links -> drehte in eine
           links stehende Ente hinein.)
        3. Sonst Strecken-Prior (diese Strecke kurvt immer links -> +1).

        Autor: Felix Faass
        """
        lines_ok = (self.center_yellow is not None and self.center_white is not None
                    and (self.white_valid or self.yellow_valid))
        if lines_ok:
            d_yellow = abs(self.image_middle - self.center_yellow)
            d_white = abs(self.image_middle - self.center_white)
            return 1 if d_white < d_yellow else -1

        if ducks_in_path:
            # naechste (unterste) Ente im Weg -> von ihr wegdrehen
            nearest = max(ducks_in_path, key=lambda d: d[3])
            duck_cx = (nearest[0] + nearest[2]) / 2.0
            return -1 if duck_cx < self._path_center() else 1

        return 1

    def _lane_px(self, y):
        """Spur-Innenbreite in px an Bildzeile y, aus der statischen Referenz.
        Das ist der MASSSTAB dieser Zeile: 1 Spurbreite ~ Bot-Breite + Toleranz.
        Damit lassen sich alle horizontalen Abstaende dieser Zeile in "Spurbreiten"
        ausdruecken -> perspektivkonsistent, ganz ohne Kamerakalibrierung.

        Autor: Felix Faass
        """
        y = max(self._ref_y0, min(self._ref_y1, float(y)))
        width = (self._ref_aR - self._ref_aL) * y + (self._ref_bR - self._ref_bL)
        return max(1.0, width)

    def fnGetLaneDuckieError(self, active_duckies, row):
        """
        Sucht die beste Fahr-Luecke in der Detektions-Zeile. Statt nur die naechste
        Ente gegen eine Linie zu betrachten, werden ALLE Enten (die diese Zeile
        schneiden) als blockierte Intervalle behandelt und die freien Luecken
        zwischen gelber Linie, den Enten und weisser Linie gebildet. Gewaehlt wird
        die breiteste Luecke und zielt in ihre MITTE (max. Abstand zu Ente UND Linie
        -> Bot passt durch, ohne dass wir die Bot-Breite kennen). Rueckgabe: (error, stop).
          - keine Linien / keine Ente im Korridor -> (None, False): kein Ausweichen.
          - Ente(n) im Weg, Bot passt durch -> (error, False): mittig durch die
            breiteste PASSIERBARE Luecke.
          - keine Luecke, durch die der Bot passt -> (error, True): Pivot noetig.
        "Passt der Bot durch" = Luecke >= Spurbreite(row) (1:1, s. Kalibrierung). Spurbreite
        kommt aus der statischen Referenz und ist der Bot-Breiten-Massstab dieser Zeile
        -> beides px in derselben Zeile, Perspektive kuerzt sich weg, keine Kalibrierung.
        Der Fehler ist in Spurbreiten normiert (siehe unten).

        Autor: Felix Faass
        """
        # ZUERST: steht eine Ente im Fahrband? Das braucht KEINE Linien - das Band kommt
        # aus der statischen Referenzbreite um _path_center() (faellt notfalls auf die
        # Bildmitte zurueck). Diese Frage muss vor allem Linien-Kram kommen, sonst wird
        # eine Ente direkt vor dem Bot ignoriert, nur weil die Linien gerade mies sind.
        in_path_ducks = [d for d in active_duckies if self._duck_in_path(d)]
        in_path = bool(in_path_ducks)

        # SPERRZONE: Ente im Fahrweg UND so nah (y2 unterhalb _blocked_zone_top), dass ein
        # PID-Ausweichbogen nicht mehr reicht - er wuerde nur noch anschneiden. Also das
        # Regler-Manoever abbrechen und stattdessen auf der Stelle drehen (Pivot).
        if any(d[3] >= self._blocked_zone_top for d in in_path_ducks):
            self.debug_largest_gap = None
            return 0.0, True

        # Sind die Linien verlaesslich genug fuer eine Luecken-Rechnung?
        lines_ok = (self.center_white is not None and self.center_yellow is not None
                    and (self.white_valid or self.yellow_valid))

        if not lines_ok:
            self.debug_largest_gap = None
            if in_path:
                # Ente steht im Weg, aber ohne echte Linien laesst sich keine Luecke
                # rechnen. NICHT ignorieren (sonst faehrt er drueber!) -> Pivot. Die
                # Richtung kommt aus _pivot_direction(), das ohne gueltige Linien den
                # Strecken-Prior nimmt. Lieber gedreht als drueber.
                return 0.0, True
            # keine Ente im Weg und keine Linien -> nichts zu tun, nichts erfinden
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
            # Ente(n) im Bild, aber NICHT im Weg -> nicht ausweichen, sondern normal die
            # Spur halten. Ueber _lane_center, damit ein Gelb-Fallback die Fahrlinie nicht
            # nach aussen zieht (genau das trug den Bot von der Strecke).
            lane_center = self._lane_center(row)
            if lane_center is None:
                lane_center = (left + right) / 2.0   # gar keine Linien -> Korridormitte
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
            # Korridor komplett zu -> definitiv kein Durchkommen -> Pivot.
            target = (left + right) / 2.0
            self.debug_largest_gap = target
            self.debug_close_to_white = False
            return self._norm_err(target), True

        # FIT-CHECK gegen die statische Spur-Referenz: passt der Bot (mit Toleranz)
        # ueberhaupt durch eine der Luecken? Beides in px in DERSELBEN Zeile -> der
        # Perspektiv-Massstab kuerzt sich weg -> keine Kalibrierung noetig.
        need = self._lane_px(row)
        passable = [g for g in gaps if (g[1] - g[0]) >= need]
        pivot_needed = not passable

        # Ziel = Mitte der breitesten PASSIERBAREN Luecke - aber mit TIEFEN-VETO: die
        # breiteste Luecke, deren Ziel keine Ente in einer anderen Tiefe trifft. Trifft
        # jede eine, bleibt es bei der breitesten (best effort, KEIN zusaetzlicher Pivot -
        # das Veto darf nur besser waehlen, nie neue Pivots ausloesen).
        ordered = sorted(passable or gaps, key=lambda g: g[1] - g[0], reverse=True)
        chosen = next((g for g in ordered
                       if not self._target_hits_duck((g[0] + g[1]) / 2.0, row, active_duckies)),
                      ordered[0])
        target = (chosen[0] + chosen[1]) / 2.0

        # AUSWEICH-DAEMPFUNG: Das Lueckenziel kann sprunghaft weit von der Spurmitte weg
        # liegen - viel weiter als beim reinen Linienfahren. Beides laeuft aber durch
        # DIESELBEN pid_duckie-Gains, deshalb sind Werte, die fuer die Linie passen, beim
        # Ausweichen zu heftig. Also nur den AUSWEICH-ANTEIL daempfen: die Verschiebung
        # von der Spurmitte zum Lueckenziel wird mit _avoid_gain skaliert, die Spurmitte
        # selbst bleibt unangetastet. avoid_gain=1.0 -> wie vorher.
        base = self._lane_center(row)
        if base is None:
            base = (left + right) / 2.0
        target_damped = base + (target - base) * self._avoid_gain
        # ABER: in die gewaehlte Luecke klemmen. Liegt die Ente zwischen Spurmitte und
        # Luecke, wuerde das gedaempfte Ziel sonst mitten AUF der Ente landen - die
        # Daempfung darf den Ausschlag verkleinern, aber nie in die Ente hineinzielen.
        target_damped = max(chosen[0], min(chosen[1], target_damped))

        self.debug_largest_gap = target_damped
        self.debug_close_to_white = target_damped > (left + right) / 2.0
        # Nur auf die halbe Bildbreite normieren (feste Konstante), NICHT durch lane_px
        # teilen: die perspektivische Stauchung ferner Seitenversaetze ist die gewollte
        # Daempfung (ferne Ente -> sanft lenken). lane_px dient nur BREITEN-Vergleichen
        # (Fit-Check/Fahrband).
        return self._norm_err(target_damped), pivot_needed

    def fnDrawFitCheck(self, img):
        """Debug-only: nutzt die STATISCHE Spur-Referenz (Bot-Breite pro Zeile) und
        zeichnet gegen die aktuellen Enten ein, WO der Bot durchpasst (gruen) und wo
        nicht (rot). Reine Anzeige, kein Steuer-Eingriff.

        Autor: Felix Faass
        """
        try:
            xl = lambda y: self._ref_aL * y + self._ref_bL
            xr = lambda y: self._ref_aR * y + self._ref_bR
            y0, y1 = int(self._ref_y0), int(self._ref_y1)

            # Referenz-Trapez in grau
            corners = np.array([[xl(y0), y0], [xr(y0), y0], [xr(y1), y1], [xl(y1), y1]], np.int32)
            cv2.polylines(img, [corners], True, (200, 200, 200), 1)

            # FAHRBAND (cyan): bot-breites Band um die SPURMITTE - das ist der Weg, auf dem
            # geprueft wird "Ente im Weg?". In der Kurve muss es sich mitverschieben; klebt
            # es auf 320, faehrt er Enten in der Kurve um.
            pc = self._path_center()
            band = np.array([[pc - self._lane_px(y0) / 2, y0], [pc + self._lane_px(y0) / 2, y0],
                             [pc + self._lane_px(y1) / 2, y1], [pc - self._lane_px(y1) / 2, y1]], np.int32)
            cv2.polylines(img, [band], True, (255, 255, 0), 2)

            # SPERRZONE (magenta): Ente mit y2 hier drin + im Fahrband -> PID-Manoever aus,
            # Pivot an. Entspricht dem pink markierten Bereich aus der Skizze.
            zt = int(self._blocked_zone_top)
            cv2.line(img, (0, zt), (img.shape[1], zt), (255, 0, 255), 2)
            cv2.putText(img, "SPERRZONE", (8, zt + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

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
        """Debug-only: zeigt die Gelb-/Weiss-Masken, mit denen fnDetectLane die Linien
        sucht (inkl. ausmaskierter Enten). Zeigt sofort, warum Y/W ggf. auf ihre
        Rand-Fallbacks fallen: sieht man in der Maske an der Suchzeile nichts, ist die
        Linie fuer den Code schlicht nicht da.
        Gelb = gelbe Maske, Weiss = weisse Maske, Grau = beide (Ueberlappung).
        Waagerecht: Suchzeile +-50 (get_x_for_driving mittelt darueber).
        Punkte: erkannte Position; gefuellt = echt gesehen, hohl = nur Fallback.

        Autor: Felix Faass
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
        # gelerntes Streckenbreiten-Verhaeltnis (in Spurbreiten) + daraus rekonstruierte
        # Fahrlinie -> zeigt, ob die Rekonstruktion bei Gelb-Fallback plausibel ist
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
                    if self.debug_blocked:
                        cv2.putText(img, "NO FIT - PIVOT", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # Debug: statische Spur-Referenz nutzen und einzeichnen, wo der Bot
                # durchpasst (gruen) / nicht (rot). Nur Anzeige, kein Steuer-Eingriff.
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