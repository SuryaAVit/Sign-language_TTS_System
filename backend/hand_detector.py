import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import cv2
import numpy as np
import base64

class HandDetector:
    def __init__(self, max_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        self._max_hands  = max_hands
        self._det_conf   = min_detection_confidence
        self._track_conf = min_tracking_confidence
        self._initialized = False
        self.hands        = None
        self.mp_hands     = None
        self.mp_drawing   = None
        print("[HandDetector] Ready (lazy init).", flush=True)

    def _ensure_init(self):
        if self._initialized:
            return self.hands is not None
        self._initialized = True
        try:
            import mediapipe as mp
            self.mp_hands   = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self._max_hands,
                min_detection_confidence=self._det_conf,
                min_tracking_confidence=self._track_conf
            )
            print("[HandDetector] MediaPipe OK.", flush=True)
            return True
        except Exception as e:
            print(f"[HandDetector] MediaPipe failed: {e}", flush=True)
            self.hands = None
            return False

    def decode_frame(self, base64_str):
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        img_bytes = base64.b64decode(base64_str)
        np_arr    = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def detect(self, base64_str):
        ready = self._ensure_init()

        # MediaPipe unavailable — real frame decode still works for camera display
        # but return no landmark data so recognition doesn't fire randomly
        if not ready:
            return {"detected": False, "landmarks": None, "annotated": None}

        try:
            frame = self.decode_frame(base64_str)
            if frame is None:
                return {"detected": False, "landmarks": None, "annotated": None}

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if not results.multi_hand_landmarks:
                return {"detected": False, "landmarks": None, "annotated": None}

            hand_landmarks = results.multi_hand_landmarks[0]
            raw = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

            wrist_x, wrist_y, wrist_z = raw[0]
            normalized = []
            for x, y, z in raw:
                normalized.extend([x - wrist_x, y - wrist_y, z - wrist_z])

            annotated_b64 = None
            try:
                af = frame.copy()
                self.mp_drawing.draw_landmarks(
                    af, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=(80,80,80), thickness=2)
                )
                _, buf        = cv2.imencode(".jpg", af)
                annotated_b64 = base64.b64encode(buf).decode("utf-8")
            except Exception:
                pass

            return {"detected": True, "landmarks": normalized, "annotated": annotated_b64}

        except Exception as e:
            print(f"[HandDetector] error: {e}", flush=True)
            return {"detected": False, "landmarks": None, "annotated": None}

    def close(self):
        if self.hands:
            try: self.hands.close()
            except Exception: pass