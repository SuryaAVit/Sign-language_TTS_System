import os
import json
import numpy as np

CONFIDENCE_THRESHOLD = 0.75

# Default label map A-Z + SPACE + DELETE
DEFAULT_LABEL_MAP = {i: chr(65+i) for i in range(26)}
DEFAULT_LABEL_MAP[26] = "SPACE"
DEFAULT_LABEL_MAP[27] = "DELETE"

SKLEARN_MODEL_PATH = "model/gesture_model.pkl"
TF_MODEL_PATH      = "model/gesture_model.h5"
LABELMAP_PATH      = "model/label_map.json"


class GestureModel:
    def __init__(self):
        self.model      = None
        self.model_type = None
        self.demo_mode  = False
        self.label_map  = self._load_label_map()
        self._load()

    def reload_model(self):
        """Force reload the label map and model files."""
        print("[GestureModel] Reloading model...", flush=True)
        self.model      = None
        self.model_type = None
        self.demo_mode  = False
        self.label_map  = self._load_label_map()
        self._load()

    def _load_label_map(self):
        if os.path.exists(LABELMAP_PATH):
            try:
                with open(LABELMAP_PATH) as f:
                    raw = json.load(f)
                lm = {int(k): v for k, v in raw.items()}
                print(f"[GestureModel] Label map loaded: {lm}", flush=True)
                return lm
            except Exception as e:
                print(f"[GestureModel] Label map load error: {e}", flush=True)
        print("[GestureModel] No label_map.json — using default A-Z.", flush=True)
        return DEFAULT_LABEL_MAP

    def _load(self):
        # Try sklearn model first — no TensorFlow dependency
        if os.path.exists(SKLEARN_MODEL_PATH):
            try:
                import joblib
                self.model      = joblib.load(SKLEARN_MODEL_PATH)
                self.model_type = "sklearn"
                print(f"[GestureModel] Sklearn model loaded from '{SKLEARN_MODEL_PATH}'.", flush=True)
                return
            except Exception as e:
                print(f"[GestureModel] Sklearn load error: {e}", flush=True)

        # Try TF/Keras model
        if os.path.exists(TF_MODEL_PATH):
            try:
                import tensorflow as tf
                self.model      = tf.keras.models.load_model(TF_MODEL_PATH)
                self.model_type = "tensorflow"
                print(f"[GestureModel] TF model loaded from '{TF_MODEL_PATH}'.", flush=True)
                return
            except Exception as e:
                print(f"[GestureModel] TF load error: {e} — DEMO mode.", flush=True)

        print("[GestureModel] No model found — DEMO mode.", flush=True)
        self.demo_mode = True

    def predict(self, landmarks):
        if not landmarks or len(landmarks) != 63:
            return {"label": None, "confidence": 0.0, "accepted": False}

        if self.demo_mode:
            arr = np.array(landmarks)
            if arr.std() < 0.005:
                return {"label": None, "confidence": 0.0, "accepted": False}
            # Hold same label for 8 frames so the browser stable-frame check fires
            if not hasattr(self, '_demo_remaining') or self._demo_remaining <= 0:
                keys = list(self.label_map.keys())
                chosen_key = int(np.random.choice(keys)) if keys else 0
                self._demo_label     = self.label_map.get(chosen_key, "?")
                self._demo_remaining = 8
            else:
                self._demo_remaining -= 1
            confidence = round(float(np.random.uniform(0.82, 0.97)), 4)
            return {"label": self._demo_label, "confidence": confidence, "accepted": True}

        features = np.array(landmarks, dtype=np.float32).reshape(1, -1)

        if self.model_type == "sklearn":
            probs = self.model.predict_proba(features)[0]
        else:
            probs = self.model.predict(features, verbose=0)[0]

        class_idx  = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        label      = self.label_map.get(class_idx, "?")
        accepted   = confidence >= CONFIDENCE_THRESHOLD

        return {"label": label, "confidence": round(confidence, 4), "accepted": accepted}
