"""
Train gesture recognition model from browser-collected landmark data.

Usage (from backend/ folder):
    python train_from_csv.py

Input:  training_data.csv  (created by the app's "Collect Training Data" feature)
Output: model/gesture_model.pkl
        model/label_map.json

No TensorFlow or MediaPipe required — uses scikit-learn only.
"""

import os
import csv
import json
import numpy as np

DATA_FILE       = "training_data.csv"
MODEL_OUTPUT    = "model/gesture_model.pkl"
LABELMAP_OUTPUT = "model/label_map.json"


def load_csv():
    if not os.path.exists(DATA_FILE):
        print(f"[Train] ERROR: '{DATA_FILE}' not found.")
        print("[Train] Use the 'Collect Training Data' button in the app first.")
        return None, None

    X, y = [], []
    with open(DATA_FILE, newline="") as f:
        for row in csv.reader(f):
            if len(row) != 64:
                continue
            y.append(row[0].strip().upper())
            X.append([float(v) for v in row[1:]])

    if not X:
        print(f"[Train] ERROR: '{DATA_FILE}' is empty or malformed.")
        return None, None

    from collections import Counter
    counts = Counter(y)
    print(f"[Train] Loaded {len(X)} samples across {len(counts)} classes:")
    for lbl in sorted(counts):
        print(f"  {lbl}: {counts[lbl]} samples")
    return np.array(X, dtype=np.float32), y


def train():
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing   import LabelEncoder
    import joblib

    print("\n" + "="*55)
    print("  Sign Language Model Training (sklearn MLP)")
    print("="*55 + "\n")

    X, y_raw = load_csv()
    if X is None:
        return

    le      = LabelEncoder()
    y       = le.fit_transform(y_raw)
    classes = list(le.classes_)
    print(f"\n[Train] Classes ({len(classes)}): {classes}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"[Train] Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    clf = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64),
        activation="relu",
        solver="adam",
        learning_rate="adaptive",
        max_iter=500,
        random_state=42,
        verbose=True,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
    )
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(f"\n[Train] OK Test accuracy: {acc*100:.1f}%")

    os.makedirs("model", exist_ok=True)
    joblib.dump(clf, MODEL_OUTPUT)
    print(f"[Train] Model saved -> {MODEL_OUTPUT}")

    label_map = {str(i): cls for i, cls in enumerate(classes)}
    with open(LABELMAP_OUTPUT, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"[Train] Label map saved -> {LABELMAP_OUTPUT}")

    print("\n" + "="*55)
    print(f"  Done!  Accuracy: {acc*100:.1f}%")
    print("  Restart Flask:   python app.py")
    print("="*55 + "\n")


if __name__ == "__main__":
    train()
