"""
Train gesture recognition model from browser-collected landmark data.

Usage (from backend/ folder):
    python train_from_csv.py

Input:  training_data.csv  (created by the app's "Collect Training Data" feature)
Output: model/gesture_model.pkl   (StandardScaler + SVM pipeline)
        model/label_map.json

No TensorFlow or MediaPipe required - uses scikit-learn only.

Model: a Pipeline of StandardScaler -> SVC(RBF, probability=True).
Tuned hyper-parameters (C=50, gamma=0.05) gave the best 5-fold cross-validated
accuracy (~97%) on the landmark dataset, clearly beating the previous raw-feature
MLP (~94%). Because the scaler is baked into the saved pipeline, the Flask backend
can keep feeding it raw 63-value wrist-relative landmarks with no code changes.
"""

import os
import csv
import json
import numpy as np

DATA_FILE       = "training_data.csv"
MODEL_OUTPUT    = "model/gesture_model.pkl"
LABELMAP_OUTPUT = "model/label_map.json"

# Tuned SVM hyper-parameters (selected by grid search, see project report)
SVC_C     = 50
SVC_GAMMA = 0.05


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
    return np.array(X, dtype=np.float64), y


def train():
    from sklearn.svm             import SVC
    from sklearn.preprocessing   import StandardScaler, LabelEncoder
    from sklearn.pipeline        import make_pipeline
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.metrics         import classification_report
    import joblib

    print("\n" + "="*55)
    print("  Sign Language Model Training (StandardScaler + SVM)")
    print("="*55 + "\n")

    X, y_raw = load_csv()
    if X is None:
        return

    le      = LabelEncoder()
    y       = le.fit_transform(y_raw)
    classes = list(le.classes_)
    print(f"\n[Train] Classes ({len(classes)}): {classes}")

    clf = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=SVC_C, gamma=SVC_GAMMA,
            probability=True, random_state=42),
    )

    # ---- Robust 5-fold cross-validated accuracy ----
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cvs = cross_val_score(clf, X, y, cv=cv, n_jobs=-1)
    print(f"\n[Train] 5-fold CV accuracy: {cvs.mean()*100:.2f}%  (+/- {cvs.std()*100:.2f})")

    # ---- Hold-out report ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"[Train] Hold-out (20%) test accuracy: {acc*100:.2f}%\n")
    print(classification_report(
        y_test, clf.predict(X_test),
        target_names=[str(c) for c in classes], zero_division=0))

    # ---- Final fit on ALL data, then save ----
    clf.fit(X, y)
    os.makedirs("model", exist_ok=True)
    joblib.dump(clf, MODEL_OUTPUT)
    print(f"[Train] Model saved -> {MODEL_OUTPUT}")

    label_map = {str(i): cls for i, cls in enumerate(classes)}
    with open(LABELMAP_OUTPUT, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"[Train] Label map saved -> {LABELMAP_OUTPUT}")

    print("\n" + "="*55)
    print(f"  Done!  CV accuracy: {cvs.mean()*100:.2f}%")
    print("  Restart Flask:   python app.py")
    print("="*55 + "\n")


if __name__ == "__main__":
    train()
