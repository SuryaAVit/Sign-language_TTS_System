"""
Sign Language Gesture Model Trainer
====================================
Compatible with popular Kaggle ASL datasets:

  1. ASL Alphabet Dataset (rohanp/asl-alphabet-test)
     folder structure: dataset/asl_alphabet_train/A/  B/  C/ ...

  2. ASL Dataset (grassknoted/asl-alphabet)
     folder structure: dataset/asl_alphabet_train/asl_alphabet_train/A/ B/ ...

  3. Sign Language MNIST
     CSV format — handled separately

Usage:
    python train_model.py --data_dir "dataset/asl_alphabet_train" --epochs 30

Output:
    model/gesture_model.h5
    model/label_map.json
"""

import os
import sys
import json
import argparse
import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.75
MODEL_OUTPUT   = "model/gesture_model.h5"
LABELMAP_OUTPUT = "model/label_map.json"
IMG_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Only keep A-Z labels (skip del/nothing/space from some Kaggle sets)
KEEP_LABELS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Map digit folder names (0-25) to letters for datasets that use numeric class folders
DIGIT_TO_LABEL = {str(i): chr(65 + i) for i in range(26)}


def collect_landmarks(data_dir, max_per_class=300):
    """
    Walk data_dir, detect hand landmarks from each image using MediaPipe.
    Returns X (landmarks), y (label strings), classes (sorted list).
    """
    import cv2
    import mediapipe as mp

    mp_h   = mp.solutions.hands
    hands  = mp_h.Hands(static_image_mode=True, max_num_hands=1,
                         min_detection_confidence=0.5)

    X, y = [], []
    skipped = 0

    # Auto-detect one level of nesting — supports both 'A'/'a' folders and '0'-'25' numeric folders
    classes_found = []
    for entry in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, entry)
        if not os.path.isdir(full):
            continue
        upper = entry.strip().upper()
        if upper in KEEP_LABELS:
            # Folder named A-Z or a-z
            classes_found.append((upper, full))
        elif entry.strip() in DIGIT_TO_LABEL:
            # Folder named 0-25 → map to A-Z
            classes_found.append((DIGIT_TO_LABEL[entry.strip()], full))

    if not classes_found:
        print(f"[Train] ERROR: No class folders found in '{data_dir}'")
        print(f"[Train] Expected folders named A, B, C ... Z inside that directory.")
        sys.exit(1)

    print(f"[Train] Found {len(classes_found)} classes: {[c[0] for c in classes_found]}")

    for label, folder in classes_found:
        count = 0
        images = [f for f in os.listdir(folder)
                  if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS]
        images = images[:max_per_class]  # cap per class

        for img_file in images:
            img_path = os.path.join(folder, img_file)
            frame    = cv2.imread(img_path)
            if frame is None:
                skipped += 1
                continue

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if not results.multi_hand_landmarks:
                skipped += 1
                continue

            raw = [(lm.x, lm.y, lm.z) for lm in results.multi_hand_landmarks[0].landmark]
            wrist_x, wrist_y, wrist_z = raw[0]
            norm = []
            for x, yy, z in raw:
                norm.extend([x - wrist_x, yy - wrist_y, z - wrist_z])

            X.append(norm)
            y.append(label)
            count += 1

        print(f"  {label}: {count} samples collected  ({len(images) - count} skipped)", flush=True)

    hands.close()
    print(f"\n[Train] Total: {len(X)} samples | {skipped} images skipped (no hand detected)")
    return np.array(X, dtype=np.float32), y


def build_model(n_classes):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(63,)),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64,  activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train(data_dir, epochs=30, max_per_class=300, output=MODEL_OUTPUT):
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing   import LabelEncoder

    print(f"\n{'='*55}")
    print(f"  Sign Language Model Training")
    print(f"  Data:   {data_dir}")
    print(f"  Epochs: {epochs}  |  Max per class: {max_per_class}")
    print(f"{'='*55}\n")

    # ── 1. Collect landmarks ──────────────────────────────────────────
    X, y_raw = collect_landmarks(data_dir, max_per_class)

    if len(X) < 50:
        print("[Train] ERROR: Not enough samples. Check your dataset path.")
        sys.exit(1)

    # ── 2. Encode labels ──────────────────────────────────────────────
    le     = LabelEncoder()
    y      = le.fit_transform(y_raw)
    classes = list(le.classes_)
    n_classes = len(classes)
    print(f"\n[Train] Classes ({n_classes}): {classes}")

    # ── 3. Train / test split ─────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)
    print(f"[Train] Train: {len(X_train)} | Test: {len(X_test)}")

    # ── 4. Build model ────────────────────────────────────────────────
    model = build_model(n_classes)
    model.summary()

    # ── 5. Callbacks ──────────────────────────────────────────────────
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    ]

    # ── 6. Train ──────────────────────────────────────────────────────
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # ── 7. Evaluate ───────────────────────────────────────────────────
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[Train] ✅ Test accuracy: {acc*100:.2f}%  |  Loss: {loss:.4f}")

    # ── 8. Save model ─────────────────────────────────────────────────
    os.makedirs("model", exist_ok=True)
    model.save(output)
    print(f"[Train] Model saved → {output}")

    # ── 9. Save label map ─────────────────────────────────────────────
    label_map = {str(i): cls for i, cls in enumerate(classes)}
    with open(LABELMAP_OUTPUT, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"[Train] Label map saved → {LABELMAP_OUTPUT}")
    print(f"\n[Train] Label map: {label_map}")
    print(f"\n{'='*55}")
    print(f"  Training complete!")
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  Now restart Flask: python app.py")
    print(f"{'='*55}\n")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASL gesture recognition model")
    parser.add_argument("--data_dir",      default="../asl_dataset",
                        help="Path to dataset folder containing A/, B/, C/ ... (or a/, b/, c/) subfolders")
    parser.add_argument("--epochs",        type=int, default=30,
                        help="Number of training epochs (default: 30)")
    parser.add_argument("--max_per_class", type=int, default=300,
                        help="Max images per class to use (default: 300)")
    parser.add_argument("--output",        default=MODEL_OUTPUT,
                        help="Output path for .h5 model file")
    args = parser.parse_args()

    train(
        data_dir      = args.data_dir,
        epochs        = args.epochs,
        max_per_class = args.max_per_class,
        output        = args.output
    )