"""
Sign Language Flask API — Simplified Architecture
==================================================
MediaPipe now runs in the browser (CDN).
Flask only handles:
  - ML gesture classification (landmarks → label)
  - TTS audio generation
  - PDF / Word export
  - SQLite database (sessions, translations, transcripts)
"""
import os, traceback, uuid
from datetime import datetime

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from database import db, Session, Translation, Transcript, GestureLabel, Document, seed_gesture_labels
from gesture_model import GestureModel
from tts_engine import generate_audio
from doc_export import export_pdf, export_word

app = Flask(__name__, static_folder="static")
CORS(app, origins=["http://localhost:5173", "http://localhost:3000"])
app.config["SQLALCHEMY_DATABASE_URI"]        = "sqlite:///sign_language.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"]             = 5 * 1024 * 1024
db.init_app(app)

with app.app_context():
    db.create_all()
    seed_gesture_labels(app)

# Only load ML model — no MediaPipe, no OpenCV needed
model = GestureModel()

print("\n" + "="*55, flush=True)
print("  Sign Language API  ->  http://localhost:5000", flush=True)
print(f"  Model: {'DEMO MODE (train to enable real recognition)' if model.demo_mode else 'LOADED OK'}", flush=True)
print("  MediaPipe: runs in BROWSER (no Python needed)", flush=True)
print("="*55 + "\n", flush=True)

# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model":  "demo" if model.demo_mode else "loaded",
        "mediapipe": "browser"
    })

@app.route("/api/session/start", methods=["POST"])
def start_session():
    try:
        token = str(uuid.uuid4())
        s = Session(session_token=token)
        db.session.add(s)
        db.session.add(Transcript(session=s, full_text="", word_count=0))
        db.session.commit()
        return jsonify({"session_id": s.session_id, "session_token": token})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/session/end", methods=["POST"])
def end_session():
    try:
        data = request.get_json()
        s    = db.session.get(Session, data.get("session_id"))
        if s:
            s.ended_at = datetime.utcnow()
            db.session.commit()
        return jsonify({"message": "ended"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/gesture/classify", methods=["POST"])
def classify_gesture():
    """
    Receive pre-computed landmarks from browser MediaPipe.
    Only run the ML model here — no frame decoding needed.

    Body: { landmarks: [63 floats], session_id: int }
    """
    try:
        data       = request.get_json(force=True)
        landmarks  = data.get("landmarks", [])
        session_id = data.get("session_id")

        if not landmarks or len(landmarks) != 63:
            return jsonify({"accepted": False, "label": None, "confidence": 0.0})

        result = model.predict(landmarks)

        # Save to DB if accepted
        if result["accepted"] and result["label"] and session_id:
            s   = db.session.get(Session, session_id)
            lbl = GestureLabel.query.filter_by(label_name=result["label"]).first()
            if s and lbl:
                db.session.add(Translation(
                    session_id=session_id,
                    label_id=lbl.label_id,
                    confidence_score=result["confidence"]
                ))
                s.total_gestures = (s.total_gestures or 0) + 1
                db.session.commit()

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"accepted": False, "error": str(e)}), 200

@app.route("/api/transcript/update", methods=["POST"])
def update_transcript():
    try:
        data = request.get_json()
        t    = Transcript.query.filter_by(session_id=data.get("session_id")).first()
        if t:
            t.full_text  = data.get("text", "")
            t.word_count = len(t.full_text.split()) if t.full_text.strip() else 0
            t.updated_at = datetime.utcnow()
            db.session.commit()
        return jsonify({"message": "updated"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/transcript/<int:session_id>")
def get_transcript(session_id):
    t = Transcript.query.filter_by(session_id=session_id).first()
    return jsonify({"text": t.full_text if t else "", "word_count": t.word_count if t else 0})

@app.route("/api/tts", methods=["POST"])
def tts():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        if not text: return jsonify({"error": "No text"}), 400
        fp = generate_audio(text, use_google=True)
        if not fp or not os.path.exists(fp):
            return jsonify({"error": "TTS failed"}), 500
        return send_file(fp, mimetype="audio/mpeg", download_name="speech.mp3")
    except Exception as e:
        traceback.print_exc(); return jsonify({"error": str(e)}), 500

@app.route("/api/export/pdf", methods=["POST"])
def export_pdf_route():
    try:
        data = request.get_json()
        s    = db.session.get(Session, data.get("session_id"))
        fp   = export_pdf(data.get("text",""), s.session_token if s else "")
        if not fp or not os.path.exists(fp): return jsonify({"error":"Failed"}), 500
        sid = data.get("session_id")
        if sid:
            db.session.add(Document(session_id=sid, format="PDF", filename=os.path.basename(fp)))
            db.session.commit()
        return send_file(fp, mimetype="application/pdf", as_attachment=True, download_name="transcript.pdf")
    except Exception as e:
        traceback.print_exc(); return jsonify({"error": str(e)}), 500

@app.route("/api/export/word", methods=["POST"])
def export_word_route():
    try:
        data = request.get_json()
        s    = db.session.get(Session, data.get("session_id"))
        fp   = export_word(data.get("text",""), s.session_token if s else "")
        if not fp or not os.path.exists(fp): return jsonify({"error":"Failed"}), 500
        sid = data.get("session_id")
        if sid:
            db.session.add(Document(session_id=sid, format="DOCX", filename=os.path.basename(fp)))
            db.session.commit()
        mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        return send_file(fp, mimetype=mime, as_attachment=True, download_name="transcript.docx")
    except Exception as e:
        traceback.print_exc(); return jsonify({"error": str(e)}), 500

@app.route("/api/training/collect", methods=["POST"])
def collect_training_sample():
    """Save a single {label, landmarks} pair to training_data.csv."""
    try:
        data      = request.get_json(force=True)
        label     = data.get("label", "").strip().upper()
        landmarks = data.get("landmarks", [])
        if not label or len(landmarks) != 63:
            return jsonify({"error": "Invalid data"}), 400
        import csv
        with open("training_data.csv", "a", newline="") as f:
            csv.writer(f).writerow([label] + [round(float(x), 6) for x in landmarks])
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/training/reset", methods=["POST"])
def reset_training_data():
    try:
        if os.path.exists("training_data.csv"):
            os.remove("training_data.csv")
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/session/<int:session_id>/history")
def session_history(session_id):
    rows = Translation.query.filter_by(session_id=session_id).order_by(Translation.recognized_at).all()
    history = []
    for t in rows:
        lbl = db.session.get(GestureLabel, t.label_id)
        history.append({
            "label":      lbl.label_name if lbl else "?",
            "confidence": t.confidence_score,
            "time":       t.recognized_at.strftime("%H:%M:%S")
        })
    return jsonify({"history": history})

# ── DATASET & TRAINING WORKAROUND ROUTES ──────────────────────────────────────

@app.route("/asl_dataset/<path:filename>")
def serve_dataset_image(filename):
    """Serve images statically from the asl_dataset directory."""
    dataset_dir = os.path.abspath(os.path.join(app.root_path, "../asl_dataset"))
    return send_from_directory(dataset_dir, filename)

@app.route("/api/dataset/images")
def get_dataset_images():
    """Scan the asl_dataset directory and return a structured list of all images."""
    try:
        dataset_dir = os.path.abspath(os.path.join(app.root_path, "../asl_dataset"))
        if not os.path.exists(dataset_dir):
            return jsonify({"error": f"Dataset directory not found at {dataset_dir}"}), 404

        images = []
        # Walk directory (categories 0-9 and a-z)
        for entry in sorted(os.listdir(dataset_dir)):
            full_path = os.path.join(dataset_dir, entry)
            if os.path.isdir(full_path):
                label = entry.strip().upper()
                if label == "ASL_DATASET":
                    continue
                for file in sorted(os.listdir(full_path)):
                    if os.path.splitext(file)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                        images.append({
                            "url": f"/asl_dataset/{entry}/{file}",
                            "label": label,
                            "filename": file
                        })
        return jsonify({"images": images, "count": len(images)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/training/train", methods=["POST"])
def trigger_training():
    """Train the scikit-learn MLP model from collected landmarks and hot-reload it."""
    try:
        csv_path = "training_data.csv"
        if not os.path.exists(csv_path):
            return jsonify({"error": "No training data found (training_data.csv missing). Please collect/process landmarks first."}), 400

        # Import and trigger the training function
        from train_from_csv import train as train_sklearn
        train_sklearn()

        # Hot-reload the trained model in GestureModel
        model.reload_model()

        return jsonify({
            "success": True,
            "message": "Model trained and hot-reloaded successfully!",
            "model_status": "demo" if model.demo_mode else "loaded"
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    os.makedirs("static/audio",   exist_ok=True)
    os.makedirs("static/exports", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)