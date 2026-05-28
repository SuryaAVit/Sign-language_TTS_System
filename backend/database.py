from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = "users"
    user_id    = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username   = db.Column(db.String(50), unique=True, nullable=False)
    email      = db.Column(db.String(100), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active  = db.Column(db.Boolean, default=True)
    sessions   = db.relationship("Session", backref="user", lazy=True)

class GestureLabel(db.Model):
    __tablename__ = "gesture_labels"
    label_id    = db.Column(db.Integer, primary_key=True, autoincrement=True)
    class_index = db.Column(db.Integer, unique=True, nullable=False)
    label_name  = db.Column(db.String(50), unique=True, nullable=False)
    label_type  = db.Column(db.String(20), default="alphabet")

class Session(db.Model):
    __tablename__ = "sessions"
    session_id     = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id        = db.Column(db.Integer, db.ForeignKey("users.user_id"), nullable=True)
    session_token  = db.Column(db.String(100), unique=True, nullable=False)
    started_at     = db.Column(db.DateTime, default=datetime.utcnow)
    ended_at       = db.Column(db.DateTime, nullable=True)
    total_gestures = db.Column(db.Integer, default=0)
    translations   = db.relationship("Translation", backref="session", lazy=True)
    transcript     = db.relationship("Transcript", backref="session", uselist=False, lazy=True)

class Translation(db.Model):
    __tablename__ = "translations"
    translation_id   = db.Column(db.Integer, primary_key=True, autoincrement=True)
    session_id       = db.Column(db.Integer, db.ForeignKey("sessions.session_id"), nullable=False)
    label_id         = db.Column(db.Integer, db.ForeignKey("gesture_labels.label_id"), nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)
    recognized_at    = db.Column(db.DateTime, default=datetime.utcnow)

class Transcript(db.Model):
    __tablename__ = "transcripts"
    transcript_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    session_id    = db.Column(db.Integer, db.ForeignKey("sessions.session_id"), unique=True, nullable=False)
    full_text     = db.Column(db.Text, default="")
    word_count    = db.Column(db.Integer, default=0)
    updated_at    = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Document(db.Model):
    __tablename__ = "documents"
    document_id   = db.Column(db.Integer, primary_key=True, autoincrement=True)
    session_id    = db.Column(db.Integer, db.ForeignKey("sessions.session_id"), nullable=False)
    format        = db.Column(db.String(10), nullable=False)
    filename      = db.Column(db.String(200), nullable=False)
    downloaded_at = db.Column(db.DateTime, default=datetime.utcnow)

class SystemLog(db.Model):
    __tablename__ = "system_logs"
    log_id      = db.Column(db.Integer, primary_key=True, autoincrement=True)
    session_id  = db.Column(db.Integer, db.ForeignKey("sessions.session_id"), nullable=True)
    module_name = db.Column(db.String(50), nullable=False)
    log_level   = db.Column(db.String(10), nullable=False)
    message     = db.Column(db.Text, nullable=False)
    logged_at   = db.Column(db.DateTime, default=datetime.utcnow)

def seed_gesture_labels(app):
    """Seed the gesture_labels table with ASL A-Z, 0-9, and SPACE/DEL if they don't exist."""
    with app.app_context():
        # 1. Letters A-Z
        for i in range(26):
            name = chr(65 + i)
            if not GestureLabel.query.filter_by(label_name=name).first():
                db.session.add(GestureLabel(class_index=i, label_name=name, label_type="alphabet"))
        # 2. Control actions
        if not GestureLabel.query.filter_by(label_name="SPACE").first():
            db.session.add(GestureLabel(class_index=26, label_name="SPACE", label_type="control"))
        if not GestureLabel.query.filter_by(label_name="DELETE").first():
            db.session.add(GestureLabel(class_index=27, label_name="DELETE", label_type="control"))
        # 3. Digits 0-9
        for i in range(10):
            name = str(i)
            if not GestureLabel.query.filter_by(label_name=name).first():
                db.session.add(GestureLabel(class_index=28 + i, label_name=name, label_type="number"))
        db.session.commit()
        print("[DB] Gesture labels checked and seeded successfully.")
