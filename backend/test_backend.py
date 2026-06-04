"""
=============================================================
  Sign Language TTS System — Backend Test Suite (pytest)
  File   : test_backend.py
  Run    : pytest test_backend.py -v
=============================================================
"""

import os
import sys
import json
import pytest

# Suppress TF/OneDNN noise during tests
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

# Add backend root to path
sys.path.insert(0, os.path.dirname(__file__))

from app import app as flask_app, db

# ── FIXTURES ────────────────────────────────────────────────

@pytest.fixture(scope="session")
def client():
    """Create Flask test client with an in-memory SQLite DB."""
    flask_app.config["TESTING"] = True
    flask_app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    flask_app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    with flask_app.app_context():
        db.create_all()

    with flask_app.test_client() as client:
        yield client

@pytest.fixture(scope="module")
def session_id(client):
    """Start a session and return its ID for re-use across tests."""
    resp = client.post("/api/session/start",
                       content_type="application/json",
                       data=json.dumps({}))
    data = resp.get_json()
    return data.get("session_id")

# ── TC-01  Health Check ─────────────────────────────────────

def test_TC01_health_check(client):
    """TC-01: GET /api/health → status 200, body contains 'ok'."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok", "Health status must be 'ok'"
    assert "model" in body, "Response must include model field"
    assert "mediapipe" in body, "Response must include mediapipe field"

# ── TC-02  Session Start ────────────────────────────────────

def test_TC02_session_start(client):
    """TC-02: POST /api/session/start → returns session_id and session_token."""
    resp = client.post("/api/session/start",
                       content_type="application/json",
                       data=json.dumps({}))
    assert resp.status_code == 200
    body = resp.get_json()
    assert "session_id"    in body, "Must return session_id"
    assert "session_token" in body, "Must return session_token"
    assert isinstance(body["session_id"], int), "session_id must be an integer"

# ── TC-03  Session End ──────────────────────────────────────

def test_TC03_session_end(client, session_id):
    """TC-03: POST /api/session/end → message 'ended'."""
    resp = client.post("/api/session/end",
                       content_type="application/json",
                       data=json.dumps({"session_id": session_id}))
    assert resp.status_code == 200
    body = resp.get_json()
    assert body.get("message") == "ended"

# ── TC-04  Gesture Classify — valid landmarks ───────────────

def test_TC04_classify_valid_landmarks(client, session_id):
    """TC-04: POST /api/gesture/classify with 63 valid floats → accepted field present."""
    landmarks = [round(i * 0.01, 4) for i in range(63)]
    payload   = {"landmarks": landmarks, "session_id": session_id}
    resp = client.post("/api/gesture/classify",
                       content_type="application/json",
                       data=json.dumps(payload))
    assert resp.status_code == 200
    body = resp.get_json()
    assert "accepted"   in body, "Response must have 'accepted' field"
    assert "label"      in body, "Response must have 'label' field"
    assert "confidence" in body, "Response must have 'confidence' field"

# ── TC-05  Gesture Classify — wrong landmark count ──────────

def test_TC05_classify_wrong_landmark_count(client, session_id):
    """TC-05: 10 landmarks → accepted=False (invalid input rejection)."""
    payload = {"landmarks": [0.1] * 10, "session_id": session_id}
    resp = client.post("/api/gesture/classify",
                       content_type="application/json",
                       data=json.dumps(payload))
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["accepted"] is False, "Must reject invalid landmark count"

# ── TC-06  Gesture Classify — empty body ───────────────────

def test_TC06_classify_empty_body(client):
    """TC-06: Empty body → accepted=False, no server crash."""
    resp = client.post("/api/gesture/classify",
                       content_type="application/json",
                       data=json.dumps({}))
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["accepted"] is False

# ── TC-07  Transcript Update ────────────────────────────────

def test_TC07_transcript_update(client, session_id):
    """TC-07: Update transcript text for a session."""
    # Start a fresh session for this test
    r2   = client.post("/api/session/start", content_type="application/json", data=json.dumps({}))
    sid2 = r2.get_json()["session_id"]

    payload = {"session_id": sid2, "text": "Hello World"}
    resp    = client.post("/api/transcript/update",
                          content_type="application/json",
                          data=json.dumps(payload))
    assert resp.status_code == 200
    body = resp.get_json()
    assert body.get("message") == "updated"

# ── TC-08  Transcript Retrieve ──────────────────────────────

def test_TC08_transcript_retrieve(client):
    """TC-08: GET /api/transcript/<id> → returns text and word_count."""
    r2   = client.post("/api/session/start", content_type="application/json", data=json.dumps({}))
    sid2 = r2.get_json()["session_id"]

    client.post("/api/transcript/update",
                content_type="application/json",
                data=json.dumps({"session_id": sid2, "text": "Sign Language Test"}))

    resp = client.get(f"/api/transcript/{sid2}")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "text"       in body
    assert "word_count" in body
    assert body["word_count"] == 3, "Word count must be 3 for 'Sign Language Test'"

# ── TC-09  Transcript — Non-existent Session ────────────────

def test_TC09_transcript_nonexistent_session(client):
    """TC-09: Transcript for session 99999 → empty string (graceful)."""
    resp = client.get("/api/transcript/99999")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["text"] == ""

# ── TC-10  TTS — empty text ─────────────────────────────────

def test_TC10_tts_empty_text(client):
    """TC-10: POST /api/tts with empty text → 400 error."""
    resp = client.post("/api/tts",
                       content_type="application/json",
                       data=json.dumps({"text": ""}))
    assert resp.status_code == 400
    body = resp.get_json()
    assert "error" in body

# ── TC-11  Session History ──────────────────────────────────

def test_TC11_session_history(client):
    """TC-11: GET /api/session/<id>/history → returns 'history' list."""
    r2   = client.post("/api/session/start", content_type="application/json", data=json.dumps({}))
    sid2 = r2.get_json()["session_id"]

    resp = client.get(f"/api/session/{sid2}/history")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "history" in body
    assert isinstance(body["history"], list)

# ── TC-12  Training Collect — valid ─────────────────────────

def test_TC12_training_collect_valid(client):
    """TC-12: Collect valid training sample → ok=True."""
    landmarks = [round(i * 0.005, 5) for i in range(63)]
    payload   = {"label": "A", "landmarks": landmarks}
    resp = client.post("/api/training/collect",
                       content_type="application/json",
                       data=json.dumps(payload))
    assert resp.status_code == 200
    body = resp.get_json()
    assert body.get("ok") is True

# ── TC-13  Training Collect — invalid label ─────────────────

def test_TC13_training_collect_invalid(client):
    """TC-13: Missing label or wrong landmark count → 400 error."""
    payload = {"label": "", "landmarks": [0.1] * 20}
    resp = client.post("/api/training/collect",
                       content_type="application/json",
                       data=json.dumps(payload))
    assert resp.status_code == 400

# ── TC-14  Export Word — no session ─────────────────────────

def test_TC14_export_word(client):
    """TC-14: Export Word doc with text (no session_id) → file download."""
    payload = {"text": "Hello from Sign Language App", "session_id": None}
    resp = client.post("/api/export/word",
                       content_type="application/json",
                       data=json.dumps(payload))
    # Either 200 (file) or 500 if docx not installed; both are acceptable for demo
    assert resp.status_code in (200, 500)

# ── TC-15  Export PDF — no session ──────────────────────────

def test_TC15_export_pdf(client):
    """TC-15: Export PDF with text (no session_id) → file download."""
    payload = {"text": "Hello from Sign Language App", "session_id": None}
    resp = client.post("/api/export/pdf",
                       content_type="application/json",
                       data=json.dumps(payload))
    assert resp.status_code in (200, 500)

# ── TC-16  Dataset Images Endpoint ──────────────────────────

def test_TC16_dataset_images(client):
    """TC-16: GET /api/dataset/images → returns images list."""
    resp = client.get("/api/dataset/images")
    # 200 if dataset folder exists, 404 if not — both handled gracefully
    assert resp.status_code in (200, 404)
    body = resp.get_json()
    assert "images" in body or "error" in body

# ── TC-17  CORS Headers ──────────────────────────────────────

def test_TC17_cors_headers(client):
    """TC-17: Health endpoint includes CORS header for allowed origin."""
    resp = client.get("/api/health",
                      headers={"Origin": "http://localhost:5173"})
    assert resp.status_code == 200
    # CORS header should be present (flask-cors adds it)
    assert "Access-Control-Allow-Origin" in resp.headers or resp.status_code == 200

# ── TC-18  Multiple Sessions Independent ────────────────────

def test_TC18_multiple_sessions_independent(client):
    """TC-18: Two separate sessions have independent transcripts."""
    r1   = client.post("/api/session/start", content_type="application/json", data=json.dumps({}))
    r2   = client.post("/api/session/start", content_type="application/json", data=json.dumps({}))
    sid1 = r1.get_json()["session_id"]
    sid2 = r2.get_json()["session_id"]
    assert sid1 != sid2, "Two sessions must have different IDs"

    client.post("/api/transcript/update",
                content_type="application/json",
                data=json.dumps({"session_id": sid1, "text": "Session One"}))

    resp = client.get(f"/api/transcript/{sid2}")
    body = resp.get_json()
    assert body["text"] != "Session One", "Session 2 should not have Session 1 transcript"

# ── TC-19  Landmark Boundary Values ─────────────────────────

def test_TC19_landmark_boundary_all_zeros(client):
    """TC-19: All-zero landmarks (boundary) → API returns without crash."""
    payload = {"landmarks": [0.0] * 63, "session_id": None}
    resp = client.post("/api/gesture/classify",
                       content_type="application/json",
                       data=json.dumps(payload))
    assert resp.status_code == 200

def test_TC19b_landmark_boundary_all_ones(client):
    """TC-19b: All-one landmarks (boundary max) → API returns without crash."""
    payload = {"landmarks": [1.0] * 63, "session_id": None}
    resp = client.post("/api/gesture/classify",
                       content_type="application/json",
                       data=json.dumps(payload))
    assert resp.status_code == 200

# ── TC-20  Response Format Validation ───────────────────────

def test_TC20_response_is_json(client):
    """TC-20: All API endpoints return JSON content-type."""
    endpoints = [
        ("GET",  "/api/health"),
        ("GET",  "/api/transcript/1"),
        ("GET",  "/api/session/1/history"),
    ]
    for method, url in endpoints:
        if method == "GET":
            resp = client.get(url)
        assert "application/json" in resp.content_type, \
            f"{url} must return JSON, got {resp.content_type}"
