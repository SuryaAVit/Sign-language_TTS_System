# 🤟 Sign Language to Audio & Text Conversion System
**Surya A | Reg No: 511924622038 | MCA Final Year | PEC, Vaniyambadi**

---

## 📁 Project Structure

```
sign-language-app/
├── backend/
│   ├── app.py                ← Flask REST API (main server)
│   ├── hand_detector.py      ← MediaPipe hand detection
│   ├── gesture_model.py      ← TensorFlow/Keras model + training script
│   ├── tts_engine.py         ← gTTS / pyttsx3 Text-to-Speech
│   ├── doc_export.py         ← PDF & Word transcript export
│   ├── database.py           ← SQLite + SQLAlchemy models
│   ├── requirements.txt      ← Python dependencies
│   └── model/
│       └── gesture_model.h5  ← (place your trained model here)
├── frontend/
│   ├── src/
│   │   ├── App.jsx           ← Main React application
│   │   ├── api.js            ← API calls to Flask backend
│   │   ├── index.css         ← Global styles
│   │   └── components/
│   │       ├── WebcamCapture.jsx
│   │       ├── SubtitleDisplay.jsx
│   │       ├── ControlPanel.jsx
│   │       ├── AudioPlayer.jsx
│   │       └── SessionHistory.jsx
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
└── README.md
```

---

## ⚙️ SETUP INSTRUCTIONS (Windows)

### STEP 1 — Install Prerequisites

Make sure you have these installed:

| Tool | Download |
|------|----------|
| Python 3.10+ | https://python.org/downloads |
| Node.js 18+  | https://nodejs.org |
| Git (optional) | https://git-scm.com |

> ✅ During Python install, check **"Add Python to PATH"**

---

### STEP 2 — Setup Backend (Python / Flask)

Open **Command Prompt** or **PowerShell** and run:

```bash
# Go to backend folder
cd sign-language-app/backend

# Create a virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

> ⚠️ TensorFlow install may take a few minutes — that's normal.

---

### STEP 3 — Setup Frontend (React)

Open a **new** Command Prompt window:

```bash
# Go to frontend folder
cd sign-language-app/frontend

# Install Node packages
npm install
```

---

### STEP 4 — Run the Application

You need **2 terminal windows** open at the same time.

#### Terminal 1 — Start Flask Backend:
```bash
cd sign-language-app/backend
venv\Scripts\activate
python app.py
```
You should see:
```
╔══════════════════════════════════════════════════╗
║  Sign Language API  —  http://localhost:5000     ║
║  Model mode: DEMO                                ║
╚══════════════════════════════════════════════════╝
```

#### Terminal 2 — Start React Frontend:
```bash
cd sign-language-app/frontend
npm run dev
```
You should see:
```
  VITE ready in 500ms
  ➜  Local:   http://localhost:5173/
```

#### Open in Browser:
```
http://localhost:5173
```

---

## 🤖 MODEL — Training Your Own Gesture Recognition Model

### Option A: Use Demo Mode (No Model Needed)
The app works **without a model** in demo mode — it shows random gesture labels.
This is useful to test the full UI flow.

### Option B: Train Your Own Model

#### 1. Collect Dataset
Create a `dataset/` folder inside `backend/`:
```
backend/dataset/
    A/  → put 100+ images of ASL letter A here
    B/  → put 100+ images of ASL letter B here
    ...
    Z/  → put 100+ images of ASL letter Z here
```
> 💡 Use your webcam to photograph your own hands.
> Each class needs at least 50–100 images for decent accuracy.

#### 2. Train the Model
```bash
cd backend
venv\Scripts\activate
python gesture_model.py --train --data_dir dataset --epochs 50
```
The trained model will be saved to: `backend/model/gesture_model.h5`

#### 3. Restart Flask
```bash
python app.py
```
You should now see: `Model mode: LOADED`

---

## 🔌 API Endpoints Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | /api/health | Check backend status |
| POST | /api/session/start | Start a new session |
| POST | /api/session/end | End a session |
| POST | /api/gesture/recognize | Send frame → get gesture label |
| POST | /api/transcript/update | Save transcript to DB |
| GET  | /api/transcript/:id | Get session transcript |
| POST | /api/tts | Convert text to speech (MP3) |
| POST | /api/export/pdf | Export transcript as PDF |
| POST | /api/export/word | Export transcript as Word |
| GET  | /api/session/:id/history | Get session gesture history |

---

## 🎯 How to Use the App

1. Open `http://localhost:5173` in Chrome
2. Click **▶ Start** — allow webcam permission
3. Hold your hand in front of the camera and sign ASL letters
4. Recognized letters appear in the subtitle area
5. Use **␣ Space** to add a space between words
6. Click **🔊 Speak Text** to hear the text spoken aloud
7. Click **📄 PDF** or **📝 Word** to download the transcript
8. Click **■ Stop** to end the session

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `Backend Offline` badge | Make sure `python app.py` is running |
| Webcam not showing | Allow camera permission in browser |
| TTS not working | Check internet connection (uses Google TTS) |
| Low accuracy | Ensure good lighting, plain background |
| Model not loading | Check `model/gesture_model.h5` exists |
| `pip install` fails | Try `pip install -r requirements.txt --upgrade` |

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React.js 18, Vite, HTML5 Canvas |
| Backend | Python Flask, SQLAlchemy |
| Hand Detection | Google MediaPipe |
| ML Model | TensorFlow 2.x / Keras |
| Text-to-Speech | gTTS (Google), pyttsx3 (offline) |
| Database | SQLite (dev) / MySQL (prod) |
| PDF Export | ReportLab |
| Word Export | python-docx |

---

*Sign Language to Audio & Text Conversion System — MCA Final Year Project — PEC, Vaniyambadi — 2025–2026*
