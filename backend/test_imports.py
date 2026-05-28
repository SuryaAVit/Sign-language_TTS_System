import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import sys
print(f"Python: {sys.version}", flush=True)

print("1. Testing cv2...", flush=True)
import cv2
print(f"   cv2 OK: {cv2.__version__}", flush=True)

print("2. Testing numpy...", flush=True)
import numpy as np
print(f"   numpy OK: {np.__version__}", flush=True)

print("3. Testing mediapipe...", flush=True)
try:
    import mediapipe as mp
    print(f"   mediapipe OK: {mp.__version__}", flush=True)
    h = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)
    print("   Hands() init OK", flush=True)
    h.close()
except Exception as e:
    print(f"   mediapipe FAILED: {e}", flush=True)
    import traceback; traceback.print_exc()

print("4. Testing flask...", flush=True)
from flask import Flask
print(f"   flask OK", flush=True)

print("5. Testing flask_sqlalchemy...", flush=True)
from flask_sqlalchemy import SQLAlchemy
print("   flask_sqlalchemy OK", flush=True)

print("6. Testing gtts...", flush=True)
from gtts import gTTS
print("   gtts OK", flush=True)

print("7. Testing pyttsx3...", flush=True)
import pyttsx3
print("   pyttsx3 OK", flush=True)

print("8. Testing reportlab...", flush=True)
from reportlab.lib.pagesizes import A4
print("   reportlab OK", flush=True)

print("9. Testing python-docx...", flush=True)
from docx import Document
print("   python-docx OK", flush=True)

print("\nAll tests done!", flush=True)
