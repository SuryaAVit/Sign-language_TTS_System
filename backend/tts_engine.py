import os
import uuid
import tempfile

AUDIO_DIR = "static/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)


def generate_audio(text: str, use_google: bool = True) -> str:
    """
    Convert text to speech and save as MP3.

    Args:
        text        : The text to speak.
        use_google  : True = gTTS (online), False = pyttsx3 (offline).

    Returns:
        Relative path to the generated MP3 file.
    """
    if not text or not text.strip():
        return None

    filename = f"{uuid.uuid4().hex}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)

    if use_google:
        try:
            from gtts import gTTS
            tts = gTTS(text=text.strip(), lang="en", slow=False)
            tts.save(filepath)
            print(f"[TTS] gTTS audio saved: {filepath}")
            return filepath
        except Exception as e:
            print(f"[TTS] gTTS failed ({e}), falling back to pyttsx3.")

    # Offline fallback — pyttsx3
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.setProperty("volume", 1.0)
        engine.save_to_file(text.strip(), filepath)
        engine.runAndWait()
        engine.stop()
        print(f"[TTS] pyttsx3 audio saved: {filepath}")
        return filepath
    except Exception as e:
        print(f"[TTS] pyttsx3 also failed: {e}")
        return None


def cleanup_audio(filepath: str):
    """Delete a generated audio file after it has been served."""
    try:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
    except Exception:
        pass
