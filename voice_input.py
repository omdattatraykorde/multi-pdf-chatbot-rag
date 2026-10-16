import hashlib
import os
from groq import Groq


def transcribe_audio(audio_file, language=None):
    """Transcribe a Streamlit audio_input recording with Groq Whisper."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.startswith("your_"):
        raise RuntimeError("GROQ_API_KEY is required for voice input. Add it to the project's .env file.")

    client = Groq(api_key=api_key)
    audio_bytes = audio_file.getvalue()
    name = getattr(audio_file, "name", "voice.wav") or "voice.wav"

    kwargs = {
        "file": (name, audio_bytes),
        "model": "whisper-large-v3-turbo",
        "response_format": "json",
        "temperature": 0.0,
    }
    if language:
        kwargs["language"] = language

    result = client.audio.transcriptions.create(**kwargs)
    return (result.text or "").strip()


def audio_hash(audio_file):
    if not audio_file:
        return None
    return hashlib.sha256(audio_file.getvalue()).hexdigest()
