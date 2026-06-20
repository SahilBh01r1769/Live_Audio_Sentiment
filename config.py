"""
Central configuration for Voice Sentiment Analyzer.

Every value can be overridden with an environment variable of the same
name, so you can tune the app for local/CPU vs. GPU/HF Spaces without
touching code, e.g.:

    export WHISPER_MODEL="Systran/faster-distil-whisper-large-v3"
    export MAX_AUDIO_MINUTES=5
"""
import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Models ---
# Must be a CTranslate2-converted Whisper checkpoint (faster-whisper does NOT
# accept raw HF Transformers Whisper repos). See README troubleshooting.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "Systran/faster-distil-whisper-large-v3")

SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest")

DEFAULT_EMOTION_MODEL = os.getenv("EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base")

SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "sshleifer/distilbart-cnn-12-6")

# --- Limits / defaults (all overridable via sidebar at runtime too) ---
MAX_AUDIO_MINUTES = int(os.getenv("MAX_AUDIO_MINUTES", "10"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
DEFAULT_CHUNK_LENGTH = int(os.getenv("DEFAULT_CHUNK_LENGTH", "30"))
DEFAULT_SENTIMENT_THRESHOLD = float(os.getenv("DEFAULT_SENTIMENT_THRESHOLD", "0.15"))
