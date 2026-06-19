import os

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Will be imported after torch

MODEL_NAME = "distil-whisper/distil-large-v3"  # Fast and accurate
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

MAX_AUDIO_LENGTH = 600  # 10 minutes max for HF Spaces
CHUNK_LENGTH = 30  # seconds for processing long files
OVERLAP = 2  # seconds overlap between chunks
