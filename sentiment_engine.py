from transformers import pipeline
import torch
import config

EMOTION_MODEL_OPTIONS = {
    "DistilRoBERTa (6 emotions, fast)": "j-hartmann/emotion-english-distilroberta-base",
    "RoBERTa GoEmotions (28 emotions, detailed)": "SamLowe/roberta-base-go_emotions",
}


class SentimentEngine:
    def __init__(self, emotion_model=None):
        emotion_model = emotion_model or config.DEFAULT_EMOTION_MODEL
        device = 0 if torch.cuda.is_available() else -1
        self.sentiment_pipe = pipeline(
            "sentiment-analysis",
            model=config.SENTIMENT_MODEL,
            device=device,
        )
        self.emotion_pipe = pipeline(
            "text-classification",
            model=emotion_model,
            top_k=None,
            device=device,
        )

    def analyze(self, text):
        if not text or len(text.strip()) < 3:
            return {"label": "neutral", "score": 0.0, "confidence": 0.5, "emotions": {}}

        # Sentiment
        sentiment_result = self.sentiment_pipe(text)[0]
        label = sentiment_result['label'].lower()
        score = sentiment_result['score']

        if label == 'negative':
            score = -score
        elif label == 'neutral':
            score = 0.0
        # positive: keep score as-is

        # Emotions
        emotions = self.emotion_pipe(text)[0]
        emotion_dict = {e['label']: round(e['score'], 3) for e in emotions}

        return {
            "label": label,
            "score": round(score, 3),
            "confidence": round(sentiment_result['score'], 3),
            "emotions": emotion_dict,
        }
