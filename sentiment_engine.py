from transformers import pipeline
import torch

class SentimentEngine:
    def __init__(self):
        self.sentiment_pipe = pipeline("sentiment-analysis", 
                                     model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                                     device=0 if torch.cuda.is_available() else -1)
        self.emotion_pipe = pipeline("text-classification", 
                                   model="j-hartmann/emotion-english-distilroberta-base",
                                   top_k=None,
                                   device=0 if torch.cuda.is_available() else -1)
    
    def analyze(self, text):
        if not text or len(text.strip()) < 3:
            return {"label": "neutral", "score": 0.5, "emotions": {}}
        
        # Sentiment
        sentiment_result = self.sentiment_pipe(text)[0]
        label = sentiment_result['label'].lower()
        score = sentiment_result['score']
        
        if label == 'negative':
            score = -score
        elif label == 'positive':
            score = score
        else:
            score = 0.0
        
        # Emotions
        emotions = self.emotion_pipe(text)[0]
        emotion_dict = {e['label']: round(e['score'], 3) for e in emotions}
        
        return {
            "label": label,
            "score": round(score, 3),
            "confidence": round(sentiment_result['score'], 3),
            "emotions": emotion_dict
        }
