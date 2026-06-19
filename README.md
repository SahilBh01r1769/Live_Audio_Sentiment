# Voice Sentiment Analyzer (File Upload Version)

A beautiful Streamlit app that transcribes audio files and performs detailed sentiment + emotion analysis.

## Features
- Upload audio files (mp3, wav, m4a, etc.)
- High-quality transcription using Distil-Whisper
- Sentiment analysis (positive/negative/neutral) with confidence
- Multi-emotion breakdown
- Interactive sentiment timeline chart
- Download transcript
- Optimized for Hugging Face Spaces and Streamlit Cloud

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment
- **Hugging Face Spaces**: Just connect this repo (very easy)
- **Streamlit Community Cloud**: Connect GitHub repo

First run will download large models (~2GB).

Perfect for analyzing meetings, interviews, podcasts, customer calls, etc.
