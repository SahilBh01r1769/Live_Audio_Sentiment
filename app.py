import streamlit as st
import torch
import os
import pandas as pd
from datetime import datetime
import tempfile

from asr_engine import ASREngine
from sentiment_engine import SentimentEngine
from utils import create_sentiment_timeline, format_timestamp

st.set_page_config(page_title="LiveVoiceSentiment - File Version", layout="wide")
st.title("🎙️ Voice Sentiment Analyzer")
st.markdown("Upload an audio file and get real-time transcription + sentiment analysis with beautiful visuals.")

# Initialize engines
@st.cache_resource
def load_models():
    asr = ASREngine()
    sentiment = SentimentEngine()
    return asr, sentiment

asr_engine, sentiment_engine = load_models()

# Sidebar
st.sidebar.header("Settings")
chunk_length = st.sidebar.slider("Chunk Length (seconds)", 15, 60, 30)
max_duration = st.sidebar.slider("Max Audio Duration (minutes)", 1, 30, 10)

# File uploader
uploaded_file = st.file_uploader("Upload Audio File (WAV, MP3, M4A, etc.)", 
                                type=['wav', 'mp3', 'm4a', 'ogg', 'flac'])

if uploaded_file:
    if uploaded_file.size > 50 * 1024 * 1024:  # 50MB limit
        st.error("File too large. Please upload smaller than 50MB.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            audio_path = tmp_file.name
        
        with st.spinner("Transcribing audio... (this may take a minute)"):
            result = asr_engine.transcribe_file(audio_path, chunk_length=chunk_length)
        
        # Process segments with sentiment
        sentiment_history = []
        full_transcript = ""
        
        st.subheader("📝 Full Transcription")
        for seg in result["segments"]:
            sentiment = sentiment_engine.analyze(seg["text"])
            full_transcript += f"[{format_timestamp(seg['start'])}] {seg['text']}\n"
            
            sentiment_history.append({
                "timestamp": datetime.now(),
                "text": seg["text"],
                "start": seg["start"],
                "label": sentiment["label"],
                "score": sentiment["score"],
                "confidence": sentiment["confidence"]
            })
        
        st.text_area("Transcript with timestamps", full_transcript, height=300)
        
        # Overall sentiment
        overall = sentiment_engine.analyze(full_transcript)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Sentiment", overall["label"].upper(), 
                     f"{overall['score']:.2f}")
        with col2:
            st.metric("Confidence", f"{overall['confidence']:.1%}")
        with col3:
            st.metric("Duration", f"{result['duration']:.1f} seconds")
        
        # Emotion breakdown
        if overall["emotions"]:
            st.subheader("😊 Emotion Distribution")
            emo_df = pd.DataFrame(list(overall["emotions"].items()), columns=["Emotion", "Score"])
            st.bar_chart(emo_df.set_index("Emotion"))
        
        # Sentiment Trend
        st.subheader("📈 Sentiment Trend")
        fig = create_sentiment_timeline(sentiment_history)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Download options
        st.download_button("Download Transcript", 
                          full_transcript, 
                          file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M')}.txt")
        
        # Clean up
        os.unlink(audio_path)
        
else:
    st.info("👆 Upload an audio file to begin analysis")
    
st.caption("Built with faster-whisper + Hugging Face Transformers + Streamlit")
