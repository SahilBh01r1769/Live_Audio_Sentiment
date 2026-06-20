import streamlit as st
import torch
import os
import shutil
import pandas as pd
from datetime import datetime
import tempfile

import config
from asr_engine import ASREngine
from sentiment_engine import SentimentEngine, EMOTION_MODEL_OPTIONS
from summary_engine import SummaryEngine
from utils import (
    create_sentiment_timeline,
    format_timestamp,
    render_transcript_heatmap,
    create_waveform_sentiment_figure,
    classify_overall,
)

st.set_page_config(page_title="Voice Sentiment Analyzer", layout="wide", page_icon="🎙️")

# Card styling only — colors/fonts come from .streamlit/config.toml theme
st.markdown(
    """
    <style>
    div[data-baseweb="radio"] label { font-weight: 500; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎙️ Voice Sentiment Analyzer")
st.markdown("Upload an audio file **or** record live, and get transcription + sentiment/emotion analysis with rich visuals.")

# --- Environment sanity check: ffmpeg is required by faster-whisper & librosa ---
if shutil.which("ffmpeg") is None:
    st.error(
        "⚠️ `ffmpeg` was not found on this system. faster-whisper and librosa both "
        "require it to decode audio. Install it with:\n\n"
        "- **macOS**: `brew install ffmpeg`\n"
        "- **Ubuntu/Debian**: `sudo apt install ffmpeg`\n"
        "- **Windows**: download from https://ffmpeg.org/download.html and add it to PATH\n\n"
        "Then restart the app."
    )
    st.stop()

# Sidebar
st.sidebar.header("⚙️ Settings")
chunk_length = st.sidebar.slider("Chunk Length (seconds)", 15, 60, config.DEFAULT_CHUNK_LENGTH)
max_duration = st.sidebar.slider("Max Audio Duration (minutes)", 1, 30, config.MAX_AUDIO_MINUTES)
generate_summary = st.sidebar.checkbox("Generate summary", value=True)

st.sidebar.subheader("Sentiment Settings")
sentiment_threshold = st.sidebar.slider(
    "Neutral zone threshold (±)",
    min_value=0.05, max_value=0.50, value=config.DEFAULT_SENTIMENT_THRESHOLD, step=0.05,
    help="Average sentiment scores within ± this value are classified as neutral overall."
)

emotion_model_label = st.sidebar.selectbox(
    "Emotion model",
    options=list(EMOTION_MODEL_OPTIONS.keys()),
    index=0,
    help="Choose which emotion-classification model to use for the emotion breakdown."
)
emotion_model_id = EMOTION_MODEL_OPTIONS[emotion_model_label]

st.sidebar.caption(f"Device: **{'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}**")


@st.cache_resource(show_spinner="Loading models (first run can take a minute)...")
def load_models(emotion_model_id):
    asr = ASREngine()
    sentiment = SentimentEngine(emotion_model=emotion_model_id)
    summarizer = SummaryEngine()
    return asr, sentiment, summarizer


asr_engine, sentiment_engine, summary_engine = load_models(emotion_model_id)

# --- Input selection: Upload vs Live mic ---
input_mode = st.radio("Choose input method", ["📁 Upload File", "🎤 Record Live"], horizontal=True)

audio_bytes = None
suffix = ".wav"
has_mic_input = hasattr(st, "audio_input")

if input_mode == "📁 Upload File":
    uploaded_file = st.file_uploader(
        "Upload Audio File (WAV, MP3, M4A, etc.)",
        type=['wav', 'mp3', 'm4a', 'ogg', 'flac']
    )
    if uploaded_file:
        if uploaded_file.size > config.MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File too large. Please upload smaller than {config.MAX_UPLOAD_MB}MB.")
        else:
            audio_bytes = uploaded_file.getvalue()
else:
    if not has_mic_input:
        st.warning(
            "Live recording requires Streamlit 1.36+ (`st.audio_input`). "
            "Run `pip install -U streamlit` (see requirements.txt), or switch to 'Upload File' for now."
        )
    else:
        st.info("Click the mic, record, then stop — your recording will be analyzed automatically.")
        mic_input = st.audio_input("Record your voice")
        if mic_input is not None:
            audio_bytes = mic_input.getvalue()


if audio_bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(audio_bytes)
        audio_path = tmp_file.name

    try:
        with st.spinner("Transcribing audio... (this may take a minute)"):
            result = asr_engine.transcribe_file(audio_path, chunk_length=chunk_length)

        if result["duration"] > max_duration * 60:
            st.warning(
                f"Audio is {result['duration']/60:.1f} min, longer than the "
                f"{max_duration} min limit set in the sidebar. Processing anyway, "
                f"but consider raising the limit or trimming the file."
            )

        if not result["segments"]:
            st.warning("No speech detected in this audio. Try a clearer recording or different file.")
        else:
            sentiment_history = []
            full_transcript = ""

            with st.spinner("Analyzing sentiment & emotion..."):
                for seg in result["segments"]:
                    sentiment = sentiment_engine.analyze(seg["text"])
                    full_transcript += f"[{format_timestamp(seg['start'])}] {seg['text']}\n"

                    sentiment_history.append({
                        "timestamp": datetime.now(),
                        "text": seg["text"],
                        "start": seg["start"],
                        "end": seg["end"],
                        "label": sentiment["label"],
                        "score": sentiment["score"],
                        "confidence": sentiment["confidence"],
                    })

            if generate_summary:
                with st.spinner("Generating summary..."):
                    summary_text = summary_engine.summarize(result["full_text"])
                st.subheader("🧾 Summary")
                st.info(summary_text)

            st.subheader("📝 Transcript (sentiment heatmap)")
            st.markdown(render_transcript_heatmap(sentiment_history), unsafe_allow_html=True)

            with st.expander("Plain text transcript"):
                st.text_area("Transcript with timestamps", full_transcript, height=200)

            scores = [s["score"] for s in sentiment_history]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            overall_label = classify_overall(
                avg_score,
                pos_threshold=sentiment_threshold,
                neg_threshold=-sentiment_threshold,
            )
            avg_confidence = sum(s["confidence"] for s in sentiment_history) / len(sentiment_history)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Sentiment", overall_label.upper(), f"{avg_score:.2f}")
            with col2:
                st.metric("Avg. Confidence", f"{avg_confidence:.1%}")
            with col3:
                st.metric("Duration", f"{result['duration']:.1f} seconds")

            overall_emotion_text = sentiment_engine.analyze(result["full_text"][:2000])
            if overall_emotion_text.get("emotions"):
                st.subheader("😊 Emotion Distribution")
                emo_df = pd.DataFrame(list(overall_emotion_text["emotions"].items()), columns=["Emotion", "Score"])
                emo_df = emo_df.sort_values("Score", ascending=False)
                st.bar_chart(emo_df.set_index("Emotion"))

            st.subheader("🌊 Waveform with Sentiment Overlay")
            wave_fig = create_waveform_sentiment_figure(audio_path, sentiment_history)
            if wave_fig:
                st.plotly_chart(wave_fig, use_container_width=True)
            else:
                st.caption("Waveform visualization unavailable (librosa not installed or unsupported audio format).")

            st.subheader("📈 Sentiment Trend")
            fig = create_sentiment_timeline(sentiment_history)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                "Download Transcript",
                full_transcript,
                file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
            )
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)

else:
    st.info("👆 Upload an audio file or record live to begin analysis")

st.caption("Built with faster-whisper + Hugging Face Transformers + Streamlit")
