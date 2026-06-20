---
title: Voice Sentiment Analyzer
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
---

# 🎙️ Voice Sentiment Analyzer

Upload an audio file or record live from your mic, and get instant transcription,
sentiment analysis, emotion breakdown, a summary, and visualizations — all running
on open-source models, no API keys required.

## Features
- 📁 Upload audio files (mp3, wav, m4a, ogg, flac) **or** 🎤 record live from your mic
- 📝 Transcription via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Distil-Whisper, CTranslate2)
- 🙂 Sentiment analysis (positive / negative / neutral) with adjustable threshold
- 😊 Emotion breakdown with a choice of models
- 🧾 Auto-generated summary
- 🔥 Sentiment heatmap directly on the transcript text
- 🌊 Waveform visualization with sentiment overlay
- 📈 Interactive sentiment timeline chart
- ⬇️ Download transcript

## Demo / Deploy options
This app runs anywhere Streamlit runs: locally, in Docker, or on Hugging Face Spaces.

---

## 1. Run locally

### Prerequisites
- Python 3.10+
- **ffmpeg** (required by faster-whisper and librosa to decode audio)
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: download from https://ffmpeg.org/download.html and add it to PATH

The app checks for ffmpeg on startup and will show a clear error if it's missing.

### Setup
```bash
git clone https://github.com/<your-username>/voice-sentiment-analyzer.git
cd voice-sentiment-analyzer

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> **No GPU?** The default `pip install torch` pulls a CUDA-enabled build that's
> large and unnecessary on CPU-only machines. For a smaller, faster CPU install:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

### Run
```bash
streamlit run app.py
```
Open the URL Streamlit prints (default `http://localhost:8501`).

First run will download ~1-2GB of models from Hugging Face and may take a
few minutes; they're cached locally afterward (`~/.cache/huggingface`).

---

## 2. Run with Docker

```bash
docker build -t voice-sentiment-analyzer .
docker run -p 8501:8501 voice-sentiment-analyzer
```
Open `http://localhost:8501`.

(For GPU acceleration, use an `nvidia/cuda` base image and run with `--gpus all`
— the default Dockerfile here is CPU-only for portability.)

---

## 3. Deploy to Hugging Face Spaces

1. Create a new Space → SDK: **Streamlit**.
2. Push this repo's contents to the Space's git remote (the YAML block at the
   top of this README is the Space config — keep it).
3. That's it; Spaces installs `requirements.txt` and runs `app.py` automatically.

---

## Configuration

All defaults can be overridden via environment variables (see `config.py`),
useful for both local runs and Spaces:

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `Systran/faster-distil-whisper-large-v3` | Must be a **CTranslate2-converted** Whisper checkpoint — see Troubleshooting |
| `SENTIMENT_MODEL` | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Sentiment classifier |
| `EMOTION_MODEL` | `j-hartmann/emotion-english-distilroberta-base` | Default emotion classifier (also switchable in the sidebar) |
| `SUMMARY_MODEL` | `sshleifer/distilbart-cnn-12-6` | Summarization model |
| `MAX_AUDIO_MINUTES` | `10` | Default max audio length (also adjustable in sidebar) |
| `MAX_UPLOAD_MB` | `50` | Max upload file size |

Example:
```bash
export MAX_AUDIO_MINUTES=20
streamlit run app.py
```

Visual theme (colors/fonts) is set in `.streamlit/config.toml` using
Streamlit's native theming — edit that file to rebrand the app.

---

## Project structure
```
.
├── app.py               # Streamlit UI and orchestration
├── asr_engine.py         # faster-whisper transcription wrapper
├── sentiment_engine.py   # sentiment + emotion classification
├── summary_engine.py     # transcript summarization
├── utils.py              # formatting, heatmap, waveform/chart helpers
├── config.py              # central config with env-var overrides
├── requirements.txt
├── Dockerfile
└── .streamlit/config.toml # theme
```

---

## Troubleshooting

**`RuntimeError: Unable to open file 'model.bin' in model ...`**
faster-whisper requires a **CTranslate2-converted** Whisper checkpoint, not a
raw Hugging Face Transformers repo. Use a repo like
`Systran/faster-distil-whisper-large-v3` (the default here), not
`distil-whisper/distil-large-v3`.

**`AttributeError: module 'streamlit' has no attribute 'audio_input'`**
Your Streamlit version is older than 1.36. Run `pip install -U streamlit` or
reinstall from `requirements.txt` (pinned to 1.38.0). The app falls back to a
warning instead of crashing if this happens.

**`KeyError: Unknown task summarization`**
This means an unusual/minimal `transformers` build is missing the `pipeline()`
task registry entry. `summary_engine.py` avoids this by loading
`AutoModelForSeq2SeqLM` directly instead of using `pipeline("summarization", ...)`.

**Audio decode errors / "ffmpeg not found"**
Install ffmpeg system-wide (see Prerequisites above) — it's a system binary,
not a pip package, so `pip install` alone won't provide it.

**Slow first load**
Three models are downloaded and cached on first run (~1-2GB total). Subsequent
runs are fast since `@st.cache_resource` keeps them in memory and Hugging Face
caches the weights on disk.

---

## License
MIT — see [LICENSE](LICENSE).
