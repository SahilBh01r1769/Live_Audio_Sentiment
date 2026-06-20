import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import html as html_lib


def create_sentiment_timeline(sentiment_history):
    """Create a Plotly timeline chart for sentiment trend"""
    if not sentiment_history:
        return None

    df = pd.DataFrame(sentiment_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    fig = px.line(
        df, x='timestamp', y='score',
        color='label',
        title='Sentiment Trend Over Time',
        labels={'score': 'Sentiment Score', 'label': 'Sentiment'},
        color_discrete_map={
            'positive': '#0E9F6E',
            'negative': '#E0445A',
            'neutral': '#8C97A8',
        },
    )
    fig.update_layout(
        height=380,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, 'Source Sans Pro', sans-serif", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def format_timestamp(seconds):
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


# Refreshed, calmer palette (muted teal / rose / slate instead of harsh red/green/blue)
SENTIMENT_COLORS = {
    "positive": (14, 159, 110),   # muted emerald/teal
    "negative": (224, 68, 90),    # soft rose/coral
    "neutral": (140, 151, 168),   # cool slate gray
}


def sentiment_rgba(label, score, base_alpha=0.12, max_alpha=0.42):
    """Return an rgba() string for a sentiment label, intensity scaled by |score|."""
    r, g, b = SENTIMENT_COLORS.get(label, SENTIMENT_COLORS["neutral"])
    intensity = min(abs(score), 1.0)
    alpha = base_alpha + intensity * (max_alpha - base_alpha)
    return f"rgba({r},{g},{b},{alpha:.2f})"


def classify_overall(avg_score, pos_threshold=0.15, neg_threshold=-0.15):
    """Apply an adjustable threshold to decide the overall sentiment label."""
    if avg_score > pos_threshold:
        return "positive"
    if avg_score < neg_threshold:
        return "negative"
    return "neutral"


def render_transcript_heatmap(sentiment_history):
    """
    Build an HTML block showing the transcript with each segment's background
    color-coded by sentiment, intensity scaled by score magnitude.

    IMPORTANT: every line is built with zero leading whitespace — Streamlit's
    markdown renderer treats 4+ leading spaces as a code block, which is why
    indented HTML here previously rendered as a raw/code block instead of HTML.
    """
    if not sentiment_history:
        return "<p>No transcript available.</p>"

    parts = ['<div style="display:flex;flex-direction:column;gap:8px;">']

    legend = (
        '<div style="margin-bottom:4px;font-size:0.78rem;color:#6b7280;'
        'display:flex;gap:10px;align-items:center;">'
        '<span style="background:rgba(14,159,110,0.30);padding:2px 10px;border-radius:12px;">Positive</span>'
        '<span style="background:rgba(224,68,90,0.30);padding:2px 10px;border-radius:12px;">Negative</span>'
        '<span style="background:rgba(140,151,168,0.30);padding:2px 10px;border-radius:12px;">Neutral</span>'
        '<span>darker = stronger sentiment</span>'
        '</div>'
    )
    parts.append(legend)

    for item in sentiment_history:
        label = item.get("label", "neutral")
        score = item.get("score", 0.0)
        text = html_lib.escape(item.get("text", ""))
        ts = format_timestamp(item.get("start", 0))
        bg = sentiment_rgba(label, score)
        r, g, b = SENTIMENT_COLORS.get(label, SENTIMENT_COLORS["neutral"])
        border_color = f"rgb({r},{g},{b})"

        row = (
            f'<div style="background:{bg};border-left:3px solid {border_color};'
            f'padding:10px 14px;border-radius:8px;font-size:0.95rem;line-height:1.5;">'
            f'<span style="opacity:0.55;font-size:0.78rem;margin-right:10px;font-variant-numeric:tabular-nums;">{ts}</span>'
            f'<span style="font-weight:600;text-transform:uppercase;font-size:0.68rem;'
            f'letter-spacing:0.05em;opacity:0.65;margin-right:10px;">{label}</span>'
            f'<span>{text}</span>'
            f'</div>'
        )
        parts.append(row)

    parts.append('</div>')
    return "".join(parts)


def create_waveform_sentiment_figure(audio_path, sentiment_history, max_points=4000):
    """
    Plot the audio waveform envelope with colored background bands per
    transcript segment, indicating sentiment over time.
    """
    try:
        import librosa
    except ImportError:
        return None

    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception:
        return None

    if len(y) == 0:
        return None

    duration = len(y) / sr

    if len(y) > max_points:
        window = len(y) // max_points
        trimmed_len = window * max_points
        y_trimmed = y[:trimmed_len].reshape(-1, window)
        y_env = np.max(np.abs(y_trimmed), axis=1)
        times = np.linspace(0, duration, num=len(y_env))
    else:
        y_env = np.abs(y)
        times = np.linspace(0, duration, num=len(y_env))

    fig = go.Figure()

    for item in sentiment_history or []:
        label = item.get("label", "neutral")
        score = item.get("score", 0.0)
        start = item.get("start", 0)
        end = item.get("end", start + 1)
        color = sentiment_rgba(label, score, base_alpha=0.12, max_alpha=0.35)
        fig.add_vrect(x0=start, x1=end, fillcolor=color, line_width=0, layer="below")

    fig.add_trace(go.Scatter(
        x=times, y=y_env, mode="lines",
        line=dict(color="rgba(51,65,85,0.85)", width=1),
        fill="tozeroy", fillcolor="rgba(51,65,85,0.12)",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=times, y=-y_env, mode="lines",
        line=dict(color="rgba(51,65,85,0.85)", width=1),
        fill="tozeroy", fillcolor="rgba(51,65,85,0.12)",
        showlegend=False,
    ))

    fig.update_layout(
        title="Waveform with Sentiment Overlay",
        xaxis_title="Time (s)",
        yaxis_title=None,
        height=280,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, 'Source Sans Pro', sans-serif", size=12),
        margin=dict(l=30, r=20, t=40, b=30),
    )
    fig.update_yaxes(showticklabels=False)

    return fig
