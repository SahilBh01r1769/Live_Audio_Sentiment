import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime

def create_sentiment_timeline(sentiment_history):
    """Create a Plotly timeline chart for sentiment trend"""
    if not sentiment_history:
        return None
    
    df = pd.DataFrame(sentiment_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig = px.line(df, x='timestamp', y='score', 
                  color='label',
                  title='Sentiment Trend Over Time',
                  labels={'score': 'Sentiment Score', 'label': 'Sentiment'},
                  color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'blue'})
    
    fig.update_layout(height=400)
    return fig

def format_timestamp(seconds):
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"
