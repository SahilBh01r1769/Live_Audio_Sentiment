from faster_whisper import WhisperModel
import torch

class ASREngine:
    def __init__(self, model_name="distil-whisper/distil-large-v3"):
        self.model = WhisperModel(model_name, 
                                device="cuda" if torch.cuda.is_available() else "cpu",
                                compute_type="float16" if torch.cuda.is_available() else "int8")
    
    def transcribe_file(self, audio_path, chunk_length=30, overlap=2):
        """Transcribe audio file with timestamps"""
        segments, info = self.model.transcribe(audio_path, 
                                             beam_size=5,
                                             word_timestamps=True,
                                             vad_filter=True,
                                             vad_parameters=dict(min_silence_duration_ms=500))
        
        transcript = []
        full_text = ""
        
        for segment in segments:
            transcript.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
            full_text += segment.text.strip() + " "
        
        return {
            "full_text": full_text.strip(),
            "segments": transcript,
            "language": info.language,
            "duration": info.duration
        }
