from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import config


class SummaryEngine:
    def __init__(self, model_name=None):
        model_name = model_name or config.SUMMARY_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _chunk_text(self, text, max_words=600):
        words = text.split()
        return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

    def _generate(self, text, max_length=120, min_length=30):
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
            )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

    def summarize(self, text, max_length=120, min_length=30):
        """Summarize text, chunking + map-reduce for long transcripts."""
        text = text.strip()
        word_count = len(text.split())

        if word_count < 40:
            return text  # too short to meaningfully summarize, return as-is

        chunks = self._chunk_text(text)

        try:
            partial_summaries = []
            for chunk in chunks:
                chunk_words = len(chunk.split())
                m_len = min(max_length, max(20, chunk_words // 2))
                m_min = min(min_length, max(10, m_len // 3))
                partial_summaries.append(self._generate(chunk, max_length=m_len, min_length=m_min))

            combined = " ".join(partial_summaries)

            if len(partial_summaries) > 1:
                combined_words = len(combined.split())
                m_len = min(max_length, max(20, combined_words // 2))
                m_min = min(min_length, max(10, m_len // 3))
                return self._generate(combined, max_length=m_len, min_length=m_min)

            return combined
        except Exception as e:
            return f"(Summary unavailable: {e})"
