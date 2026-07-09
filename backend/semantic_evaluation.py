"""
backend/semantic_evaluation.py

Semantic understanding evaluation using Sentence-BERT embeddings, plus
optional qualitative feedback generation via Google Gemini. Falls back
to a rule-based feedback generator if no Gemini API key is configured.
"""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from typing import List, Dict, Any

from config import settings

COMMON_FILLER_WORDS = {
    "um", "uh", "umm", "uhh", "like", "you know", "sort of", "kind of",
    "basically", "actually", "literally", "so yeah", "i mean",
}


@dataclass
class SemanticResult:
    similarity_score: float  # 0-1
    matched_key_terms: List[str]
    missing_key_terms: List[str]
    feedback: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "similarity_score": self.similarity_score,
            "matched_key_terms": self.matched_key_terms,
            "missing_key_terms": self.missing_key_terms,
            "feedback": self.feedback,
        }


@functools.lru_cache(maxsize=1)
def _load_sbert_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def count_filler_words(text: str) -> int:
    lower = text.lower()
    count = 0
    for filler in COMMON_FILLER_WORDS:
        count += len(re.findall(rf"\b{re.escape(filler)}\b", lower))
    return count


def _extract_key_terms(reference_text: str, top_n: int = 8) -> List[str]:
    """Very lightweight keyword extraction: longest unique non-stopword tokens."""
    stopwords = {
        "the", "a", "an", "is", "are", "of", "to", "and", "in", "on", "for",
        "that", "this", "it", "with", "as", "by", "or", "be", "was", "were",
        "which", "at", "from", "its", "these", "those", "such",
    }
    words = re.findall(r"[a-zA-Z]+", reference_text.lower())
    unique_terms = sorted(set(w for w in words if w not in stopwords and len(w) > 3),
                           key=lambda w: -len(w))
    return unique_terms[:top_n]


class SemanticEvaluator:
    """Compares a spoken explanation against a reference concept description."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.SENTENCE_BERT_MODEL
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _load_sbert_model(self.model_name)
        return self._model

    def compute_similarity(self, reference_text: str, spoken_text: str) -> float:
        """Cosine similarity between reference and spoken text embeddings, scaled 0-1."""
        from sentence_transformers import util
        embeddings = self.model.encode([reference_text, spoken_text], convert_to_tensor=True)
        cosine_sim = util.cos_sim(embeddings[0], embeddings[1]).item()
        # cosine similarity is typically in [-1, 1]; clamp/rescale to [0, 1]
        return max(0.0, min(1.0, (cosine_sim + 1) / 2 if cosine_sim < 0 else cosine_sim))

    def evaluate(self, reference_text: str, spoken_text: str) -> SemanticResult:
        similarity = self.compute_similarity(reference_text, spoken_text)
        key_terms = _extract_key_terms(reference_text)
        spoken_lower = spoken_text.lower()

        matched = [t for t in key_terms if t in spoken_lower]
        missing = [t for t in key_terms if t not in spoken_lower]

        feedback = generate_feedback(reference_text, spoken_text, similarity, matched, missing)

        return SemanticResult(
            similarity_score=round(similarity, 3),
            matched_key_terms=matched,
            missing_key_terms=missing,
            feedback=feedback,
        )


def generate_feedback(reference_text: str, spoken_text: str, similarity: float,
                       matched_terms: List[str], missing_terms: List[str]) -> str:
    """Generate qualitative feedback via Gemini if configured, else rule-based fallback."""
    if settings.GEMINI_API_KEY:
        try:
            return _gemini_feedback(reference_text, spoken_text, similarity, missing_terms)
        except Exception:
            pass  # fall through to rule-based feedback on any API error

    return _rule_based_feedback(similarity, matched_terms, missing_terms)


def _gemini_feedback(reference_text: str, spoken_text: str, similarity: float,
                      missing_terms: List[str]) -> str:
    import google.generativeai as genai

    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = (
        "You are an educational assessor. A learner was asked to explain a concept "
        "out loud. Compare their spoken explanation to the reference definition and "
        "give concise, encouraging, constructive feedback (3-4 sentences max). "
        "Mention what they got right and what key ideas they missed.\n\n"
        f"Reference concept:\n{reference_text}\n\n"
        f"Learner's spoken explanation (transcribed):\n{spoken_text}\n\n"
        f"Computed semantic similarity score: {similarity:.2f} (0-1 scale)\n"
        f"Key terms possibly missing: {', '.join(missing_terms) if missing_terms else 'none'}\n\n"
        "Feedback:"
    )
    response = model.generate_content(prompt)
    return response.text.strip()


def _rule_based_feedback(similarity: float, matched_terms: List[str],
                          missing_terms: List[str]) -> str:
    if similarity >= 0.8:
        base = "Excellent explanation — your response closely matches the core concept."
    elif similarity >= 0.6:
        base = "Good explanation overall, with a solid grasp of the main idea."
    elif similarity >= 0.4:
        base = "Partial understanding shown — some important aspects were underdeveloped."
    else:
        base = "The explanation diverges significantly from the reference concept."

    details = []
    if matched_terms:
        details.append(f"You correctly referenced: {', '.join(matched_terms)}.")
    if missing_terms:
        details.append(f"Consider also covering: {', '.join(missing_terms)}.")

    return " ".join([base] + details)
