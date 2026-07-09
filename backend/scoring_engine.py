"""
backend/scoring_engine.py

Combines semantic understanding, fluency, and clarity metrics into a
single weighted score with a letter grade, used to assess a spoken
concept explanation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from config import settings


@dataclass
class ScoreBreakdown:
    understanding_score: float  # 0-100
    fluency_score: float        # 0-100
    clarity_score: float        # 0-100
    overall_score: float        # 0-100
    grade: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "understanding_score": self.understanding_score,
            "fluency_score": self.fluency_score,
            "clarity_score": self.clarity_score,
            "overall_score": self.overall_score,
            "grade": self.grade,
        }


def _score_to_grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def _fluency_subscore(speaking_rate_wpm: float, pause_ratio_per_30s: float,
                       filler_ratio_pct: float) -> float:
    """
    Ideal conversational speaking rate: ~110-160 wpm.
    Penalize excessive pauses and filler words.
    """
    # Rate score: 100 at ideal range, decreasing outside it
    if 110 <= speaking_rate_wpm <= 160:
        rate_score = 100.0
    else:
        distance = min(abs(speaking_rate_wpm - 110), abs(speaking_rate_wpm - 160))
        rate_score = max(0.0, 100 - distance * 1.2)

    pause_penalty = min(40.0, pause_ratio_per_30s * 8)
    filler_penalty = min(30.0, filler_ratio_pct * 3)

    score = rate_score - pause_penalty - filler_penalty
    return max(0.0, min(100.0, score))


def _clarity_subscore(pitch_std_hz: float, zero_crossing_rate: float,
                       rms_energy_mean: float, silence_ratio: float) -> float:
    """
    Clarity approximated from vocal expressiveness (pitch variation),
    signal energy consistency, and low excessive silence.
    """
    # Some pitch variation is good (expressiveness); too flat or too erratic is worse.
    if 10 <= pitch_std_hz <= 60:
        pitch_score = 100.0
    else:
        distance = min(abs(pitch_std_hz - 10), abs(pitch_std_hz - 60))
        pitch_score = max(0.0, 100 - distance * 1.5)

    energy_score = max(0.0, min(100.0, rms_energy_mean * 4000))  # scaled heuristic
    silence_penalty = min(40.0, silence_ratio * 80)

    score = (pitch_score * 0.5 + energy_score * 0.5) - silence_penalty
    return max(0.0, min(100.0, score))


def compute_scores(semantic_similarity: float, fluency_metrics: Dict[str, Any],
                    audio_features: Dict[str, Any]) -> ScoreBreakdown:
    """
    Args:
        semantic_similarity: 0-1 similarity score from SemanticEvaluator
        fluency_metrics: dict from compute_fluency_metrics()
        audio_features: dict from AudioFeatures.to_dict()

    Returns:
        ScoreBreakdown with 0-100 scaled scores and overall grade.
    """
    understanding_score = round(semantic_similarity * 100, 1)

    fluency_score = round(_fluency_subscore(
        fluency_metrics.get("speaking_rate_wpm", 0),
        fluency_metrics.get("pause_ratio_per_30s", 0),
        fluency_metrics.get("filler_ratio_pct", 0),
    ), 1)

    clarity_score = round(_clarity_subscore(
        audio_features.get("pitch_std_hz", 0),
        audio_features.get("zero_crossing_rate", 0),
        audio_features.get("rms_energy_mean", 0),
        audio_features.get("silence_ratio", 0),
    ), 1)

    overall = (
        understanding_score * settings.WEIGHT_UNDERSTANDING +
        fluency_score * settings.WEIGHT_FLUENCY +
        clarity_score * settings.WEIGHT_CLARITY
    )
    overall = round(overall, 1)

    return ScoreBreakdown(
        understanding_score=understanding_score,
        fluency_score=fluency_score,
        clarity_score=clarity_score,
        overall_score=overall,
        grade=_score_to_grade(overall),
    )
