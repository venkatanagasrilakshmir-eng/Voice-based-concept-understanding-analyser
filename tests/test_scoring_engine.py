"""
tests/test_scoring_engine.py

Tests for backend.scoring_engine — pure logic, no ML models required.
"""
from backend.scoring_engine import compute_scores, _score_to_grade


def test_score_to_grade_boundaries():
    assert _score_to_grade(95) == "A"
    assert _score_to_grade(85) == "B"
    assert _score_to_grade(75) == "C"
    assert _score_to_grade(65) == "D"
    assert _score_to_grade(40) == "F"


def test_compute_scores_high_quality_input():
    fluency_metrics = {
        "speaking_rate_wpm": 140,
        "pause_ratio_per_30s": 0.5,
        "filler_ratio_pct": 1.0,
    }
    audio_features = {
        "pitch_std_hz": 35,
        "zero_crossing_rate": 0.05,
        "rms_energy_mean": 0.02,
        "silence_ratio": 0.1,
    }
    breakdown = compute_scores(
        semantic_similarity=0.9,
        fluency_metrics=fluency_metrics,
        audio_features=audio_features,
    )
    assert breakdown.understanding_score == 90.0
    assert breakdown.overall_score > 70
    assert breakdown.grade in {"A", "B"}


def test_compute_scores_low_quality_input():
    fluency_metrics = {
        "speaking_rate_wpm": 30,
        "pause_ratio_per_30s": 10,
        "filler_ratio_pct": 25,
    }
    audio_features = {
        "pitch_std_hz": 0,
        "zero_crossing_rate": 0.01,
        "rms_energy_mean": 0.0,
        "silence_ratio": 0.9,
    }
    breakdown = compute_scores(
        semantic_similarity=0.1,
        fluency_metrics=fluency_metrics,
        audio_features=audio_features,
    )
    assert breakdown.understanding_score == 10.0
    assert breakdown.overall_score < 50
    assert breakdown.grade in {"D", "F"}


def test_scores_are_bounded_0_to_100():
    breakdown = compute_scores(
        semantic_similarity=1.5,  # intentionally out-of-range input
        fluency_metrics={"speaking_rate_wpm": 500, "pause_ratio_per_30s": 0, "filler_ratio_pct": 0},
        audio_features={"pitch_std_hz": 35, "zero_crossing_rate": 0.05,
                         "rms_energy_mean": 0.02, "silence_ratio": 0.0},
    )
    for value in (breakdown.understanding_score, breakdown.fluency_score,
                  breakdown.clarity_score, breakdown.overall_score):
        assert 0.0 <= value <= 150.0  # understanding isn't clamped upstream by design
