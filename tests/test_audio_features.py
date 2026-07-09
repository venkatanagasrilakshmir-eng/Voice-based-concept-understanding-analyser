"""
tests/test_audio_features.py

Tests for backend.audio_features using a synthetic tone (no network/model
downloads required).
"""
from backend.audio_features import extract_features, compute_fluency_metrics


def test_extract_features_returns_expected_fields(synthetic_wav):
    features = extract_features(synthetic_wav)
    data = features.to_dict()

    assert data["duration_sec"] > 0
    assert "tempo_bpm" in data
    assert "pitch_mean_hz" in data
    assert "rms_energy_mean" in data
    assert "zero_crossing_rate" in data
    assert 0.0 <= data["silence_ratio"] <= 1.0
    assert isinstance(data["mfcc_mean"], list)
    assert len(data["mfcc_mean"]) == 13


def test_compute_fluency_metrics_basic():
    metrics = compute_fluency_metrics(
        word_count=150, duration_sec=60.0, pause_count=2, filler_words_found=3
    )
    assert metrics["speaking_rate_wpm"] == 150.0
    assert metrics["pause_ratio_per_30s"] == 1.0
    assert metrics["filler_ratio_pct"] == round((3 / 150) * 100, 2)


def test_compute_fluency_metrics_handles_zero_duration():
    metrics = compute_fluency_metrics(word_count=0, duration_sec=0.0, pause_count=0)
    assert metrics["speaking_rate_wpm"] >= 0
    assert metrics["filler_ratio_pct"] == 0
