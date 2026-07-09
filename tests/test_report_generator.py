"""
tests/test_report_generator.py

Verifies the PDF report is generated successfully and produces a
non-trivial file on disk.
"""
import os
from backend.report_generator import generate_pdf_report


def test_generate_pdf_report_creates_file(tmp_path):
    output_path = tmp_path / "test_report.pdf"

    score_breakdown = {
        "understanding_score": 82.0,
        "fluency_score": 75.5,
        "clarity_score": 68.0,
        "overall_score": 76.9,
        "grade": "B",
    }
    fluency_metrics = {
        "speaking_rate_wpm": 135.0,
        "pause_ratio_per_30s": 1.2,
        "filler_ratio_pct": 2.1,
    }
    audio_features = {
        "duration_sec": 45.0,
        "pitch_std_hz": 28.0,
        "silence_ratio": 0.15,
    }

    result_path = generate_pdf_report(
        output_path=str(output_path),
        reference_concept="Photosynthesis is how plants convert light into energy.",
        transcript="Plants use sunlight to make food through photosynthesis.",
        score_breakdown=score_breakdown,
        fluency_metrics=fluency_metrics,
        audio_features=audio_features,
        feedback="Good explanation with clear structure.",
        waveform_image_path=None,
        audio_filename="test.wav",
    )

    assert os.path.exists(result_path)
    assert os.path.getsize(result_path) > 500  # non-trivial PDF content
