"""
backend/pipeline.py

High-level orchestration: given an audio file + reference concept text,
runs transcription -> semantic evaluation -> audio feature extraction ->
fluency scoring -> overall scoring -> PDF report generation -> DB persistence.

Both the Streamlit frontend and the FastAPI service layer call into this
single pipeline so behavior stays consistent across interfaces.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from config import settings
from backend.transcription import Transcriber
from backend.semantic_evaluation import SemanticEvaluator, count_filler_words
from backend.audio_features import extract_features, compute_fluency_metrics, generate_waveform_plot
from backend.scoring_engine import compute_scores
from backend.report_generator import generate_pdf_report
from backend import database


class AnalysisPipeline:
    def __init__(self):
        self.transcriber = Transcriber()
        self.semantic_evaluator = SemanticEvaluator()

    def run(self, audio_path: str, reference_concept: str,
             original_filename: Optional[str] = None,
             generate_report: bool = True) -> Dict[str, Any]:
        """
        Execute the full analysis pipeline on an audio file.

        Returns a dict with transcript, audio_features, fluency_metrics,
        semantic_result, score_breakdown, and (optionally) report_path.
        """
        original_filename = original_filename or os.path.basename(audio_path)

        # 1. Transcription
        transcription = self.transcriber.transcribe(audio_path)

        # 2. Semantic evaluation
        semantic_result = self.semantic_evaluator.evaluate(reference_concept, transcription.text)

        # 3. Audio feature extraction
        audio_feats = extract_features(audio_path)

        # 4. Fluency metrics (uses transcript word count + audio timing)
        filler_count = count_filler_words(transcription.text)
        fluency = compute_fluency_metrics(
            word_count=transcription.word_count,
            duration_sec=audio_feats.duration_sec or transcription.duration,
            pause_count=audio_feats.pause_count,
            filler_words_found=filler_count,
        )

        # 5. Scoring
        score_breakdown = compute_scores(
            semantic_similarity=semantic_result.similarity_score,
            fluency_metrics=fluency,
            audio_features=audio_feats.to_dict(),
        )

        report_path = None
        if generate_report:
            session_id = str(uuid.uuid4())[:8]
            waveform_path = str(settings.REPORT_DIR / f"waveform_{session_id}.png")
            generate_waveform_plot(audio_path, waveform_path)

            report_path = str(settings.REPORT_DIR / f"report_{session_id}.pdf")
            generate_pdf_report(
                output_path=report_path,
                reference_concept=reference_concept,
                transcript=transcription.text,
                score_breakdown=score_breakdown.to_dict(),
                fluency_metrics=fluency,
                audio_features=audio_feats.to_dict(),
                feedback=semantic_result.feedback,
                waveform_image_path=waveform_path,
                audio_filename=original_filename,
            )

        # 6. Persist to DB
        database.save_session(
            audio_filename=original_filename,
            reference_concept=reference_concept,
            transcript=transcription.text,
            language=transcription.language,
            duration_sec=audio_feats.duration_sec or transcription.duration,
            audio_features=audio_feats.to_dict(),
            fluency_metrics=fluency,
            semantic_result=semantic_result.to_dict(),
            score_breakdown=score_breakdown.to_dict(),
            report_path=report_path,
        )

        return {
            "transcript": transcription.to_dict(),
            "audio_features": audio_feats.to_dict(),
            "fluency_metrics": fluency,
            "semantic_result": semantic_result.to_dict(),
            "score_breakdown": score_breakdown.to_dict(),
            "report_path": report_path,
        }
