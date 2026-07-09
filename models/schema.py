"""
models/schema.py

Pydantic models describing API request/response payloads.
"""
from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ScoreBreakdownSchema(BaseModel):
    understanding_score: float
    fluency_score: float
    clarity_score: float
    overall_score: float
    grade: str


class SemanticResultSchema(BaseModel):
    similarity_score: float
    matched_key_terms: List[str]
    missing_key_terms: List[str]
    feedback: str


class AnalysisResponse(BaseModel):
    session_id: Optional[str] = None
    transcript: str
    language: str
    duration_sec: float
    audio_features: Dict[str, Any]
    fluency_metrics: Dict[str, Any]
    semantic_result: SemanticResultSchema
    score_breakdown: ScoreBreakdownSchema
    report_available: bool = False


class SessionSummary(BaseModel):
    id: str
    created_at: Optional[str]
    audio_filename: Optional[str]
    reference_concept: Optional[str]
    overall_score: Optional[float]
    grade: Optional[str]


class HealthResponse(BaseModel):
    status: str = "ok"
    whisper_model: str
    sbert_model: str
