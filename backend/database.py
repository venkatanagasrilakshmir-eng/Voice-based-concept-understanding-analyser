"""
backend/database.py

SQLAlchemy-based persistence layer for VBCUA sessions: stores transcripts,
audio features, evaluation scores, and metadata for each analysis run.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

from config import settings

Base = declarative_base()


class SessionRecord(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow)

    audio_filename = Column(String)
    reference_concept = Column(Text)
    transcript = Column(Text)
    language = Column(String)
    duration_sec = Column(Float)

    # JSON-serialized blobs for flexible nested data
    audio_features_json = Column(Text)
    fluency_metrics_json = Column(Text)
    semantic_result_json = Column(Text)
    score_breakdown_json = Column(Text)

    understanding_score = Column(Float)
    fluency_score = Column(Float)
    clarity_score = Column(Float)
    overall_score = Column(Float)
    grade = Column(String)

    report_path = Column(String, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "audio_filename": self.audio_filename,
            "reference_concept": self.reference_concept,
            "transcript": self.transcript,
            "language": self.language,
            "duration_sec": self.duration_sec,
            "audio_features": json.loads(self.audio_features_json or "{}"),
            "fluency_metrics": json.loads(self.fluency_metrics_json or "{}"),
            "semantic_result": json.loads(self.semantic_result_json or "{}"),
            "score_breakdown": json.loads(self.score_breakdown_json or "{}"),
            "understanding_score": self.understanding_score,
            "fluency_score": self.fluency_score,
            "clarity_score": self.clarity_score,
            "overall_score": self.overall_score,
            "grade": self.grade,
            "report_path": self.report_path,
        }


_engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False}
                         if settings.DATABASE_URL.startswith("sqlite") else {})
_SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=_engine)


def get_db_session():
    return _SessionLocal()


def save_session(
    audio_filename: str,
    reference_concept: str,
    transcript: str,
    language: str,
    duration_sec: float,
    audio_features: Dict[str, Any],
    fluency_metrics: Dict[str, Any],
    semantic_result: Dict[str, Any],
    score_breakdown: Dict[str, Any],
    report_path: Optional[str] = None,
) -> SessionRecord:
    db = get_db_session()
    try:
        record = SessionRecord(
            audio_filename=audio_filename,
            reference_concept=reference_concept,
            transcript=transcript,
            language=language,
            duration_sec=duration_sec,
            audio_features_json=json.dumps(audio_features),
            fluency_metrics_json=json.dumps(fluency_metrics),
            semantic_result_json=json.dumps(semantic_result),
            score_breakdown_json=json.dumps(score_breakdown),
            understanding_score=score_breakdown.get("understanding_score"),
            fluency_score=score_breakdown.get("fluency_score"),
            clarity_score=score_breakdown.get("clarity_score"),
            overall_score=score_breakdown.get("overall_score"),
            grade=score_breakdown.get("grade"),
            report_path=report_path,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record
    finally:
        db.close()


def get_all_sessions(limit: int = 100) -> List[Dict[str, Any]]:
    db = get_db_session()
    try:
        records = (
            db.query(SessionRecord)
            .order_by(SessionRecord.created_at.desc())
            .limit(limit)
            .all()
        )
        return [r.to_dict() for r in records]
    finally:
        db.close()


def get_session_by_id(session_id: str) -> Optional[Dict[str, Any]]:
    db = get_db_session()
    try:
        record = db.query(SessionRecord).filter(SessionRecord.id == session_id).first()
        return record.to_dict() if record else None
    finally:
        db.close()
