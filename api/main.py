"""
api/main.py

Optional FastAPI service layer exposing the VBCUA analysis pipeline as a
REST API. Run with:

    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

The Streamlit frontend (frontend/app.py) can run standalone without this
service (it calls the pipeline directly), but this API is useful for
integrating VBCUA into other applications or mobile clients.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from config import settings
from backend import database
from backend.pipeline import AnalysisPipeline
from models.schema import AnalysisResponse, SessionSummary, HealthResponse

app = FastAPI(
    title="VBCUA API",
    description="Voice-Based Concept Understanding Analyser - REST API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = AnalysisPipeline()


@app.on_event("startup")
def on_startup():
    database.init_db()


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        whisper_model=settings.WHISPER_MODEL_SIZE,
        sbert_model=settings.SENTENCE_BERT_MODEL,
    )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    reference_concept: str = Form(..., description="The concept the speaker should explain"),
    audio_file: UploadFile = File(..., description="Recorded audio (wav/mp3/m4a)"),
):
    if not audio_file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    temp_name = f"{uuid.uuid4().hex}_{audio_file.filename}"
    dest_path = settings.UPLOAD_DIR / temp_name
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(audio_file.file, f)

    try:
        result = pipeline.run(
            audio_path=str(dest_path),
            reference_concept=reference_concept,
            original_filename=audio_file.filename,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    return AnalysisResponse(
        transcript=result["transcript"]["text"],
        language=result["transcript"]["language"],
        duration_sec=result["transcript"]["duration"],
        audio_features=result["audio_features"],
        fluency_metrics=result["fluency_metrics"],
        semantic_result=result["semantic_result"],
        score_breakdown=result["score_breakdown"],
        report_available=result["report_path"] is not None,
    )


@app.get("/sessions", response_model=list[SessionSummary])
def list_sessions(limit: int = 50):
    sessions = database.get_all_sessions(limit=limit)
    return [
        SessionSummary(
            id=s["id"],
            created_at=s["created_at"],
            audio_filename=s["audio_filename"],
            reference_concept=s["reference_concept"],
            overall_score=s["overall_score"],
            grade=s["grade"],
        )
        for s in sessions
    ]


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    session = database.get_session_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.get("/sessions/{session_id}/report")
def download_report(session_id: str):
    session = database.get_session_by_id(session_id)
    if not session or not session.get("report_path"):
        raise HTTPException(status_code=404, detail="Report not found")

    report_path = Path(session["report_path"])
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report file missing on disk")

    return FileResponse(report_path, media_type="application/pdf", filename=report_path.name)
