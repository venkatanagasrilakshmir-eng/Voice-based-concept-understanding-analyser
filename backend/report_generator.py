"""
backend/report_generator.py

Generates a polished PDF report summarizing a VBCUA analysis session:
scores, transcript, waveform visualization, and educational feedback.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
)


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="VBCUATitle", fontSize=20, leading=24, spaceAfter=6,
        textColor=colors.HexColor("#1F2937"), fontName="Helvetica-Bold"
    ))
    styles.add(ParagraphStyle(
        name="VBCUASubtitle", fontSize=11, textColor=colors.HexColor("#6B7280"),
        spaceAfter=14
    ))
    styles.add(ParagraphStyle(
        name="SectionHeader", fontSize=13, spaceBefore=16, spaceAfter=8,
        textColor=colors.HexColor("#111827"), fontName="Helvetica-Bold"
    ))
    styles.add(ParagraphStyle(
        name="BodyTextVBCUA", fontSize=10, leading=15, textColor=colors.HexColor("#374151")
    ))
    return styles


def _grade_color(grade: str) -> colors.Color:
    return {
        "A": colors.HexColor("#16A34A"),
        "B": colors.HexColor("#65A30D"),
        "C": colors.HexColor("#CA8A04"),
        "D": colors.HexColor("#EA580C"),
        "F": colors.HexColor("#DC2626"),
    }.get(grade, colors.HexColor("#6B7280"))


def generate_pdf_report(
    output_path: str,
    reference_concept: str,
    transcript: str,
    score_breakdown: Dict[str, Any],
    fluency_metrics: Dict[str, Any],
    audio_features: Dict[str, Any],
    feedback: str,
    waveform_image_path: Optional[str] = None,
    audio_filename: str = "recording",
) -> str:
    """Build and save a PDF report. Returns the output_path."""
    styles = _styles()
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm
    )
    story = []

    # --- Header ---
    story.append(Paragraph("Voice-Based Concept Understanding Analyser", styles["VBCUATitle"]))
    story.append(Paragraph(
        f"Assessment Report &nbsp;|&nbsp; Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} "
        f"&nbsp;|&nbsp; File: {audio_filename}",
        styles["VBCUASubtitle"]
    ))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#E5E7EB")))
    story.append(Spacer(1, 10))

    # --- Overall score badge ---
    overall = score_breakdown.get("overall_score", 0)
    grade = score_breakdown.get("grade", "-")
    score_table = Table(
        [[f"Overall Score: {overall}/100", f"Grade: {grade}"]],
        colWidths=[9 * cm, 6 * cm]
    )
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#EFF6FF")),
        ("BACKGROUND", (1, 0), (1, 0), _grade_color(grade)),
        ("TEXTCOLOR", (1, 0), (1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 12),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 16))

    # --- Score breakdown ---
    story.append(Paragraph("Score Breakdown", styles["SectionHeader"]))
    breakdown_data = [
        ["Metric", "Score (0-100)"],
        ["Conceptual Understanding", score_breakdown.get("understanding_score")],
        ["Fluency", score_breakdown.get("fluency_score")],
        ["Clarity", score_breakdown.get("clarity_score")],
    ]
    bt = Table(breakdown_data, colWidths=[10 * cm, 5 * cm])
    bt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(bt)
    story.append(Spacer(1, 14))

    # --- Reference concept & transcript ---
    story.append(Paragraph("Reference Concept", styles["SectionHeader"]))
    story.append(Paragraph(reference_concept or "-", styles["BodyTextVBCUA"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Spoken Transcript", styles["SectionHeader"]))
    story.append(Paragraph(transcript or "-", styles["BodyTextVBCUA"]))
    story.append(Spacer(1, 8))

    # --- Waveform image ---
    if waveform_image_path and os.path.exists(waveform_image_path):
        story.append(Paragraph("Audio Waveform", styles["SectionHeader"]))
        story.append(Image(waveform_image_path, width=15 * cm, height=4.2 * cm))
        story.append(Spacer(1, 8))

    # --- Fluency & audio metrics ---
    story.append(Paragraph("Fluency & Audio Metrics", styles["SectionHeader"]))
    metrics_data = [
        ["Speaking rate (wpm)", fluency_metrics.get("speaking_rate_wpm")],
        ["Pauses / 30s", fluency_metrics.get("pause_ratio_per_30s")],
        ["Filler word ratio (%)", fluency_metrics.get("filler_ratio_pct")],
        ["Pitch variation (Hz)", audio_features.get("pitch_std_hz")],
        ["Silence ratio", audio_features.get("silence_ratio")],
        ["Duration (s)", audio_features.get("duration_sec")],
    ]
    mt = Table([["Metric", "Value"]] + metrics_data, colWidths=[10 * cm, 5 * cm])
    mt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(mt)
    story.append(Spacer(1, 14))

    # --- Feedback ---
    story.append(Paragraph("Educational Feedback", styles["SectionHeader"]))
    story.append(Paragraph(feedback or "-", styles["BodyTextVBCUA"]))

    doc.build(story)
    return output_path
