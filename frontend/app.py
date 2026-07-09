"""
frontend/app.py

Streamlit frontend for the Voice-Based Concept Understanding Analyser (VBCUA).
Run with:

    streamlit run frontend/app.py

Features:
- Audio upload & playback
- Waveform visualization
- Real-time scoring (understanding, fluency, clarity)
- Understanding analysis (matched / missing key terms + feedback)
- PDF report download
- Session history browser
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import streamlit as st

# Allow running via `streamlit run frontend/app.py` from repo root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings
from backend import database
from backend.pipeline import AnalysisPipeline
from backend.audio_features import generate_waveform_plot

st.set_page_config(
    page_title="VBCUA - Concept Understanding Analyser",
    page_icon="🎙️",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_pipeline() -> AnalysisPipeline:
    database.init_db()
    return AnalysisPipeline()


def score_badge(label: str, value: float, suffix: str = "/100"):
    color = "#16A34A" if value >= 80 else "#CA8A04" if value >= 60 else "#DC2626"
    st.markdown(
        f"""
        <div style="border:1px solid #E5E7EB;border-radius:10px;padding:14px 16px;text-align:center;">
            <div style="font-size:13px;color:#6B7280;">{label}</div>
            <div style="font-size:28px;font-weight:700;color:{color};">{value}{suffix}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.title("🎙️ Voice-Based Concept Understanding Analyser")
    st.caption(
        "Speak an explanation of a concept, and get AI-powered scoring on "
        "understanding, fluency, and clarity — with a downloadable PDF report."
    )

    pipeline = get_pipeline()

    tab_analyze, tab_history = st.tabs(["🧠 Analyze", "📜 History"])

    # ---------------------------------------------------------------
    # ANALYZE TAB
    # ---------------------------------------------------------------
    with tab_analyze:
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("1. Reference Concept")
            reference_concept = st.text_area(
                "Enter the concept / definition the speaker should explain",
                placeholder="e.g. 'Photosynthesis is the process by which plants convert "
                            "sunlight, water, and carbon dioxide into glucose and oxygen.'",
                height=140,
            )

            st.subheader("2. Upload Audio")
            audio_file = st.file_uploader(
                "Upload a recording (wav, mp3, m4a, flac, ogg)",
                type=["wav", "mp3", "m4a", "flac", "ogg"],
            )

            if audio_file is not None:
                st.audio(audio_file)

            analyze_clicked = st.button(
                "🔍 Analyze Recording", type="primary", use_container_width=True,
                disabled=not (audio_file and reference_concept.strip())
            )

        with col_right:
            st.subheader("Results")
            results_placeholder = st.container()

        if analyze_clicked and audio_file and reference_concept.strip():
            with st.spinner("Transcribing speech, scoring understanding, and analyzing audio..."):
                temp_name = f"{uuid.uuid4().hex}_{audio_file.name}"
                audio_path = settings.UPLOAD_DIR / temp_name
                with open(audio_path, "wb") as f:
                    f.write(audio_file.getbuffer())

                try:
                    result = pipeline.run(
                        audio_path=str(audio_path),
                        reference_concept=reference_concept,
                        original_filename=audio_file.name,
                    )
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
                    return

            with results_placeholder:
                score_breakdown = result["score_breakdown"]
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    score_badge("Understanding", score_breakdown["understanding_score"])
                with c2:
                    score_badge("Fluency", score_breakdown["fluency_score"])
                with c3:
                    score_badge("Clarity", score_breakdown["clarity_score"])
                with c4:
                    score_badge("Overall", score_breakdown["overall_score"])
                    st.markdown(
                        f"<div style='text-align:center;margin-top:4px;'>Grade: "
                        f"<b>{score_breakdown['grade']}</b></div>",
                        unsafe_allow_html=True,
                    )

                st.divider()

                # Waveform
                waveform_path = settings.REPORT_DIR / f"waveform_preview_{uuid.uuid4().hex[:6]}.png"
                generate_waveform_plot(str(audio_path), str(waveform_path))
                st.image(str(waveform_path), caption="Audio Waveform", use_container_width=True)

                # Transcript
                with st.expander("📝 Transcript", expanded=True):
                    st.write(result["transcript"]["text"] or "_No speech detected._")

                # Understanding analysis
                with st.expander("🧩 Understanding Analysis", expanded=True):
                    sem = result["semantic_result"]
                    st.metric("Semantic Similarity", f"{sem['similarity_score']:.2f}")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**✅ Key terms matched:**")
                        st.write(", ".join(sem["matched_key_terms"]) or "_None_")
                    with col_b:
                        st.markdown("**⚠️ Key terms missing:**")
                        st.write(", ".join(sem["missing_key_terms"]) or "_None_")
                    st.markdown("**Feedback:**")
                    st.info(sem["feedback"])

                # Fluency & audio metrics
                with st.expander("📊 Fluency & Audio Metrics"):
                    fm = result["fluency_metrics"]
                    af = result["audio_features"]
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Speaking rate", f"{fm['speaking_rate_wpm']} wpm")
                    m2.metric("Pauses / 30s", fm["pause_ratio_per_30s"])
                    m3.metric("Filler word ratio", f"{fm['filler_ratio_pct']}%")
                    m4, m5, m6 = st.columns(3)
                    m4.metric("Pitch variation", f"{af['pitch_mean_hz']} Hz")
                    m5.metric("Silence ratio", af["silence_ratio"])
                    m6.metric("Duration", f"{af['duration_sec']}s")

                # PDF download
                if result["report_path"]:
                    with open(result["report_path"], "rb") as f:
                        st.download_button(
                            "⬇️ Download PDF Report",
                            data=f.read(),
                            file_name=Path(result["report_path"]).name,
                            mime="application/pdf",
                            use_container_width=True,
                        )

    # ---------------------------------------------------------------
    # HISTORY TAB
    # ---------------------------------------------------------------
    with tab_history:
        st.subheader("Past Sessions")
        sessions = database.get_all_sessions(limit=50)
        if not sessions:
            st.info("No sessions yet — analyze a recording to see history here.")
        else:
            for s in sessions:
                with st.expander(
                    f"{s['created_at']} — {s['audio_filename']} — "
                    f"Score: {s['overall_score']} ({s['grade']})"
                ):
                    st.write(f"**Reference concept:** {s['reference_concept']}")
                    st.write(f"**Transcript:** {s['transcript']}")
                    cols = st.columns(4)
                    cols[0].metric("Understanding", s["understanding_score"])
                    cols[1].metric("Fluency", s["fluency_score"])
                    cols[2].metric("Clarity", s["clarity_score"])
                    cols[3].metric("Overall", s["overall_score"])
                    if s.get("report_path") and Path(s["report_path"]).exists():
                        with open(s["report_path"], "rb") as f:
                            st.download_button(
                                "⬇️ Download Report",
                                data=f.read(),
                                file_name=Path(s["report_path"]).name,
                                mime="application/pdf",
                                key=f"dl_{s['id']}",
                            )


if __name__ == "__main__":
    main()
