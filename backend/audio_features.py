"""
backend/audio_features.py

Extracts acoustic / prosodic features from an audio recording using
Librosa, and derives fluency-related metrics (speaking rate, pauses,
pitch variation, energy) used by the scoring engine.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple

import numpy as np


@dataclass
class AudioFeatures:
    duration_sec: float
    tempo_bpm: float
    pitch_mean_hz: float
    pitch_std_hz: float
    rms_energy_mean: float
    zero_crossing_rate: float
    silence_ratio: float
    pause_count: int
    mfcc_mean: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_audio(audio_path: str, sr: int = 16000):
    import librosa
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    return y, sr


def _detect_silence_intervals(y: "np.ndarray", sr: int, top_db: int = 30) -> Tuple[float, int]:
    """Return (silence_ratio, pause_count) using librosa's non-silent interval split."""
    import librosa
    non_silent = librosa.effects.split(y, top_db=top_db)
    if len(non_silent) == 0:
        return 1.0, 0

    total_len = len(y)
    voiced_len = sum(end - start for start, end in non_silent)
    silence_ratio = 1.0 - (voiced_len / total_len if total_len else 0)

    # A "pause" is a gap between consecutive voiced segments longer than ~300ms
    pause_count = 0
    min_pause_samples = int(0.3 * sr)
    for i in range(1, len(non_silent)):
        gap = non_silent[i][0] - non_silent[i - 1][1]
        if gap >= min_pause_samples:
            pause_count += 1

    return float(silence_ratio), pause_count


def extract_features(audio_path: str) -> AudioFeatures:
    """Compute a bundle of acoustic features for the given audio file."""
    import librosa

    y, sr = load_audio(audio_path)
    duration = float(librosa.get_duration(y=y, sr=sr))

    # Tempo (speaking rhythm proxy)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    # Pitch (fundamental frequency) via pyin
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
        )
        f0_voiced = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        pitch_mean = float(np.mean(f0_voiced)) if len(f0_voiced) else 0.0
        pitch_std = float(np.std(f0_voiced)) if len(f0_voiced) else 0.0
    except Exception:
        pitch_mean, pitch_std = 0.0, 0.0

    # Energy
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))

    # Zero crossing rate (articulation / noisiness proxy)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = float(np.mean(zcr))

    # Silence / pause analysis
    silence_ratio, pause_count = _detect_silence_intervals(y, sr)

    # MFCCs (timbre summary, 13 coefficients averaged)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1).tolist()

    return AudioFeatures(
        duration_sec=round(duration, 2),
        tempo_bpm=round(tempo, 2),
        pitch_mean_hz=round(pitch_mean, 2),
        pitch_std_hz=round(pitch_std, 2),
        rms_energy_mean=round(rms_mean, 5),
        zero_crossing_rate=round(zcr_mean, 5),
        silence_ratio=round(silence_ratio, 3),
        pause_count=pause_count,
        mfcc_mean=[round(v, 3) for v in mfcc_mean],
    )


def compute_fluency_metrics(word_count: int, duration_sec: float, pause_count: int,
                             filler_words_found: int = 0) -> Dict[str, Any]:
    """
    Derive fluency metrics from transcript + audio timing.

    speaking_rate_wpm: words per minute
    pause_ratio: pauses per 30 seconds of speech (normalized)
    filler_ratio: filler words per 100 words
    """
    minutes = max(duration_sec / 60.0, 1e-6)
    speaking_rate_wpm = word_count / minutes

    pause_ratio = pause_count / max(duration_sec / 30.0, 1e-6)
    filler_ratio = (filler_words_found / max(word_count, 1)) * 100

    return {
        "speaking_rate_wpm": round(speaking_rate_wpm, 1),
        "pause_ratio_per_30s": round(pause_ratio, 2),
        "filler_ratio_pct": round(filler_ratio, 2),
    }


def generate_waveform_plot(audio_path: str, output_path: str) -> str:
    """Save a waveform visualization PNG for the given audio file. Returns output_path."""
    import librosa
    import librosa.display
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y, sr = load_audio(audio_path)
    fig, ax = plt.subplots(figsize=(8, 2.5))
    librosa.display.waveshow(y, sr=sr, ax=ax, color="#4F8BF9")
    ax.set_title("Audio Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path
