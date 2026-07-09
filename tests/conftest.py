"""
tests/conftest.py

Shared pytest fixtures, including a synthetically generated short WAV
file so audio-processing tests don't depend on external assets.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def synthetic_wav(tmp_path_factory) -> str:
    """Generate a 2-second 440Hz sine tone WAV file for testing audio pipelines."""
    import soundfile as sf

    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)

    out_dir = tmp_path_factory.mktemp("audio")
    out_path = out_dir / "tone.wav"
    sf.write(str(out_path), tone, sr)
    return str(out_path)
