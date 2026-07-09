"""
tests/test_transcription.py

Tests for backend.transcription. The Whisper model is mocked so tests
run fast and offline (no multi-hundred-MB model download needed in CI).
"""
from backend.transcription import Transcriber, TranscriptionResult


def test_transcription_result_to_dict():
    result = TranscriptionResult(
        text="hello world", language="en", duration=1.5,
        segments=[{"start": 0.0, "end": 1.5, "text": "hello world"}],
        word_count=2,
    )
    data = result.to_dict()
    assert data["text"] == "hello world"
    assert data["word_count"] == 2
    assert data["duration"] == 1.5


def test_transcriber_transcribe_with_mocked_model(mocker):
    transcriber = Transcriber(model_size="base")

    fake_model = mocker.Mock()
    fake_model.transcribe.return_value = {
        "text": " Photosynthesis converts sunlight into energy. ",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 2.0, "text": " Photosynthesis converts sunlight "},
            {"start": 2.0, "end": 3.5, "text": "into energy. "},
        ],
    }
    mocker.patch.object(Transcriber, "model", new_callable=mocker.PropertyMock,
                         return_value=fake_model)

    result = transcriber.transcribe("fake_path.wav")

    assert result.text == "Photosynthesis converts sunlight into energy."
    assert result.language == "en"
    assert result.duration == 3.5
    assert result.word_count == 5
    assert len(result.segments) == 2
