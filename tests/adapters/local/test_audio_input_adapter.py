from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

from sluchawa.adapters.local.audio_input_adapter import LocalAudioInputAdapter


def _make_adapter() -> LocalAudioInputAdapter:
    return LocalAudioInputAdapter(domain_port=MagicMock())


def test_save_track_writes_ogg_opus(tmp_path):
    # Given
    adapter = _make_adapter()
    adapter._output_dir = tmp_path
    adapter._chunks = [np.zeros((1024, 1), dtype=np.float32)]

    # When
    path = adapter._save_track()

    # Then
    info = sf.info(path)
    assert path.suffix == ".ogg"
    assert info.format == "OGG"
    assert info.subtype == "OPUS"
    assert info.samplerate == 48_000
    assert info.frames == 1024


def test_start_recording_starts_stream_and_returns(tmp_path):
    # Given
    adapter = _make_adapter()
    stream = MagicMock()

    # When
    with patch(
        "sluchawa.adapters.local.audio_input_adapter.sd.InputStream",
        return_value=stream,
    ) as input_stream:
        adapter.start_recording()

    # Then
    input_stream.assert_called_once()
    stream.start.assert_called_once()
    assert adapter._stream is stream


def test_start_recording_ignores_second_call():
    # Given
    adapter = _make_adapter()
    adapter._stream = MagicMock()

    # When
    with patch(
        "sluchawa.adapters.local.audio_input_adapter.sd.InputStream",
    ) as input_stream:
        adapter.start_recording()

    # Then
    input_stream.assert_not_called()


def test_stop_recording_stops_stream_and_submits_audio(tmp_path):
    # Given
    adapter = _make_adapter()
    adapter._output_dir = tmp_path
    adapter._chunks = [np.zeros((1024, 1), dtype=np.float32)]
    stream = MagicMock()
    adapter._stream = stream

    # When
    adapter.stop_recording()

    # Then
    stream.stop.assert_called_once()
    stream.close.assert_called_once()
    assert adapter._stream is None

    adapter._domain_port.submit_audio.assert_called_once()
    submitted_paths = adapter._domain_port.submit_audio.call_args.args[0]
    assert len(submitted_paths) == 1
    assert submitted_paths[0].suffix == ".ogg"


def test_stop_recording_without_audio_does_not_submit(tmp_path):
    # Given
    adapter = _make_adapter()
    adapter._output_dir = tmp_path

    # When
    adapter.stop_recording()

    # Then
    adapter._domain_port.submit_audio.assert_not_called()
