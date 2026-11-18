import threading
import time
from typing import Self
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

from sluchawa.adapters.local.audio_input_adapter import (
    BLOCK_SIZE,
    RATE,
    LocalAudioInputAdapter,
    _Recording,
)

MODULE_PATH = "sluchawa.adapters.local.audio_input_adapter"


class FakeRecorder:
    """Returns ``count`` blocks, then parks until the test releases it."""

    def __init__(self, count: int = 4, fail: bool = False) -> None:
        self.count = count
        self.fail = fail
        self.release = threading.Event()
        self.entered = threading.Event()
        self.calls = 0

    def __enter__(self) -> Self:
        if self.fail:
            raise OSError("device disappeared")
        self.entered.set()
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def record(self, numframes: int) -> np.ndarray:
        if self.calls >= self.count:
            self.release.wait(timeout=5)
        self.calls += 1
        time.sleep(0.0005)
        return np.zeros((numframes, 1), dtype=np.float32)


class FakeSource:
    def __init__(self, **recorder_kwargs: object) -> None:
        self._recorder_kwargs = recorder_kwargs
        self.recorder_kwargs: dict | None = None
        self.recorders: list[FakeRecorder] = []

    @property
    def recorder_obj(self) -> FakeRecorder | None:
        return self.recorders[-1] if self.recorders else None

    def recorder(self, **kwargs: object) -> FakeRecorder:
        self.recorder_kwargs = kwargs
        recorder = FakeRecorder(**self._recorder_kwargs)
        self.recorders.append(recorder)
        return recorder


def wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


@pytest.fixture
def port() -> MagicMock:
    return MagicMock()


@pytest.fixture
def adapter(port, tmp_path, mocker) -> LocalAudioInputAdapter:
    mocker.patch(f"{MODULE_PATH}.tempfile.gettempdir", return_value=str(tmp_path))
    return LocalAudioInputAdapter(domain_port=port)


@pytest.fixture
def wire(adapter):
    created: list[FakeSource] = []

    def _wire(mic=None, system=None):
        mic_source = FakeSource(**(mic or {}))
        system_source = FakeSource(**(system or {}))
        adapter._resolve_mic = lambda: mic_source
        adapter._resolve_system = lambda: system_source
        created.extend([mic_source, system_source])
        return mic_source, system_source

    yield _wire

    for source in created:
        for recorder in source.recorders:
            recorder.release.set()
    adapter.stop_recording()


def test_save_track_writes_ogg_opus(adapter, tmp_path):
    # Given
    recording = _Recording(output_dir=tmp_path)
    recording.chunks["mic"] = [np.zeros((1024, 1), dtype=np.float32)]

    # When
    path = adapter._save_track(recording, "mic")

    # Then
    info = sf.info(path)
    assert path.suffix == ".ogg"
    assert (info.format, info.subtype) == ("OGG", "OPUS")
    assert (info.samplerate, info.channels, info.frames) == (RATE, 1, 1024)


def test_capture_uses_16khz_mono_and_block_size(adapter, wire):
    # Given
    mic, system = wire()

    # When
    adapter.start_recording()
    assert wait_for(lambda: mic.recorder_obj is not None)
    assert wait_for(lambda: system.recorder_obj is not None)

    # Then
    for source in (mic, system):
        assert source.recorder_kwargs == {
            "samplerate": RATE,
            "channels": 1,
            "blocksize": BLOCK_SIZE,
        }


def test_stop_recording_submits_both_tracks(adapter, port, wire):
    # Given
    mic, system = wire({"count": 3}, {"count": 3})
    adapter.start_recording()
    session_dir = adapter._recording.output_dir
    assert wait_for(lambda: mic.recorder_obj.calls >= 3)
    assert wait_for(lambda: system.recorder_obj.calls >= 3)

    # When
    mic.recorder_obj.release.set()
    system.recorder_obj.release.set()
    adapter.stop_recording()

    # Then
    port.submit_audio.assert_called_once()
    paths = port.submit_audio.call_args.args[0]
    assert [path.name for path in paths] == ["mic.ogg", "system.ogg"]
    assert all(path.parent == session_dir for path in paths)
    assert all(sf.info(path).subtype == "OPUS" for path in paths)
    assert adapter._recording is None


def test_failed_capture_still_submits_healthy_track(adapter, port, wire, mocker):
    # Given
    error_log = mocker.patch(f"{MODULE_PATH}.logger.error")
    mic, _system = wire({"count": 2}, {"fail": True})

    # When
    adapter.start_recording()
    assert wait_for(lambda: mic.recorder_obj.calls >= 2)
    mic.recorder_obj.release.set()
    adapter.stop_recording()

    # Then
    paths = port.submit_audio.call_args.args[0]
    assert [path.name for path in paths] == ["mic.ogg"]
    assert "system capture failed" in str(error_log.call_args_list)


def test_no_audio_does_not_submit(adapter, port, tmp_path):
    # When
    adapter._add_audio_to_queue(_Recording(output_dir=tmp_path), error=None)

    # Then
    port.submit_audio.assert_not_called()


def test_stop_recording_before_start_is_noop(adapter, port):
    # When
    adapter.stop_recording()

    # Then
    port.submit_audio.assert_not_called()


def test_start_recording_while_recording_is_rejected(adapter, wire):
    # Given
    wire({"count": 1}, {"count": 1})
    adapter.start_recording()
    first_session = adapter.session_id
    first_recording = adapter._recording

    # When
    adapter.start_recording()

    # Then
    assert adapter.session_id == first_session
    assert adapter._recording is first_recording


def test_stuck_capture_thread_is_detached(adapter, port, wire, mocker):
    # Given
    mocker.patch(f"{MODULE_PATH}.STOP_TIMEOUT", 0.05)
    warning = mocker.patch(f"{MODULE_PATH}.logger.warning")
    mic, system = wire({"count": 0}, {"count": 0})
    adapter.start_recording()
    first_recording = adapter._recording
    assert wait_for(lambda: mic.recorder_obj.entered.is_set())
    assert wait_for(lambda: system.recorder_obj.entered.is_set())

    # When
    adapter.stop_recording()

    # Then
    assert adapter._recording is None
    warning.assert_called_once()
    assert "sluchawa-system" in warning.call_args.args[0]
    port.submit_audio.assert_not_called()

    # A new session must work while the old threads are still parked.
    adapter.start_recording()
    second_recording = adapter._recording
    assert second_recording is not first_recording

    # Late writes from the abandoned session must not reach the new one.
    for recorder in (mic.recorders[0], system.recorders[0]):
        recorder.release.set()
    assert wait_for(lambda: len(first_recording.chunks["mic"]) == 1)
    assert second_recording.chunks["mic"] == []


def test_add_file_to_queue_accepts_ogg_opus(adapter, port, tmp_path):
    # Given
    path = tmp_path / "existing.ogg"
    sf.write(
        path,
        np.zeros(1024, dtype=np.float32),
        RATE,
        format="OGG",
        subtype="OPUS",
    )

    # When
    adapter.add_file_to_queue(path)

    # Then
    port.submit_audio.assert_called_once_with([path])


def test_add_file_to_queue_rejects_invalid_files(adapter, port, tmp_path, mocker):
    # Given
    wrong_format = tmp_path / "wrong.wav"
    sf.write(wrong_format, np.zeros(1024, dtype=np.float32), RATE, format="WAV")
    unreadable = tmp_path / "broken.ogg"
    unreadable.write_text("this is not audio")
    error_log = mocker.patch(f"{MODULE_PATH}.logger.error")

    # When
    adapter.add_file_to_queue(wrong_format)
    adapter.add_file_to_queue(unreadable)

    # Then
    port.submit_audio.assert_not_called()
    assert "format must be" in str(error_log.call_args_list)
    assert "Cannot read" in str(error_log.call_args_list)