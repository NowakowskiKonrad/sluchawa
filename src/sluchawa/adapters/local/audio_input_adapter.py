import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundcard as sc
import soundfile as sf
from loguru import logger

from sluchawa.ports.audio_input_port import AudioInputPort

TRACKS = ("mic", "system")
RATE = 16_000
STOP_TIMEOUT = 2.0
BLOCK_SIZE = 1024


@dataclass
class _Recording:
    """State owned by exactly one recording session."""

    output_dir: Path
    stop: threading.Event = field(default_factory=threading.Event)
    chunks: dict[str, list[np.ndarray]] = field(
        default_factory=lambda: {key: [] for key in TRACKS}
    )
    errors: list[Exception] = field(default_factory=list)
    threads: list[threading.Thread] = field(default_factory=list)

class LocalAudioInputAdapter:
    def __init__(self, domain_port: AudioInputPort, mic_name: str | None = None, speaker_name: str | None = None):
        self._domain_port: AudioInputPort = domain_port
        self._speaker_name: str | None = speaker_name
        self._mic_name: str | None = mic_name

        self.session_id: str | None = None
        self._recording: _Recording | None = None


    def start_recording(self) -> None:
        if self._recording is not None:
            logger.error("Only one local recording can be run at the same time.")
            return

        sources = {
            "mic": self._resolve_mic(),
            "system": self._resolve_system(),
        }

        session_id = str(uuid.uuid4())
        output_dir = Path(tempfile.gettempdir()) / "sluchawa_records" / session_id
        output_dir.mkdir(parents=True, exist_ok=True)

        recording = _Recording(output_dir=output_dir)
        self.session_id = session_id
        self._recording = recording

        recording.threads = [
            threading.Thread(
                target=self._capture,
                args=(sources[key], key, recording),
                name=f"sluchawa-{key}",
                daemon=True,
            )
            for key in TRACKS
        ]

        for thread in recording.threads:
            thread.start()


    def _capture(self, source, key: str, recording: _Recording) -> None:
        try:
            with source.recorder(samplerate=RATE,channels=1,blocksize=BLOCK_SIZE) as recorder:
                while not recording.stop.is_set():
                    recording.chunks[key].append(
                        recorder.record(numframes=BLOCK_SIZE)
                    )
        except Exception as ex:  # noqa: BLE001 - never let a capture thread die silently
            logger.error(f"{key} capture failed: {ex}")
            recording.errors.append(ex)


    def stop_recording(self) -> None:
        recording, self._recording = self._recording, None
        if recording is None:
            return

        recording.stop.set()
        deadline = time.monotonic() + STOP_TIMEOUT
        for thread in recording.threads:
            thread.join(max(0.0, deadline - time.monotonic()))

        stuck = [thread.name for thread in recording.threads if thread.is_alive()]
        if stuck:
            logger.warning(
                f"Capture threads did not stop in time, "
                f"their last audio is discarded: {stuck}"
            )

        self._add_audio_to_queue(
            recording,
            error=recording.errors[0] if recording.errors else None,
        )

    def add_file_to_queue(self, file_path: str | Path) -> None:
        path = Path(file_path)

        if not path.is_file():
            logger.error(f"Element in path ({path}) is not a file")
            return

        try:
            audio = sf.info(path)
        except (RuntimeError, sf.LibsndfileError) as ex:
            logger.error(f"Cannot read {path}: {ex}")
            return

        if audio.format != "OGG":
            logger.error(f"File {audio.name} format must be \'OGG\'")
            return

        if audio.subtype != "OPUS":
            logger.error(f"File {audio.name} subtype must be \'OPUS\'")
            return

        self._domain_port.submit_audio([path])

    def _add_audio_to_queue(
        self, recording: _Recording, error: Exception | None
    ) -> None:
        if error:
            logger.error(f"There was an error while recording: {error}")

        paths: list[Path] = []
        for key in TRACKS:
            try:
                path = self._save_track(recording, key)
            except Exception as ex:  # noqa: BLE001
                logger.error(f"Failed to save {key}: {ex}")
                continue
            if path is not None:
                paths.append(path)

        if paths:
            self._domain_port.submit_audio(paths)


    def _save_track(self, recording: _Recording, key: str) -> Path | None:
        # Snapshot: a timed-out capture thread may still append to this list.
        chunks = list(recording.chunks.get(key, []))
        if not chunks:
            return None

        audio = np.concatenate(chunks, axis=0)
        audio = audio.mean(axis=1) if audio.ndim > 1 else audio

        if audio.size == 0:
            return None

        path = recording.output_dir / f"{key}.ogg"
        sf.write(
            path,
            audio,
            RATE,
            format="OGG",
            subtype="OPUS",
            compression_level=0.85,
        )
        return path

    def _resolve_mic(self):
        if self._mic_name:
            return sc.get_microphone(self._mic_name)
        return sc.default_microphone()

    def _resolve_system(self):
        speaker = (
            sc.get_speaker(self._speaker_name)
            if self._speaker_name
            else sc.default_speaker()
        )
        return sc.get_microphone(f"{speaker.id}.monitor", include_loopback=True)
