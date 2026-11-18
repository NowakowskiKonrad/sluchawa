import queue
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import soundcard as sc
import soundfile as sf
from loguru import logger

from sluchawa.ports.audio_input_port import AudioInputPort

TRACKS = ("mic", "system")
RATE = 16_000
STOP_TIMEOUT = 2.0
WRITER_TIMEOUT = 2.0
BLOCK_SIZE = 1024
# Bounds RAM: 512 blocks * 1024 frames * 4 B = 2 MiB per track.
# 512 blocks at 16 kHz is roughly 32 s of audio in flight.
QUEUE_BLOCKS = 512

_STOP = object()


@dataclass
class _Track:
    """One output track: a bounded queue drained by a writer thread."""

    part_path: Path
    final_path: Path
    blocks: queue.Queue = field(
        default_factory=lambda: queue.Queue(maxsize=QUEUE_BLOCKS)
    )
    failed: threading.Event = field(default_factory=threading.Event)
    writer: threading.Thread | None = None
    frames: int = 0
    dropped: int = 0
    closed: bool = False


@dataclass
class _Recording:
    """State owned by exactly one recording session."""

    output_dir: Path
    stop: threading.Event = field(default_factory=threading.Event)
    tracks: dict[str, _Track] = field(default_factory=dict)
    errors: list[Exception] = field(default_factory=list)
    threads: list[threading.Thread] = field(default_factory=list)


class LocalAudioInputAdapter:
    """Records the microphone and the system output as OGG/Opus files.

    Each track is captured and encoded in its own thread, so audio goes to disk as it arrives.
    On stop, the finished files are handed to the domain port via ``submit_audio``.
    """
    def __init__(
        self,
        domain_port: AudioInputPort,
        mic_name: str | None = None,
        speaker_name: str | None = None,
        output_root: Path | None = None,
    ):
        self._domain_port: AudioInputPort = domain_port
        self._speaker_name: str | None = speaker_name
        self._mic_name: str | None = mic_name
        self._output_root: Path = (
            Path(output_root) if output_root else Path(tempfile.gettempdir()) / "sluchawa_records"
        )

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
        output_dir = self._output_root / session_id
        output_dir.mkdir(parents=True, exist_ok=True)

        recording = _Recording(output_dir=output_dir)
        recording.tracks = {
            key: _Track(
                part_path=output_dir / f"{key}.ogg.part",
                final_path=output_dir / f"{key}.ogg",
            )
            for key in TRACKS
        }

        self.session_id = session_id
        self._recording = recording

        for key, track in recording.tracks.items():
            track.writer = threading.Thread(
                target=self._write,
                args=(recording, key),
                name=f"sluchawa-{key}-writer",
                daemon=True,
            )
            track.writer.start()

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
        track = recording.tracks[key]
        try:
            with source.recorder(
                samplerate=RATE, channels=1, blocksize=BLOCK_SIZE
            ) as recorder:
                while not recording.stop.is_set() and not track.failed.is_set():
                    block = recorder.record(numframes=BLOCK_SIZE)
                    if block.ndim > 1:
                        block = block.mean(axis=1)
                    try:
                        track.blocks.put_nowait(block)
                    except queue.Full:
                        # Never block the mic on a slow disk or encoder.
                        track.dropped += 1
        except Exception as ex:  # noqa: BLE001 - never let a capture thread die silently
            logger.error(f"{key} capture failed: {ex}")
            recording.errors.append(ex)

    def _write(self, recording: _Recording, key: str) -> None:
        track = recording.tracks[key]
        try:
            with sf.SoundFile(
                track.part_path,
                mode="w",
                samplerate=RATE,
                channels=1,
                format="OGG",
                subtype="OPUS",
                compression_level=0.9,
            ) as out:
                while (item := track.blocks.get()) is not _STOP:
                    out.write(item)
                track.frames = out.frames
        except Exception as ex:  # noqa: BLE001 - writer must not die silently
            logger.error(f"{key} writer failed: {ex}")
            recording.errors.append(ex)
            track.failed.set()
            return

        if track.frames:
            track.part_path.replace(track.final_path)
            track.closed = True
        else:
            track.part_path.unlink(missing_ok=True)

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
                f"audio captured from now on is discarded: {stuck}"
            )

        for key, track in recording.tracks.items():
            self._stop_writer(key, track)

        self._add_audio_to_queue(
            recording,
            error=recording.errors[0] if recording.errors else None,
        )

    def _stop_writer(self, key: str, track: _Track) -> None:
        writer = track.writer
        if writer is None:
            return

        if writer.is_alive() and not self._enqueue_stop(track):
            logger.warning(
                f"{key} writer did not drain its queue, "
                f"keeping {track.part_path.name} for salvage"
            )

        writer.join(WRITER_TIMEOUT)
        if writer.is_alive():
            logger.warning(
                f"{key} writer did not stop in time, "
                f"keeping {track.part_path.name} for salvage"
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
            logger.error(f"File {audio.name} format must be 'OGG'")
            return

        if audio.subtype != "OPUS":
            logger.error(f"File {audio.name} subtype must be 'OPUS'")
            return

        self._domain_port.submit_audio([path])

    def _add_audio_to_queue(
        self, recording: _Recording, error: Exception | None
    ) -> None:
        if error:
            logger.error(f"There was an error while recording: {error}")

        paths: list[Path] = []
        for key in TRACKS:
            track = recording.tracks.get(key)
            if track is None:
                continue

            if track.dropped:
                logger.warning(
                    f"{key}: dropped {track.dropped} blocks "
                    f"(~{track.dropped * BLOCK_SIZE / RATE:.1f}s), "
                    f"recording is not continuous"
                )

            if not track.closed or not track.frames:
                continue

            paths.append(track.final_path)

        if paths:
            self._domain_port.submit_audio(paths)

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

    def _enqueue_stop(self, track: _Track, timeout: float = 1.0) -> bool:
        """Push the sentinel after the queued blocks, within a timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                track.blocks.put(_STOP, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False
