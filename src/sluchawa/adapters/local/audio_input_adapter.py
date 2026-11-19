import queue
import shutil
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
# Writers re-check their shutdown flag at least this often while idle.
WRITER_POLL = 0.1
BLOCK_SIZE = 1024
# Bounds RAM: 512 blocks * 1024 frames * 4 B = 2 MiB per track.
# 512 blocks at 16 kHz is roughly 32 s of audio in flight.
QUEUE_BLOCKS = 512
# Session directories in the default temp root older than this are swept on start.
RETENTION_SECONDS = 24 * 60 * 60


@dataclass
class _Track:
    """One output track: a bounded queue drained by a writer thread."""

    part_path: Path
    final_path: Path
    blocks: queue.Queue = field(
        default_factory=lambda: queue.Queue(maxsize=QUEUE_BLOCKS)
    )
    shutdown: threading.Event = field(default_factory=threading.Event)
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

    System capture uses the PulseAudio monitor source, so this adapter targets Linux.
    Recordings are written to a private session directory and are not deleted
    automatically: callers own the files after ``submit_audio``, and sessions in the
    default temp root are swept after ``RETENTION_SECONDS``.
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
        self._owns_output_root: bool = output_root is None
        self._output_root: Path = (
            Path(output_root) if output_root else Path(tempfile.gettempdir()) / "sluchawa_records"
        )
        self._lock = threading.RLock()

        self.session_id: str | None = None
        self._recording: _Recording | None = None

    def start_recording(self) -> bool:
        """Start capturing every source that is available.

        Returns:
            True if at least one track started, False otherwise.
        """
        with self._lock:
            if self._recording is not None:
                logger.error("Only one local recording can be run at the same time.")
                return False

            sources: dict[str, object] = {}
            errors: list[Exception] = []
            for key, resolver in (
                ("mic", self._resolve_mic),
                ("system", self._resolve_system),
            ):
                try:
                    sources[key] = resolver()
                except Exception as ex:  # noqa: BLE001 - one bad device must not kill the other
                    logger.error(f"{key} source unavailable: {ex}")
                    errors.append(ex)

            if not sources:
                logger.error("No audio source is available, recording not started.")
                return False

            session_id = str(uuid.uuid4())
            output_dir = self._output_root / session_id
            try:
                self._secure_root()
                self._cleanup_stale_sessions()
                output_dir.mkdir(mode=0o700)
            except OSError as ex:
                logger.error(f"Could not create recording directory {output_dir}: {ex}")
                return False

            recording = _Recording(output_dir=output_dir)
            recording.errors.extend(errors)
            recording.tracks = {
                key: _Track(
                    part_path=output_dir / f"{key}.ogg.part",
                    final_path=output_dir / f"{key}.ogg",
                )
                for key in TRACKS
            }

            self.session_id = session_id
            self._recording = recording

            try:
                for key in sources:
                    track = recording.tracks[key]
                    track.writer = threading.Thread(
                        target=self._write,
                        args=(recording, key),
                        name=f"sluchawa-{key}-writer",
                        daemon=True,
                    )
                    track.writer.start()

                for key, source in sources.items():
                    thread = threading.Thread(
                        target=self._capture,
                        args=(source, key, recording),
                        name=f"sluchawa-{key}",
                        daemon=True,
                    )
                    thread.start()
                    recording.threads.append(thread)
            except Exception as ex:  # noqa: BLE001 - roll back a half-started session
                logger.error(f"Could not start recording threads: {ex}")
                self._abort_recording(recording)
                self._recording = None
                self.session_id = None
                return False

            return True

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
                while True:
                    try:
                        item = track.blocks.get(timeout=WRITER_POLL)
                    except queue.Empty:
                        if track.shutdown.is_set():
                            break
                        continue
                    out.write(item)
                track.frames = out.frames
        except Exception as ex:  # noqa: BLE001 - writer must not die silently
            logger.error(f"{key} writer failed: {ex}")
            recording.errors.append(ex)
            track.failed.set()
            return

        try:
            if track.frames:
                track.part_path.replace(track.final_path)
                track.closed = True
            else:
                track.part_path.unlink(missing_ok=True)
        except OSError as ex:
            logger.error(
                f"{key} could not finalize {track.part_path.name}: {ex}; "
                f"file kept for salvage"
            )
            recording.errors.append(ex)
            track.failed.set()

    def stop_recording(self) -> None:
        with self._lock:
            recording, self._recording = self._recording, None
            if recording is None:
                return

            self.session_id = None
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

            for track in recording.tracks.values():
                track.shutdown.set()

            deadline = time.monotonic() + WRITER_TIMEOUT
            for key, track in recording.tracks.items():
                self._stop_writer(key, track, deadline)

            self._add_audio_to_queue(
                recording,
                error=recording.errors[0] if recording.errors else None,
            )

    def _abort_recording(self, recording: _Recording) -> None:
        """Stop and discard a session whose threads failed to start."""
        recording.stop.set()
        for track in recording.tracks.values():
            track.shutdown.set()

        for thread in recording.threads:
            thread.join(STOP_TIMEOUT)
        for track in recording.tracks.values():
            if track.writer is not None:
                track.writer.join(WRITER_TIMEOUT)

        shutil.rmtree(recording.output_dir, ignore_errors=True)

    def _secure_root(self) -> None:
        """Create the output root and keep the default temp root private."""
        self._output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not self._owns_output_root:
            return
        try:
            self._output_root.chmod(0o700)
        except OSError as ex:
            logger.warning(
                f"Could not restrict permissions on {self._output_root}: {ex}"
            )

    def _cleanup_stale_sessions(self) -> None:
        """Remove old session dirs from the default temp root only."""
        if not self._owns_output_root or not self._output_root.is_dir():
            return

        cutoff = time.time() - RETENTION_SECONDS
        for entry in self._output_root.iterdir():
            try:
                if entry.is_symlink() or not entry.is_dir():
                    continue
                if entry.stat().st_mtime >= cutoff:
                    continue
            except OSError as ex:
                logger.warning(f"Could not inspect {entry}: {ex}")
                continue

            logger.info(f"Removing stale recording session {entry.name}")
            shutil.rmtree(entry, ignore_errors=True)

    def _stop_writer(self, key: str, track: _Track, deadline: float) -> None:
        writer = track.writer
        if writer is None:
            return

        writer.join(max(0.0, deadline - time.monotonic()))
        if writer.is_alive():
            logger.warning(
                f"{key} writer is still draining after {WRITER_TIMEOUT}s; "
                f"it will keep writing {track.final_path.name} in the background, "
                f"but this track will not be submitted"
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
        source = (
            sc.get_microphone(self._mic_name)
            if self._mic_name
            else sc.default_microphone()
        )
        if source is None:
            raise RuntimeError("No microphone is available")
        return source

    def _resolve_system(self):
        speaker = (
            sc.get_speaker(self._speaker_name)
            if self._speaker_name
            else sc.default_speaker()
        )
        if speaker is None:
            raise RuntimeError("No output device is available for system capture")
        return sc.get_microphone(f"{speaker.id}.monitor", include_loopback=True)
