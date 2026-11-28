import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf
from loguru import logger

if TYPE_CHECKING:
    import sounddevice as sd

from sluchawa.ports.audio_input_port import AudioInputPort

# Only microphones that support 48 kHz will work because of opus.
# This class is only for simple tests, so there is no need to support other mics.
RATE = 48_000
BLOCK_SIZE = 1024


class LocalAudioInputAdapter:
    def __init__(self, domain_port: AudioInputPort, mic_name: str | None = None):
        self._domain_port: AudioInputPort = domain_port
        self._mic_name: str | None = mic_name

        self._output_dir: Path | None = None
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._stop_requested = False

    def start_recording(self) -> None:
        if self._stream is not None:
            logger.warning("Recording is already in progress")
            return

        import sounddevice as sd

        self._output_dir = Path("../sluchawa_recordings")

        self._chunks = []
        self._stop_requested = False

        def callback(in_data: np.ndarray, frames: int, time_info, status) -> None:
            if status:
                logger.warning("Audio input status: {}", status)
            if not self._stop_requested:
                self._chunks.append(in_data.copy())

        try:
            self._stream = sd.InputStream(
                device=self._resolve_device(self._mic_name),
                samplerate=RATE,
                channels=1,
                dtype="float32",
                blocksize=BLOCK_SIZE,
                callback=callback,
            )
            self._stream.start()
        except ValueError as ex:
            logger.error("Recording was not started: {}", ex)
            self._abort_start()
        except sd.PortAudioError as ex:
            logger.error(
                "Microphone capture failed (device must support {} Hz mono): {}",
                RATE,
                ex,
            )
            self._abort_start()

    def stop_recording(self) -> None:
        self._stop_requested = True
        try:
            self._close_stream()
        finally:
            try:
                if self._chunks and self._output_dir is not None:
                    path = self._save_track()
                    if path is not None:
                        self.add_file_to_queue(path)
            finally:
                self._chunks = []
                self._output_dir = None

    def add_file_to_queue(self, file_path: str | Path) -> None:
        """
        Validate an audio file and submit it to the domain port.

        The file must exist and be OGG/OPUS encoded.
        Invalid paths or formats are logged.

        Args:
            file_path: Path to the audio file to enqueue.
        """
        path = Path(file_path)

        if not path.is_file():
            logger.error("Element in path ({}) is not a file", path)
            return

        try:
            audio = sf.info(path)
        except (sf.LibsndfileError, OSError) as ex:
            logger.error("File {} is not a valid audio file: {}", path, ex)
            return

        if audio.format != "OGG":
            logger.error("File {} format must be 'OGG'", path)
            return

        if audio.subtype != "OPUS":
            logger.error("File {} subtype must be 'OPUS'", path)
            return

        self._domain_port.submit_audio([path])

    def _close_stream(self) -> None:
        stream = self._stream
        self._stream = None

        if stream is None:
            return

        import sounddevice as sd

        try:
            stream.stop()
        except sd.PortAudioError:
            logger.exception("Could not stop the microphone stream")
        finally:
            try:
                stream.close()
            except sd.PortAudioError:
                logger.exception("Could not close the microphone stream")

    def _save_track(self) -> Path | None:
        if self._output_dir is None:
            logger.error("Cannot save a track before starting recording.")
            return None

        if not self._chunks:
            logger.error("Cannot save a track without recorded audio")
            return None

        path = self._output_dir / time.strftime("%Y-%m-%d_%H-%M-%S_mic.ogg")
        try:
            audio = np.concatenate(self._chunks, axis=0)
            audio = audio.mean(axis=1, dtype=np.float32) if audio.ndim >= 2 else audio

            self._output_dir.mkdir(parents=True, exist_ok=True)
            sf.write(
                path,
                audio,
                RATE,
                format="OGG",
                subtype="OPUS",
                compression_level=0.9,
            )
        except (OSError, RuntimeError, ValueError):
            logger.exception("Could not save recording to {}", path)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.exception("Could not remove {}", path)
            return None

        return path

    def _abort_start(self) -> None:
        self._output_dir = None
        self._chunks = []
        self._close_stream()

    @staticmethod
    def _resolve_device(mic_name: str | None) -> int | None:
        if not mic_name:
            return None

        import sounddevice as sd

        input_devices: list[str] = []

        for index, device in enumerate(sd.query_devices()):
            if device["max_input_channels"] <= 0:
                continue

            input_devices.append(device["name"])
            if mic_name.lower() in device["name"].lower():
                return index

        available = ", ".join(input_devices) or "none"
        raise ValueError(
            f"microphone {mic_name!r} not found (available input devices: {available})"
        )
