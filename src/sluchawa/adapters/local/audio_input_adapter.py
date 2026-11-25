import tempfile
import uuid
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from loguru import logger

from sluchawa.ports.audio_input_port import AudioInputPort

# Right now only microphones that support 48 kHz will work.
# Whisper only supports 8\12\16\24\48 kHz.
# Also, this class is used only for tests right now.
# With that in mind, there is no need to create resampler or support other mics.
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

        self._output_dir = Path(tempfile.gettempdir()) / f"sluchawa_records/{uuid.uuid4()}"
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._chunks = []
        self._stop_requested = False

        def callback(in_data: np.ndarray, frames: int, time_info, status) -> None:
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
        except (sd.PortAudioError, ValueError):
            logger.exception("Microphone capture failed")
            self._output_dir = None
            self._chunks = []
            self._close_stream()

    def stop_recording(self) -> None:
        self._stop_requested = True
        try:
            self._close_stream()
        finally:
            if self._chunks and self._output_dir is not None:
                path = self._save_track()
                self._chunks = []
                if path is not None:
                    self.add_file_to_queue(path)

    def _close_stream(self) -> None:
        stream = self._stream
        self._stream = None

        if stream is None:
            return

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

        audio = np.concatenate(self._chunks, axis=0)
        audio = audio.mean(axis=1) if audio.ndim >= 2 else audio

        path = self._output_dir / "microphone.ogg"
        try:
            sf.write(
                path,
                audio,
                RATE,
                format="OGG",
                subtype="OPUS",
                compression_level=0.9,
            )
        except (OSError, RuntimeError):
            logger.exception("Could not save recording to {}", path)
            path.unlink(missing_ok=True)
            return None

        return path

    @staticmethod
    def _resolve_device(mic_name: str | None) -> int | None:
        if mic_name is None:
            return None

        for index, device in enumerate(sd.query_devices()):
            if (
                device["max_input_channels"] > 0
                and mic_name.lower() in device["name"].lower()
            ):
                return index

        raise ValueError(f"Microphone not found: {mic_name}")

    def add_file_to_queue(self, file_path: str | Path) -> None:
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
