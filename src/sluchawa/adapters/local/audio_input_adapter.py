import threading
import uuid

import numpy as np
import soundcard as sc
import soundfile as sf

from pathlib import Path
from loguru import logger

from sluchawa.ports.audio_input_port import AudioInputPort

TRACKS = ("mic", "system")
BITRATE = 16_000

class LocalAudioInputAdapter:
    def __init__(self, mic_name: str | None = None, speaker_name: str | None = None):
        # self._domain_port: AudioInputPort = domain_port
        self._speaker_name: str | None = speaker_name
        self._mic_name: str | None = mic_name

        self.session_id: str | None = None
        self._output_dir: Path | None = None
        self._stop = threading.Event()
        self._chunks: dict[str, list[np.ndarray]] = {}
        self._errors: list[Exception] = []
        self._threads: list[threading.Thread] = []


    def start_recording(self):
        self.session_id = str(uuid.uuid4())
        self._output_dir = Path(f"/tmp/sluchawa_records/{self.session_id}")
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._stop.clear()
        self._chunks = { key: [] for key in TRACKS}
        self._errors = []

        sources = {
            "mic": self._resolve_mic(),
            "system": self._resolve_system()
        }

        self._threads = [
            threading.Thread(
                target=self._capture,
                args=(sources[key], key),
                name=f"sluchawa-{key}",
                daemon=True,
            ) for key in TRACKS
        ]

        for thread in self._threads:
            thread.start()


    def _capture(self, source, key: str):
        try:
            with source.recorder(samplerate=BITRATE) as recorder:
                while not self._stop.is_set():
                    self._chunks[key].append(recorder.record(numframes=1024))
        except Exception as ex:
            logger.error(f"{key} capture failed: {ex}")
            self._errors.append(ex)


    def stop_recording(self):
        if not self._threads:
            return

        self._stop.set()
        for thread in self._threads:
            thread.join()

        self._threads = []
        self.add_audio_to_queue(
            error=self._errors[0] if self._errors else None
        )

    def add_audio_to_queue(self, error: Exception | None):
        if error:
            logger.error(f"There was an error while recording: {error}")

        if not self._output_dir:
            return

        paths = [
            path for key in TRACKS
            if (path := self._save_track(key)) is not None
        ]

        if paths:
            # self._domain_port.submit_audio(paths)
            pass

    def _save_track(self, key: str) -> Path | None:
        chunks = self._chunks.get(key, [])
        if not chunks or self._output_dir is None:
            return None

        audio = np.concatenate(chunks, axis=0)
        audio = audio.mean(axis=1)

        path = self._output_dir / f"{key}.ogg"
        sf.write(
            path,
            audio,
            BITRATE,
            format="OGG",
            subtype="OPUS",
            compression_level=0.9
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

if __name__ == "__main__":
    import time

    adapter = LocalAudioInputAdapter(mic_name="alsa_input.pci-0000_00_1b.0.analog-stereo", speaker_name=None)
    adapter.start_recording()
    print("Recording for 10 seconds...")
    time.sleep(10)
    adapter.stop_recording()
    print(f"Saved to: {adapter._output_dir}")
