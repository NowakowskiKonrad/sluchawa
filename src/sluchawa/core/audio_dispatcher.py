from pathlib import Path
import queue
import threading
from loguru import logger


class AudioDispatcher:
    def __init__(self, audio_queue: queue.Queue[Path]) -> None:
        self._audio_queue = audio_queue
        self._stop_event = threading.Event()

    def submit_audio(self, paths: list[Path]) -> None:
        if self._stop_event.is_set():
            logger.warning("Audio dispatcher stopped")
            return

        valid_paths: list[Path] = []
        for path in paths:
            valid_path = self._validate_audio_file(path)
            if valid_path is not None:
                valid_paths.append(valid_path)

            if not valid_paths:
                logger.warning("No valid audio files in batch. Nothing dispatched")
                return

            try:
                self._audio_queue.put_nowait(valid_paths)
                logger.info(f"Audio dispatched {valid_path}")
            except queue.Full:
                logger.error(f"Queue full - maxsize={self._audio_queue.maxsize}. File {valid_path} rejected")

    def _validate_audio_file(self, path: Path) -> Path | None:

        if not path.exists() or not path.is_file():
            logger.error(f"{path} does not exist or is not a file.")
            return None

        if path.suffix.lower() != ".ogg":
            logger.error(
                f"Error - unsupported format '{path.suffix}' (required .ogg): {path}"
            )
            return None

        if path.stat().st_size == 0:
            logger.error(f"Error - empty file: {path}")
            return None

        return path
