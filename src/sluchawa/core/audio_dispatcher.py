from pathlib import Path
import queue
from loguru import logger


class AudioDispatcher:
    def __init__(self, audio_queue: queue.Queue[list[Path]]) -> None:
        self._audio_queue = audio_queue

    def submit_audio(self, paths: list[Path]) -> None:
        valid_paths = [path for path in paths if self._is_valid(path)]
        for path in paths:
            valid_path = self._validate_audio_file(path)
            if valid_path is not None:
                valid_paths.append(valid_path)

            if not valid_paths:
                logger.warning("No valid audio files in batch. Nothing dispatched")
                return

        try:
            self._audio_queue.put_nowait(valid_paths)
            logger.info(f"Audio dispatched {path}")
        except queue.Full:
            logger.error(f"Queue full - maxsize={self._audio_queue.maxsize}. File {path} rejected")

    def _is_audio_file_valid(self, path: Path) -> bool:
        if not path.exists() or not path.is_file():
            logger.error(f"{path} does not exist or is not a file.")
            return False

        if path.suffix.lower() != ".ogg":
            logger.error(
                f"Error - unsupported format '{path.suffix}' (required .ogg): {path}"
            )
            return False

        if path.stat().st_size == 0:
            logger.error(f"Error - empty file: {path}")
            return False

        return True
