from pathlib import Path
import queue
import threading
from loguru import logger

log = logger.bind(name="AudioDispatcher")


class AudioDispatcher:
    def __init__(
        self,
        audio_queue: queue.Queue[Path],
        stop_event: threading.Event | None = None,
    ) -> None:
        self._audio_queue = audio_queue
        self._stop_event = stop_event or threading.Event()

    def submit_audio(self, paths: list[Path]) -> None:
        if self._stop_event.is_set():
            logger.warning("Audio dispatcher stopped")
            return

        for path in paths:
            valid_path = self._validate_audio_file(path)
            if valid_path is None:
                continue

            try:
                self._audio_queue.put_nowait(valid_path)
                logger.info(f"Audio dispatched {valid_path}")
            except queue.Full:
                logger.error(f"Queue full - maxsize={self._audio_queue.maxsize}. File {valid_path} rejected")


    def _validate_audio_file(self, path: Path) -> Path | None:
        resolved_path = Path(path)

        if not resolved_path.exists() or not resolved_path.is_file():
            log.error(f"Audio file {resolved_path} does not exist or is not a file.")
            return None

        if resolved_path.suffix.lower() != ".ogg":
            log.error(
                f"Error - unsupported format '{resolved_path.suffix}' (requierd .ogg): {resolved_path}"
            )
            return None

        if resolved_path.stat().st_size == 0:
            log.error(f"Error - empty file: {resolved_path}")
            return None

        return resolved_path