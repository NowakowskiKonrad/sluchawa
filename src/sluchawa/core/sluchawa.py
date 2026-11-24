import queue
import threading
from pathlib import Path
from loguru import logger

class Sluchawa:
    def __init__(self, audio_queue: queue.Queue[list[Path]]) -> None:
        self._audio_queue = audio_queue
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

    def start(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            logger.warning("Sluchawa worker thread is already running.")
            return

        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._run,
            name="SluchawaDomainWorker",
            daemon=True
        )
        self._worker_thread.start()
        logger.info("Sluchawa domain worker thread started.")

    def stop(self) -> None:
        if not self._worker_thread or not self._worker_thread.is_alive():
            return

        logger.info("Stopping Sluchawa domain worker thread...")
        self._stop_event.set()
        self._worker_thread.join()
        logger.info("Sluchawa domain worker thread stopped.")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                audio_paths = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self._process_session(audio_paths)
            except Exception:
                logger.exception(f"Error while processing audio session.")
            finally:
                self._audio_queue.task_done()

    def _process_session(self, audio_paths: list[Path]) -> None:
        logger.info(f"Processing {len(audio_paths)} audio file(s)...")
        transcript = self._transcribe(audio_paths)
        note = self._generate_note(transcript)
        self._send_note(note)

    # dalej są placeholdery na przyszłe moduły

    # TODO: podpiąć whispera
    def _transcribe(self, audio_paths: list[Path]) -> str:
        logger.debug(f"[placeholder] Transcribing files: {audio_paths}")
        return ""

    # TODO: podpiąć LLMPort Igora
    def _generate_note(self, transcript: str) -> str:
        logger.debug("[placeholder] Generating note from transcript.")
        return ""

    # TODO: podpiąć message port
    def _send_note(self, note: str) -> None:
        logger.debug("[placeholder] Sending generated note.")
