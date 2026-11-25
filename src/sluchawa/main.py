from pathlib import Path
import queue
import threading
import time
from loguru import logger

from ports.audio_input_port import AudioInputPort
from core.audio_dispatcher import AudioDispatcher

"""
def main():
    
    Create domain audio dispatcher
    Create event loop
    Create discord audio input adapter (passing the loop and dispatcher)
    Create discord notification adapter (passing the loop)
    Create domain audio processing (passing notificaion adapter)
    Run audio processing in separate thread

    Wait for input
    """

class LocalAudioAdapter:
    def __init__(self, input_port: AudioInputPort, stop_event: threading.Event) -> None:
        self._input_port = input_port
        self._stop_event = stop_event

    def run(self) -> None:
        logger.info("Starting local audio adapter")
        test_file = Path("recordings/test_sample.ogg")
        if not self._stop_event.is_set():
            self._input_port.submit_audio([test_file])

def main() -> None:
    stop_event = threading.Event()
    audio_queue: queue.Queue[Path] = queue.Queue(maxsize=100)

    audio_input_port: AudioInputPort = AudioDispatcher(
        audio_queue=audio_queue,
        stop_event=stop_event,
    )

    local_adapter = LocalAudioAdapter(
        input_port=audio_input_port,
        stop_event=stop_event,
    )

    local_input_thread = threading.Thread(
        target=local_adapter.run,
        name="LocalInputThread",
        daemon=True,
    )
    local_input_thread.start()

    try:
        while local_input_thread.is_alive():
            time.sleep(0.2)
    except KeyboardInterrupt:
        logger.info("Closing local audio adapter")
        stop_event.set()
    finally:
        local_input_thread.join(timeout=2.0)
        logger.info("Local audio adapter stopped")


if __name__ == "__main__":
    main()