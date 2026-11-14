from pathlib import Path
from typing import Protocol

class AudioInputPort(Protocol):
    def submit_audio(self, paths: list[Path]) -> None: ...
