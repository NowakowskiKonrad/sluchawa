import tempfile
import uuid
from pathlib import Path

from loguru import logger

class LocalNoteAdapter:
    def __init__(self, output_dir: Path | str):
        self._output_dir: Path = Path(output_dir)

    def send(self, message: str) -> None:
        output_dir = self._output_dir
        if not Path.is_dir(output_dir):
            output_dir = Path(tempfile.gettempdir()) / f"sluchawa_notes/{uuid.uuid4()}"
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.warning("Can't find chosen directory. Instead file will be saved in: {}", output_dir)

        with open(output_dir / "note.txt", "w", encoding="utf-8") as file:
            file.write(message)