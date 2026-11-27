import tempfile
import time
from pathlib import Path

from loguru import logger

TEMP_PATH = Path(tempfile.gettempdir()) / "sluchawa_notes"


class LocalNoteAdapter:
    def __init__(self, output_dir: Path | str = Path("../sluchawa_notes")):
        self._output_dir: Path = Path(output_dir)

    def send(self, message: str) -> None:
        output_dir = self._resolve_output_dir()

        if output_dir is None:
            return

        file_name = time.strftime("%Y-%m-%d_%H-%M-%S_note.md", time.localtime())

        try:
            (output_dir / file_name).write_text(message, encoding="utf-8")
            logger.success(
                "Local note has been sent successfully at {}", output_dir / file_name
            )
        except Exception:  # noqa: BLE001
            logger.exception("There is an error while saving note at {}", output_dir / file_name)

    def _resolve_output_dir(self) -> Path | None:
        output_dir = self._output_dir

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir
        except Exception:  # noqa: BLE001
            if output_dir.is_file():
                logger.error(
                    "Can't access selected directory at {}, because selected directory is a file. "
                    "Trying fallback directory: {}",
                    output_dir,
                    TEMP_PATH,
                )
            else:
                logger.exception(
                    "There is an error while creating directory at {}", output_dir
                )

        if output_dir == TEMP_PATH:
            logger.error("Fallback directory {} is not usable either.", TEMP_PATH)
            return None

        try:
            TEMP_PATH.mkdir(parents=True, exist_ok=True)
        except Exception:  # noqa: BLE001
            logger.exception(
                "There is an error while creating fallback directory at {}", TEMP_PATH
            )
            return None

        logger.warning("Note will be saved in fallback directory: {}", TEMP_PATH)
        return TEMP_PATH
