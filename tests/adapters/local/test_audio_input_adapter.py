from unittest.mock import MagicMock

import numpy as np
import soundfile as sf

from sluchawa.adapters.local.audio_input_adapter import LocalAudioInputAdapter

def test_save_track_writes_ogg_opus(tmp_path):
    # Given
    adapter = LocalAudioInputAdapter()
    adapter._output_dir = tmp_path
    adapter._chunks = {
        "mic": [np.zeros((1024, 1), dtype=np.float32)],
        "system": [],
    }

    # When
    path = adapter._save_track("mic")

    # Then
    info = sf.info(path)
    assert path.suffix == ".ogg"
    assert info.format == "OGG"
    assert info.subtype == "OPUS"
    assert info.samplerate == 16_000
    assert info.frames == 1024