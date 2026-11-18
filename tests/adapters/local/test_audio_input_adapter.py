from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

from sluchawa.adapters.local.audio_input_adapter import (
    LocalAudioInputAdapter,
    _Recording,
)

@pytest.fixture
def mock_domain_port():
    return MagicMock()

@pytest.fixture
def adapter(mock_domain_port):
    # Given
    return LocalAudioInputAdapter(domain_port=mock_domain_port)

def test_save_track_writes_ogg_opus(adapter, tmp_path):
    # Given
    recording = _Recording(output_dir=tmp_path)
    recording.chunks = {
        "mic": [np.zeros((1024, 1), dtype=np.float32)],
        "system": [],
    }

    # When
    path = adapter._save_track(recording, "mic")

    # Then
    info = sf.info(path)
    assert path.suffix == ".ogg"
    assert info.format == "OGG"
    assert info.subtype == "OPUS"
    assert info.samplerate == 16_000
    assert info.frames == 1024