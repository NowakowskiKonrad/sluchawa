import pytest
from pathlib import Path
from unittest.mock import MagicMock, call

from sluchawa.adapters.discord.audio_input_adapter import DiscordAudioInputAdapter

MODULE_PATH = "sluchawa.adapters.discord.audio_input_adapter"

@pytest.fixture
def mock_domain_port():
    return MagicMock()


@pytest.fixture
def mock_voice_client():
    return MagicMock()


@pytest.fixture
def adapter(mock_domain_port, mock_voice_client) -> DiscordAudioInputAdapter:
    # Given
    return DiscordAudioInputAdapter(
        domain_port=mock_domain_port,
        voice_client=mock_voice_client
    )


def test_start_recording_initializes_and_calls_voice_client(adapter, mock_voice_client, mocker):
    # Given
    mock_uuid = mocker.patch(f"{MODULE_PATH}.uuid.uuid4", return_value="fake-session-123")

    mock_mkdir = mocker.patch(f"{MODULE_PATH}.Path.mkdir")

    mock_sink_instance = MagicMock()
    mocker.patch(f"{MODULE_PATH}.discord.sinks.OGGSink", return_value=mock_sink_instance)

    # When
    adapter.start_recording()

    # Then
    assert adapter.session_id == "fake-session-123"
    assert adapter._output_dir == Path("/tmp/sluchawa_records/fake-session-123")

    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    mock_voice_client.start_recording.assert_called_once_with(
        mock_sink_instance,
        adapter.add_audio_to_queue
    )


def test_stop_recording_calls_voice_client(adapter, mock_voice_client):
    # When
    adapter.stop_recording()

    # Then
    mock_voice_client.stop_recording.assert_called_once()


def test_add_audio_to_queue_logs_error_if_present(adapter, mocker):
    # Given
    mock_logger = mocker.patch(f"{MODULE_PATH}.logger.error")
    test_error = Exception("Discord API crashed")

    # When
    adapter.add_audio_to_queue(error=test_error)

    # Then
    mock_logger.assert_called_once_with("There was an error while recording: Discord API crashed")


def test_add_audio_to_queue_returns_early_if_uninitialized(adapter, mock_domain_port):
    # Given

    # When
    adapter.add_audio_to_queue(error=None)

    # Then
    mock_domain_port.submit_audio.assert_not_called()


def test_add_audio_to_queue_saves_files_and_submits_to_domain(adapter, mock_domain_port, mocker):
    # Given
    adapter.session_id = "test-session"
    adapter._output_dir = Path("/tmp/sluchawa_records/test-session")

    mock_audio_1 = MagicMock()
    mock_audio_1.file.read.return_value = b"fake_binary_audio_1"

    mock_audio_2 = MagicMock()
    mock_audio_2.file.read.return_value = b"fake_binary_audio_2"

    adapter._sink = MagicMock()
    adapter._sink.audio_data = {
        "user_alice_999": mock_audio_1,
        "user_bob_888": mock_audio_2
    }

    mock_file_open = mocker.patch("builtins.open", mocker.mock_open())

    # When
    adapter.add_audio_to_queue(error=None)

    # Then
    expected_path_1 = Path("/tmp/sluchawa_records/test-session/user_alice_999.ogg")
    expected_path_2 = Path("/tmp/sluchawa_records/test-session/user_bob_888.ogg")

    mock_file_open.assert_has_calls([
        call(expected_path_1, "wb"),
        call(expected_path_2, "wb")
    ], any_order=True)

    mock_domain_port.submit_audio.assert_called_once_with([expected_path_1, expected_path_2])