import pytest
import httpx
from unittest.mock import MagicMock

from sluchawa.adapters.discord.message_adapter import DiscordMessageAdapter

MODULE_PATH = "sluchawa.adapters.discord.message_adapter"

@pytest.fixture
def adapter() -> DiscordMessageAdapter:
    # Given
    return DiscordMessageAdapter(token="fake_token", channel_id="12345")


@pytest.fixture
def mock_httpx_client(mocker):
    mock_client_instance = MagicMock()
    mock_client_instance.__enter__.return_value = mock_client_instance

    mocker.patch(f"{MODULE_PATH}.httpx.Client", return_value=mock_client_instance)
    return mock_client_instance


def test_send_message_success(adapter: DiscordMessageAdapter, mock_httpx_client, mocker):
    # Given
    message = "Hello Domain!"
    mock_logger_info = mocker.patch(f"{MODULE_PATH}.logger.info")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_httpx_client.post.return_value = mock_response

    # When
    adapter.send(message)

    # Then
    mock_httpx_client.post.assert_called_once_with(
        "https://discord.com/api/v10/channels/12345/messages",
        headers={
            "Authorization": "Bot fake_token",
            "Content-Type": "application/json"
        },
        json={"content": message}
    )
    mock_logger_info.assert_called_once_with("Successfully sent message to Discord channel 12345")


def test_send_message_handles_403_forbidden(adapter: DiscordMessageAdapter, mock_httpx_client, mocker):
    # Given
    mock_logger_error = mocker.patch(f"{MODULE_PATH}.logger.error")

    mock_response = MagicMock()
    mock_response.status_code = 403

    error = httpx.HTTPStatusError("Forbidden", request=MagicMock(), response=mock_response)
    mock_response.raise_for_status.side_effect = error
    mock_httpx_client.post.return_value = mock_response

    # When
    adapter.send("Test message")

    # Then
    mock_logger_error.assert_called_once_with(
        "403 ERROR: The bot does not have permission to post in the channel 12345"
    )


def test_send_message_handles_429_rate_limit(adapter: DiscordMessageAdapter, mock_httpx_client, mocker):
    # Given
    mock_logger_error = mocker.patch(f"{MODULE_PATH}.logger.error")

    mock_response = MagicMock()
    mock_response.status_code = 429

    error = httpx.HTTPStatusError("Rate Limit", request=MagicMock(), response=mock_response)
    mock_response.raise_for_status.side_effect = error
    mock_httpx_client.post.return_value = mock_response

    # When
    adapter.send("Test message")

    # Then
    mock_logger_error.assert_called_once_with(
        "429 ERROR: Discord rate limit. Too many messages in a short amount of time."
    )


def test_send_message_handles_network_error(adapter: DiscordMessageAdapter, mock_httpx_client, mocker):
    # Given
    mock_logger_error = mocker.patch(f"{MODULE_PATH}.logger.error")

    error = httpx.RequestError("No internet connection")
    mock_httpx_client.post.side_effect = error

    # When
    adapter.send("Test message")

    # Then
    mock_logger_error.assert_called_once_with("Network ERROR while Discord API connecting: No internet connection")