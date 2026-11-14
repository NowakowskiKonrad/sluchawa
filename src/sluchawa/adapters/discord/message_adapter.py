import logging
import httpx

from sluchawa.ports.message_port import MessagePort

class DiscordMessageAdapter:
    def __init__(self, token: str, channel_id: str | int):
        self._token = token
        self._channel_id = channel_id
        self._base_url = "https://discord.com/api/v10"

    async def send(self, message: str) -> None:
        url = f"{self._base_url}/channels/{self._channel_id}/messages"
        headers = {
            "Authorization": f"Bot {self._token}",
            "Content-Type": "application/json"
        }
        payload = {
            "content": message
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status == 403:
                    logger.error(f"403 ERROR: The bot does not have permission to post in the channel {self._channel_id}")
                elif status == 429:
                    logger.error("429 ERROR: Discord rate limit. Too many messages in a short amount of time.")
                else:
                    logger.error(f"HTTP ERROR {status} while sending: {e.response.text}")
            except httpx.RequestError as e:
                logger.error(f"Network ERROR while Discord API connecting: {e}")
