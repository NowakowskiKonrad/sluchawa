from asyncio import AbstractEventLoop

"""
This classes job is just to send messages from out domain to discord.
"""


class DiscordMessageAdapter:
    def __init__(self, bot_loop: AbstractEventLoop | None = None):
        self._bot_loop = bot_loop

    def send(self, message: str) -> None:
        # TODO: Implement proper message sending, if bot event loop exists use it, otherwise make a http POST request.
        pass
