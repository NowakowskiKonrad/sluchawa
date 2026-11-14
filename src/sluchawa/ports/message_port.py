from typing import Protocol

class MessagePort(Protocol):
    async def send(self, message: str) -> None: ...
