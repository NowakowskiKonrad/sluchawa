from typing import Protocol


class MessagePort(Protocol):
    def send(self, message: str) -> None: ...
