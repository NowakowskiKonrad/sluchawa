from typing import Protocol


class LLMPort(Protocol):
    def generate_note(self, prompt: str) -> str: ...
