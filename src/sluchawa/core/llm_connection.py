from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import SecretStr

SYSTEM_PROMPT = Path("config/SYSTEM_PROMPT.txt").read_text(encoding="utf-8")


class LLMClient:
    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.0,
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        self._system_prompt = SYSTEM_PROMPT
        self.llm = ChatGroq(
            model=model,
            temperature=temperature,
            api_key=SecretStr(api_key) if api_key else None,
            timeout=timeout,
        )

    def generate_note(self, prompt: str | None = None) -> str:
        messages: list[tuple[str, str]] = [
            ("system", self._system_prompt),
            ("human", "{prompt}"),
        ]

        chat_prompt = ChatPromptTemplate.from_messages(messages).format_messages(
            prompt=prompt
        )
        response = self.llm.invoke(chat_prompt)
        return str(response.content)
