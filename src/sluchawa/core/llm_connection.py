from langchain_groq import ChatGroq


class LLMClient:
    def __init__(
        self, model: str = "llama-3.3-70b-versatile", temperature: float = 0.2
    ):
        self.llm = ChatGroq(model=model, temperature=temperature)

    def generate_note(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content
