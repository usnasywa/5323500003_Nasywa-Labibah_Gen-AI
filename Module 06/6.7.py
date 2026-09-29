from abc import ABC, abstractmethod
from dataclasses import dataclass
import anthropic
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ChatMessage:
    role: str # "user" or "assistant"
    content: str

@dataclass
class ChatResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str

class BaseLLMClient(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ChatResponse: ...

class AnthropicClient(BaseLLMClient):
    def __init__(self, model: str = "qwen/qwen3.8-max:free"):
        self.model = model
        # self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._client = anthropic.Anthropic(
            api_key=os.environ["XKIRO_KEY"],
            base_url="https://api.xkiro.com"
        )

    def chat(self, messages, system="", max_tokens=1024, temperature=0.7) -> ChatResponse:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        return ChatResponse(
            text=resp.content[0].text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=self.model,
        )

class OpenAIClient(BaseLLMClient):
    def __init__(self, model: str = "openai/gpt-4o-mini"):
        self.model = model
        # self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._client = OpenAI(
            api_key=os.environ["OPENROUTER_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )

    def chat(self, messages, system="", max_tokens=1024, temperature=0.7) -> ChatResponse:
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages += [{"role": m.role, "content": m.content} for m in messages]

        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=api_messages,
        )
        return ChatResponse(
            text=resp.choices[0].message.content,
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            model=self.model,
        )

if __name__ == "__main__":
    if not os.getenv("XKIRO_KEY"):
        print("ANTHROPIC_API_KEY is not set. Add it to a .env file in this folder.")
    else:
        client: BaseLLMClient = AnthropicClient()
        msgs = [ChatMessage(role="user", content="What is a vector database?")]
        result = client.chat(msgs, system="Be concise.")
        print(result.text)
        print(f"Cost estimate: {result.input_tokens} in, {result.output_tokens} out")
