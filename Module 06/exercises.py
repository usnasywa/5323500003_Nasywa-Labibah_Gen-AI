import asyncio
import os
import time
from pathlib import Path
import anthropic
import httpx
from dotenv import load_dotenv

load_dotenv()

def retry_on_rate_limit(
    client,
    messages: list[dict],
    max_retries: int = 5,
    model: str = "qwen/qwen3.8-max:free",
    max_tokens: int = 512,
    base_delay: float = 0.05,   # short delay so the offline demo runs fast
):
    """Call client.messages.create(), retrying with exponential backoff on
    RateLimitError. Re-raises the last error if max_retries is exhausted."""
    for attempt in range(1, max_retries + 1):
        try:
            return client.messages.create(model=model, max_tokens=max_tokens, messages=messages)
        except anthropic.RateLimitError:
            if attempt == max_retries:
                raise
            wait = base_delay * (2 ** (attempt - 1))
            print(f"  Rate limited (attempt {attempt}/{max_retries}). Retrying in {wait:.2f}s...")
            time.sleep(wait)

class _MockRateLimitedClient:
    """Fake Anthropic client: fails with RateLimitError `fail_times` times,
    then succeeds. Lets us test retry_on_rate_limit() with no network/API key."""

    class _Messages:
        def __init__(self, outer):
            self._outer = outer

        def create(self, **kwargs):
            self._outer.calls += 1
            if self._outer.calls <= self._outer.fail_times:
                fake_response = httpx.Response(
                    status_code=429,
                    request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
                )
                raise anthropic.RateLimitError(
                    "rate limit exceeded (mock)", response=fake_response, body=None
                )
            return {"content": [{"text": "mock response after retries"}]}

    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls = 0
        self.messages = self._Messages(self)

class BudgetExceeded(Exception):
    """Raised when recording a call would exceed the configured token budget."""

class TokenBudgetManager:
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.used_tokens = 0

    def record(self, input_tokens: int, output_tokens: int) -> None:
        total = input_tokens + output_tokens
        if self.used_tokens + total > self.max_tokens:
            raise BudgetExceeded(
                f"Budget exceeded: {self.used_tokens + total} > {self.max_tokens} tokens"
            )
        self.used_tokens += total

    @property
    def remaining(self) -> int:
        return self.max_tokens - self.used_tokens

async def compare_models(prompt: str, models: list[str]):
    """Requires ANTHROPIC_API_KEY (claude-* models) and/or OPENAI_API_KEY
    (gpt-* models) depending on which models are passed in."""
    import pandas as pd
    from anthropic import AsyncAnthropic
    from openai import AsyncOpenAI

    # anthropic_client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    # openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    anthropic_client = AsyncAnthropic(
        api_key=os.environ["XKIRO_KEY"],
        base_url="https://api.xkiro.com"
    )
    openai_client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )

    async def call_one(model: str) -> dict:
        start = time.perf_counter()
        if model.startswith("claude"):
            resp = await anthropic_client.messages.create(
                model=model, max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text
            input_tokens, output_tokens = resp.usage.input_tokens, resp.usage.output_tokens
        else:
            resp = await openai_client.chat.completions.create(
                model=model, max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content
            input_tokens = resp.usage.prompt_tokens
            output_tokens = resp.usage.completion_tokens

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {
            "model": model,
            "response_text": text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
        }

    results = await asyncio.gather(*(call_one(m) for m in models))
    return pd.DataFrame(results)

def stream_to_file(prompt: str, output_path: str, model: str = "qwen/qwen3.8-max:free") -> None:
    client = anthropic.Anthropic(
        api_key=os.environ["XKIRO_KEY"],
        base_url="https://api.xkiro.com"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        with client.messages.stream(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                f.write(text)
                f.flush()   # write to disk as tokens arrive, not just at the end

if __name__ == "__main__":
    print("=== Exercise 1: retry_on_rate_limit (offline mock) ===")
    mock_client = _MockRateLimitedClient(fail_times=2)   # fails twice, then succeeds
    result = retry_on_rate_limit(mock_client, messages=[{"role": "user", "content": "hi"}])
    print("Result:", result)

    print("\n=== Exercise 2: TokenBudgetManager ===")
    budget = TokenBudgetManager(max_tokens=1000)
    budget.record(input_tokens=300, output_tokens=200)
    print(f"Used: {budget.used_tokens}, remaining: {budget.remaining}")
    try:
        budget.record(input_tokens=400, output_tokens=300)   # would push total to 1200 > 1000
    except BudgetExceeded as e:
        print(f"Raised as expected: {e}")

    print("\n=== Exercise 3: compare_models (needs API keys) ===")
    if os.getenv("XKIRO_KEY") or os.getenv("OPENROUTER_KEY"):
        df = asyncio.run(compare_models("What is RAG?", ["openrouter/free", "openai/gpt-4o-mini"]))
        print(df)
    else:
        print("Skipped - no ANTHROPIC_API_KEY/OPENAI_API_KEY set.")

    print("\n=== Exercise 4: stream_to_file (needs API key) ===")
    if os.getenv("XKIRO_KEY"):
        out_path = Path(__file__).parent / "stream_output.txt"
        stream_to_file("Write one sentence about embeddings.", str(out_path))
        print(f"Streamed to {out_path}")
    else:
        print("Skipped - no ANTHROPIC_API_KEY set.")