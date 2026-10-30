import json
import os
from typing import Type
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables (.env)
load_dotenv()


# 1. Definisikan Custom Tools
class TokenCostCalculatorInput(BaseModel):
  model: str = Field(description="LLM model name, e.g. 'gpt-4o'")
  input_tokens: int = Field(description="Number of input tokens")
  output_tokens: int = Field(description="Number of output tokens")


class TokenCostCalculator(BaseTool):
  name: str = "Token Cost Calculator"
  description: str = (
      "Calculate the USD cost of an LLM API call given model name and token"
      " counts."
  )
  args_schema: Type[BaseModel] = TokenCostCalculatorInput

  def _run(self, model: str, input_tokens: int, output_tokens: int) -> str:
    pricing = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    }
    if model not in pricing:
      return f"Unknown model: {model}"
    p = pricing[model]
    cost = (
        input_tokens * p["input"] + output_tokens * p["output"]
    ) / 1_000_000
    return json.dumps({
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6),
    })


class ModelContextLookupInput(BaseModel):
  model: str = Field(description="LLM model name")


class ModelContextLookup(BaseTool):
  name: str = "Model Context Window Lookup"
  description: str = "Look up the context window size (in tokens) of an LLM."
  args_schema: Type[BaseModel] = ModelContextLookupInput

  def _run(self, model: str) -> str:
    windows = {
        "gpt-4o": 128_000,
        "gpt-4o-mini": 128_000,
        "claude-sonnet-4-5": 200_000,
        "claude-opus-4-5": 200_000,
        "gemini-1.5-pro": 1_000_000,
    }
    tokens = windows.get(model, "Unknown")
    return json.dumps({"model": model, "context_window_tokens": tokens})


# Instansiasi Tool
cost_tool = TokenCostCalculator()
context_tool = ModelContextLookup()

# 2. Inisialisasi LLM via OpenRouter
llm = LLM(
    model="openrouter/openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

# 3. Buat Agen dan Pasang Tools
researcher = Agent(
    role="AI Research Analyst",
    goal="Look up model context sizes and calculate usage costs accurately.",
    backstory=(
        "You are an AI infrastructure analyst who helps developers calculate"
        " API costs and understand model limits."
    ),
    llm=llm,
    tools=[context_tool, cost_tool],  # Menghubungkan tool ke agen
    verbose=True,
)

# 4. Buat Tugas yang Membutuhkan Penggunaan Kedua Tool Tersebut
task = Task(
    description=(
        "1. Look up the context window for 'gpt-4o-mini'.\n2. Calculate the"
        " cost for 'gpt-4o-mini' using 500,000 input tokens and 100,000 output"
        " tokens.\nSummarize both findings."
    ),
    expected_output=(
        "A brief summary showing the context window size and calculated cost"
        " in USD."
    ),
    agent=researcher,
)

# 5. Gabungkan ke dalam Crew dan Eksekusi
crew = Crew(
    agents=[researcher],
    tasks=[task],
    process=Process.sequential,
    verbose=True,
)

if __name__ == "__main__":
  result = crew.kickoff()
  print(result)