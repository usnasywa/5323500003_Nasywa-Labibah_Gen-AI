import json
import os
from typing import Type

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ==========================================
# 1. PERSIAPAN ENVIRONMENT
# ==========================================
load_dotenv()

# ==========================================
# 2. DEFINISI TOOLS (Dari 14.3.1_Tools.py)
# ==========================================
class TokenCostCalculatorInput(BaseModel):
    model: str = Field(description="LLM model name, e.g. 'gpt-4o'")
    input_tokens: int = Field(description="Number of input tokens")
    output_tokens: int = Field(description="Number of output tokens")

class TokenCostCalculator(BaseTool):
    name: str = "Token Cost Calculator"
    description: str = "Calculate the USD cost of an LLM API call given model name and token counts."
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
        cost = (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000
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

cost_tool = TokenCostCalculator()
context_tool = ModelContextLookup()

# ==========================================
# 3. DEFINISI LLM & AGENTS (Dari 14.2.1_Agent.py & Gambar 14.5)
# ==========================================
# Menggunakan LLM bawaan CrewAI via OpenRouter
llm = LLM(
    model="openrouter/openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

researcher = Agent(
    role="AI Research Analyst",
    goal="Find accurate information on AI topics.",
    backstory="Senior AI research analyst with expertise in LLM systems.",
    llm=llm,
    tools=[context_tool],  # Tool dari 14.3
    verbose=False,
    max_iter=5,
)

writer = Agent(
    role="Technical Content Writer",
    goal="Write clear technical content for developers.",
    backstory="Technical writer with a software engineering background.",
    llm=llm,
    tools=[cost_tool],  # Tool dari 14.3
    verbose=False,
    max_iter=5,
)

reviewer = Agent(
    role="Content Quality Reviewer",
    goal="Ensure content is accurate and publication-ready.",
    backstory="Meticulous editor who reviews technical AI content.",
    llm=llm,
    verbose=False,
    max_iter=5,
)

# ==========================================
# 4. DEFINISI TASKS (Dari 14.4.1_Tasks.py)
# ==========================================
def make_tasks(researcher, writer, reviewer):
    research_task = Task(
        description="""Research the current state of RAG (Retrieval-Augmented Generation) in 2024-2025.
        Cover:
        1. What RAG is and how it works
        2. The main variants (naive RAG, advanced RAG, modular RAG)
        3. Common production challenges and how practitioners solve them
        4. Three real-world use cases with specific results
        
        Be specific and factual. Avoid vague generalities.""",
        expected_output="""A structured research report with:
        - A clear definition and architecture overview
        - Descriptions of the three RAG variants with key differences
        - Three specific production challenges with concrete solutions
        - Three real-world use cases with measurable outcomes
        Format: Markdown with headers.""",
        agent=researcher,
    )

    writing_task = Task(
        description="""Using the research report provided, write a developer guide titled
        'Building Production RAG Systems'.
        
        The guide should:
        - Be written for senior developers with Python experience
        - Include a conceptual overview, implementation considerations, and common pitfalls
        - Use active voice and concrete examples
        - Be 600-800 words""",
        expected_output="A complete, publication-ready developer guide in Markdown format.",
        agent=writer,
        context=[research_task],
    )

    review_task = Task(
        description="""Review the developer guide for:
        1. Technical accuracy - flag any claims that are incorrect or unsupported
        2. Completeness - identify important topics that are missing
        3. Clarity - suggest any sections that need rewriting
        4. Audience fit - check that the level is appropriate for senior developers
        
        Provide a structured review with specific, actionable feedback.""",
        expected_output="""A quality review report containing:
        - Overall assessment (Ready / Needs Revision)
        - Specific accuracy issues (if any) with corrections
        - Missing content recommendations
        - Clarity improvement suggestions
        - Final verdict""",
        agent=reviewer,
        context=[writing_task],
    )

    return research_task, writing_task, review_task

# ==========================================
# 5. EKSEKUSI CREW (Dari Gambar 14.5)
# ==========================================
# Buat objek tugas
research_task, writing_task, review_task = make_tasks(researcher, writer, reviewer)

# Inisialisasi Crew
crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, writing_task, review_task],
    process=Process.sequential,
    verbose=False,
    memory=False,
)

if __name__ == "__main__":
    # Kick off the crew
    result = crew.kickoff()
    print("=== CREW OUTPUT ===")
    print(result)

    # Access individual task outputs
    print("\n=== TASK OUTPUTS ===")
    for task in crew.tasks:
        print(f"\n--- {task.agent.role} ---")
        if task.output:
            print(str(task.output)[:200], "...")