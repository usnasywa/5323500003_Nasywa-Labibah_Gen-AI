import os
from crewai import Agent, LLM
from dotenv import load_dotenv

load_dotenv()

# Solusi Paling Aman: Menggunakan LLM bawaan CrewAI
llm = LLM(
    model="openrouter/openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

researcher = Agent(
    role="AI Research Analyst",
    goal="Find accurate and up-to-date information on AI topics.",
    backstory="You are a senior AI research analyst with 10 years of experience...",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,
)

# Writing specialist
writer = Agent(
    role="Technical Content Writer",
    goal="Write clear, accurate, and engaging technical content for developers.",
    backstory="""You are a technical writer with a background in software engineering.
    You excel at turning complex AI concepts into practical, well-structured guides.
    You always write in active voice, use concrete examples, and avoid jargon.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# Quality reviewer
reviewer = Agent(
    role="Content Quality Reviewer",
    goal="Ensure content is accurate, complete, and publication-ready.",
    backstory="""You are a meticulous editor who reviews technical content for
    accuracy, clarity, and completeness. You check facts, flag unsupported claims,
    and ensure the writing matches the target audience's level.""",
    llm=llm,
    verbose=True,
    allow_delegation=True,
)

if __name__ == "__main__":
    print("Agents configured:", researcher.role, "|", writer.role, "|", reviewer.role)