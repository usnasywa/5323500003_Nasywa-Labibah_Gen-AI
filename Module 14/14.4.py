import os
from crewai import Agent, LLM, Task
from dotenv import load_dotenv

# 1. Load environment variables (.env)
load_dotenv()

# 2. Inisialisasi LLM kustom via OpenRouter
llm = LLM(
    model="openrouter/openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

# 3. Definisikan Agen-Agen (Agents) terlebih dahulu
researcher = Agent(
    role="AI Research Analyst",
    goal="Find accurate and up-to-date information on AI topics.",
    backstory="You are a senior AI research analyst with 10 years of experience...",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,
)

writer = Agent(
    role="Technical Content Writer",
    goal="Write clear, accurate, and engaging technical content for developers.",
    backstory="You are a technical writer with a background in software engineering...",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

reviewer = Agent(
    role="Content Quality Reviewer",
    goal="Ensure content is accurate, complete, and publication-ready.",
    backstory="You are a meticulous editor who reviews technical content...",
    llm=llm,
    verbose=True,
    allow_delegation=True,
)


# 4. Definisikan Fungsi Tasks
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
      expected_output=(
          "A complete, publication-ready developer guide in Markdown format."
      ),
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


# 5. Panggil make_tasks setelah researcher, writer, dan reviewer terdefinisi
t1, t2, t3 = make_tasks(researcher, writer, reviewer)

# 6. Pengecekan Output Sederhana
print("✅ Fungsi make_tasks berhasil dijalankan!")
print(f"Task 1: {t1.description[:60]}...")
print(f"Task 2: {t2.description[:60]}...")
print(f"Task 3: {t3.description[:60]}...")