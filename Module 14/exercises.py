import os
from typing import Type
from dotenv import load_dotenv

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("XKIRO_KEY")
if not api_key:
    raise ValueError("API Key (OPENROUTER_API_KEY / XKIRO_KEY) tidak ditemukan di .env!")

# Inisialisasi Native LLM CrewAI
llm = LLM(
    model="openrouter/openai/gpt-4o-mini",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,
    max_tokens=512,
)

# EXERCISE 1: Custom BaseTool with Pydantic Input Schema
# 1. Definisi Schema Input
class TopicSearchInput(BaseModel):
    topic: str = Field(description="Topik teknologi atau sains yang ingin dicari keterangannya.")

# 2. Definisi Custom Tool
class TechKnowledgeBaseTool(BaseTool):
    name: str = "Tech Knowledge Base Search"
    description: str = "Mencari fakta dan penjelasan singkat mengenai topik teknologi dari basis data internal."
    args_schema: Type[BaseModel] = TopicSearchInput

    def _run(self, topic: str) -> str:
        db = {
            "crewai": "CrewAI adalah framework berorientasi agen untuk mendesain dan mengoordinasikan tim agen AI yang kolaboratif.",
            "rag": "Retrieval-Augmented Generation (RAG) menggabungkan model bahasa dengan pencarian basis pengetahuan eksternal.",
            "langgraph": "LangGraph adalah pustaka untuk membangun aplikasi LLM berbasis stateful dan berarah siklik (graph).",
        }
        key = topic.lower().strip()
        result = db.get(key)
        if result:
            return f"[Data Internal]: {result}"
        return f"[Data Internal]: Informasi tentang '{topic}' tidak ditemukan di basis data."

# Instansiasi Tool
tech_tool = TechKnowledgeBaseTool()

# Uji coba fungsionalitas tool secara independen
print("Uji Coba Tool:")
print(tech_tool.run(topic="CrewAI"))

# EXERCISE 2: Sequential Process with Context Dependencies
# Agen 1: Peneliti Topik
researcher = Agent(
    role="Senior Tech Researcher",
    goal="Mengumpulkan fakta penting mengenai topik {topic} secara akurat.",
    backstory="Anda adalah peneliti senior yang berpengalaman menyaring informasi fakta teknologi.",
    tools=[tech_tool],
    verbose=True,
    llm=llm,
    allow_delegation=False,
)

# Agen 2: Penulis Ringkasan
writer = Agent(
    role="Technical Content Writer",
    goal="Menulis ringkasan eksekutif berdasarkan hasil riset yang diberikan.",
    backstory="Anda adalah penulis teknis handal yang mampu mengubah poin-poin riset menjadi tulisan ringkas dan tajam.",
    verbose=True,
    llm=llm,
    allow_delegation=False,
)

# Task 1: Riset Topik
task_research = Task(
    description="Gunakan tool 'Tech Knowledge Base Search' untuk mencari fakta mengenai topik '{topic}'. Berikan poin-poin kunci.",
    expected_output="Poin-poin fakta utama dari topik yang dicari.",
    agent=researcher,
)

# Task 2: Penulisan Ringkasan (Menerima context dari task_research)
task_write = Task(
    description="Buatkan ringkasan 2 kalimat untuk audiens eksekutif berdasarkan poin riset berikut.",
    expected_output="Ringkasan eksekutif 2 kalimat.",
    agent=writer,
    context=[task_research],  # Dependency konteks antar Task
)

# Sequential Crew
sequential_crew = Crew(
    agents=[researcher, writer],
    tasks=[task_research, task_write],
    process=Process.sequential,
    verbose=True,
)

# Eksekusi Sequential Crew
result_ex2 = sequential_crew.kickoff(inputs={"topic": "CrewAI"})
print("\n[Hasil Akhir Exercise 2]:")
print(result_ex2)

# EXERCISE 3 & 4: Agent Delegation & Hierarchical / Manager Process
# Agen Pakar Kode
code_expert = Agent(
    role="Python Developer Specialist",
    goal="Menulis contoh kode Python yang efisien dan bersih.",
    backstory="Anda adalah pengembang senior Python yang ahli dalam membuat arsitektur kode modern.",
    verbose=True,
    llm=llm,
    allow_delegation=False,
)

# Agen Penguji & Delegator (Manager)
lead_reviewer = Agent(
    role="Lead Code Auditor & Manager",
    goal="Memastikan permintaan kode dari pengguna ditangani dengan baik dan meninjau kualitasnya.",
    backstory="Anda adalah ketua tim teknis. Anda berwenang mendelegasikan tugas penulisan kode kepada pakar kode.",
    verbose=True,
    llm=llm,
    allow_delegation=True,  # Mengizinkan delegasi tugas ke agen lain
)

# Task Delegasi
task_delegation = Task(
    description="Minta pakar kode untuk menulis fungsi Python singkat yang menghitung jumlah kata dalam teks, lalu berikan ulasan ringkas atas kode tersebut.",
    expected_output="Kode Python beserta ulasan ringkas 1 kalimat.",
    agent=lead_reviewer,
)

# Hierarchical / Manager Crew
delegation_crew = Crew(
    agents=[lead_reviewer, code_expert],
    tasks=[task_delegation],
    process=Process.sequential,
    verbose=True,
)

# Eksekusi Delegation Crew
result_ex3 = delegation_crew.kickoff()
print("\n[Hasil Akhir Exercise 3 & 4]:")
print(result_ex3)