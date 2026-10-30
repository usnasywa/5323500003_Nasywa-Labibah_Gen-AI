import os
from dotenv import load_dotenv
from llama_index.core import Document, Settings, SummaryIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

# Configuration LLM via OpenRouter API (tanpa prefix 'openai/')
Settings.llm = OpenAI(
    model="gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    api_base="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/",
        "X-Title": "Python For GenAI",
    },
)

# Configuration Embedding via OpenRouter API
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.environ["OPENROUTER_API_KEY"],
    api_base="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/",
        "X-Title": "Python For GenAI",
    },
)

# Simulate a long document split into multiple sections
sections = [
    Document(
        text="Introduction: This report analyses the performance of five LLMs on coding, reasoning, and language tasks."
    ),
    Document(
        text="Section 1 - Coding: Claude Sonnet 4.5 scored highest on HumanEval with 94.2%. GPT-4o followed at 90.1%."
    ),
    Document(
        text="Section 2 - Reasoning: GPT-4o led on GSM8K with 97.1%. Claude Sonnet 4.5 achieved 96.8%."
    ),
    Document(
        text="Section 3 - Language: Both models performed similarly on MMLU, within 1% of each other."
    ),
    Document(
        text="Conclusion: For coding tasks, prefer Claude Sonnet 4.5. For balanced performance, GPT-4o is the safer choice."
    ),
]

index = SummaryIndex.from_documents(sections)
query_engine = index.as_query_engine()

# SummaryIndex reads ALL nodes, not just top-k - good for document-wide questions
response = query_engine.query("Which model should I use for a coding project?")
print(response)