import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

# Configure LLM dengan OpenRouter
Settings.llm = OpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    api_base="https://openrouter.ai/api/v1",
    temperature=0.1,
    max_tokens=1024,
    default_headers={
        "HTTP-Referer": "https://github.com/",
        "X-Title": "Python For GenAI",
    },
)

# Configure Embed Model dengan OpenRouter API Key
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",  # atau "openai/text-embedding-3-small"
    api_key=os.environ["OPENROUTER_API_KEY"],
    api_base="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/",
        "X-Title": "Python For GenAI",
    },
)

Settings.chunk_size = 512
Settings.chunk_overlap = 64

# Tambahkan di baris paling bawah untuk verifikasi:
print("LLM Model:", Settings.llm.model)
print("Chunk Size:", Settings.chunk_size)
print("Chunk Overlap:", Settings.chunk_overlap)