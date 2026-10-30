import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

# OpenRouter menggunakan format OpenAI-compatible API
llm_openrouter = ChatOpenAI(
    model="openai/gpt-4o",
    temperature=0.7,
    max_tokens=1024,
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# Jika XKIRO menyediakan endpoint kompatibel OpenAI
llm_xkiro = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=1024,
    api_key=os.environ.get("XKIRO_KEY"),
)

# Pesan yang digunakan sama untuk semua provider
messages = [
    SystemMessage(content="You are a concise technical assistant."),
    HumanMessage(content="What is LangChain Expression Language?"),
]

# 1. Invoke (Blocking Call)
response = llm_openrouter.invoke(messages)
print("--- Response (Invoke) ---")
print(response.content)

# 2. Stream Response
print("\n--- Response (Stream) ---")
for chunk in llm_openrouter.stream(messages):
    print(chunk.content, end="", flush=True)
print()

# 3. Batch (Menjalankan beberapa input secara paralel)
responses = llm_openrouter.batch(
    [
        [HumanMessage(content="What is RAG?")],
        [HumanMessage(content="What is a vector database?")],
    ]
)

print("\n--- Response (Batch) ---")
for r in responses:
    print(r.content[:80])