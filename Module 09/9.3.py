import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o",  # Sesuaikan model yang aktif pada API Anda
    temperature=0.7,
    max_tokens=1024,
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",  # Sesuaikan base_url jika menggunakan XKIRO
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a {role}. Answer in {language}."),
        ("human", "{question}"),
    ]
)

# Format ke dalam objek messages (untuk diinspeksi)
msgs = prompt.format_messages(
    role="Python tutor",
    language="Indonesian",
    question="What is a generator?",
)

print("--- 1. Simple Template Formatted Messages ---")
print(msgs)

chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Today's context: {context}",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

formatted = chat_prompt.format_messages(
    context="User is learning about RAG",
    history=[
        HumanMessage(content="What is retrieval?"),
    ],
    input="How is it different from fine-tuning?",
)

print("\n--- 2. Chat Template with History Formatted Messages ---")
print(formatted)
print(f"\nTotal messages in formatted prompt: {len(formatted)}")

# Menggabungkan Prompt Template dengan LLM
chain = prompt | llm
response = chain.invoke(
    {
        "role": "Python tutor",
        "language": "Indonesian",
        "question": "Apa itu generator dalam Python?",
    }
)

print("\n--- 3. Hasil Eksekusi Prompt via LLM ---")
print(response.content)