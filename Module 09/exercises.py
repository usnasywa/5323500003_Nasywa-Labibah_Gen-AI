# Module 09 - LangChain
# 9.4 LCEL - LangChain Expression Language

import os
from dotenv import load_dotenv

# Impor FakeListChatModel dengan penanganan kompatibilitas versi LangChain
try:
    from langchain_community.chat_models.fake import FakeListChatModel
except ImportError:
    try:
        from langchain_core.language_models.fake_chat import FakeListChatModel
    except ImportError:
        from langchain_community.chat_models import FakeListChatModel

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI

load_dotenv()

SAMPLE_TEXT = (
    "LangChain is a framework for developing applications powered by large "
    "language models. It provides abstractions for prompts, models, memory, and "
    "retrieval so developers can compose complex AI workflows from simple "
    "building blocks without writing boilerplate."
)


# ── Basic chain: prompt | model | parser ───────────────────────────────────────

def build_summary_chain(llm: Runnable) -> Runnable:
    """prompt | model | parser - input: {"sentences": int, "text": str}."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarise the following text in exactly {sentences} sentences."),
        ("human", "{text}"),
    ])
    return prompt | llm | StrOutputParser()


# ── Parallel branches ──────────────────────────────────────────────────────────

def build_parallel_chain(llm: Runnable) -> Runnable:
    """Run two chains in parallel on the same input and combine the results
    into one dict - input: {"text": str}, output: {"summary", "keywords"}."""
    summarise_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarise this in one sentence."),
        ("human", "{text}"),
    ])
    keywords_prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract 3 keywords from this text. Return them comma-separated."),
        ("human", "{text}"),
    ])
    parser = StrOutputParser()

    return RunnableParallel(
        summary=summarise_prompt | llm | parser,
        keywords=keywords_prompt | llm | parser,
    )


# ── Conditional routing ────────────────────────────────────────────────────────

CODE_KEYWORDS = ["write", "code", "implement", "example", "function"]


def route(input_dict: dict) -> str:
    """Route to code or prose chain based on the question."""
    q = input_dict["input"].lower()
    if any(w in q for w in CODE_KEYWORDS):
        return "code"
    return "prose"


def build_router_chain(llm: Runnable) -> Runnable:
    """Pick the code chain or the prose chain per question - input: {"input": str}."""
    parser = StrOutputParser()

    code_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Python expert. Provide a complete, working code example."),
        ("human", "{input}"),
    ])
    prose_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a clear technical writer. Explain conceptually."),
        ("human", "{input}"),
    ])

    code_chain = code_prompt | llm | parser
    prose_chain = prose_prompt | llm | parser

    def run_routed(input_dict: dict) -> str:
        choice = route(input_dict)
        if choice == "code":
            return code_chain.invoke(input_dict)
        return prose_chain.invoke(input_dict)

    return RunnableLambda(run_routed)


# ── Demos ──────────────────────────────────────────────────────────────────────

def run_demos(llm: Runnable) -> None:
    print("--- Basic chain ---")
    chain = build_summary_chain(llm)
    print(chain.invoke({"sentences": 2, "text": SAMPLE_TEXT}))

    print("\n--- Parallel chain ---")
    parallel_chain = build_parallel_chain(llm)
    result = parallel_chain.invoke({
        "text": "Retrieval-Augmented Generation (RAG) improves LLM answers "
                "by fetching relevant documents at inference time."
    })
    print("Summary: ", result["summary"])
    print("Keywords:", result["keywords"])

    print("\n--- Conditional routing ---")
    router_chain = build_router_chain(llm)
    print(router_chain.invoke({"input": "Write a function to chunk text with overlap."}))
    print(router_chain.invoke({"input": "What is attention in transformers?"}))


def get_real_llm() -> ChatOpenAI:
    """Inisialisasi model nyata menggunakan OpenRouter / XKIRO."""
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("XKIRO_KEY")
    if not api_key:
        raise ValueError("API Key (OPENROUTER_API_KEY atau XKIRO_KEY) tidak ditemukan di .env")

    return ChatOpenAI(
        model="openrouter/free",  # Menggunakan model gratis dari OpenRouter
        temperature=0.7,
        max_tokens=512,  # Dibatasi agar tidak melampaui kuota
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


if __name__ == "__main__":
    print("=== Offline demo (fake model, no API key) ===")
    fake_responses = [
        "(fake model) summary of the requested length",
        "(fake model) one-sentence summary",
        "(fake model) langchain, retrieval, prompts",
        "(fake model) code chain answered",
        "(fake model) prose chain answered",
    ]
    fake_llm = FakeListChatModel(responses=fake_responses)
    run_demos(fake_llm)

    print("\n=== Real model demo ===")
    try:
        real_llm = get_real_llm()
        run_demos(real_llm)
    except Exception as e:
        print(f"Skipped real model demo due to error or missing key: {e}")