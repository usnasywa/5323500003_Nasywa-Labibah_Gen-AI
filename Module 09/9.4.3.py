import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Inisialisasi model yang disesuaikan dengan OpenRouter
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/",
        "X-Title": "Python For GenAI",
    },
)

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

def route(input_dict: dict) -> str:
    """Route to code or prose chain based on the question."""
    q = input_dict["input"].lower()
    if any(w in q for w in ["write", "code", "implement", "example", "function"]):
        return "code"
    return "prose"

def run_routed(input_dict: dict) -> str:
    choice = route(input_dict)
    if choice == "code":
        return code_chain.invoke(input_dict)
    return prose_chain.invoke(input_dict)

router_chain = RunnableLambda(run_routed)

print(router_chain.invoke({"input": "Write a function to chunk text with overlap."}))
print(router_chain.invoke({"input": "What is attention in transformers?"}))