import os
from dotenv import load_dotenv
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

# Basic chain: prompt | model | parser
prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarise the following text in exactly {sentences} sentences."),
    ("human", "{text}"),
])

chain = prompt | llm | StrOutputParser()

result = chain.invoke({
    "sentences": 2,
    "text": """LangChain is a framework for developing applications powered by large language models.
It provides abstractions for prompts, models, memory, and retrieval so developers can
    compose complex AI workflows from simple building blocks without writing boilerplate."""
})

print(result)