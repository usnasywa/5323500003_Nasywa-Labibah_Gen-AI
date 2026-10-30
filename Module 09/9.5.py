import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

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

# Pydantic schema for structured output
class ModelComparison(BaseModel):
    models: List[str] = Field(description="Model names compared")
    best_for_cost: str = Field(description="Most cost-effective model")
    best_for_quality: str = Field(description="Highest quality model")
    recommendation: str = Field(description="Overall recommendation")

# Use with_structured_output for schema-enforced responses
structured_llm = llm.with_structured_output(ModelComparison)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI model comparison expert."),
    ("human", "Compare these models for a startup on a budget:{models}"),
])

chain = prompt | structured_llm

result: ModelComparison = chain.invoke({
    "models": "gpt-4o, gpt-4o-mini, claude-sonnet-4-5"
})

print(f"Best for cost: {result.best_for_cost}")
print(f"Best for quality: {result.best_for_quality}")
print(f"Recommendation: {result.recommendation}")