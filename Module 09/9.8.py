import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# 1. Documents
docs = [
    Document(page_content="RAG stands for Retrieval-Augmented Generation. It connects LLMs to external knowledge bases."),
    Document(page_content="The retrieval step uses embeddings and vector search to find relevant documents for a given query."),
    Document(page_content="The generation step takes the retrieved context and the user question to produce a grounded answer."),
    Document(page_content="Vector databases like ChromaDB, Pinecone, and Weaviate store embeddings for fast similarity search."),
    Document(page_content="Fine-tuning bakes knowledge into model weights. RAG retrieves knowledge at inference time. Both have trade-offs."),
]

# 2. Split
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
chunks = splitter.split_documents(docs)

# 3. Embed and store in Chroma (in-memory) - disesuaikan dengan OpenRouter
embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. RAG prompt
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer the question using ONLY the provided context.
If the context does not contain the answer, say "I don't have that information."

Context:
{context}"""),
    ("human", "{question}"),
])

# 5. Chain - disesuaikan dengan OpenRouter
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

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | parser
)

# 6. Ask questions
questions = [
    "What does RAG stand for?",
    "How is RAG different from fine-tuning?",
    "What is the capital of France?", # Not in context - should say so
]

for q in questions:
    print(f"Q: {q}")
    print(f"A: {rag_chain.invoke(q)}\n")