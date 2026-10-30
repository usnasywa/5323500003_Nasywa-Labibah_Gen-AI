import os
from dotenv import load_dotenv
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
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

documents = [
    Document(
        text="ChromaDB is an open-source embedding database with a simple Python API.",
        metadata={"tag": "vector-db"},
    ),
    Document(
        text="Pinecone is a managed vector database designed for production workloads.",
        metadata={"tag": "vector-db"},
    ),
    Document(
        text="FAISS is a library by Meta for efficient similarity search on dense vectors.",
        metadata={"tag": "vector-search"},
    ),
    Document(
        text="pgvector adds vector similarity search to PostgreSQL.",
        metadata={"tag": "vector-db"},
    ),
    Document(
        text="Weaviate is an open-source vector database with a GraphQL API.",
        metadata={"tag": "vector-db"},
    ),
]

index = VectorStoreIndex.from_documents(documents)
retriever = VectorIndexRetriever(index=index, similarity_top_k=3)

# Retrieve without synthesising an answer
nodes = retriever.retrieve("open source vector databases")

print("Retrieved nodes:")
for node in nodes:
    print(f"  [{node.score:.4f}] {node.text}")
    print(f"    tag: {node.metadata.get('tag')}")