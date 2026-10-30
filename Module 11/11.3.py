import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
# Requires: OPENROUTER_API_KEY and PINECONE_API_KEY in .env

# Menggunakan client OpenAI yang telah disesuaikan ke OpenRouter / API kustom
client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/",
        "X-Title": "Python For GenAI",
    },
)

def embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [e.embedding for e in sorted(resp.data, key=lambda x: x.index)]

# --- Connect to Pinecone ---
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

INDEX_NAME = "ai-knowledge-base"

# Create index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # text-embedding-3-small dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(INDEX_NAME)

# --- Upsert vectors ---
documents = [
    {"id": "d1", "text": "RAG connects LLMs to external knowledge bases."},
    {"id": "d2", "text": "Fine-tuning updates model weights on task-specific data."},
    {"id": "d3", "text": "ChromaDB is an open-source local vector database."},
]

vectors = [
    {
        "id": doc["id"],
        "values": embed([doc["text"]])[0],
        "metadata": {"text": doc["text"], "source": "guide"},
    }
    for doc in documents
]

index.upsert(vectors=vectors)
print(index.describe_index_stats())

# --- Query ---
q_vector = embed(["how to add external knowledge to an LLM?"])[0]
results = index.query(
    vector=q_vector,
    top_k=3,
    include_metadata=True,
)

for match in results["matches"]:
    print(f"[{match['score']:.4f}] {match['metadata']['text']}")

# --- Metadata filter ---
filtered = index.query(
    vector=q_vector,
    top_k=3,
    filter={"source": {"$eq": "guide"}},
    include_metadata=True,
)