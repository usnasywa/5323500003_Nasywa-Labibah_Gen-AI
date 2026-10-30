import os
from dotenv import load_dotenv
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer,
)
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
        text="HNSW (Hierarchical Navigable Small World) is the graph algorithm powering most modern vector search."
    ),
    Document(
        text="IVF (Inverted File Index) partitions vectors into clusters for fast approximate search."
    ),
    Document(
        text="PQ (Product Quantisation) compresses vectors to reduce memory usage at the cost of recall."
    ),
    Document(
        text="Exact nearest-neighbour search guarantees correctness but is O(n) - unusable at scale."
    ),
]

index = VectorStoreIndex.from_documents(documents)
retriever = VectorIndexRetriever(index=index, similarity_top_k=2)

# Different response modes
for mode in [ResponseMode.COMPACT, ResponseMode.TREE_SUMMARIZE]:
    synthesiser = get_response_synthesizer(response_mode=mode)
    engine = RetrieverQueryEngine(
        retriever=retriever, response_synthesizer=synthesiser
    )
    resp = engine.query("What algorithms are used in vector search?")
    print(f"[{mode}]{str(resp)[:120]}\n")