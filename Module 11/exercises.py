import os
from dotenv import load_dotenv

# Import LangChain & Vector Store Components
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("XKIRO_KEY")
if not api_key:
    raise ValueError("API Key (OPENROUTER_API_KEY / XKIRO_KEY) tidak ditemukan di .env!")

# Inisialisasi Embeddings (text-embedding-3-small via OpenRouter)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

# Inisialisasi Chat Model (menggunakan gpt-4o-mini / openrouter free)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=512,
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

parser = StrOutputParser()

# EXERCISE 1: Custom Embedding Generation & Distance Metrics Comparison
import numpy as np

# 1. Buat Teks Sampel dan Dapatkan Embedding Vektornya
text_a = "Sistem basis data vektor menyimpan data dalam bentuk embedding."
text_b = "Vector database sangat penting untuk aplikasi RAG berbasis AI."
text_c = "Resep membuat nasi goreng spesial yang lezat dan gurih."

vec_a = np.array(embeddings.embed_query(text_a))
vec_b = np.array(embeddings.embed_query(text_b))
vec_c = np.array(embeddings.embed_query(text_c))

# 2. Fungsi Perhitungan Cosine Similarity dan Euclidean Distance
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

print("--- Hasil Perbandingan Kemiripan Vektor ---")
print(f"Cosine Sim (A vs B - Relevan)   : {cosine_similarity(vec_a, vec_b):.4f}")
print(f"Cosine Sim (A vs C - Irrelevan) : {cosine_similarity(vec_a, vec_c):.4f}")
print(f"Euclidean Dist (A vs B)         : {euclidean_distance(vec_a, vec_b):.4f}")
print(f"Euclidean Dist (A vs C)         : {euclidean_distance(vec_a, vec_c):.4f}")

# EXERCISE 2: Pinecone / Vector Store Indexing with Metadata Filtering
# Membuat dataset dokumen dengan metadata bidang ilmu dan tahun
docs = [
    Document(
        page_content="Model Transformer menggunakan mekanisme self-attention untuk pemrosesan teks.",
        metadata={"topic": "AI", "year": 2023, "category": "paper"}
    ),
    Document(
        page_content="Penyimpanan Vektor memungkinkan pencarian berbasis makna secara cepat.",
        metadata={"topic": "Database", "year": 2024, "category": "guide"}
    ),
    Document(
        page_content="Algoritma HNSW mempermudah pencarian Approximate Nearest Neighbor.",
        metadata={"topic": "Database", "year": 2022, "category": "paper"}
    ),
]

# Menggunakan InMemoryVectorStore untuk mensimulasikan mekanisme Pinecone
vectorstore = InMemoryVectorStore.from_documents(docs, embeddings)

# Lakukan pencarian similarity dengan metadata filter (Topic: Database)
# Gunakan lambda function untuk memeriksa metadata
filtered_results = vectorstore.similarity_search(
    "Bagaimana cara melakukan pencarian vektor?",
    k=2,
    filter=lambda doc: doc.metadata.get("topic") == "Database"
)

print("Hasil Pencarian dengan Metadata Filter (Topic: Database):")
for i, doc in enumerate(filtered_results, 1):
    print(f"{i}. {doc.page_content} | Metadata: {doc.metadata}")

# EXERCISE 3: Hybrid Search Simulation (Vector + Keyword Search)
def hybrid_search(query: str, top_k: int = 2):
    # 1. Vector / Dense Search
    dense_docs = vectorstore.similarity_search(query, k=top_k)
    
    # 2. Keyword / Sparse Search (Simulasi Sederhana)
    query_words = set(query.lower().split())
    sparse_docs = []
    for doc in docs:
        content_words = set(doc.page_content.lower().split())
        if query_words.intersection(content_words):
            sparse_docs.append(doc)
            
    # 3. Penggabungan & Reciprocal Rank Fusion (RRF) sederhanakan deduplikasi
    unique_docs = {}
    for doc in dense_docs + sparse_docs:
        unique_docs[doc.page_content] = doc
        
    return list(unique_docs.values())[:top_k]

results_hybrid = hybrid_search("mekanisme attention transformer")
print("Hasil Hybrid Search:")
for i, doc in enumerate(results_hybrid, 1):
    print(f"{i}. {doc.page_content}")

# EXERCISE 4: Complete RAG Pipeline with Vector Store Context
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Jawab pertanyaan berikut hanya berdasarkan konteks yang diberikan:\n\n{context}"),
    ("human", "{question}")
])

def format_docs(documents):
    return "\n\n".join(doc.page_content for doc in documents)

# Menyusun LCEL Pipeline RAG
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | parser
)

question = "Apa fungsi dari mekanisme self-attention?"
response_rag = rag_chain.invoke(question)

print(f"Pertanyaan : {question}")
print(f"Jawaban RAG: {response_rag}")