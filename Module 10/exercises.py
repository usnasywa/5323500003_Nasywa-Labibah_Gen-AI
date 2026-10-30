# Module 10 - LlamaIndex
# 10.9 Exercises (1 - 4) Combined Implementation

import os
import shutil
from dotenv import load_dotenv

# Import LlamaIndex Core & Components
from llama_index.core import (
    Document,
    Settings,
    VectorStoreIndex,
    StorageContext,
    SimpleDirectoryReader,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import MetadataMode
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

load_dotenv()

api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("XKIRO_KEY")
if not api_key:
    raise ValueError("API Key (OPENROUTER_API_KEY / XKIRO_KEY) tidak ditemukan di .env!")

# Konfigurasi LLM di LlamaIndex via OpenRouter
Settings.llm = OpenAI(
    model="gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    api_base="https://openrouter.ai/api/v1",
    default_headers={"HTTP-Referer": "https://github.com/", "X-Title": "Python For GenAI"},
)

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.environ["OPENROUTER_API_KEY"],
    api_base="https://openrouter.ai/api/v1",
    default_headers={"HTTP-Referer": "https://github.com/", "X-Title": "Python For GenAI"},
)

# EXERCISE 1: Custom Document Ingestion with Metadata & Node Chunking
# 1. Buat Dokumen LlamaIndex dengan metadata kustom
raw_docs = [
    Document(
        text="LlamaIndex adalah data framework untuk menghubungkan LLM dengan sumber data eksternal.",
        metadata={"category": "framework", "source": "official_doc", "author": "LlamaIndex Team"},
        excluded_embed_metadata_keys=["author"],  # Abaikan metadata 'author' saat embedding
    ),
    Document(
        text="Chunking yang tepat membantu peningkatkan performa pencarian konteks pada sistem RAG.",
        metadata={"category": "technique", "source": "best_practices", "author": "RAG Expert"},
    ),
]

# 2. Lakukan Chunking / Parsing menggunakan SentenceSplitter
parser = SentenceSplitter(chunk_size=100, chunk_overlap=20)
nodes = parser.get_nodes_from_documents(raw_docs)

print(f"[Log] Total Node yang dihasilkan: {len(nodes)}")
for i, node in enumerate(nodes, 1):
    print(f"\n--- Node {i} ---")
    print(f"Content        : {node.get_content(metadata_mode=MetadataMode.NONE)}")
    print(f"Embed Metadata : {node.get_content(metadata_mode=MetadataMode.EMBED)}")

# EXERCISE 2: Persistent Vector Indexing with ChromaDB Vector Store
chroma_dir = "./chroma_db_10_9"
if os.path.exists(chroma_dir):
    shutil.rmtree(chroma_dir)  # Bersihkan direktori lokal lama jika ada

# 1. Inisialisasi Chroma Client Persistent
db = chromadb.PersistentClient(path=chroma_dir)
chroma_collection = db.get_or_create_collection("exercise_collection")

# 2. Buat StorageContext dengan ChromaVectorStore
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 3. Build Vector Index dan Simpan ke ChromaDB
index_chroma = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
)

print("[Log] Indeks berhasil disimpan secara permanen di direktori ChromaDB.")

# 4. Verifikasi dengan memuat ulang indeks dari ChromaDB
reloaded_vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
reloaded_index = VectorStoreIndex.from_vector_store(reloaded_vector_store)
query_engine_chroma = reloaded_index.as_query_engine()

response_chroma = query_engine_chroma.query("Apa fungsi utama dari LlamaIndex?")
print(f"\nQuery Test Result:\n{response_chroma}")

# EXERCISE 3: Advanced Query Engine with Similarity Postprocessor
# Buat Query Engine dengan Similarity Threshold Postprocessor (Threshold: 0.6)
# Hanya node dengan skor kemiripan >= 0.6 yang akan diserahkan ke LLM
similarity_cutoff = SimilarityPostprocessor(similarity_cutoff=0.6)

query_engine_advanced = index_chroma.as_query_engine(
    similarity_top_k=3,
    node_postprocessors=[similarity_cutoff],
)

# Eksekusi Query
response_advanced = query_engine_advanced.query("Bagaimana cara meningkatkan performa RAG?")

print("Query Result (With Similarity Filter):")
print(response_advanced)

print("\n--- Retrieved Source Nodes ---")
for node in response_advanced.source_nodes:
    print(f"- Score: {node.score:.4f} | Text: {node.node.get_content()}")

# EXERCISE 4: Conversational Chat Engine (Context Chat Mode)
# Mengubah Indeks menjadi Chat Engine interaktif bertipe 'context'
chat_engine = index_chroma.as_chat_engine(
    chat_mode="context",
    system_prompt="Anda adalah asisten teknis yang membantu menjawab pertanyaan berdasarkan dokumen yang ada secara jelas dan singkat.",
)

# Putaran 1 Percakapan
reply_1 = chat_engine.chat("Apa itu LlamaIndex?")
print(f"User   : Apa itu LlamaIndex?")
print(f"Bot    : {reply_1}\n")

# Putaran 2 Percakapan (Sistem mengingat konteks sebelumnya)
reply_2 = chat_engine.chat("Apakah itu bisa dihubungkan dengan vector store seperti ChromaDB?")
print(f"User   : Apakah itu bisa dihubungkan dengan vector store seperti ChromaDB?")
print(f"Bot    : {reply_2}")