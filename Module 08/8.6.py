import os
from dataclasses import dataclass
import numpy as np
from dotenv import load_dotenv

load_dotenv()

@dataclass
class RetrievalEvalCase:
    query: str
    relevant_doc_ids: list[str]  # ground-truth relevant documents

# ── Metrics Calculations ──────────────────────────────────────────────────

def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of top-k results that are relevant."""
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k

def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of all relevant docs found in top-k."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)

def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """MRR: reciprocal of the rank of the first relevant result."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0

def evaluate_retrieval(store, eval_cases: list[RetrievalEvalCase], k: int = 5) -> dict:
    """Run all eval cases and return aggregate metrics."""
    p_scores, r_scores, mrr_scores = [], [], []

    for case in eval_cases:
        results = store.search(case.query, k=k)
        retrieved_ids = [r.document.id for r in results]

        p_scores.append(precision_at_k(retrieved_ids, case.relevant_doc_ids, k))
        r_scores.append(recall_at_k(retrieved_ids, case.relevant_doc_ids, k))
        mrr_scores.append(mean_reciprocal_rank(retrieved_ids, case.relevant_doc_ids))

    return {
        f"precision@{k}": round(float(np.mean(p_scores)), 4),
        f"recall@{k}":    round(float(np.mean(r_scores)), 4),
        "MRR":            round(float(np.mean(mrr_scores)), 4),
    }

# ── Standalone Fallback Store ─────────────────────────────────────────────

@dataclass
class Document:
    id: str
    text: str

@dataclass
class SearchResult:
    document: Document
    score: float

class StandaloneVectorStore:
    """Simulasi VectorStore lokal tanpa perlu import file 8.3.py."""
    def __init__(self):
        self.documents = []

    def add_documents(self, docs):
        self.documents.extend(docs)

    def search(self, query: str, k: int = 5):
        # Simulasi return hasil berdasarkan urutan dokumen terdaftar
        results = [SearchResult(document=doc, score=0.9) for doc in self.documents[:k]]
        return results

# Sample CORPUS untuk pengujian lokal
MOCK_CORPUS = [
    Document("d01", "RAG combines retrieval with generation."),
    Document("d02", "Vector databases store embeddings."),
    Document("d03", "Fine-tuning adapts models to specific tasks."),
    Document("d06", "Index structure in vector DBs."),
    Document("d08", "Retrieval Augmented Generation details."),
    Document("d09", "Transformers model attention mechanisms."),
    Document("d10", "Agents use tools and language models."),
]

# ── Main Execution ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Offline: precision_at_k / recall_at_k / MRR on mock retrieval results ===")
    mock_retrieved = ["d08", "d03", "d01", "d05"]
    mock_relevant = ["d01", "d08"]
    print("precision@3:", precision_at_k(mock_retrieved, mock_relevant, k=3))
    print("recall@3:   ", recall_at_k(mock_retrieved, mock_relevant, k=3))
    print("MRR:        ", mean_reciprocal_rank(mock_retrieved, mock_relevant))

    print("\n=== evaluate_retrieval() against VectorStore ===")
    
    # Menjalankan evaluasi menggunakan StandaloneVectorStore
    store = StandaloneVectorStore()
    store.add_documents(MOCK_CORPUS)

    eval_cases = [
        RetrievalEvalCase("How does RAG work?", ["d01", "d08"]),
        RetrievalEvalCase("What are vector databases?", ["d02", "d06"]),
        RetrievalEvalCase("How do agents use language models?", ["d10"]),
        RetrievalEvalCase("What is fine-tuning?", ["d03"]),
        RetrievalEvalCase("How do transformers model token relationships?", ["d09"]),
    ]
    
    metrics = evaluate_retrieval(store, eval_cases, k=3)
    print("Evaluation Metrics:", metrics)