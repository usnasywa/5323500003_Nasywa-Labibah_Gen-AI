from openai import OpenAI
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def embed(texts: list[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Embed a list of texts. Returns array of shape (n, dim)."""
    # client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    client = OpenAI(
        api_key=os.environ["OPENROUTER_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )

    # API accepts up to 2048 texts per call
    response = client.embeddings.create(input=texts, model=model)
    # Sort by index to guarantee order matches input
    vectors = sorted(response.data, key=lambda e: e.index)
    return np.array([v.embedding for v in vectors], dtype=np.float32)

TEXTS = [
    "Retrieval-Augmented Generation combines search with LLMs.",
    "RAG retrieves documents then generates an answer from them.",
    "The Eiffel Tower is in Paris.",
    "Python is a popular programming language.",
    "Fine-tuning trains a model on new data.",
]

if __name__ == "__main__":
    if not os.getenv("OPENROUTER_KEY"):
        print("OPENAI_API_KEY is not set. Add it to a .env file in this folder.")
    else:
        embeddings = embed(TEXTS)
        print(f"Shape: {embeddings.shape}")   # (5, 1536)
        print(f"Norm of first vector: {np.linalg.norm(embeddings[0]):.4f}")   # ~1.0

    # NOTE on Voyage AI (Anthropic's recommended embedding provider):
    # pip install voyageai, then:
    #
    # import voyageai
    # vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    # result = vo.embed(
    #     ["What is RAG?", "Explain vector databases."],
    #     model="voyage-3",
    #     input_type="document",   # "document" for corpus, "query" for search queries
    # )
    # embeddings = np.array(result.embeddings, dtype=np.float32)
    #
    # Use text-embedding-3-small if you are already on OpenAI; use voyage-3
    # for Anthropic/Claude stacks. Both are excellent general-purpose choices.
