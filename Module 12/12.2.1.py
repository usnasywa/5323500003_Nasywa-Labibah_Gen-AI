from transformers import pipeline

# Text classification (sentiment analysis)

classifier = pipeline(
    task="text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1,    # -1 = CPU; 0 = first GPU
)

texts = [
    "This LLM API is incredibly fast and easy to use.",
    "The documentation is confusing and the latency is terrible.",
    "The model works fine for most tasks.",
]

results = classifier(texts)
for text, result in zip(texts, results):
    print(f"[{result['label']} |{result['score']:.3f}]{text[:60]}")