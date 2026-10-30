from transformers import pipeline

# Text generation (small local model)
generator = pipeline(
    task="text-generation",
    model="distilgpt2",        # small model - runs on CPU
    device=-1,
)

output = generator(
    "The future of AI agents is",
    max_new_tokens=50,
    num_return_sequences=2,
    temperature=0.8,
    do_sample=True,
    pad_token_id=50256,        # suppress warning for GPT-2
)

for i, seq in enumerate(output):
    print(f"Sequence{i+1}:{seq['generated_text']}")