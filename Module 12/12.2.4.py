import torch
from transformers import pipeline

# Inisialisasi pipeline text-generation
summariser = pipeline(
    task="text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    device=-1,  # -1 = CPU; 0 = GPU
)

article = """
LangChain is a framework for developing applications powered by large language models. 
It provides abstractions for prompts, models, memory, and retrieval so developers can 
compose complex AI workflows from simple building blocks without writing boilerplate.
"""

messages = [
    {"role": "system", "content": "You are a helpful assistant that summarizes text concisely."},
    {"role": "user", "content": f"Summarize the following text in 1-2 sentences:\n\n{article}"}
]

# Jalankan generasi ringkasan
summary = summariser(
    messages,
    max_new_tokens=60,
    do_sample=False
)

# Cetak hasil ringkasan menggunakan kunci yang tepat
print(summary[0]["generated_text"][-1]["content"])