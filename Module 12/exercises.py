import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline,
)
from peft import LoraConfig, TaskType, get_peft_model

# EXERCISE 1: Pipeline Analysis & Task Adaptation
# 1. Pipeline Sentiment Analysis (Text Classification)
sentiment_pipeline = pipeline(
    task="text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1,  # CPU
)

texts_to_analyze = [
    "Generative AI makes development so much faster and more intuitive!",
    "Debugging CUDA out of memory errors is quite frustrating.",
]

print("--- Hasil Analisis Sentimen ---")
for text in texts_to_analyze:
    res = sentiment_pipeline(text)
    print(f"Teks : {text}")
    print(f"Hasil: {res[0]['label']} (Score: {res[0]['score']:.4f})\n")

# 2. Pipeline Perangkuman Teks (Text Generation via Chat/Instruction Model)
summarizer = pipeline(
    task="text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    device=-1,
)

article = """
LangChain and LlamaIndex are complementary frameworks in the LLM ecosystem. 
While LangChain provides broad abstractions for building complex agentic workflows and chains, 
LlamaIndex focuses deeply on data ingestion, indexing, and retrieval-augmented generation (RAG).
"""

messages = [
    {"role": "system", "content": "You are a concise technical assistant."},
    {"role": "user", "content": f"Summarize the following text in one clear sentence:\n\n{article}"}
]

summary_result = summarizer(messages, max_new_tokens=60, do_sample=False)
print("--- Hasil Perangkuman Teks ---")
print(summary_result[0]["generated_text"][-1]["content"])

# EXERCISE 2: Tokenization Inspection & ID Mapping
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

sample_sentence = "Fine-tuning LLMs with LoRA saves GPU memory!"

# Tokenisasi kalimat menjadi ID Token
tokens = tokenizer.tokenize(sample_sentence)
input_ids = tokenizer.encode(sample_sentence)
decoded_text = tokenizer.decode(input_ids)

print(f"Kalimat Asli : {sample_sentence}")
print(f"Sub-tokens   : {tokens}")
print(f"Input IDs    : {input_ids}")
print(f"Hasil Decode : {decoded_text}")
print(f"Ukuran Vocab : {tokenizer.vocab_size} tokens")

# EXERCISE 3: Model Loading & Precision/Memory Management
# Memuat model dengan presisi FP16 (torch.float16) untuk menghemat RAM/VRAM
try:
    print("Memuat model dengan presisi torch.float16...")
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    
    # Hitung estimasi penggunaan memori parameter
    param_size = sum(p.nelement() * p.element_size() for p in model_fp16.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model_fp16.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    print(f"Model berhasil dimuat!")
    print(f"Estimasi Ukuran Memori Parameter (FP16): {size_mb:.2f} MB")
except Exception as e:
    print(f"Gagal memuat model FP16: {e}")

# EXERCISE 4: PEFT / LoRA Configuration & Parameter Counting
# 1. Konfigurasi LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                       # Rank LoRA
    lora_alpha=16,             # Scaling factor
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],  # Modul target perhatian (attention)
)

print(f"Konfigurasi LoRA: Rank (r)={peft_config.r}, Alpha={peft_config.lora_alpha}")

# 2. Menerapkan LoRA ke Model Base
if 'model_fp16' in locals():
    lora_model = get_peft_model(model_fp16, peft_config)
    
    print("\n--- Analisis Parameter Setelah LoRA ---")
    lora_model.print_trainable_parameters()