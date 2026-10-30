# pip install peft transformers torch datasets accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from datasets import Dataset
import torch

# ── 1. Load base model ───────────────────────────────────────────────────────
model_name = "distilgpt2" 

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Secara eksplisit paksa model masuk ke CPU
model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")

# ── 2. Configure LoRA ────────────────────────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                       # rank - higher = more parameters, more capacity
    lora_alpha=32,             # scaling factor (usually 2x to 4x r)
    target_modules=["c_attn"], # which layers to adapt (model-specific)
    lora_dropout=0.1,
    bias="none",
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
# Output: trainable params: 294,912 || all params: 82,206,720 || trainable%: 0.3588

# ── 3. Prepare dataset ────────────────────────────────────────────────────────
training_examples = [
    "Question: What is RAG? Answer: RAG stands for Retrieval-Augmented Generation.",
    "Question: What is fine-tuning? Answer: Fine-tuning trains a pre-trained model on new data.",
    "Question: What is an embedding? Answer: An embedding is a dense vector representing text meaning.",
    "Question: What is a vector database? Answer: A vector database stores embeddings for similarity search.",
    "Question: What is LangChain? Answer: LangChain is a framework for building LLM applications.",
]

def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )

dataset = Dataset.from_dict({"text": training_examples})
tokenized = dataset.map(tokenize, batched=True)
tokenized = tokenized.map(lambda x: {"labels": x["input_ids"]})

# ── 4. Training arguments ─────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="/tmp/lora_output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=False,                 
    logging_steps=5,
    save_strategy="epoch",
    report_to="none",
    use_cpu=True,               # <--- Ganti use_mps_device=False menjadi use_cpu=True
)

# ── 5. Train ──────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized,
)
trainer.train()

# ── 6. Save adapter only (small file, not the full model) ─────────────────────
peft_model.save_pretrained("/tmp/lora_adapter")
tokenizer.save_pretrained("/tmp/lora_adapter")
print("LoRA adapter saved.")

# ── 7. Load adapter for inference ─────────────────────────────────────────────
base = AutoModelForCausalLM.from_pretrained(model_name)
merged = PeftModel.from_pretrained(base, "/tmp/lora_adapter")
merged.eval()

inputs = tokenizer("Question: What is RAG? Answer:", return_tensors="pt")
with torch.no_grad():
    out = merged.generate(**inputs, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id)

print(tokenizer.decode(out[0], skip_special_tokens=True))