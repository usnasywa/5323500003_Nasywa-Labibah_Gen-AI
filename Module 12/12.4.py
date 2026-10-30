from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load a small model for local inference (CPU-friendly)
model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # float16 for GPU, float32 for CPU
)
model.eval()

prompt = "The key advantage of local LLMs is"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=40,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(generated)