import anthropic, os
from dotenv import load_dotenv

load_dotenv()
# client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
client = anthropic.Anthropic(
    api_key=os.environ["XKIRO_KEY"],
    base_url="https://api.xkiro.com"
)

with client.messages.stream(
    model="qwen/qwen3.8-max:free",
    max_tokens=512,
    messages=[{"role": "user", "content": "List 5 use cases for vector databases."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
        
print() # newline after stream ends

# Access final message and usage after stream completes
final = stream.get_final_message()
print(f"\nTotal tokens:{final.usage.input_tokens + final.usage.output_tokens}")