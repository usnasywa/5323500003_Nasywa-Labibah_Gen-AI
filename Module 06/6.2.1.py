import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

# client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
client = anthropic.Anthropic(
    api_key=os.environ["XKIRO_KEY"],
    base_url="https://api.xkiro.com"
)

message = client.messages.create(
    model="qwen/qwen3.8-max:free",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "What is retrieval-augmented generation?"}
    ]
)

# Response text is in the first content block
print(message.content[0].text)

# Usage stats
print(f"Input tokens:{message.usage.input_tokens}")
print(f"Output tokens:{message.usage.output_tokens}")