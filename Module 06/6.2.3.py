import anthropic, os
from dotenv import load_dotenv

load_dotenv()
# client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
client = anthropic.Anthropic(
    api_key=os.environ["XKIRO_KEY"],
    base_url="https://api.xkiro.com"
)

def chat(system: str) -> None:
    """Simple interactive multi-turn chat loop."""
    history = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        history.append({"role": "user", "content": user_input})
        response = client.messages.create(
            model="qwen/qwen3.8-max:free",
            max_tokens=1024,
            system=system,
            messages=history,
        )
        assistant_text = response.content[0].text
        history.append({"role": "assistant", "content": assistant_text})
        print(f"Claude:{assistant_text}\n")

chat(system="You are a helpful Python tutor.")