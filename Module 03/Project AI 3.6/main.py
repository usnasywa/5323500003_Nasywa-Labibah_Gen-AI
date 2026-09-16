from my_ai_project import LLMConfig, ConversationHistory

config = LLMConfig(
    model="claude-sonnet-4-5",
    temperature=0.3,
    system_prompt="You are a concise Python tutor."
)

history = ConversationHistory(
    max_turns=5,
    system_prompt=config.system_prompt
)

history.add("user", "What is RAG?")
history.add("assistant", "RAG is Retrieval-Augmented Generation.")

print(config)
print()
print(history)
print()
print(history.to_api_payload())