from typing import Optional


class ConversationHistory:
    """Manages a rolling window of messages for LLM context."""

    def __init__(self, max_turns: int = 10, system_prompt: str = ""):
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self._messages: list[dict] = []

    def add(self, role: str, content: str) -> None:
        """Add a message and trim to max_turns."""
        if role not in ("user", "assistant"):
            raise ValueError(f"Invalid role: {role!r}")

        self._messages.append({
            "role": role,
            "content": content
        })

        # Keep only the latest max_turns * 2 messages
        if len(self._messages) > self.max_turns * 2:
            self._messages = self._messages[-(self.max_turns * 2):]

    def to_api_payload(self) -> list[dict]:
        """Return messages formatted for an LLM API call."""
        messages = []

        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })

        messages.extend(self._messages)

        return messages

    def clear(self) -> None:
        """Clear all conversation messages."""
        self._messages = []

    def __len__(self) -> int:
        """Return the number of stored messages."""
        return len(self._messages)

    def __repr__(self) -> str:
        """Return a readable representation of the object."""
        return (
            f"ConversationHistory("
            f"messages={len(self._messages)}, "
            f"max_turns={self.max_turns})"
        )


# Usage
history = ConversationHistory(
    max_turns=5,
    system_prompt="Be concise."
)

history.add("user", "What is an embedding?")
history.add(
    "assistant",
    "An embedding is a vector representation of data."
)
history.add("user", "Give a use case.")

print(history)
print(len(history))
print(history.to_api_payload())