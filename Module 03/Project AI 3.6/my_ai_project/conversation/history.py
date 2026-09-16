class ConversationHistory :
    def __init__(self, max_turns: int = 10, system_prompt: str = ""):
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self._messages = []

    def add(self, role: str, content: str) -> None:
        if role not in ("user", "assistant"):
            raise ValueError(f"Invalid role: {role!r}")

        self._messages.append({
            "role": role,
            "content": content
        })

        if len(self._messages) > self.max_turns * 2:
            self._messages = self._messages[-(self.max_turns * 2):]

    def to_api_payload(self) -> list[dict]:
        messages = []

        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })

        messages.extend(self._messages)
        return messages

    def clear(self) -> None:
        self._messages = []

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return (
            f"ConversationHistory("
            f"messages={len(self._messages)}, "
            f"max_turns={self.max_turns})"
        )