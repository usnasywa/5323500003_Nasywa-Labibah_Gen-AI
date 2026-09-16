responses = [
    {"model": "gpt-4o", "tokens": 540, "latency_ms": 450},
    {"model": "claude-sonnet-4-5", "tokens": 310, "latency_ms": 580},
    {"model": "gemini-1.5-pro", "tokens": 820, "latency_ms": 300},
    {"model": "gpt-4o-mini", "tokens": 200, "latency_ms": 400},
]

result = sorted(
    [r for r in responses if r["latency_ms"] < 500],
    key=lambda r: r["tokens"]
)

print(result)

def conversation_stats(messages: list[dict]) -> dict:
    total_messages = len(messages)

    user_turns = sum(
        1 for m in messages
        if m["role"] == "user"
    )

    assistant_turns = sum(
        1 for m in messages
        if m["role"] == "assistant"
    )

    total_words = sum(
        len(m["content"].split())
        for m in messages
    )

    avg_words_per_message = (
        total_words / total_messages
        if total_messages > 0
        else 0.0
    )

    return {
        "total_messages": total_messages,
        "user_turns": user_turns,
        "assistant_turns": assistant_turns,
        "avg_words_per_message": avg_words_per_message,
    }

def batch_items(items, batch_size):
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    batch = []

    for item in items:
        batch.append(item)

        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch

fast_models = [
    "gpt-4o",
    "claude-sonnet-4-5",
    "gemini-1.5-pro",
]

cheap_models = [
    "gpt-4o-mini",
    "gpt-4o",
    "gemini-1.5-pro",
]

both = set(fast_models) & set(cheap_models)

print(both)