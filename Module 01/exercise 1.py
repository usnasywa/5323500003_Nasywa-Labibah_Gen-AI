def token_cost(tokens: int, model: str) -> float:
    # Dictionary harga per 1K token (harga contoh)
    costs_per_1k = {
        "gpt-4o": 0.005,
        "claude-3.5-sonnet": 0.003,
        "gemini-1.5-pro": 0.0025
    }
    
    if model not in costs_per_1k:
        raise ValueError(f"Unknown model: '{model}'")
        
    return (tokens / 1000) * costs_per_1k[model]

# --- Contoh Penggunaan ---
print(f"Cost: ${token_cost(5000, 'gpt-4o')}") 
# print(token_cost(100, 'unknown-model')) # Ini akan memicu ValueError