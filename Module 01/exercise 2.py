import time
from functools import wraps

def retry(n: int, sleep_seconds: float = 1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, n + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == n:
                        print(f"Attempt {attempt} failed. Max retries reached.")
                        raise e # Lempar error jika percobaan terakhir tetap gagal
                    print(f"Attempt {attempt} failed with error: {e}. Retrying in {sleep_seconds}s...")
                    time.sleep(sleep_seconds)
        return wrapper
    return decorator

# --- Contoh Penggunaan ---
# Variabel bantuan untuk mensimulasikan kegagalan
attempts_made = 0 

@retry(n=3, sleep_seconds=0.5)
def test_unstable_api():
    global attempts_made
    attempts_made += 1
    if attempts_made < 3:
        raise ConnectionError("Connection Timeout!")
    return "API Call Success!"

print(test_unstable_api())