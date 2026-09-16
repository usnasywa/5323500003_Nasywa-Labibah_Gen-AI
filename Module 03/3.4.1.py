import functools
import time


def retry(max_attempts: int = 3, delay: float = 0.1):
    """Parametrised retry decorator."""

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_error = e

                    print(
                        f"Attempt {attempt}/{max_attempts} failed: {e}"
                    )

                    if attempt < max_attempts:
                        time.sleep(delay)

            raise last_error

        return wrapper

    return decorator


@retry(max_attempts=3, delay=0.05)
def flaky_api_call(prompt: str) -> str:
    """Simulate an unreliable API call."""

    import random

    if random.random() < 0.6:
        # Fails 60% of the time
        raise ConnectionError("Simulated network error")

    return f"Success! Response for: {prompt}"


# Test the function
try:
    result = flaky_api_call("Hello AI")
    print(result)

except ConnectionError as e:
    print(f"API call failed after all retries: {e}")