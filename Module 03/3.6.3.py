import functools
import time


def retry(max_attempts: int = 3, delay: float = 0.1):
    """Retry a function when it raises an exception."""

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


# Test function
attempt_count = 0


@retry(max_attempts=3, delay=0.1)
def test_function():
    global attempt_count

    attempt_count += 1

    if attempt_count <= 2:
        raise ConnectionError(
            f"Simulated error on attempt {attempt_count}"
        )

    return "Success!"


# Run
result = test_function()
print(result)