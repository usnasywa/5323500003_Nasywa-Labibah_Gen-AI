import time


class RateLimiter:
    def __init__(self, max_calls: int):
        if max_calls < 1:
            raise ValueError("max_calls must be >= 1")

        self.max_calls = max_calls
        self.call_times = []

    def check_and_wait(self) -> None:
        """Wait if the rate limit has been reached."""
        now = time.monotonic()

        # Remove calls older than 60 seconds
        self.call_times = [
            t for t in self.call_times
            if now - t < 60
        ]

        if len(self.call_times) >= self.max_calls:
            wait_time = 60 - (now - self.call_times[0])

            print(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
            time.sleep(wait_time)

            # Remove expired calls after sleeping
            now = time.monotonic()
            self.call_times = [
                t for t in self.call_times
                if now - t < 60
            ]

        self.call_times.append(time.monotonic())


# Test: 5 calls rapidly
limiter = RateLimiter(max_calls=3)

for i in range(5):
    limiter.check_and_wait()
    print(f"Call {i + 1}")