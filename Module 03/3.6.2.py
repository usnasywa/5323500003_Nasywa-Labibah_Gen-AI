import time


class RateLimiter:
    def __init__(self, max_calls: int, period: float = 60):
        self.max_calls = max_calls
        self.period = period
        self.call_times = []

    def check_and_wait(self):
        now = time.monotonic()

        # Hapus timestamp yang sudah melewati periode
        self.call_times = [
            t for t in self.call_times
            if now - t < self.period
        ]

        if len(self.call_times) >= self.max_calls:
            wait_time = self.period - (
                now - self.call_times[0]
            )

            print(
                f"Rate limit reached. "
                f"Waiting {wait_time:.2f} seconds..."
            )

            time.sleep(wait_time)

            now = time.monotonic()

            self.call_times = [
                t for t in self.call_times
                if now - t < self.period
            ]

        self.call_times.append(time.monotonic())


# Testing
limiter = RateLimiter(
    max_calls=3,
    period=5
)

for i in range(5):
    limiter.check_and_wait()
    print(f"Call {i + 1}")