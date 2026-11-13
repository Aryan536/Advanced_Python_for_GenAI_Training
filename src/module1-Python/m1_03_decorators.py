import time
import random
import functools

def timer(func):
    """Decorator: prints how long a function took."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"[timer] {func.__name__} took {time.perf_counter() - start:.3f} s")
        return result
    return wrapper

def retry(max_attempts=3, wait_seconds=0.2):
    """Decorator factory: retries a function when it raises ConnectionError."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except ConnectionError as err:
                    print(f"[retry] attempt {attempt} failed: {err}")
                    time.sleep(wait_seconds)
            raise RuntimeError(f"{func.__name__} failed after {max_attempts} attempts")
        return wrapper
    return decorator

random.seed(7)

@timer
@retry(max_attempts=5)
def call_llm_api(prompt):
    """Simulates an unreliable LLM API that fails about half the time."""
    if random.random() < 0.5:
        raise ConnectionError("API timeout")
    return f"LLM answer to: '{prompt}'"

print(call_llm_api("Summarise yesterday's sales in one line"))
