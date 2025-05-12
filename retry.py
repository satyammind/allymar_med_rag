# allymar/retry.py

import functools
import random
import time
from typing import Callable, Tuple, Type, TypeVar

T = TypeVar("T")

def retry(
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    *,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    jitter: float = 0.10,
    verbose: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Retry a synchronous function with exponential back‑off.

    Args:
        max_retries: Total attempts, including the first call. Must be ≥ 1.
        backoff_factor: Base wait in seconds. Actual wait is
                        backoff_factor * 2**(attempt‑1) ± jitter.
        exceptions: Tuple of exception classes that trigger a retry.
        jitter: Fractional jitter width. The sleep time is adjusted by
                ±(sleep_time * jitter). Pass 0 to disable.
        verbose: If True, prints the caught exception, the function name,
                 and the sleep time for each retry.
    """
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")
    if backoff_factor < 0:
        raise ValueError("backoff_factor cannot be negative")
    if jitter < 0:
        raise ValueError("jitter cannot be negative")

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exc: BaseException | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_retries:
                        break  # no more retries

                    # Exponential back‑off
                    sleep_time = backoff_factor * (2 ** (attempt - 1))

                    # Optional ± jitter
                    if jitter:
                        sleep_time += sleep_time * random.uniform(-jitter, jitter)

                    if verbose:
                        print(
                            f"[retry] {func.__qualname__} raised {type(exc).__name__}: {exc} – "
                            f"retry {attempt}/{max_retries} in {sleep_time:.2f}s"
                            )


                    time.sleep(sleep_time)

            # Re‑raise the original exception
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator