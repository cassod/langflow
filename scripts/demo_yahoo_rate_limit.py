import time
import os

# Optionally speed up for demo by increasing YFINANCE_RATE_PER_MIN
# os.environ["YFINANCE_RATE_PER_MIN"] = "60"

from lfx.src.lfx.components.tools.yahoo_finance import _call_with_rate_and_retry, _token_bucket

def fake_network_call(i):
    """Simulated network call — returns a string with the call index and timestamp."""
    ts = time.time()
    print(f"[{i}] fake_network_call at {ts:.3f}")
    return {"i": i, "ts": ts}

def main():
    print("Token bucket initial tokens (approx):", _token_bucket._tokens)
    calls = 6
    start = time.time()
    results = []
    for i in range(calls):
        print(f"Requesting call #{i}")
        res = _call_with_rate_and_retry(lambda i=i: fake_network_call(i))
        results.append(res)
        print(f"Returned: {res}")
    elapsed = time.time() - start
    print(f"Completed {calls} calls in {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()