import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"

health = requests.get(f"{BASE}/health", timeout=5)
if health.status_code != 200:
    print(f"/health answered {health.status_code}; start the container first")
    sys.exit(1)
schema = requests.get(f"{BASE}/openapi.json", timeout=5).json()
body = schema["components"]["schemas"]["LoanApplication"]
example = (body.get("examples") or [body.get("example")])[0]
if not example:
    print("LoanApplication has no worked example in /docs; add one before benchmarking")
    sys.exit(1)

times = []
for _ in range(200):
    t0 = time.perf_counter()
    requests.post(f"{BASE}/predict", json=example)
    times.append((time.perf_counter() - t0) * 1000)
times.sort()
median_ms = times[len(times) // 2]
p95_ms = times[int(len(times) * 0.95)]

t0 = time.perf_counter()
for _ in range(100):
    requests.post(f"{BASE}/predict", json=example)
hundred_singles_ms = (time.perf_counter() - t0) * 1000
t0 = time.perf_counter()
requests.post(f"{BASE}/predict/batch", json=[example] * 100)
one_batch_ms = (time.perf_counter() - t0) * 1000


def hammer(seconds):
    count, deadline = 0, time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        if requests.post(f"{BASE}/predict", json=example).status_code == 200:
            count += 1
    return count


with ThreadPoolExecutor(max_workers=5) as pool:
    total = sum(pool.map(hammer, [5] * 5))

results = {
    "single_median_ms": round(median_ms, 1),
    "single_p95_ms": round(p95_ms, 1),
    "hundred_singles_ms": round(hundred_singles_ms),
    "one_batch_of_100_ms": round(one_batch_ms),
    "requests_per_second_5_clients": round(total / 5),
}
print(json.dumps(results, indent=2))
