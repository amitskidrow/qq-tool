#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import statistics
import time

import httpx

UDS = os.getenv("QQ_UDS", "/run/qq.sock")


def main():
    c = httpx.Client(transport=httpx.HTTPTransport(uds=UDS), base_url="http://qq.local", timeout=30.0)
    qs = [f"probe {i}" for i in range(500)]
    times = []
    for q in qs:
        t0 = time.perf_counter()
        r = c.post("/query", json={"q": q, "k": 6, "alpha": 0.55, "rerank": False})
        r.raise_for_status()
        data = r.json()
        times.append(float(data.get("timings", {}).get("total_ms", 0.0)))
    p50 = statistics.median(times)
    p95 = statistics.quantiles(times, n=100)[94]
    p99 = statistics.quantiles(times, n=100)[98]
    out = {"p50_ms": p50, "p95_ms": p95, "p99_ms": p99}
    print(json.dumps(out, indent=2))
    assert p95 < 1000.0, f"p95 too high: {p95:.2f} ms"


if __name__ == "__main__":
    main()

