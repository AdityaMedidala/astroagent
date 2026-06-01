"""Prove chart caching works. Usage: cd backend && python scripts/test_cache.py"""
from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.tools.chart import compute_birth_chart

INPUT = {"date": "1990-03-15", "time": "08:30", "place": "New York City, USA"}

print("Call 1 (cold — geocode + kerykeion)…")
t0 = time.perf_counter()
result1 = compute_birth_chart.invoke(INPUT)
t1 = time.perf_counter()
cold_ms = (t1 - t0) * 1000
print(f"  {cold_ms:.1f} ms")

print("Call 2 (cache hit)…")
t0 = time.perf_counter()
result2 = compute_birth_chart.invoke(INPUT)
t1 = time.perf_counter()
warm_ms = (t1 - t0) * 1000
print(f"  {warm_ms:.1f} ms")

assert result1 == result2, "Cache returned different data!"
print(f"\nResults identical: ✓")
print(f"Speedup: {cold_ms / warm_ms:.0f}x  ({cold_ms:.1f} ms → {warm_ms:.1f} ms)")
