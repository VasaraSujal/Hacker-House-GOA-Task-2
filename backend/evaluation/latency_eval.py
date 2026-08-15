from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


@dataclass
class LatencyReport:
    name: str
    n: int
    min_ms: float
    mean_ms: float
    median_ms: float
    p50_ms: float
    p70_ms: float
    p90_ms: float
    p95_ms: float
    p100_ms: float
    max_ms: float


def summarize(name: str, values: list[float]) -> LatencyReport:
    return LatencyReport(
        name=name,
        n=len(values),
        min_ms=float(np.min(values)) if values else 0.0,
        mean_ms=float(np.mean(values)) if values else 0.0,
        median_ms=percentile(values, 50),
        p50_ms=percentile(values, 50),
        p70_ms=percentile(values, 70),
        p90_ms=percentile(values, 90),
        p95_ms=percentile(values, 95),
        p100_ms=percentile(values, 100),
        max_ms=float(np.max(values)) if values else 0.0,
    )
