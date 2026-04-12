from __future__ import annotations

import os
import psutil
from typing import Dict, List


def rss_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


class ForwardMemoryTracer:
    def __init__(self) -> None:
        self.snapshots: List[Dict[str, float]] = []

    def record(self, name: str) -> None:
        self.snapshots.append({"step": name, "rss_mb": rss_mb()})
