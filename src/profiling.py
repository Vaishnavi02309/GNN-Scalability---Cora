"""
profiling.py  –  Process-RSS RAM profiling for GNN scalability analysis.

What it measures
----------------
For each labelled phase (for example: forward, backward, step), this module records:

- start_mb   : RSS right before the phase starts
- end_mb     : RSS right after the phase ends
- peak_mb    : highest RSS observed during the phase
- duration_s : wall-clock time of the phase

Derived values:
- delta_mb    = end_mb - start_mb
- overhead_mb = peak_mb - start_mb

Important note
--------------
This measures PROCESS RAM (RSS), not exact tensor-only memory.
So it is useful for practical / operational RAM analysis, but not for exact
activation-only or gradient-only decomposition.

Important fix
-------------
If the same phase name appears multiple times in one epoch (for example, many
mini-batch forwards in GraphSAINT / ClusterGCN), phase results are AGGREGATED
within the epoch instead of being overwritten.
"""

from __future__ import annotations

import os
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import psutil

# Tuneable polling interval.
POLL_MS: float = 1.0

_PROC = psutil.Process(os.getpid())


def _rss_mb() -> float:
    """Return current process RSS in megabytes."""
    return _PROC.memory_info().rss / (1024 * 1024)


@dataclass
class PhaseResult:
    name: str
    start_mb: float
    end_mb: float
    peak_mb: float
    duration_s: float

    @property
    def delta_mb(self) -> float:
        """Net change across the phase: end - start."""
        return self.end_mb - self.start_mb

    @property
    def overhead_mb(self) -> float:
        """Extra RAM needed during the phase beyond its starting baseline."""
        return self.peak_mb - self.start_mb


def _merge_phase_results(old: PhaseResult, new: PhaseResult) -> PhaseResult:
    """
    Merge two results of the same phase within one epoch.

    Aggregation rule:
    - start_mb   = first start
    - end_mb     = last end
    - peak_mb    = max peak across all occurrences
    - duration_s = sum of durations
    """
    if old.name != new.name:
        raise ValueError(f"Cannot merge different phases: {old.name} vs {new.name}")

    return PhaseResult(
        name=old.name,
        start_mb=old.start_mb,
        end_mb=new.end_mb,
        peak_mb=max(old.peak_mb, new.peak_mb),
        duration_s=old.duration_s + new.duration_s,
    )


class _PeakPoller(threading.Thread):
    """Background thread that polls RSS and stores the maximum seen."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._stop_event = threading.Event()
        self.peak_mb: float = _rss_mb()

    def run(self) -> None:
        while not self._stop_event.is_set():
            current = _rss_mb()
            if current > self.peak_mb:
                self.peak_mb = current
            time.sleep(POLL_MS / 1000.0)

    def stop(self) -> None:
        self._stop_event.set()
        self.join()


class RamPhase:
    """
    Context manager for profiling one code phase.

    Example:
        with RamPhase("forward") as p:
            out = model(x, edge_index)
        print(p.result.peak_mb, p.result.overhead_mb)

    Note:
        force_gc defaults to False because forced garbage collection before every
        phase can make RAM traces look artificially clean and hide realistic
        training memory behavior.
    """

    def __init__(self, name: str, force_gc: bool = False) -> None:
        self.name = name
        self.force_gc = force_gc
        self.result: Optional[PhaseResult] = None

    def __enter__(self) -> "RamPhase":
        if self.force_gc:
            import gc
            gc.collect()

        self._start_mb = _rss_mb()
        self._t0 = time.perf_counter()

        self._poller = _PeakPoller()
        self._poller.peak_mb = self._start_mb
        self._poller.start()
        return self

    def __exit__(self, *_) -> None:
        self._poller.stop()

        end_mb = _rss_mb()
        peak_mb = max(self._poller.peak_mb, end_mb)
        duration_s = time.perf_counter() - self._t0

        self.result = PhaseResult(
            name=self.name,
            start_mb=self._start_mb,
            end_mb=end_mb,
            peak_mb=peak_mb,
            duration_s=duration_s,
        )


@dataclass
class EpochRecord:
    epoch: int
    phases: Dict[str, PhaseResult] = field(default_factory=dict)


class EpochProfiler:
    """
    Stores per-phase profiling results across epochs.

    Important:
    If a phase appears multiple times in one epoch, those results are merged.
    This is necessary for sampled methods that do many mini-batches per epoch.
    """

    def __init__(self) -> None:
        self._records: List[EpochRecord] = []
        self._current: Optional[EpochRecord] = None

    def begin_epoch(self, epoch: int = 0) -> None:
        self._current = EpochRecord(epoch=epoch)

    def end_epoch(self) -> None:
        if self._current is not None:
            self._records.append(self._current)
            self._current = None

    def phase(self, name: str, force_gc: bool = False) -> "RamPhase":
        """
        Return a RamPhase context manager whose result is stored automatically
        in the current epoch record.

        If the same phase name is encountered multiple times within the same
        epoch, results are merged instead of overwritten.
        """
        outer = self

        class _StoringRamPhase(RamPhase):
            def __exit__(self, *args):
                super().__exit__(*args)
                if outer._current is not None and self.result is not None:
                    existing = outer._current.phases.get(self.name)
                    if existing is None:
                        outer._current.phases[self.name] = self.result
                    else:
                        outer._current.phases[self.name] = _merge_phase_results(existing, self.result)

        return _StoringRamPhase(name=name, force_gc=force_gc)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        Compute mean/std stats across all recorded epochs.
        """
        phase_names: List[str] = []
        for rec in self._records:
            for phase_name in rec.phases:
                if phase_name not in phase_names:
                    phase_names.append(phase_name)

        out: Dict[str, Dict[str, float]] = {}

        for phase_name in phase_names:
            vals: Dict[str, List[float]] = defaultdict(list)

            for rec in self._records:
                if phase_name not in rec.phases:
                    continue
                r = rec.phases[phase_name]
                vals["peak_mb"].append(r.peak_mb)
                vals["overhead_mb"].append(r.overhead_mb)
                vals["delta_mb"].append(r.delta_mb)
                vals["duration_s"].append(r.duration_s)

            def _stats(key: str) -> tuple[float, float]:
                v = vals[key]
                mean = statistics.mean(v)
                std = statistics.pstdev(v) if len(v) > 1 else 0.0
                return mean, std

            out[phase_name] = {
                "peak_mb_mean": _stats("peak_mb")[0],
                "peak_mb_std": _stats("peak_mb")[1],
                "overhead_mb_mean": _stats("overhead_mb")[0],
                "overhead_mb_std": _stats("overhead_mb")[1],
                "delta_mb_mean": _stats("delta_mb")[0],
                "delta_mb_std": _stats("delta_mb")[1],
                "duration_s_mean": _stats("duration_s")[0],
                "duration_s_std": _stats("duration_s")[1],
            }

        return out

    def slim_summary(self, warmup_epochs: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Same as summary(), but skips the first warmup_epochs records.
        """
        saved = self._records
        self._records = saved[warmup_epochs:]
        result = self.summary()
        self._records = saved
        return result

    def print_summary(self, warmup_epochs: int = 5) -> None:
        stats = self.slim_summary(warmup_epochs=warmup_epochs)

        col = 18
        print()
        print("=" * 72)
        print("RAM PROFILING SUMMARY  (mean ± std across epochs, warmup skipped)")
        print("=" * 72)
        header = (
            f"{'Phase':<{col}}  {'Peak MB':>12}  {'Overhead MB':>14}  "
            f"{'Delta MB':>12}  {'Time (s)':>10}"
        )
        print(header)
        print("-" * 72)

        for phase_name, s in stats.items():
            row = (
                f"{phase_name:<{col}}"
                f"  {s['peak_mb_mean']:>7.2f}±{s['peak_mb_std']:>4.2f}"
                f"  {s['overhead_mb_mean']:>9.2f}±{s['overhead_mb_std']:>4.2f}"
                f"  {s['delta_mb_mean']:>7.2f}±{s['delta_mb_std']:>4.2f}"
                f"  {s['duration_s_mean']:>6.4f}±{s['duration_s_std']:>6.4f}"
            )
            print(row)

        print("=" * 72)
        print()