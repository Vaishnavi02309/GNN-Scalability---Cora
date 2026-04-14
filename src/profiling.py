"""
profiling.py  –  Research-grade RAM profiling for GNN scalability analysis.

Strategy
--------
psutil point-in-time snapshots miss peaks that occur between samples.
Instead we run a lightweight background thread that polls process RSS
every POLL_MS milliseconds throughout a labelled phase.  The thread
records the TRUE peak RSS seen during that window, not just the value
at the end.

Public API
----------
    with RamPhase("forward") as p:
        logits = model(x, edge_index)
    print(p.peak_mb, p.delta_mb)

    profiler = EpochProfiler()
    profiler.begin_epoch()
    with profiler.phase("forward"):   ...
    with profiler.phase("backward"):  ...
    with profiler.phase("step"):      ...
    profiler.end_epoch()

    summary = profiler.summary()   # dict of stats across all epochs
"""

from __future__ import annotations

import gc
import os
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import psutil

# ── tuneable constants ────────────────────────────────────────────────────────
POLL_MS: float = 1.0          # background thread polls every 5 ms
_PROC   = psutil.Process(os.getpid())
# ─────────────────────────────────────────────────────────────────────────────


def _rss_mb() -> float:
    """Current process RSS in megabytes (reads /proc/self/status on Linux)."""
    return _PROC.memory_info().rss / (1024 * 1024)


# ── per-phase result ──────────────────────────────────────────────────────────

@dataclass
class PhaseResult:
    name:       str
    start_mb:   float   # RSS right before phase
    end_mb:     float   # RSS right after phase
    peak_mb:    float   # true peak seen by background thread during phase
    duration_s: float   # wall-clock seconds

    @property
    def delta_mb(self) -> float:
        """Net change: end − start.  Negative means GC freed memory."""
        return self.end_mb - self.start_mb

    @property
    def overhead_mb(self) -> float:
        """Extra RAM needed mid-phase beyond what you had at the start."""
        return self.peak_mb - self.start_mb


# ── background polling thread ─────────────────────────────────────────────────

class _PeakPoller(threading.Thread):
    """Polls RSS in the background; records the maximum seen."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._stop_event = threading.Event()
        self.peak_mb: float = _rss_mb()

    def run(self) -> None:
        while not self._stop_event.is_set():
            self.peak_mb = max(self.peak_mb, _rss_mb())
            time.sleep(POLL_MS / 1000.0)

    def stop(self) -> None:
        self._stop_event.set()
        self.join()


# ── context-manager for a single phase ───────────────────────────────────────

class RamPhase:
    """
    Context manager that measures peak RSS for one labelled phase.

    Usage::

        with RamPhase("forward") as p:
            logits = model(x, edge_index)
        print(f"peak={p.result.peak_mb:.2f} MB  overhead={p.result.overhead_mb:.2f} MB")
    """

    def __init__(self, name: str, force_gc: bool = True) -> None:
        self.name     = name
        self.force_gc = force_gc
        self.result: Optional[PhaseResult] = None

    def __enter__(self) -> "RamPhase":
        if self.force_gc:
            gc.collect()
        self._start_mb  = _rss_mb()
        self._t0        = time.perf_counter()
        self._poller    = _PeakPoller()
        self._poller.peak_mb = self._start_mb   # initialise to current
        self._poller.start()
        return self

    def __exit__(self, *_) -> None:
        self._poller.stop()
        end_mb     = _rss_mb()
        peak_mb    = max(self._poller.peak_mb, end_mb)
        duration_s = time.perf_counter() - self._t0
        self.result = PhaseResult(
            name       = self.name,
            start_mb   = self._start_mb,
            end_mb     = end_mb,
            peak_mb    = peak_mb,
            duration_s = duration_s,
        )


# ── epoch-level profiler ──────────────────────────────────────────────────────

@dataclass
class EpochRecord:
    epoch:   int
    phases:  Dict[str, PhaseResult] = field(default_factory=dict)


class EpochProfiler:
    """
    Tracks per-phase RAM across every epoch so you get mean ± std
    for a proper research table.

    Usage::

        profiler = EpochProfiler()

        for epoch in range(1, epochs + 1):
            profiler.begin_epoch(epoch)

            with profiler.phase("forward"):
                logits = model(x, edge_index)

            loss = criterion(logits[mask], y[mask])

            with profiler.phase("backward"):
                loss.backward()

            with profiler.phase("step"):
                optimizer.step()
                optimizer.zero_grad()

            profiler.end_epoch()

        stats = profiler.summary()
        profiler.print_summary()
    """

    def __init__(self) -> None:
        self._records:      List[EpochRecord] = []
        self._current:      Optional[EpochRecord] = None
        self._active_phase: Optional[RamPhase]    = None

    # ── epoch lifecycle ───────────────────────────────────────────────────────

    def begin_epoch(self, epoch: int = 0) -> None:
        self._current = EpochRecord(epoch=epoch)

    def end_epoch(self) -> None:
        if self._current is not None:
            self._records.append(self._current)
            self._current = None

    # ── phase context manager (returns self so callers can nest) ──────────────

    def phase(self, name: str, force_gc: bool = True) -> "RamPhase":
        """
        Returns a RamPhase context manager whose result is automatically
        stored in the current epoch record when the block exits.
        """
        outer = self   # capture for the inner class

        class _Storing(RamPhase):
            def __exit__(self, *args):
                super().__exit__(*args)
                if outer._current is not None and self.result is not None:
                    outer._current.phases[self.name] = self.result

        return _Storing(name, force_gc=force_gc)

    # ── statistics ────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        Returns a dict:
          phase_name → {
              "peak_mb_mean", "peak_mb_std",
              "overhead_mb_mean", "overhead_mb_std",
              "delta_mb_mean",  "delta_mb_std",
              "duration_s_mean","duration_s_std",
          }
        Computed across ALL recorded epochs (burn-in included).
        For cleaner stats, call with records[warmup_epochs:] via slim_summary().
        """
        phase_names: List[str] = []
        for rec in self._records:
            for n in rec.phases:
                if n not in phase_names:
                    phase_names.append(n)

        out: Dict[str, Dict[str, float]] = {}
        for pn in phase_names:
            vals: Dict[str, List[float]] = defaultdict(list)
            for rec in self._records:
                if pn in rec.phases:
                    r = rec.phases[pn]
                    vals["peak_mb"].append(r.peak_mb)
                    vals["overhead_mb"].append(r.overhead_mb)
                    vals["delta_mb"].append(r.delta_mb)
                    vals["duration_s"].append(r.duration_s)

            def _stats(key: str) -> tuple:
                v = vals[key]
                mean = statistics.mean(v)
                std  = statistics.pstdev(v) if len(v) > 1 else 0.0
                return mean, std

            out[pn] = {
                "peak_mb_mean":      _stats("peak_mb")[0],
                "peak_mb_std":       _stats("peak_mb")[1],
                "overhead_mb_mean":  _stats("overhead_mb")[0],
                "overhead_mb_std":   _stats("overhead_mb")[1],
                "delta_mb_mean":     _stats("delta_mb")[0],
                "delta_mb_std":      _stats("delta_mb")[1],
                "duration_s_mean":   _stats("duration_s")[0],
                "duration_s_std":    _stats("duration_s")[1],
            }
        return out

    def slim_summary(self, warmup_epochs: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Same as summary() but skips the first `warmup_epochs` records,
        removing JIT / allocator warm-up noise from your statistics.
        """
        saved = self._records
        self._records = saved[warmup_epochs:]
        result = self.summary()
        self._records = saved
        return result

    def print_summary(self, warmup_epochs: int = 5) -> None:
        """Pretty-prints a research-table-style summary to stdout."""
        stats = self.slim_summary(warmup_epochs)
        col = 18
        print()
        print("=" * 72)
        print("RAM PROFILING SUMMARY  (mean ± std across epochs, warmup skipped)")
        print("=" * 72)
        header = f"{'Phase':<{col}}  {'Peak MB':>12}  {'Overhead MB':>14}  {'Delta MB':>12}  {'Time (s)':>10}"
        print(header)
        print("-" * 72)
        for pn, s in stats.items():
            row = (
                f"{pn:<{col}}"
                f"  {s['peak_mb_mean']:>7.2f}±{s['peak_mb_std']:>4.2f}"
                f"  {s['overhead_mb_mean']:>9.2f}±{s['overhead_mb_std']:>4.2f}"
                f"  {s['delta_mb_mean']:>7.2f}±{s['delta_mb_std']:>4.2f}"
                f"  {s['duration_s_mean']:>6.4f}±{s['duration_s_std']:>6.4f}"
            )
            print(row)
        print("=" * 72)
        print()