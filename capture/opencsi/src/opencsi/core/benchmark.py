"""
Benchmark Utilities for OpenCSI.

Measures actual speedup from parallelization and GPU acceleration
in your specific environment.

Usage:
    # Command line
    python -m opencsi.core.benchmark

    # Programmatic
    from opencsi.core.benchmark import run_benchmarks
    results = run_benchmarks()
    print(results.summary())

Environment Variables:
    OPENCSI_BENCHMARK_DURATION: Seconds per test (default: 2)
    OPENCSI_BENCHMARK_SIZES: Comma-separated sizes (default: "10,100,1000")
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from opencsi.core.env import config
from opencsi.core.parallel import (
    BACKEND_NONE,
    BACKEND_PROCESS,
    BACKEND_THREAD,
    ParallelExecutor,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_DURATION_S = 2.0
DEFAULT_SIZES = [10, 100, 1000]
MIN_ITERATIONS = 3

# Simulated workload parameters
SUBCARRIERS_HT20 = 52
SUBCARRIERS_HE20 = 242
FFT_SIZE = 256

# Calibrator parameters
EMA_ALPHA = 0.01  # Exponential moving average smoothing factor
EPSILON = 1e-6  # Small value to prevent division by zero

# Benchmark configuration
MAX_LINK_WORKERS = 8
QUICK_BENCHMARK_DURATION_S = 0.5
QUICK_BENCHMARK_SIZE = 100
DEFAULT_GPU_SIZES = [1000, 10000, 100000]
PERFORMANCE_THRESHOLD_RATIO = 1.2  # Ratio to determine significant performance difference


# =============================================================================
# Workload Simulations
# =============================================================================

def _simulate_light_work(csi_data: np.ndarray) -> float:
    """Light workload: simple statistics on CSI data."""
    return float(np.mean(csi_data) + np.std(csi_data))


def _simulate_medium_work(csi_data: np.ndarray) -> np.ndarray:
    """Medium workload: phase correction simulation."""
    phase = np.angle(csi_data)
    unwrapped = np.unwrap(phase)
    x = np.arange(len(unwrapped))
    coeffs = np.polyfit(x, unwrapped, 1)
    corrected = unwrapped - np.polyval(coeffs, x)
    return corrected


def _simulate_heavy_work(csi_data: np.ndarray) -> np.ndarray:
    """Heavy workload: FFT-based processing with filtering."""
    fft_result = np.fft.fft(csi_data, n=FFT_SIZE)
    filtered = np.fft.ifft(fft_result * np.hamming(FFT_SIZE))
    return np.abs(filtered)


def _create_test_data(n_items: int, n_subcarriers: int = SUBCARRIERS_HT20) -> List[np.ndarray]:
    """Create test CSI-like data."""
    return [
        np.random.randn(n_subcarriers) + 1j * np.random.randn(n_subcarriers)
        for _ in range(n_items)
    ]


# =============================================================================
# Benchmark Results
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    backend: str
    n_items: int
    n_workers: int
    total_time_s: float
    iterations: int
    items_per_second: float = 0.0
    time_per_item_us: float = 0.0

    def __post_init__(self):
        if self.iterations > 0 and self.n_items > 0 and self.total_time_s > 0:
            total_items = self.iterations * self.n_items
            self.items_per_second = total_items / self.total_time_s
            self.time_per_item_us = (self.total_time_s / total_items) * 1e6


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    results: List[BenchmarkResult] = field(default_factory=list)
    system_info: Dict[str, Any] = field(default_factory=dict)

    def add(self, result: BenchmarkResult) -> None:
        self.results.append(result)

    def get_speedup(self, workload: str, n_items: int, backend: str) -> float:
        """Calculate speedup compared to sequential baseline."""
        baseline = self._find_result(workload, n_items, BACKEND_NONE)
        parallel = self._find_result(workload, n_items, backend)

        if baseline and parallel and parallel.time_per_item_us > 0:
            return baseline.time_per_item_us / parallel.time_per_item_us
        return 1.0

    def _find_result(
        self, name: str, n_items: int, backend: str
    ) -> Optional[BenchmarkResult]:
        for r in self.results:
            if r.name == name and r.n_items == n_items and r.backend == backend:
                return r
        return None

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            "OpenCSI Parallelization Benchmark Results",
            "=" * 70,
            "",
            f"System: {self.system_info.get('cpu_count', '?')} CPUs, "
            f"Backend default: {self.system_info.get('default_backend', '?')}",
            "",
        ]

        # Group by workload
        workloads = {}
        for r in self.results:
            if r.name not in workloads:
                workloads[r.name] = []
            workloads[r.name].append(r)

        for workload, results in workloads.items():
            lines.append(f"## {workload}")
            lines.append("-" * 50)
            lines.append(
                f"{'Backend':<12} {'Items':<8} {'Workers':<8} "
                f"{'Items/s':<12} {'μs/item':<10} {'Speedup':<8}"
            )
            lines.append("-" * 50)

            for r in sorted(results, key=lambda x: (x.n_items, x.backend)):
                speedup = self.get_speedup(workload, r.n_items, r.backend)
                speedup_str = f"{speedup:.2f}x" if r.backend != BACKEND_NONE else "baseline"
                lines.append(
                    f"{r.backend:<12} {r.n_items:<8} {r.n_workers:<8} "
                    f"{r.items_per_second:<12.1f} {r.time_per_item_us:<10.1f} {speedup_str:<8}"
                )
            lines.append("")

        # Recommendations
        lines.append("=" * 70)
        lines.append("Recommendations")
        lines.append("=" * 70)
        lines.extend(self._generate_recommendations())

        return "\n".join(lines)

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []

        # Find best backend for heavy workload at 100 items
        heavy_results = [
            r for r in self.results
            if r.name == "heavy_work" and r.n_items == 100
        ]

        if heavy_results:
            best = max(heavy_results, key=lambda r: r.items_per_second)
            if best.backend != BACKEND_NONE:
                speedup = self.get_speedup("heavy_work", 100, best.backend)
                recommendations.append(
                    f"- For CPU-intensive work: Use OPENCSI_PARALLEL_BACKEND={best.backend} "
                    f"({speedup:.1f}x speedup)"
                )
            else:
                recommendations.append(
                    "- Parallelization overhead exceeds benefit for your workload size"
                )

        # Check if thread vs process matters
        thread_heavy = self._find_result("heavy_work", 100, BACKEND_THREAD)
        process_heavy = self._find_result("heavy_work", 100, BACKEND_PROCESS)

        if thread_heavy and process_heavy:
            if process_heavy.items_per_second > thread_heavy.items_per_second * PERFORMANCE_THRESHOLD_RATIO:
                recommendations.append(
                    "- Process pool outperforms threads (GIL contention detected)"
                )
            elif thread_heavy.items_per_second > process_heavy.items_per_second * PERFORMANCE_THRESHOLD_RATIO:
                recommendations.append(
                    "- Thread pool outperforms processes (pickling overhead detected)"
                )

        # Check for small workload overhead
        light_10_none = self._find_result("light_work", 10, BACKEND_NONE)
        light_10_thread = self._find_result("light_work", 10, BACKEND_THREAD)

        if light_10_none and light_10_thread:
            if light_10_none.items_per_second > light_10_thread.items_per_second:
                recommendations.append(
                    "- For small batches (<10 items): Use OPENCSI_PARALLEL_BACKEND=none"
                )

        # GPU recommendation
        recommendations.append(
            "- For maximum throughput: Consider OPENCSI_ARRAY_BACKEND=cupy (requires GPU)"
        )

        return recommendations


# =============================================================================
# Benchmark Runner
# =============================================================================

def _run_single_benchmark(
    name: str,
    func: Callable[[np.ndarray], Any],
    data: List[np.ndarray],
    backend: str,
    n_workers: int,
    duration_s: float,
) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    iterations = 0
    start = time.perf_counter()

    with ParallelExecutor(backend=backend, max_workers=n_workers) as executor:
        while True:
            _ = executor.map(func, data)
            iterations += 1

            elapsed = time.perf_counter() - start
            if elapsed >= duration_s and iterations >= MIN_ITERATIONS:
                break

    return BenchmarkResult(
        name=name,
        backend=backend,
        n_items=len(data),
        n_workers=n_workers if backend != BACKEND_NONE else 1,
        total_time_s=elapsed,
        iterations=iterations,
    )


def run_benchmarks(
    duration_s: Optional[float] = None,
    sizes: Optional[List[int]] = None,
    backends: Optional[List[str]] = None,
    verbose: bool = True,
) -> BenchmarkSuite:
    """
    Run parallelization benchmarks.

    Args:
        duration_s: Seconds per test (default: 2)
        sizes: List of item counts to test (default: [10, 100, 1000])
        backends: Backends to test (default: all)
        verbose: Print progress

    Returns:
        BenchmarkSuite with results and recommendations
    """
    duration_s = duration_s or float(os.environ.get("OPENCSI_BENCHMARK_DURATION", DEFAULT_DURATION_S))
    sizes = sizes or DEFAULT_SIZES

    env_sizes = os.environ.get("OPENCSI_BENCHMARK_SIZES")
    if env_sizes:
        sizes = [int(s.strip()) for s in env_sizes.split(",")]

    backends = backends or [BACKEND_NONE, BACKEND_THREAD, BACKEND_PROCESS]
    n_workers = config.max_workers

    suite = BenchmarkSuite(
        system_info={
            "cpu_count": os.cpu_count(),
            "default_backend": config.parallel_backend,
            "max_workers": n_workers,
        }
    )

    workloads = [
        ("light_work", _simulate_light_work),
        ("medium_work", _simulate_medium_work),
        ("heavy_work", _simulate_heavy_work),
    ]

    total_tests = len(workloads) * len(sizes) * len(backends)
    test_num = 0

    for workload_name, workload_func in workloads:
        for n_items in sizes:
            # Create test data once per size
            test_data = _create_test_data(n_items)

            for backend in backends:
                test_num += 1
                if verbose:
                    print(
                        f"[{test_num}/{total_tests}] "
                        f"{workload_name} n={n_items} backend={backend}...",
                        end=" ",
                        flush=True,
                    )

                result = _run_single_benchmark(
                    name=workload_name,
                    func=workload_func,
                    data=test_data,
                    backend=backend,
                    n_workers=n_workers,
                    duration_s=duration_s,
                )
                suite.add(result)

                if verbose:
                    print(f"{result.items_per_second:.0f} items/s")

    return suite


def quick_benchmark(verbose: bool = True) -> BenchmarkSuite:
    """
    Run a quick benchmark with shorter duration.

    Good for getting a rough idea of parallelization benefit.
    """
    return run_benchmarks(
        duration_s=QUICK_BENCHMARK_DURATION_S,
        sizes=[QUICK_BENCHMARK_SIZE],
        verbose=verbose,
    )


# =============================================================================
# Realistic Link-Level Benchmark
# =============================================================================

class SimulatedLinkCalibrator:
    """Simulates a per-link calibrator with state."""

    def __init__(self, link_id: str, n_subcarriers: int = SUBCARRIERS_HT20):
        self.link_id = link_id
        self.n_subcarriers = n_subcarriers
        # Simulate baseline state
        self.baseline_mean = np.zeros(n_subcarriers)
        self.baseline_std = np.ones(n_subcarriers)
        self.sample_count = 0

    def process_packet(self, csi: np.ndarray) -> Dict[str, float]:
        """Process a single CSI packet through calibration pipeline."""
        # 1. Extract amplitude and phase
        amplitude = np.abs(csi)
        phase = np.angle(csi)

        # 2. Phase correction (unwrap + linear fit)
        unwrapped = np.unwrap(phase)
        x = np.arange(len(unwrapped))
        coeffs = np.polyfit(x, unwrapped, 1)
        corrected_phase = unwrapped - np.polyval(coeffs, x)

        # 3. Update baseline (exponential moving average)
        self.baseline_mean = (1 - EMA_ALPHA) * self.baseline_mean + EMA_ALPHA * amplitude
        self.baseline_std = (1 - EMA_ALPHA) * self.baseline_std + EMA_ALPHA * np.abs(amplitude - self.baseline_mean)
        self.sample_count += 1

        # 4. Compute Z-scores
        z_scores = (amplitude - self.baseline_mean) / (self.baseline_std + EPSILON)

        # 5. Motion detection
        motion_score = float(np.mean(np.abs(z_scores)))

        return {
            "link_id": self.link_id,
            "motion_score": motion_score,
            "mean_amplitude": float(np.mean(amplitude)),
        }


def _process_link_batch(args: Tuple[str, List[np.ndarray]]) -> List[Dict[str, float]]:
    """Process a batch of packets for a single link."""
    link_id, packets = args
    calibrator = SimulatedLinkCalibrator(link_id)
    return [calibrator.process_packet(pkt) for pkt in packets]


def benchmark_link_parallel(
    n_links: int = 8,
    packets_per_link: int = 100,
    duration_s: float = 2.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Benchmark link-level parallelism.

    This is the realistic use case: multiple TX-RX links processed independently,
    each with their own calibrator state.

    Args:
        n_links: Number of simulated links
        packets_per_link: Packets per link per iteration
        duration_s: Test duration
        verbose: Print progress

    Returns:
        Dict with throughput and speedup for each backend
    """
    # Create test data: each link gets its own packet batch
    link_data = {
        f"link_{i}": _create_test_data(packets_per_link)
        for i in range(n_links)
    }

    results = {}
    backends = [BACKEND_NONE, BACKEND_THREAD, BACKEND_PROCESS]

    for backend in backends:
        if verbose:
            print(f"Link-parallel benchmark: {n_links} links, backend={backend}...", end=" ", flush=True)

        iterations = 0
        start = time.perf_counter()

        with ParallelExecutor(backend=backend, max_workers=min(n_links, MAX_LINK_WORKERS)) as executor:
            while True:
                # Process all links
                link_args = list(link_data.items())
                _ = executor.map(_process_link_batch, link_args)
                iterations += 1

                elapsed = time.perf_counter() - start
                if elapsed >= duration_s and iterations >= MIN_ITERATIONS:
                    break

        total_packets = iterations * n_links * packets_per_link
        packets_per_second = total_packets / elapsed

        results[backend] = {
            "packets_per_second": packets_per_second,
            "iterations": iterations,
            "elapsed_s": elapsed,
        }

        if verbose:
            print(f"{packets_per_second:.0f} packets/s")

    # Calculate speedups
    baseline = results[BACKEND_NONE]["packets_per_second"]
    for backend in backends:
        results[backend]["speedup"] = results[backend]["packets_per_second"] / baseline

    return results


# =============================================================================
# GPU Benchmark (if CuPy available)
# =============================================================================

def benchmark_gpu(
    sizes: Optional[List[int]] = None,
    duration_s: float = DEFAULT_DURATION_S,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Benchmark GPU vs CPU for array operations.

    Requires CuPy installation.

    Returns:
        Dict with numpy_time, cupy_time, speedup
    """
    sizes = sizes or DEFAULT_GPU_SIZES
    results = {}

    try:
        import cupy as cp
    except ImportError:
        if verbose:
            print("CuPy not installed, skipping GPU benchmark")
        return {"error": "CuPy not installed"}

    for size in sizes:
        if verbose:
            print(f"GPU benchmark n={size}...", end=" ", flush=True)

        # Create test data
        np_data = np.random.randn(size) + 1j * np.random.randn(size)
        cp_data = cp.asarray(np_data)

        # Benchmark NumPy
        iterations = 0
        start = time.perf_counter()
        while time.perf_counter() - start < duration_s:
            _ = np.fft.fft(np_data)
            iterations += 1
        np_time = (time.perf_counter() - start) / iterations

        # Benchmark CuPy
        cp.cuda.Stream.null.synchronize()  # Warm up
        iterations = 0
        start = time.perf_counter()
        while time.perf_counter() - start < duration_s:
            _ = cp.fft.fft(cp_data)
            cp.cuda.Stream.null.synchronize()
            iterations += 1
        cp_time = (time.perf_counter() - start) / iterations

        speedup = np_time / cp_time if cp_time > 0 else 0

        results[size] = {
            "numpy_time_ms": np_time * 1000,
            "cupy_time_ms": cp_time * 1000,
            "speedup": speedup,
        }

        if verbose:
            print(f"NumPy: {np_time*1000:.2f}ms, CuPy: {cp_time*1000:.2f}ms, Speedup: {speedup:.1f}x")

    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

def main() -> None:
    """Run benchmarks from command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="OpenCSI Parallelization Benchmark"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run quick benchmark (shorter duration)",
    )
    parser.add_argument(
        "--gpu", "-g",
        action="store_true",
        help="Include GPU benchmark (requires CuPy)",
    )
    parser.add_argument(
        "--links", "-l",
        action="store_true",
        help="Run link-parallel benchmark (realistic use case)",
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=DEFAULT_DURATION_S,
        help=f"Seconds per test (default: {DEFAULT_DURATION_S})",
    )
    parser.add_argument(
        "--sizes", "-s",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SIZES),
        help=f"Comma-separated item counts (default: {','.join(str(s) for s in DEFAULT_SIZES)})",
    )
    parser.add_argument(
        "--n-links",
        type=int,
        default=MAX_LINK_WORKERS,
        help=f"Number of links for link-parallel benchmark (default: {MAX_LINK_WORKERS})",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("OpenCSI Parallelization Benchmark")
    print("=" * 70)
    print(f"CPUs: {os.cpu_count()}, Workers: {config.max_workers}")
    print(f"Default backend: {config.parallel_backend}")
    print()

    if args.links:
        # Run link-parallel benchmark (most realistic)
        print("=" * 70)
        print("Link-Parallel Benchmark (Realistic Use Case)")
        print("=" * 70)
        print(f"Simulating {args.n_links} TX-RX links with independent calibrators")
        print()

        for n_links in [2, 4, 8, 16]:
            if n_links > args.n_links * 2:
                break
            print(f"\n--- {n_links} links ---")
            results = benchmark_link_parallel(
                n_links=n_links,
                packets_per_link=100,
                duration_s=args.duration,
            )
            print(f"  Speedup: thread={results['thread']['speedup']:.2f}x, "
                  f"process={results['process']['speedup']:.2f}x")

    elif args.quick:
        suite = quick_benchmark()
        print()
        print(suite.summary())
    else:
        sizes = [int(s.strip()) for s in args.sizes.split(",")]
        suite = run_benchmarks(duration_s=args.duration, sizes=sizes)
        print()
        print(suite.summary())

    if args.gpu:
        print()
        print("=" * 70)
        print("GPU Benchmark (FFT)")
        print("=" * 70)
        benchmark_gpu()


if __name__ == "__main__":
    main()
