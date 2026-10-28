"""
Memory safety utilities for the analysis scripts.

Prevents OOM kills on memory-constrained machines (e.g., 16 GB) by:
1. Capping multiprocessing workers based on available RAM
2. Setting per-process virtual memory limits via resource.setrlimit
3. Providing a safe Pool wrapper that applies limits automatically
"""

import logging
import multiprocessing
import os
import resource
from typing import Optional

logger = logging.getLogger(__name__)

# Estimated peak RSS per worker process (bytes).
# Each worker loads one session and builds per-link data structures.
# Empirical peak: ~2-4 GB per worker.
ESTIMATED_WORKER_RSS_BYTES = 4 * 1024 ** 3  # 4 GB conservative estimate

# Env var name for overriding ESTIMATED_WORKER_RSS_BYTES per machine, e.g. on
# a machine whose measured peak RSS differs from the conservative default.
# Unset by default.
WORKER_RSS_ENV_VAR = "ESTIMATED_WORKER_RSS_BYTES"


def _resolve_worker_rss_bytes(explicit: Optional[int]) -> int:
    """Explicit argument wins; otherwise WORKER_RSS_ENV_VAR; otherwise the
    module default. Resolved at call time (not at def-time) so setting the
    env var after import still takes effect."""
    if explicit is not None:
        return explicit
    env_val = os.environ.get(WORKER_RSS_ENV_VAR)
    if env_val:
        try:
            return int(env_val)
        except ValueError:
            logger.warning(f"Ignoring invalid {WORKER_RSS_ENV_VAR}={env_val!r}")
    return ESTIMATED_WORKER_RSS_BYTES

# Fraction of total RAM the analysis processes may use.
# Leave headroom for OS, Docker, and the parent process.
RAM_BUDGET_FRACTION = 0.70

# Hard per-process virtual memory cap (bytes).
# Triggers MemoryError instead of letting the kernel OOM-kill.
PER_PROCESS_VM_LIMIT_BYTES = 10 * 1024 ** 3  # 10 GB


def get_available_ram_bytes() -> int:
    """Return available RAM in bytes (MemAvailable from /proc/meminfo)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024  # kB -> bytes
    except (OSError, ValueError):
        pass
    # Fallback: total RAM * fraction (less accurate but portable)
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return int(total * RAM_BUDGET_FRACTION)
    except (ValueError, OSError):
        return 8 * 1024 ** 3  # Assume 8 GB if all else fails


def safe_worker_count(requested: int, per_worker_bytes: Optional[int] = None) -> int:
    """
    Return a worker count that fits in available RAM.

    Args:
        requested: User-requested worker count (0=sequential, -1=all CPUs).
        per_worker_bytes: Estimated peak memory per worker. If not given,
            falls back to the WORKER_RSS_ENV_VAR env var, then the module
            default (see _resolve_worker_rss_bytes).

    Returns:
        Clamped worker count (minimum 1).
    """
    if requested == 0:
        return 0  # Sequential mode explicitly requested

    per_worker_bytes = _resolve_worker_rss_bytes(per_worker_bytes)
    available = get_available_ram_bytes()
    budget = int(available * RAM_BUDGET_FRACTION)
    max_by_ram = max(1, budget // per_worker_bytes)
    max_by_cpu = multiprocessing.cpu_count()

    if requested < 0:
        requested = max_by_cpu

    safe = min(requested, max_by_ram, max_by_cpu)
    safe = max(1, safe)

    if safe < requested:
        logger.warning(
            f"Clamped workers {requested} -> {safe} "
            f"(available RAM: {available / 1024**3:.1f} GB, "
            f"budget per worker: {per_worker_bytes / 1024**3:.1f} GB)"
        )

    return safe


def set_vm_limit(limit_bytes: int = PER_PROCESS_VM_LIMIT_BYTES) -> None:
    """
    Set per-process virtual memory limit.

    Causes MemoryError on allocation beyond the limit instead of
    letting the kernel OOM-kill the process (and everything else).
    """
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        new_soft = min(limit_bytes, hard) if hard != resource.RLIM_INFINITY else limit_bytes
        resource.setrlimit(resource.RLIMIT_AS, (new_soft, hard))
        logger.debug(f"Set RLIMIT_AS soft={new_soft / 1024**3:.1f} GB")
    except (ValueError, OSError) as e:
        logger.debug(f"Could not set RLIMIT_AS: {e}")


def _worker_init() -> None:
    """Initializer for Pool workers — sets per-process memory limit."""
    set_vm_limit()


def safe_pool_map(func, items, requested_workers: int, per_worker_bytes: Optional[int] = None):
    """
    Like pool.map() but with automatic worker count capping and memory limits.

    Args:
        func: Worker function.
        items: Iterable of arguments.
        requested_workers: User-requested worker count.
        per_worker_bytes: Estimated peak memory per worker. If not given,
            falls back to the WORKER_RSS_ENV_VAR env var, then the module
            default (see _resolve_worker_rss_bytes).

    Returns:
        List of results (same as pool.map).
    """
    items_list = list(items)
    n_workers = safe_worker_count(requested_workers, per_worker_bytes)

    if n_workers <= 1:
        # Sequential fallback
        logger.info("Running sequentially (1 worker)")
        return [func(item) for item in items_list]

    logger.info(f"Using {n_workers} parallel workers (memory-safe)")
    with multiprocessing.Pool(
        processes=n_workers,
        initializer=_worker_init,
    ) as pool:
        return pool.map(func, items_list)
