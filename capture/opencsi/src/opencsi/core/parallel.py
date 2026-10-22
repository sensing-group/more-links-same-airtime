"""
Parallel Processing Utilities for OpenCSI.

Provides multi-threading and multi-processing support for high-throughput
CSI processing. Follows patterns from OpenMPI's flexible backend selection.

Usage:
    from opencsi.core.parallel import ParallelExecutor, parallel_map

    # Process links in parallel
    with ParallelExecutor() as executor:
        results = executor.map(process_link, links)

    # Or use the convenience function
    results = parallel_map(process_link, links)

Environment Variables:
    OPENCSI_PARALLEL_BACKEND: "none", "thread", "process"
    OPENCSI_MAX_WORKERS: Number of workers or "auto"
    OPENCSI_BATCH_SIZE: Batch size for chunked processing
"""

import logging
from concurrent.futures import (
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    TypeVar,
)

from opencsi.core.env import config

logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar("T")
R = TypeVar("R")

# =============================================================================
# Constants
# =============================================================================

BACKEND_NONE = "none"
BACKEND_THREAD = "thread"
BACKEND_PROCESS = "process"

VALID_BACKENDS = {BACKEND_NONE, BACKEND_THREAD, BACKEND_PROCESS}


# =============================================================================
# Executor Factory
# =============================================================================

def _create_executor(
    backend: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> Optional[Executor]:
    """
    Create an executor based on configuration.

    Args:
        backend: Override config.parallel_backend
        max_workers: Override config.max_workers

    Returns:
        Executor instance or None if backend is "none"
    """
    backend = backend or config.parallel_backend
    max_workers = max_workers or config.max_workers

    if backend == BACKEND_NONE:
        return None
    elif backend == BACKEND_THREAD:
        return ThreadPoolExecutor(max_workers=max_workers)
    elif backend == BACKEND_PROCESS:
        return ProcessPoolExecutor(max_workers=max_workers)
    else:
        logger.warning(
            f"Unknown parallel backend '{backend}', falling back to thread"
        )
        return ThreadPoolExecutor(max_workers=max_workers)


# =============================================================================
# Parallel Executor Context Manager
# =============================================================================

class ParallelExecutor:
    """
    Context manager for parallel execution.

    Automatically selects thread or process pool based on configuration.
    Falls back to sequential execution if backend is "none".

    Usage:
        with ParallelExecutor() as executor:
            results = executor.map(func, items)

        # Or with explicit backend
        with ParallelExecutor(backend="process", max_workers=8) as executor:
            results = executor.map(cpu_bound_func, large_dataset)
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        max_workers: Optional[int] = None,
    ):
        """
        Initialize parallel executor.

        Args:
            backend: "none", "thread", or "process" (default: from config)
            max_workers: Number of workers (default: from config)
        """
        self.backend = backend or config.parallel_backend
        self.max_workers = max_workers or config.max_workers
        self._executor: Optional[Executor] = None

    def __enter__(self) -> "ParallelExecutor":
        self._executor = _create_executor(self.backend, self.max_workers)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def map(
        self,
        func: Callable[[T], R],
        items: Iterable[T],
        chunksize: int = 1,
    ) -> List[R]:
        """
        Apply function to items in parallel.

        Args:
            func: Function to apply
            items: Items to process
            chunksize: Items per chunk (for process backend)

        Returns:
            List of results in same order as inputs
        """
        items_list = list(items)

        if self._executor is None:
            # Sequential fallback
            return [func(item) for item in items_list]

        if isinstance(self._executor, ProcessPoolExecutor):
            # ProcessPoolExecutor.map supports chunksize
            return list(self._executor.map(func, items_list, chunksize=chunksize))
        else:
            # ThreadPoolExecutor.map doesn't support chunksize
            return list(self._executor.map(func, items_list))

    def submit(self, func: Callable[..., R], *args: Any, **kwargs: Any) -> "Future[R]":
        """
        Submit a single task for execution.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Future representing the result
        """
        if self._executor is None:
            # Create a completed future for sequential execution
            future: Future[R] = Future()
            try:
                result = func(*args, **kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            return future

        return self._executor.submit(func, *args, **kwargs)

    def map_unordered(
        self,
        func: Callable[[T], R],
        items: Iterable[T],
    ) -> Generator[R, None, None]:
        """
        Apply function to items, yielding results as they complete.

        Results are yielded in completion order, not input order.
        This can be faster when processing times vary.

        Args:
            func: Function to apply
            items: Items to process

        Yields:
            Results as they complete
        """
        items_list = list(items)

        if self._executor is None:
            # Sequential fallback
            for item in items_list:
                yield func(item)
            return

        futures = {self._executor.submit(func, item): item for item in items_list}

        for future in as_completed(futures):
            yield future.result()


# =============================================================================
# Convenience Functions
# =============================================================================

def parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    backend: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> List[R]:
    """
    Apply function to items in parallel.

    Convenience function that handles executor lifecycle.

    Args:
        func: Function to apply to each item
        items: Items to process
        backend: Override parallel backend
        max_workers: Override max workers

    Returns:
        List of results in same order as inputs

    Example:
        from opencsi.core.parallel import parallel_map

        def process_link(link_id):
            return calibrate(link_id)

        results = parallel_map(process_link, link_ids)
    """
    with ParallelExecutor(backend=backend, max_workers=max_workers) as executor:
        return executor.map(func, items)


def parallel_map_unordered(
    func: Callable[[T], R],
    items: Iterable[T],
    backend: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> Generator[R, None, None]:
    """
    Apply function to items, yielding results as they complete.

    Args:
        func: Function to apply
        items: Items to process
        backend: Override parallel backend
        max_workers: Override max workers

    Yields:
        Results in completion order
    """
    with ParallelExecutor(backend=backend, max_workers=max_workers) as executor:
        yield from executor.map_unordered(func, items)


# =============================================================================
# Batching Utilities
# =============================================================================

def batch_iter(
    items: Sequence[T],
    batch_size: Optional[int] = None,
) -> Iterator[Sequence[T]]:
    """
    Split items into batches.

    Args:
        items: Items to batch
        batch_size: Size of each batch (default: from config)

    Yields:
        Batches of items
    """
    batch_size = batch_size or config.batch_size

    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


@dataclass
class BatchAccumulator:
    """
    Accumulate items and process in batches.

    Usage:
        batch = BatchAccumulator(batch_size=32)

        for packet in stream:
            batch.add(packet)
            if batch.ready():
                process(batch.flush())

        # Process remaining
        if batch.pending():
            process(batch.flush())
    """

    batch_size: int = 0
    _items: List[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            self.batch_size = config.batch_size

    def add(self, item: Any) -> None:
        """Add an item to the batch."""
        self._items.append(item)

    def ready(self) -> bool:
        """Check if batch is full and ready to process."""
        return len(self._items) >= self.batch_size

    def pending(self) -> bool:
        """Check if there are any pending items."""
        return len(self._items) > 0

    def flush(self) -> List[Any]:
        """Get and clear accumulated items."""
        items = self._items
        self._items = []
        return items

    def __len__(self) -> int:
        return len(self._items)


# =============================================================================
# Link-Level Parallelism
# =============================================================================

@contextmanager
def link_parallel_context(
    link_ids: Sequence[str],
    backend: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> Generator[ParallelExecutor, None, None]:
    """
    Context manager for processing links in parallel.

    Automatically adjusts worker count based on number of links.

    Args:
        link_ids: List of link identifiers
        backend: Override parallel backend
        max_workers: Override max workers (capped at link count)

    Yields:
        ParallelExecutor configured for link processing
    """
    n_links = len(link_ids)
    max_workers = max_workers or config.max_workers

    # Don't use more workers than links
    effective_workers = min(max_workers, n_links)

    # Use sequential for single link
    if n_links <= 1:
        effective_backend = BACKEND_NONE
    else:
        effective_backend = backend or config.parallel_backend

    logger.debug(
        f"Link parallel: {n_links} links, {effective_workers} workers, "
        f"backend={effective_backend}"
    )

    with ParallelExecutor(
        backend=effective_backend,
        max_workers=effective_workers,
    ) as executor:
        yield executor
