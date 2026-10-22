"""
OpenCSI Environment Configuration.

Centralized environment variable handling inspired by OpenMPI's MCA framework.
All configuration can be overridden via OPENCSI_* environment variables.

Environment Variables:
    OPENCSI_DEFAULT_HARDWARE: Default hardware type for get_extractor()
        Values: "esp32", "esp32_c6", "intel_5300", "nexmon", "atheros"
        Default: "esp32"

    OPENCSI_LOG_LEVEL: Logging verbosity
        Values: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        Default: "WARNING"

    OPENCSI_PROFILE_PATH: Custom path to hardware profiles JSON
        Values: Absolute or relative path to profiles.json
        Default: Built-in profiles.json

    OPENCSI_DEBUG: Enable debug mode with verbose output
        Values: "1", "true", "yes" (case insensitive)
        Default: "0" (disabled)

    OPENCSI_BASELINE_CACHE: Enable baseline caching to disk
        Values: "1", "true", "yes" (case insensitive)
        Default: "0" (disabled)

    OPENCSI_CACHE_DIR: Directory for cached data
        Values: Absolute or relative path
        Default: ".opencsi_cache"

    OPENCSI_ARRAY_BACKEND: Numerical backend for array operations
        Values: "numpy", "cupy" (requires CuPy installation)
        Default: "numpy"

    OPENCSI_PARALLEL_BACKEND: Parallelization strategy
        Values: "none", "thread", "process"
        Default: "thread"

    OPENCSI_MAX_WORKERS: Maximum parallel workers
        Values: Positive integer, or "auto" for CPU count
        Default: "auto"

    OPENCSI_BATCH_SIZE: Packet batch size for parallel processing
        Values: Positive integer
        Default: "32"

Usage:
    from opencsi.core.env import config

    # Access configuration
    hardware = config.default_hardware
    if config.debug:
        print("Debug mode enabled")

    # Override at runtime (not recommended, use env vars)
    config.set("default_hardware", "intel_5300")
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

ENV_PREFIX = "OPENCSI_"

# Default values
DEFAULT_HARDWARE = "esp32"
DEFAULT_LOG_LEVEL = "WARNING"
DEFAULT_CACHE_DIR = ".opencsi_cache"
DEFAULT_ARRAY_BACKEND = "numpy"
DEFAULT_PARALLEL_BACKEND = "thread"
DEFAULT_BATCH_SIZE = 32

# Boolean true values (case insensitive)
TRUTHY_VALUES = {"1", "true", "yes", "on", "enabled"}


def _parse_bool(value: Optional[str]) -> bool:
    """Parse environment variable as boolean."""
    if value is None:
        return False
    return value.lower() in TRUTHY_VALUES


def _get_env(name: str, default: str = "") -> str:
    """Get environment variable with OPENCSI_ prefix."""
    return os.environ.get(f"{ENV_PREFIX}{name}", default)


def _get_max_workers() -> int:
    """Get max workers from env or auto-detect CPU count."""
    env_val = _get_env("MAX_WORKERS", "auto")
    if env_val.lower() == "auto":
        return os.cpu_count() or 4
    try:
        return max(1, int(env_val))
    except ValueError:
        return os.cpu_count() or 4


@dataclass
class OpenCSIConfig:
    """
    Centralized configuration for OpenCSI.

    Reads from environment variables on initialization.
    Values can be overridden at runtime via set() method.

    Attributes:
        default_hardware: Default hardware type for get_extractor()
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        profile_path: Custom path to hardware profiles JSON
        debug: Enable debug mode with verbose output
        baseline_cache: Enable baseline caching to disk
        cache_dir: Directory for cached data
        array_backend: Numerical backend ("numpy" or "cupy")
        parallel_backend: Parallelization strategy ("none", "thread", "process")
        max_workers: Maximum number of parallel workers
        batch_size: Packet batch size for parallel processing
    """

    default_hardware: str = field(default_factory=lambda: _get_env("DEFAULT_HARDWARE", DEFAULT_HARDWARE))
    log_level: str = field(default_factory=lambda: _get_env("LOG_LEVEL", DEFAULT_LOG_LEVEL))
    profile_path: Optional[str] = field(default_factory=lambda: _get_env("PROFILE_PATH") or None)
    debug: bool = field(default_factory=lambda: _parse_bool(_get_env("DEBUG")))
    baseline_cache: bool = field(default_factory=lambda: _parse_bool(_get_env("BASELINE_CACHE")))
    cache_dir: str = field(default_factory=lambda: _get_env("CACHE_DIR", DEFAULT_CACHE_DIR))
    array_backend: str = field(default_factory=lambda: _get_env("ARRAY_BACKEND", DEFAULT_ARRAY_BACKEND))
    parallel_backend: str = field(default_factory=lambda: _get_env("PARALLEL_BACKEND", DEFAULT_PARALLEL_BACKEND))
    max_workers: int = field(default_factory=_get_max_workers)
    batch_size: int = field(default_factory=lambda: int(_get_env("BATCH_SIZE", str(DEFAULT_BATCH_SIZE))))

    # Runtime overrides (not from env)
    _overrides: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Configure logging based on settings."""
        self._configure_logging()
        if self.debug:
            logger.debug("OpenCSI debug mode enabled")
            logger.debug(f"Configuration: {self}")

    def _configure_logging(self) -> None:
        """Configure OpenCSI logging based on log_level."""
        level = getattr(logging, self.log_level.upper(), logging.WARNING)
        logging.getLogger("opencsi").setLevel(level)

    def set(self, key: str, value: Any) -> None:
        """
        Override a configuration value at runtime.

        Args:
            key: Configuration key (e.g., "default_hardware")
            value: New value

        Note:
            Prefer environment variables for configuration.
            Runtime overrides are primarily for testing.
        """
        if hasattr(self, key):
            self._overrides[key] = value
            object.__setattr__(self, key, value)
            if key == "log_level":
                self._configure_logging()
            logger.debug(f"Config override: {key}={value}")
        else:
            raise KeyError(f"Unknown configuration key: {key}")

    def reset(self, key: Optional[str] = None) -> None:
        """
        Reset configuration to environment variable values.

        Args:
            key: Specific key to reset, or None to reset all
        """
        if key is not None:
            if key in self._overrides:
                del self._overrides[key]
            # Re-read from environment with proper type coercion
            # (re-run the same factory logic as field defaults)
            _defaults = {
                "default_hardware": lambda: _get_env("DEFAULT_HARDWARE", DEFAULT_HARDWARE),
                "log_level": lambda: _get_env("LOG_LEVEL", DEFAULT_LOG_LEVEL),
                "profile_path": lambda: _get_env("PROFILE_PATH") or None,
                "debug": lambda: _parse_bool(_get_env("DEBUG")),
                "baseline_cache": lambda: _parse_bool(_get_env("BASELINE_CACHE")),
                "cache_dir": lambda: _get_env("CACHE_DIR", DEFAULT_CACHE_DIR),
                "array_backend": lambda: _get_env("ARRAY_BACKEND", DEFAULT_ARRAY_BACKEND),
                "parallel_backend": lambda: _get_env("PARALLEL_BACKEND", DEFAULT_PARALLEL_BACKEND),
                "max_workers": _get_max_workers,
                "batch_size": lambda: int(_get_env("BATCH_SIZE", str(DEFAULT_BATCH_SIZE))),
            }
            if key in _defaults:
                object.__setattr__(self, key, _defaults[key]())
        else:
            self._overrides.clear()
            # Reinitialize from environment
            self.__init__()

    def get_cache_path(self, filename: str) -> Path:
        """
        Get path for cached file.

        Args:
            filename: Name of the cache file

        Returns:
            Path object for the cache file
        """
        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / filename

    def get_array_module(self) -> Any:
        """
        Get the array module (numpy or cupy) based on configuration.

        Returns:
            numpy or cupy module

        Raises:
            ImportError: If cupy is requested but not installed
        """
        if self.array_backend == "cupy":
            try:
                import cupy as cp
                return cp
            except ImportError as e:
                raise ImportError(
                    "CuPy requested via OPENCSI_ARRAY_BACKEND=cupy but not installed. "
                    "Install with: pip install cupy-cuda12x (adjust for your CUDA version)"
                ) from e
        else:
            import numpy as np
            return np

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            "default_hardware": self.default_hardware,
            "log_level": self.log_level,
            "profile_path": self.profile_path,
            "debug": self.debug,
            "baseline_cache": self.baseline_cache,
            "cache_dir": self.cache_dir,
            "array_backend": self.array_backend,
            "parallel_backend": self.parallel_backend,
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenCSIConfig":
        """Create configuration from dictionary."""
        cfg = cls()
        for key, value in data.items():
            if hasattr(cfg, key):
                cfg.set(key, value)
        return cfg


# =============================================================================
# Global Configuration Instance
# =============================================================================

# Singleton configuration instance
# Import this in other modules: from opencsi.core.env import config
config = OpenCSIConfig()


def get_config() -> OpenCSIConfig:
    """Get the global configuration instance."""
    return config


def reload_config() -> OpenCSIConfig:
    """
    Reload configuration from environment variables.

    Useful after modifying os.environ at runtime.

    Returns:
        Fresh OpenCSIConfig instance
    """
    global config
    config = OpenCSIConfig()
    return config


# =============================================================================
# Convenience Functions
# =============================================================================

def is_debug() -> bool:
    """Check if debug mode is enabled."""
    return config.debug


def get_default_hardware() -> str:
    """Get the default hardware type."""
    return config.default_hardware


def get_array_module() -> Any:
    """
    Get the configured array module (numpy or cupy).

    This is the main entry point for GPU acceleration.
    Code should use this instead of importing numpy directly
    for operations that can benefit from GPU acceleration.

    Example:
        from opencsi.core.env import get_array_module
        xp = get_array_module()
        data = xp.zeros(100)  # Uses GPU if OPENCSI_ARRAY_BACKEND=cupy
    """
    return config.get_array_module()
