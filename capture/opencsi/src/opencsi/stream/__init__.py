"""Streaming and callback adapters for OpenCSI."""

from opencsi.stream.adapter import CallbackAdapter
from opencsi.stream.async_stream import CalibratedStream

__all__ = [
    "CallbackAdapter",
    "CalibratedStream",
]

# Note: These adapters work with any AutoCalibrator instance.
# Import AutoCalibrator from opencsi.core for full functionality.
