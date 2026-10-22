"""
Hardware Registry for OpenCSI.

Provides a plugin-based discovery system for CSI extractors.
Supports:
- Built-in extractors (ESP32, Intel 5300, Nexmon, Atheros)
- Entry points for community plugins
- Factory function for creating extractors
- Environment variable configuration (OPENCSI_DEFAULT_HARDWARE)

Example:
    # Get extractor by hardware type
    extractor = get_extractor("esp32")
    extractor = get_extractor("esp32_c6", wifi6=True)

    # Use default from environment (OPENCSI_DEFAULT_HARDWARE)
    extractor = get_extractor()

    # Register a custom extractor
    HardwareRegistry.register("my_hardware", MyExtractor)

    # List available extractors
    available = HardwareRegistry.list_available()
"""

import logging
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING

from opencsi.core.env import config

if TYPE_CHECKING:
    from opencsi.hardware.base import CSIExtractor

logger = logging.getLogger(__name__)

# Entry point group for community plugins
ENTRY_POINT_GROUP = "opencsi.hardware"


class HardwareRegistry:
    """
    Registry for hardware-specific CSI extractors.

    Maintains a mapping from hardware type names to extractor classes.
    Supports lazy loading of entry points for community plugins.

    Usage:
        # Register an extractor
        HardwareRegistry.register("my_hardware", MyExtractor)

        # Get an extractor class
        cls = HardwareRegistry.get("esp32")
        extractor = cls()

        # Check available extractors
        available = HardwareRegistry.list_available()
    """

    _extractors: Dict[str, Type['CSIExtractor']] = {}
    _plugins_loaded: bool = False

    @classmethod
    def register(
        cls,
        name: str,
        extractor_class: Type['CSIExtractor'],
        aliases: Optional[List[str]] = None
    ) -> None:
        """
        Register an extractor class.

        Args:
            name: Primary hardware type identifier (e.g., "esp32")
            extractor_class: The CSIExtractor subclass
            aliases: Optional list of alternative names
        """
        name_lower = name.lower()
        cls._extractors[name_lower] = extractor_class
        logger.debug(f"Registered hardware extractor: {name_lower}")

        if aliases:
            for alias in aliases:
                cls._extractors[alias.lower()] = extractor_class

    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        Unregister an extractor.

        Args:
            name: Hardware type identifier

        Returns:
            True if extractor was removed, False if not found
        """
        name_lower = name.lower()
        if name_lower in cls._extractors:
            del cls._extractors[name_lower]
            return True
        return False

    @classmethod
    def get(cls, name: str) -> Optional[Type['CSIExtractor']]:
        """
        Get extractor class by name.

        Loads plugins on first access if not already loaded.

        Args:
            name: Hardware type identifier

        Returns:
            CSIExtractor subclass, or None if not found
        """
        cls._ensure_plugins_loaded()

        name_lower = name.lower()
        if name_lower in cls._extractors:
            return cls._extractors[name_lower]

        # Try partial match
        for key, extractor_cls in cls._extractors.items():
            if name_lower in key or key in name_lower:
                return extractor_cls

        return None

    @classmethod
    def list_available(cls) -> List[str]:
        """
        List all available hardware types.

        Returns:
            List of registered hardware type names
        """
        cls._ensure_plugins_loaded()
        return list(cls._extractors.keys())

    @classmethod
    def _ensure_plugins_loaded(cls) -> None:
        """Load built-in extractors and entry point plugins."""
        if cls._plugins_loaded:
            return

        cls._load_builtin_extractors()
        cls._load_entry_points()
        cls._plugins_loaded = True

    @classmethod
    def _load_builtin_extractors(cls) -> None:
        """Register built-in extractors."""
        try:
            from opencsi.hardware.esp32 import ESP32Extractor
            cls.register("esp32", ESP32Extractor, aliases=[
                "esp32_wroom", "esp32_wrover", "esp32_s2", "esp32_s3", "esp32_c3"
            ])
            # ESP32-C6 uses the same class with wifi6=True flag
            cls.register("esp32_c6", ESP32Extractor)
        except ImportError as e:
            logger.debug(f"ESP32 extractor not available: {e}")

        # Register stub extractors if available
        try:
            from opencsi.hardware.intel5300 import Intel5300Extractor
            cls.register("intel_5300", Intel5300Extractor, aliases=["intel5300", "iwl5300"])
        except ImportError:
            logger.debug("Intel 5300 extractor not available")

        try:
            from opencsi.hardware.nexmon import NexmonExtractor
            cls.register("nexmon", NexmonExtractor, aliases=[
                "nexmon_bcm4339", "nexmon_bcm43455", "nexmon_bcm43456", "bcm43455"
            ])
        except ImportError:
            logger.debug("Nexmon extractor not available")

        try:
            from opencsi.hardware.atheros import AtherosExtractor
            cls.register("atheros", AtherosExtractor, aliases=[
                "atheros_ar9380", "atheros_ar9580", "atheros_qca9558", "ath9k"
            ])
        except ImportError:
            logger.debug("Atheros extractor not available")

        try:
            from opencsi.hardware.iwlwifi import IwlWifiExtractor
            cls.register("iwlwifi", IwlWifiExtractor, aliases=[
                "ax200", "ax201", "ax210", "ax211", "be200", "be202",
                "intel_ax200", "intel_ax210", "intel_be200", "wifi6e", "wifi7"
            ])
        except ImportError:
            logger.debug("IwlWifi extractor not available")

    @classmethod
    def _load_entry_points(cls) -> None:
        """Load extractors from setuptools entry points."""
        try:
            # Python 3.10+ has importlib.metadata in stdlib
            from importlib.metadata import entry_points
        except ImportError:
            try:
                # Fallback for Python 3.9
                from importlib_metadata import entry_points
            except ImportError:
                logger.debug("importlib.metadata not available, skipping entry points")
                return

        try:
            # Python 3.10+ style
            eps = entry_points(group=ENTRY_POINT_GROUP)
        except TypeError:
            # Python 3.9 style
            eps = entry_points().get(ENTRY_POINT_GROUP, [])

        for ep in eps:
            try:
                extractor_class = ep.load()
                cls.register(ep.name, extractor_class)
                logger.info(f"Loaded plugin extractor: {ep.name}")
            except Exception as e:
                logger.warning(f"Failed to load plugin '{ep.name}': {e}")

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (for testing)."""
        cls._extractors.clear()
        cls._plugins_loaded = False


def get_extractor(
    hardware_type: Optional[str] = None,
    **kwargs: Any
) -> 'CSIExtractor':
    """
    Factory function for creating hardware extractors.

    This is the primary API for obtaining a CSI extractor.

    Args:
        hardware_type: Hardware identifier (e.g., "esp32", "intel_5300", "nexmon").
            If None, uses OPENCSI_DEFAULT_HARDWARE environment variable.
        **kwargs: Additional arguments passed to the extractor constructor
            - wifi6: bool - Enable WiFi 6 mode for ESP32-C6

    Returns:
        Configured CSIExtractor instance

    Raises:
        ValueError: If hardware type is not found

    Examples:
        # Basic ESP32
        extractor = get_extractor("esp32")

        # ESP32-C6 with WiFi 6
        extractor = get_extractor("esp32_c6", wifi6=True)

        # Intel 5300
        extractor = get_extractor("intel_5300")

        # Use default from OPENCSI_DEFAULT_HARDWARE
        extractor = get_extractor()
    """
    if hardware_type is None:
        hardware_type = config.default_hardware
        logger.debug(f"Using default hardware from config: {hardware_type}")

    extractor_cls = HardwareRegistry.get(hardware_type)

    if extractor_cls is None:
        available = HardwareRegistry.list_available()
        raise ValueError(
            f"Unknown hardware type: '{hardware_type}'. "
            f"Available: {available}"
        )

    # Handle special cases
    hardware_lower = hardware_type.lower()

    if hardware_lower == "esp32_c6" and "wifi6" not in kwargs:
        kwargs["wifi6"] = True

    # Handle iwlwifi device variants
    iwlwifi_devices = {"ax200", "ax201", "ax210", "ax211", "be200", "be202"}
    if hardware_lower in iwlwifi_devices and "device" not in kwargs:
        kwargs["device"] = hardware_lower

    return extractor_cls(**kwargs)


def list_extractors() -> List[str]:
    """
    List all available hardware extractors.

    Returns:
        List of hardware type names that can be used with get_extractor()
    """
    return HardwareRegistry.list_available()
