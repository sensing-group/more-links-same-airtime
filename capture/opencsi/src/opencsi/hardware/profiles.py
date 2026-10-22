"""
Hardware Profile Loader for OpenCSI.

This module loads hardware-specific defaults for cold start initialization.
When no learned baseline exists yet, OpenCSI uses these defaults to
provide reasonable performance from day 0.

Supports schema v2 with:
- Subcarrier configurations per bandwidth
- Profile inheritance
- Hardware capabilities generation
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from opencsi.hardware.capabilities import HardwareCapabilities, SubcarrierConfig

logger = logging.getLogger(__name__)


# Default profile path (relative to this module)
DEFAULT_PROFILES_PATH = Path(__file__).parent / "profiles.json"

# Default calibration values for cold-start when no profile is available
DEFAULT_NOISE_FLOOR_DBM = -95.0
DEFAULT_AMPLITUDE_SIGMA = 2.5
DEFAULT_PHASE_SIGMA_RAD = 0.8
DEFAULT_SNR_DB = 25.0


class HardwareProfileLoader:
    """
    Loads and manages hardware profiles for OpenCSI AutoCalibrator.

    Hardware profiles provide sensible defaults for different ESP32 variants,
    allowing the calibrator to work reasonably well before learning baselines.
    """

    def __init__(self, profiles_path: Optional[Path] = None):
        """
        Initialize the profile loader.

        Args:
            profiles_path: Path to profiles.json (uses default if None)
        """
        self._profiles_path = profiles_path or DEFAULT_PROFILES_PATH
        self._profiles: Dict[str, dict] = {}
        self._loaded = False

    def _load_profiles(self):
        """Load profiles from JSON file."""
        if self._loaded:
            return

        try:
            with open(self._profiles_path, 'r') as f:
                data = json.load(f)

            # Filter out metadata keys (starting with _)
            self._profiles = {
                k: v for k, v in data.items()
                if not k.startswith('_')
            }
            self._loaded = True
            logger.info(f"Loaded {len(self._profiles)} hardware profiles from {self._profiles_path}")

        except FileNotFoundError:
            logger.debug(f"Hardware profiles not found at {self._profiles_path}, using defaults")
            self._profiles = self._get_default_profiles()
            self._loaded = True

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in hardware profiles: {e}")
            self._profiles = self._get_default_profiles()
            self._loaded = True

    def _get_default_profiles(self) -> Dict[str, dict]:
        """Return built-in default profiles."""
        return {
            "GENERIC_ESP32": {
                "description": "Generic ESP32 defaults",
                "typical_noise_floor_dbm": -95,
                "typical_amplitude_sigma": 2.5,
                "typical_phase_sigma_rad": 0.8,
                "typical_snr_db": 25
            },
            "ESP32_WROOM": {
                "description": "ESP32-WROOM module",
                "typical_noise_floor_dbm": -95,
                "typical_amplitude_sigma": 2.5,
                "typical_phase_sigma_rad": 0.8,
                "typical_snr_db": 25
            },
            "ESP32_C3": {
                "description": "ESP32-C3 RISC-V",
                "typical_noise_floor_dbm": -94,
                "typical_amplitude_sigma": 2.3,
                "typical_phase_sigma_rad": 0.75,
                "typical_snr_db": 26
            },
            "ESP32_S3": {
                "description": "ESP32-S3 dual-core",
                "typical_noise_floor_dbm": -94,
                "typical_amplitude_sigma": 2.2,
                "typical_phase_sigma_rad": 0.7,
                "typical_snr_db": 27
            }
        }

    def get_profile(self, hardware_type: str) -> Optional[dict]:
        """
        Get profile for a specific hardware type.

        Args:
            hardware_type: Hardware identifier (e.g., "ESP32_WROOM", "ESP32_S3")

        Returns:
            Profile dict with hardware defaults, or None if not found
        """
        self._load_profiles()
        hw_upper = self._normalize_hardware_type(hardware_type)

        # Try exact match first
        if hw_upper in self._profiles:
            return self._profiles[hw_upper]

        # Try partial match
        for key, profile in self._profiles.items():
            if hw_upper in key or key in hw_upper:
                logger.info(f"Using profile '{key}' for hardware '{hardware_type}'")
                return profile

        # Fall back to generic
        if "GENERIC_ESP32" in self._profiles:
            logger.info(f"Using GENERIC_ESP32 profile for unknown hardware '{hardware_type}'")
            return self._profiles["GENERIC_ESP32"]

        logger.warning(f"No profile found for hardware '{hardware_type}'")
        return None

    def get_profile_for_mac(
        self,
        mac_address: str,
        mac_to_hardware_map: Optional[Dict[str, str]] = None
    ) -> Optional[dict]:
        """
        Get profile for a specific device by MAC address.

        Args:
            mac_address: Device MAC address
            mac_to_hardware_map: Optional mapping of MAC to hardware type

        Returns:
            Profile dict, or None if no mapping exists
        """
        if mac_to_hardware_map and mac_address in mac_to_hardware_map:
            hardware_type = mac_to_hardware_map[mac_address]
            return self.get_profile(hardware_type)

        # Default to generic
        return self.get_profile("GENERIC_ESP32")

    def list_profiles(self) -> list:
        """List available hardware profile names."""
        self._load_profiles()
        return list(self._profiles.keys())

    def get_all_profiles(self) -> Dict[str, dict]:
        """Get all loaded profiles."""
        self._load_profiles()
        return self._profiles.copy()

    def _normalize_hardware_type(self, hardware_type: str) -> str:
        """Normalize hardware type string to uppercase with underscores."""
        return hardware_type.upper().replace("-", "_").replace(" ", "_")

    def get_capabilities(self, hardware_type: str) -> Optional['HardwareCapabilities']:
        """
        Get HardwareCapabilities for a specific hardware type.

        Creates a HardwareCapabilities object from the JSON profile,
        enabling dynamic configuration of the sanitization pipeline.

        Args:
            hardware_type: Hardware identifier (e.g., "ESP32_C6", "INTEL_5300")

        Returns:
            HardwareCapabilities instance, or None if profile not found
        """
        # Import here to avoid circular imports
        from opencsi.hardware.capabilities import (
            HardwareCapabilities,
            SubcarrierConfig,
        )

        profile = self._get_resolved_profile(hardware_type)
        if profile is None:
            return None

        # Check if profile has subcarrier_configs (schema v2)
        if 'subcarrier_configs' not in profile:
            # Schema v1 profile - create minimal capabilities
            return self._create_v1_capabilities(hardware_type, profile)

        # Build SubcarrierConfig for each bandwidth
        subcarrier_configs: Dict[int, SubcarrierConfig] = {}
        for bw_str, config in profile['subcarrier_configs'].items():
            bw_mhz = int(bw_str)
            subcarrier_configs[bw_mhz] = SubcarrierConfig(
                num_subcarriers=config['num_subcarriers'],
                pilot_indices=tuple(config['pilot_indices']),
                dc_index=config.get('dc_index'),
                guard_indices=tuple(config.get('guard_indices', [])),
            )

        # Get calibration defaults
        cal_defaults = profile.get('calibration_defaults', {})

        return HardwareCapabilities(
            hardware_type=hardware_type.lower(),
            supported_bandwidths=profile.get('supported_bandwidths', [20]),
            supported_bands=profile.get('supported_bands', ['2.4GHz']),
            max_tx_antennas=profile.get('max_tx_antennas', 1),
            max_rx_antennas=profile.get('max_rx_antennas', 1),
            subcarrier_configs=subcarrier_configs,
            supports_mimo=profile.get('supports_mimo', False),
            has_reliable_agc=profile.get('has_reliable_agc', False),
            typical_noise_floor_dbm=float(
                cal_defaults.get('typical_noise_floor_dbm', DEFAULT_NOISE_FLOOR_DBM)
            ),
        )

    def _get_resolved_profile(self, hardware_type: str) -> Optional[dict]:
        """
        Get a profile with inheritance resolved.

        If a profile has 'inherits' field, merge parent profile values.
        """
        self._load_profiles()
        hw_upper = self._normalize_hardware_type(hardware_type)

        # Try exact match first
        if hw_upper not in self._profiles:
            # Try partial match
            for key in self._profiles:
                if hw_upper in key or key in hw_upper:
                    hw_upper = key
                    break
            else:
                return None

        profile = self._profiles[hw_upper].copy()

        # Resolve inheritance
        if 'inherits' in profile:
            parent_name = profile['inherits']
            parent = self._get_resolved_profile(parent_name)
            if parent:
                # Merge: parent values, then child overrides
                merged = parent.copy()
                for key, value in profile.items():
                    if key == 'inherits':
                        continue
                    if key == 'calibration_defaults' and key in merged:
                        # Merge calibration defaults
                        merged[key] = {**merged[key], **value}
                    else:
                        merged[key] = value
                profile = merged

        return profile

    def _create_v1_capabilities(
        self,
        hardware_type: str,
        profile: dict
    ) -> 'HardwareCapabilities':
        """Create capabilities from a v1 (legacy) profile."""
        from opencsi.hardware.capabilities import (
            HardwareCapabilities,
            DEFAULT_HT20_CONFIG,
        )

        return HardwareCapabilities(
            hardware_type=hardware_type.lower(),
            supported_bandwidths=[20],
            supported_bands=['2.4GHz'],
            max_tx_antennas=1,
            max_rx_antennas=1,
            subcarrier_configs={20: DEFAULT_HT20_CONFIG},
            supports_mimo=False,
            has_reliable_agc=False,
            typical_noise_floor_dbm=float(
                profile.get('typical_noise_floor_dbm', DEFAULT_NOISE_FLOOR_DBM)
            ),
        )

    def detect_hardware(self, device_info: dict) -> str:
        """
        Attempt to detect hardware type from device info.

        Args:
            device_info: Dict with device information (model, chip_id, etc.)

        Returns:
            Hardware type string for profile lookup
        """
        model = device_info.get('model', '').upper()
        chip = device_info.get('chip', '').upper()

        # ESP32-S3 detection
        if 'S3' in model or 'S3' in chip:
            return 'ESP32_S3'

        # ESP32-C3 detection
        if 'C3' in model or 'C3' in chip:
            return 'ESP32_C3'

        # ESP32-C6 detection
        if 'C6' in model or 'C6' in chip:
            return 'ESP32_C6'

        # WROVER detection (has PSRAM)
        if 'WROVER' in model or device_info.get('has_psram', False):
            return 'ESP32_WROVER'

        # WROOM detection
        if 'WROOM' in model:
            return 'ESP32_WROOM'

        # Default
        return 'GENERIC_ESP32'


# Module-level convenience function
_default_loader: Optional[HardwareProfileLoader] = None


def get_hardware_profile(hardware_type: str = "GENERIC_ESP32") -> dict:
    """
    Convenience function to get a hardware profile.

    Args:
        hardware_type: Hardware identifier

    Returns:
        Profile dict with hardware defaults
    """
    global _default_loader
    if _default_loader is None:
        _default_loader = HardwareProfileLoader()

    profile = _default_loader.get_profile(hardware_type)
    if profile is None:
        # Return minimal defaults
        return {
            "typical_noise_floor_dbm": DEFAULT_NOISE_FLOOR_DBM,
            "typical_amplitude_sigma": DEFAULT_AMPLITUDE_SIGMA,
            "typical_phase_sigma_rad": DEFAULT_PHASE_SIGMA_RAD,
            "typical_snr_db": DEFAULT_SNR_DB
        }
    return profile
