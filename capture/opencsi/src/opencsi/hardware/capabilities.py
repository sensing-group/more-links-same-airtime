"""
Hardware Capabilities Abstraction for OpenCSI.

This module defines hardware-agnostic capability descriptors that allow
the calibration layer to dynamically adapt to different WiFi CSI hardware.

Supports WiFi 4/5/6/7 with bandwidths from 20 MHz to 320 MHz.

Example:
    # ESP32 HT20 configuration
    config = SubcarrierConfig(
        num_subcarriers=52,
        pilot_indices=(6, 20, 32, 46),
        dc_index=26
    )

    capabilities = HardwareCapabilities(
        hardware_type="esp32",
        supported_bandwidths=[20, 40],
        supported_bands=["2.4GHz", "5GHz"],
        subcarrier_configs={20: config, 40: ht40_config}
    )
"""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


# =============================================================================
# Standard Feature Names
# =============================================================================
# Use these constants when querying capabilities.has_feature()

# Antenna and MIMO features
FEATURE_MIMO = "mimo"                          # Full MIMO matrix available
FEATURE_MULTI_ANTENNA_RX = "multi_antenna_rx"  # Multiple RX antennas
FEATURE_MULTI_ANTENNA_TX = "multi_antenna_tx"  # Multiple TX antennas
FEATURE_BEAMFORMING = "beamforming"            # Beamforming feedback

# Band and bandwidth features
FEATURE_BAND_2_4GHZ = "band_2.4ghz"            # 2.4 GHz support
FEATURE_BAND_5GHZ = "band_5ghz"                # 5 GHz support
FEATURE_BAND_6GHZ = "band_6ghz"                # 6 GHz support (WiFi 6E/7)
FEATURE_BW_40MHZ = "bw_40mhz"                  # 40 MHz bandwidth
FEATURE_BW_80MHZ = "bw_80mhz"                  # 80 MHz bandwidth
FEATURE_BW_160MHZ = "bw_160mhz"                # 160 MHz bandwidth
FEATURE_BW_320MHZ = "bw_320mhz"                # 320 MHz bandwidth (WiFi 7)

# WiFi standard features
FEATURE_WIFI4 = "wifi4"                        # 802.11n (HT)
FEATURE_WIFI5 = "wifi5"                        # 802.11ac (VHT)
FEATURE_WIFI6 = "wifi6"                        # 802.11ax (HE)
FEATURE_WIFI7 = "wifi7"                        # 802.11be (EHT)

# Data quality features
FEATURE_RELIABLE_AGC = "reliable_agc"          # AGC state reliably reported
FEATURE_PHASE_CALIBRATION = "phase_calibration"  # Hardware phase calibration
FEATURE_TIMESTAMP_SYNC = "timestamp_sync"      # Precise timestamps available
FEATURE_RAW_CSI = "raw_csi"                    # Unprocessed CSI available

# Processing features
FEATURE_HIGH_SUBCARRIER_COUNT = "high_subcarrier_count"  # >100 subcarriers
FEATURE_GPU_ACCELERATION = "gpu_acceleration"  # GPU processing supported

# Threshold for high subcarrier count
HIGH_SUBCARRIER_THRESHOLD = 100

# WiFi 6 (HE) subcarrier counts for auto-detection
# These are the standard HE resource unit sizes across bandwidths
WIFI6_HE_SUBCARRIER_COUNTS = frozenset({242, 484, 996, 1992})


# =============================================================================
# WiFi Standard Reference
# =============================================================================
# | Standard  | Bandwidths (MHz)     | Max Subcarriers | Notes              |
# |-----------|----------------------|-----------------|---------------------|
# | WiFi 4    | 20, 40               | 114 (HT40)      | Legacy HT           |
# | WiFi 5    | 20, 40, 80, 160      | 512 (VHT160)    | VHT                 |
# | WiFi 6    | 20, 40, 80, 160      | 1024 (HE160)    | HE, OFDMA           |
# | WiFi 6E/7 | 20-320               | 2048 (EHT320)   | 6 GHz band          |
# =============================================================================


@dataclass(frozen=True)
class SubcarrierConfig:
    """
    Subcarrier configuration for a specific bandwidth.

    Different hardware and bandwidth modes have different numbers of
    subcarriers and pilot positions. This configuration allows the
    sanitization layer to adapt dynamically.

    Attributes:
        num_subcarriers: Total number of data + pilot subcarriers
        pilot_indices: Tuple of pilot subcarrier positions (0-indexed)
        dc_index: DC (center) subcarrier index, if applicable
        guard_indices: Guard band subcarrier indices (typically nulled)

    Example (HT20 - 802.11n/WiFi 4):
        SubcarrierConfig(
            num_subcarriers=52,
            pilot_indices=(6, 20, 32, 46),  # 4 pilots
            dc_index=26
        )

    Example (HE80 - 802.11ax/WiFi 6):
        SubcarrierConfig(
            num_subcarriers=242,
            pilot_indices=(...),  # 8 pilots
            dc_index=121
        )
    """
    num_subcarriers: int
    pilot_indices: Tuple[int, ...]
    dc_index: Optional[int] = None
    guard_indices: Tuple[int, ...] = ()

    def __post_init__(self):
        """Validate configuration."""
        if self.num_subcarriers <= 0:
            raise ValueError(f"num_subcarriers must be positive, got {self.num_subcarriers}")

        for idx in self.pilot_indices:
            if idx < 0 or idx >= self.num_subcarriers:
                raise ValueError(
                    f"Pilot index {idx} out of range [0, {self.num_subcarriers})"
                )

        if self.dc_index is not None:
            if self.dc_index < 0 or self.dc_index >= self.num_subcarriers:
                raise ValueError(
                    f"DC index {self.dc_index} out of range [0, {self.num_subcarriers})"
                )

    @property
    def num_pilots(self) -> int:
        """Number of pilot subcarriers."""
        return len(self.pilot_indices)

    @property
    def num_guards(self) -> int:
        """Number of guard subcarriers."""
        return len(self.guard_indices)

    @property
    def num_data_subcarriers(self) -> int:
        """Number of data subcarriers (total - pilots - guards - dc)."""
        excluded = self.num_pilots + self.num_guards
        if self.dc_index is not None:
            excluded += 1
        return self.num_subcarriers - excluded


@dataclass
class HardwareCapabilities:
    """
    Hardware capabilities descriptor for a CSI extractor.

    This descriptor enables dynamic configuration throughout the OpenCSI
    pipeline. The calibration and sanitization layers query these capabilities
    to adapt their behavior without hardcoded assumptions.

    Attributes:
        hardware_type: Unique identifier (e.g., "esp32", "intel_5300", "nexmon")
        supported_bandwidths: List of supported bandwidths in MHz
        supported_bands: List of supported frequency bands
        max_tx_antennas: Maximum number of TX antennas
        max_rx_antennas: Maximum number of RX antennas
        subcarrier_configs: Mapping from bandwidth (MHz) to SubcarrierConfig
        supports_mimo: Whether hardware reports full MIMO matrix
        has_reliable_agc: Whether AGC state is reliably reported
        typical_noise_floor_dbm: Expected noise floor for cold-start

    Example:
        caps = HardwareCapabilities(
            hardware_type="esp32",
            supported_bandwidths=[20, 40],
            supported_bands=["2.4GHz"],
            subcarrier_configs={
                20: SubcarrierConfig(52, (6, 20, 32, 46), 26),
                40: SubcarrierConfig(114, (6, 20, 32, 46, 60, 74, 86, 100))
            }
        )

        # Query capabilities dynamically
        pilot_indices = caps.get_pilot_indices(bandwidth_mhz=20)
        num_subcarriers = caps.get_num_subcarriers(bandwidth_mhz=20)
    """
    hardware_type: str
    supported_bandwidths: List[int]
    supported_bands: List[str]
    max_tx_antennas: int = 1
    max_rx_antennas: int = 1
    subcarrier_configs: Dict[int, SubcarrierConfig] = field(default_factory=dict)
    supports_mimo: bool = False
    has_reliable_agc: bool = False
    typical_noise_floor_dbm: float = -95.0

    def __post_init__(self):
        """Validate capabilities."""
        if not self.hardware_type:
            raise ValueError("hardware_type cannot be empty")

        if not self.supported_bandwidths:
            raise ValueError("supported_bandwidths cannot be empty")

        # Verify configs exist for all bandwidths
        for bw in self.supported_bandwidths:
            if bw not in self.subcarrier_configs:
                raise ValueError(
                    f"Missing subcarrier_config for bandwidth {bw} MHz"
                )

    def _get_config(self, bandwidth_mhz: int) -> SubcarrierConfig:
        """Get subcarrier config for bandwidth, raising KeyError if unsupported."""
        if bandwidth_mhz not in self.subcarrier_configs:
            raise KeyError(
                f"Bandwidth {bandwidth_mhz} MHz not supported. "
                f"Supported: {self.supported_bandwidths}"
            )
        return self.subcarrier_configs[bandwidth_mhz]

    def get_pilot_indices(self, bandwidth_mhz: int = 20) -> Tuple[int, ...]:
        """
        Get pilot subcarrier indices for a given bandwidth.

        Args:
            bandwidth_mhz: Bandwidth in MHz (default: 20)

        Returns:
            Tuple of pilot indices

        Raises:
            KeyError: If bandwidth not supported
        """
        return self._get_config(bandwidth_mhz).pilot_indices

    def get_num_subcarriers(self, bandwidth_mhz: int = 20) -> int:
        """
        Get number of subcarriers for a given bandwidth.

        Args:
            bandwidth_mhz: Bandwidth in MHz (default: 20)

        Returns:
            Number of subcarriers

        Raises:
            KeyError: If bandwidth not supported
        """
        return self._get_config(bandwidth_mhz).num_subcarriers

    def get_dc_index(self, bandwidth_mhz: int = 20) -> Optional[int]:
        """
        Get DC subcarrier index for a given bandwidth.

        Args:
            bandwidth_mhz: Bandwidth in MHz (default: 20)

        Returns:
            DC index or None if not applicable

        Raises:
            KeyError: If bandwidth not supported
        """
        return self._get_config(bandwidth_mhz).dc_index

    def get_subcarrier_config(self, bandwidth_mhz: int = 20) -> SubcarrierConfig:
        """
        Get full subcarrier configuration for a bandwidth.

        Args:
            bandwidth_mhz: Bandwidth in MHz (default: 20)

        Returns:
            SubcarrierConfig for the bandwidth

        Raises:
            KeyError: If bandwidth not supported
        """
        return self._get_config(bandwidth_mhz)

    def supports_bandwidth(self, bandwidth_mhz: int) -> bool:
        """Check if a bandwidth is supported."""
        return bandwidth_mhz in self.supported_bandwidths

    def supports_band(self, band: str) -> bool:
        """Check if a frequency band is supported."""
        return band in self.supported_bands

    def has_feature(self, feature: str) -> bool:
        """
        Check if hardware has a specific feature.

        This method enables algorithms to adapt based on hardware capabilities
        without hardcoded assumptions. Features are auto-detected from the
        capabilities or can be explicitly set.

        Args:
            feature: Feature name (use FEATURE_* constants)

        Returns:
            True if feature is available

        Examples:
            if caps.has_feature(FEATURE_MIMO):
                process_mimo_matrix(csi)

            if caps.has_feature(FEATURE_WIFI6):
                use_ofdma_processing()

            if caps.has_feature(FEATURE_GPU_ACCELERATION):
                enable_gpu_backend()
        """
        feature_lower = feature.lower()

        # Check explicit features first (supports custom features)
        if hasattr(self, '_explicit_features'):
            if feature_lower in self._explicit_features:
                return True

        # Auto-detect standard features
        return self._detect_feature(feature_lower)

    def _detect_feature(self, feature: str) -> bool:
        """Auto-detect feature from capabilities."""
        # Direct attribute mappings
        if feature == FEATURE_MIMO:
            return self.supports_mimo
        if feature == FEATURE_MULTI_ANTENNA_RX:
            return self.max_rx_antennas > 1
        if feature == FEATURE_MULTI_ANTENNA_TX:
            return self.max_tx_antennas > 1
        if feature == FEATURE_RELIABLE_AGC:
            return self.has_reliable_agc

        # Band features - check membership in supported_bands
        band_mapping = {
            FEATURE_BAND_2_4GHZ: "2.4GHz",
            FEATURE_BAND_5GHZ: "5GHz",
            FEATURE_BAND_6GHZ: "6GHz",
        }
        if feature in band_mapping:
            return band_mapping[feature] in self.supported_bands

        # Bandwidth features - check membership in supported_bandwidths
        bandwidth_mapping = {
            FEATURE_BW_40MHZ: 40,
            FEATURE_BW_80MHZ: 80,
            FEATURE_BW_160MHZ: 160,
            FEATURE_BW_320MHZ: 320,
        }
        if feature in bandwidth_mapping:
            return bandwidth_mapping[feature] in self.supported_bandwidths

        # WiFi standard features (inferred from bandwidths/subcarriers)
        if feature == FEATURE_WIFI4:
            return bool(self.supported_bandwidths)
        if feature == FEATURE_WIFI5:
            return 80 in self.supported_bandwidths
        if feature == FEATURE_WIFI6:
            return self._has_wifi6_subcarrier_count()
        if feature == FEATURE_WIFI7:
            return 320 in self.supported_bandwidths

        # Processing features
        if feature == FEATURE_HIGH_SUBCARRIER_COUNT:
            return self._get_max_subcarrier_count() > HIGH_SUBCARRIER_THRESHOLD

        return False

    def _has_wifi6_subcarrier_count(self) -> bool:
        """Check if any subcarrier config matches WiFi 6 (HE) counts."""
        return any(
            config.num_subcarriers in WIFI6_HE_SUBCARRIER_COUNTS
            for config in self.subcarrier_configs.values()
        )

    def _get_max_subcarrier_count(self) -> int:
        """Get the maximum subcarrier count across all configurations."""
        return max(
            (config.num_subcarriers for config in self.subcarrier_configs.values()),
            default=0
        )

    def list_features(self) -> FrozenSet[str]:
        """
        List all detected features for this hardware.

        Returns:
            Frozenset of feature names

        Example:
            features = caps.list_features()
            print(f"Hardware features: {features}")
            # {'wifi4', 'band_2.4ghz', 'bw_40mhz', ...}
        """
        all_features = [
            FEATURE_MIMO,
            FEATURE_MULTI_ANTENNA_RX,
            FEATURE_MULTI_ANTENNA_TX,
            FEATURE_BAND_2_4GHZ,
            FEATURE_BAND_5GHZ,
            FEATURE_BAND_6GHZ,
            FEATURE_BW_40MHZ,
            FEATURE_BW_80MHZ,
            FEATURE_BW_160MHZ,
            FEATURE_BW_320MHZ,
            FEATURE_WIFI4,
            FEATURE_WIFI5,
            FEATURE_WIFI6,
            FEATURE_WIFI7,
            FEATURE_RELIABLE_AGC,
            FEATURE_HIGH_SUBCARRIER_COUNT,
        ]

        detected = frozenset(f for f in all_features if self.has_feature(f))

        # Add explicit features if any
        if hasattr(self, '_explicit_features'):
            detected = detected | self._explicit_features

        return detected

    def add_feature(self, feature: str) -> None:
        """
        Explicitly add a feature.

        Use this to mark custom or undetectable features.

        Args:
            feature: Feature name
        """
        if not hasattr(self, '_explicit_features'):
            object.__setattr__(self, '_explicit_features', set())
        self._explicit_features.add(feature.lower())

    def require_features(self, *features: str) -> None:
        """
        Assert that all specified features are available.

        Raises:
            ValueError: If any feature is missing

        Example:
            caps.require_features(FEATURE_MIMO, FEATURE_BW_80MHZ)
        """
        missing = [f for f in features if not self.has_feature(f)]
        if missing:
            available = self.list_features()
            raise ValueError(
                f"Hardware '{self.hardware_type}' missing required features: {missing}. "
                f"Available: {available}"
            )


# =============================================================================
# Default Configurations for Common Hardware
# =============================================================================

# HT20 (802.11n, 20 MHz) - Used by ESP32, Intel 5300 (subset)
DEFAULT_HT20_CONFIG = SubcarrierConfig(
    num_subcarriers=52,
    pilot_indices=(6, 20, 32, 46),
    dc_index=26
)

# HT40 (802.11n, 40 MHz) - ESP32 with HT40 mode
DEFAULT_HT40_CONFIG = SubcarrierConfig(
    num_subcarriers=114,
    pilot_indices=(6, 20, 32, 46, 60, 74, 86, 100),
    dc_index=57
)

# VHT80 (802.11ac, 80 MHz) - Nexmon, modern hardware
# Guard indices from 802.11ac null carrier definitions (0-based array indices)
DEFAULT_VHT80_CONFIG = SubcarrierConfig(
    num_subcarriers=256,
    pilot_indices=(25, 53, 89, 117, 139, 167, 203, 231),
    dc_index=128,
    guard_indices=(0, 1, 2, 3, 4, 5, 251, 252, 253, 254, 255),
)

# VHT160 (802.11ac, 160 MHz) - High-end hardware
DEFAULT_VHT160_CONFIG = SubcarrierConfig(
    num_subcarriers=512,
    pilot_indices=(25, 53, 89, 117, 139, 167, 203, 231,
                   281, 309, 345, 373, 395, 423, 459, 487),
    dc_index=256
)

# HE80 (802.11ax/WiFi 6, 80 MHz) - WiFi 6 hardware
DEFAULT_HE80_CONFIG = SubcarrierConfig(
    num_subcarriers=242,
    pilot_indices=(24, 48, 72, 96, 145, 169, 193, 217),
    dc_index=121
)

# Intel 5300 - Special case with only 30 subcarrier groups
INTEL_5300_HT20_CONFIG = SubcarrierConfig(
    num_subcarriers=30,
    pilot_indices=(7, 21),  # Approximate pilot positions
    dc_index=15
)

# Nexmon BCM43455 configurations
# Guard indices from 802.11 null carrier definitions mapped to 0-based array indices.
# Nexmon outputs all subcarriers including guards/nulls.
NEXMON_HT20_CONFIG = SubcarrierConfig(
    num_subcarriers=64,
    pilot_indices=(11, 25, 39, 53),
    dc_index=32,
    guard_indices=(0, 1, 2, 3, 59, 60, 61, 62, 63),
)

NEXMON_HT40_CONFIG = SubcarrierConfig(
    num_subcarriers=128,
    pilot_indices=(11, 39, 53, 75, 89, 117),
    dc_index=64,
    guard_indices=(0, 1, 2, 3, 4, 5, 123, 124, 125, 126, 127),
)


def create_esp32_capabilities() -> HardwareCapabilities:
    """Create default ESP32 capabilities."""
    return HardwareCapabilities(
        hardware_type="esp32",
        supported_bandwidths=[20, 40],
        supported_bands=["2.4GHz"],
        max_tx_antennas=1,
        max_rx_antennas=1,
        subcarrier_configs={
            20: DEFAULT_HT20_CONFIG,
            40: DEFAULT_HT40_CONFIG,
        },
        supports_mimo=False,
        has_reliable_agc=False,
        typical_noise_floor_dbm=-95.0
    )


def create_intel5300_capabilities() -> HardwareCapabilities:
    """Create default Intel 5300 capabilities."""
    return HardwareCapabilities(
        hardware_type="intel_5300",
        supported_bandwidths=[20],
        supported_bands=["2.4GHz", "5GHz"],
        max_tx_antennas=3,
        max_rx_antennas=3,
        subcarrier_configs={
            20: INTEL_5300_HT20_CONFIG,
        },
        supports_mimo=True,
        has_reliable_agc=True,
        typical_noise_floor_dbm=-92.0
    )


def create_nexmon_capabilities() -> HardwareCapabilities:
    """Create default Nexmon (BCM43455) capabilities."""
    return HardwareCapabilities(
        hardware_type="nexmon",
        supported_bandwidths=[20, 40, 80],
        supported_bands=["2.4GHz", "5GHz"],
        max_tx_antennas=1,
        max_rx_antennas=1,
        subcarrier_configs={
            20: NEXMON_HT20_CONFIG,
            40: NEXMON_HT40_CONFIG,
            80: DEFAULT_VHT80_CONFIG,
        },
        supports_mimo=False,
        has_reliable_agc=False,
        typical_noise_floor_dbm=-93.0
    )
