"""
Tests for OpenCSI Hardware Abstraction Layer.

Tests the multi-hardware support including:
- HardwareCapabilities and SubcarrierConfig
- Dynamic pilot indices in sanitizer
- Hardware registry and factory function
- Dynamic baseline arrays
- Wavelength conversion functions
"""

import pytest
import numpy as np


class TestSubcarrierConfig:
    """Tests for SubcarrierConfig dataclass."""

    def test_ht20_config(self):
        """Test HT20 configuration."""
        from opencsi.hardware.capabilities import SubcarrierConfig

        config = SubcarrierConfig(
            num_subcarriers=52,
            pilot_indices=(6, 20, 32, 46),
            dc_index=26
        )

        assert config.num_subcarriers == 52
        assert config.num_pilots == 4
        assert config.dc_index == 26
        assert 6 in config.pilot_indices

    def test_he20_config(self):
        """Test WiFi 6 HE20 configuration."""
        from opencsi.hardware.capabilities import SubcarrierConfig

        config = SubcarrierConfig(
            num_subcarriers=242,
            pilot_indices=(24, 48, 72, 96, 145, 169, 193, 217),
            dc_index=121
        )

        assert config.num_subcarriers == 242
        assert config.num_pilots == 8

    def test_invalid_pilot_index_raises(self):
        """Test that invalid pilot index raises ValueError."""
        from opencsi.hardware.capabilities import SubcarrierConfig

        with pytest.raises(ValueError):
            SubcarrierConfig(
                num_subcarriers=52,
                pilot_indices=(6, 20, 32, 100),  # 100 is out of range
                dc_index=26
            )


class TestHardwareCapabilities:
    """Tests for HardwareCapabilities class."""

    def test_esp32_capabilities(self):
        """Test ESP32 capabilities creation."""
        from opencsi.hardware.capabilities import create_esp32_capabilities

        caps = create_esp32_capabilities()

        assert caps.hardware_type == "esp32"
        assert 20 in caps.supported_bandwidths
        assert 40 in caps.supported_bandwidths
        assert "2.4GHz" in caps.supported_bands

    def test_get_pilot_indices(self):
        """Test dynamic pilot index retrieval."""
        from opencsi.hardware.capabilities import create_esp32_capabilities

        caps = create_esp32_capabilities()

        pilots_20 = caps.get_pilot_indices(20)
        pilots_40 = caps.get_pilot_indices(40)

        assert len(pilots_20) == 4
        assert len(pilots_40) == 8

    def test_get_num_subcarriers(self):
        """Test dynamic subcarrier count retrieval."""
        from opencsi.hardware.capabilities import create_esp32_capabilities

        caps = create_esp32_capabilities()

        assert caps.get_num_subcarriers(20) == 52
        assert caps.get_num_subcarriers(40) == 114

    def test_unsupported_bandwidth_raises(self):
        """Test that unsupported bandwidth raises KeyError."""
        from opencsi.hardware.capabilities import create_esp32_capabilities

        caps = create_esp32_capabilities()

        with pytest.raises(KeyError):
            caps.get_pilot_indices(80)  # ESP32 doesn't support 80 MHz

    def test_intel5300_capabilities(self):
        """Test Intel 5300 capabilities."""
        from opencsi.hardware.capabilities import create_intel5300_capabilities

        caps = create_intel5300_capabilities()

        assert caps.hardware_type == "intel_5300"
        assert caps.supports_mimo
        assert caps.max_tx_antennas == 3
        assert caps.get_num_subcarriers(20) == 30

    def test_nexmon_capabilities(self):
        """Test Nexmon capabilities."""
        from opencsi.hardware.capabilities import create_nexmon_capabilities

        caps = create_nexmon_capabilities()

        assert caps.hardware_type == "nexmon"
        assert 80 in caps.supported_bandwidths
        assert "5GHz" in caps.supported_bands


class TestESP32Extractor:
    """Tests for ESP32Extractor."""

    def test_wifi4_mode(self):
        """Test WiFi 4 mode (default)."""
        from opencsi.hardware.esp32 import ESP32Extractor

        extractor = ESP32Extractor()

        assert extractor.hardware_type == "esp32"
        assert extractor.num_subcarriers == 52
        assert extractor.get_pilot_indices(20) == (6, 20, 32, 46)

    def test_wifi6_mode(self):
        """Test WiFi 6 mode for ESP32-C6."""
        from opencsi.hardware.esp32 import ESP32Extractor

        extractor = ESP32Extractor(wifi6=True)

        assert extractor.hardware_type == "esp32_c6"
        assert extractor.num_subcarriers == 242  # HE20
        assert len(extractor.get_pilot_indices(20)) == 8

    def test_capabilities_property(self):
        """Test capabilities property returns HardwareCapabilities."""
        from opencsi.hardware.esp32 import ESP32Extractor
        from opencsi.hardware.capabilities import HardwareCapabilities

        extractor = ESP32Extractor()

        assert isinstance(extractor.capabilities, HardwareCapabilities)
        assert extractor.capabilities.hardware_type == "esp32"


class TestHardwareRegistry:
    """Tests for HardwareRegistry and get_extractor factory."""

    def test_get_esp32_extractor(self):
        """Test getting ESP32 extractor via factory."""
        from opencsi.hardware.registry import get_extractor

        extractor = get_extractor("esp32")

        assert extractor.hardware_type == "esp32"

    def test_get_esp32_c6_extractor(self):
        """Test getting ESP32-C6 extractor auto-enables wifi6."""
        from opencsi.hardware.registry import get_extractor

        extractor = get_extractor("esp32_c6")

        assert extractor.hardware_type == "esp32_c6"
        assert extractor.capabilities.get_num_subcarriers(20) == 242

    def test_list_extractors(self):
        """Test listing available extractors."""
        from opencsi.hardware.registry import list_extractors

        available = list_extractors()

        assert "esp32" in available
        # Stub extractors should also be listed
        assert len(available) >= 1

    def test_unknown_hardware_raises(self):
        """Test that unknown hardware raises ValueError."""
        from opencsi.hardware.registry import get_extractor

        with pytest.raises(ValueError):
            get_extractor("nonexistent_hardware")


class TestSanitizerDynamicPilots:
    """Tests for sanitizer with dynamic pilot indices."""

    def test_sanitizer_config_default_pilots(self):
        """Test SanitizerConfig returns default pilots without capabilities."""
        from opencsi.sanitization.sanitizer import SanitizerConfig

        config = SanitizerConfig()

        assert config.get_pilot_indices(20) == (6, 20, 32, 46)
        assert config.get_num_subcarriers(20) == 52

    def test_sanitizer_config_with_capabilities(self):
        """Test SanitizerConfig uses hardware capabilities."""
        from opencsi.sanitization.sanitizer import SanitizerConfig
        from opencsi.hardware.esp32 import ESP32Extractor

        extractor = ESP32Extractor(wifi6=True)
        config = SanitizerConfig(hardware_capabilities=extractor.capabilities)

        # Should return WiFi 6 pilot indices
        pilots = config.get_pilot_indices(20)
        assert len(pilots) == 8  # HE20 has 8 pilots

        # Should return WiFi 6 subcarrier count
        assert config.get_num_subcarriers(20) == 242


class TestDynamicBaseline:
    """Tests for dynamic baseline array sizing."""

    def test_baseline_default_size(self):
        """Test baseline initializes with default 52 subcarriers."""
        from opencsi.core.types import PerLinkBaseline

        baseline = PerLinkBaseline(tx_mac="AA:BB:CC:DD:EE:01", rx_mac="AA:BB:CC:DD:EE:02")

        assert baseline.n_subcarriers == 52
        assert len(baseline.amplitude_mean) == 52

    def test_baseline_ensure_arrays(self):
        """Test ensure_arrays resizes correctly."""
        from opencsi.core.types import PerLinkBaseline

        baseline = PerLinkBaseline(tx_mac="AA:BB:CC:DD:EE:01", rx_mac="AA:BB:CC:DD:EE:02")

        # Resize to WiFi 6 HE20
        baseline.ensure_arrays(242)

        assert baseline.n_subcarriers == 242
        assert len(baseline.amplitude_mean) == 242
        assert len(baseline.phase_mean) == 242

    def test_baseline_resize_resets_initialization(self):
        """Test that resizing resets is_initialized."""
        from opencsi.core.types import PerLinkBaseline

        baseline = PerLinkBaseline(tx_mac="AA:BB:CC:DD:EE:01", rx_mac="AA:BB:CC:DD:EE:02")
        baseline.is_initialized = True

        # Resize should reset initialization
        baseline.ensure_arrays(114)

        assert not baseline.is_initialized


class TestWavelengthConversion:
    """Tests for wavelength conversion functions."""

    def test_channel_to_frequency_2_4ghz(self):
        """Test 2.4 GHz channel to frequency conversion."""
        from opencsi.core.config import channel_to_frequency_mhz

        # Channel 1 = 2412 MHz
        assert channel_to_frequency_mhz(1) == 2412.0

        # Channel 6 = 2437 MHz
        assert channel_to_frequency_mhz(6) == 2437.0

        # Channel 14 = 2484 MHz (Japan exception)
        assert channel_to_frequency_mhz(14) == 2484.0

    def test_channel_to_frequency_5ghz(self):
        """Test 5 GHz channel to frequency conversion."""
        from opencsi.core.config import channel_to_frequency_mhz

        # Channel 36 = 5180 MHz
        assert channel_to_frequency_mhz(36, "5GHz") == 5180.0

        # Channel 149 = 5745 MHz
        assert channel_to_frequency_mhz(149, "5GHz") == 5745.0

    def test_channel_to_wavelength(self):
        """Test channel to wavelength conversion."""
        from opencsi.core.config import channel_to_wavelength

        # 2.4 GHz should be ~0.124 m
        wavelength_2_4 = channel_to_wavelength(1)
        assert 0.12 < wavelength_2_4 < 0.13

        # 5 GHz should be ~0.058 m
        wavelength_5 = channel_to_wavelength(36, "5GHz")
        assert 0.05 < wavelength_5 < 0.06

    def test_get_band_wavelength(self):
        """Test band wavelength constants."""
        from opencsi.core.config import get_band_wavelength

        assert get_band_wavelength("2.4GHz") == 0.125
        assert get_band_wavelength("5GHz") == 0.060
        assert get_band_wavelength("6GHz") == 0.050


class TestFresnelZoneDynamic:
    """Tests for dynamic Fresnel zone calculations."""

    def test_detection_width_for_channel(self):
        """Test Fresnel zone with channel-based wavelength."""
        from opencsi.geometry.fresnel import compute_detection_width_for_channel

        # 2.4 GHz should have larger detection width than 5 GHz
        width_2_4 = compute_detection_width_for_channel(5.0, 25.0, channel=6)
        width_5 = compute_detection_width_for_channel(5.0, 25.0, channel=36, band="5GHz")

        assert width_2_4 > width_5

    def test_detection_width_band_fallback(self):
        """Test detection width with band fallback."""
        from opencsi.geometry.fresnel import compute_detection_width_for_channel

        # Without channel, should use band default
        width = compute_detection_width_for_channel(5.0, 25.0, band="2.4GHz")

        assert width > 0


class TestProfileLoader:
    """Tests for HardwareProfileLoader."""

    def test_load_profiles(self):
        """Test loading profiles from JSON."""
        from opencsi.hardware.profiles import HardwareProfileLoader

        loader = HardwareProfileLoader()
        profiles = loader.list_profiles()

        assert len(profiles) > 0
        assert "ESP32" in profiles or "esp32" in [p.lower() for p in profiles]

    def test_get_capabilities_from_profile(self):
        """Test getting HardwareCapabilities from profile."""
        from opencsi.hardware.profiles import HardwareProfileLoader
        from opencsi.hardware.capabilities import HardwareCapabilities

        loader = HardwareProfileLoader()
        caps = loader.get_capabilities("ESP32")

        assert isinstance(caps, HardwareCapabilities)
        assert caps.get_num_subcarriers(20) == 52

    def test_profile_inheritance(self):
        """Test profile inheritance works."""
        from opencsi.hardware.profiles import HardwareProfileLoader

        loader = HardwareProfileLoader()

        # ESP32_WROOM inherits from ESP32
        profile = loader._get_resolved_profile("ESP32_WROOM")

        assert profile is not None
        assert "subcarrier_configs" in profile


class TestBackwardCompatibility:
    """Tests for backward compatibility."""

    def test_old_esp32_extractor_usage(self):
        """Test old-style ESP32Extractor usage still works via submodule."""
        from opencsi.hardware.esp32 import ESP32Extractor

        # Old pattern: direct instantiation
        extractor = ESP32Extractor()

        assert extractor.hardware_type == "esp32"
        assert extractor.num_subcarriers == 52

    def test_old_sanitizer_usage(self):
        """Test old-style sanitizer usage still works via submodule."""
        from opencsi.sanitization.sanitizer import CSISanitizer, SanitizerConfig

        # Old pattern: no hardware_capabilities
        config = SanitizerConfig()
        sanitizer = CSISanitizer(config)

        # Should work with default HT20 settings
        assert sanitizer.config.get_pilot_indices() == (6, 20, 32, 46)

    def test_module_exports(self):
        """Test the trimmed top-level API exports only essential symbols."""
        import opencsi

        # --- Symbols that MUST be at top level ---
        assert hasattr(opencsi, 'AutoCalibrator')
        assert hasattr(opencsi, 'AutoCalibratorConfig')
        assert hasattr(opencsi, 'CalibratedFrame')
        assert hasattr(opencsi, 'CalibratedLink')
        assert hasattr(opencsi, 'RawCSIPacket')
        assert hasattr(opencsi, 'CalibrationSnapshot')
        assert hasattr(opencsi, 'QualityGrade')
        assert hasattr(opencsi, 'CSIExtractor')
        assert hasattr(opencsi, 'CSIPacket')
        assert hasattr(opencsi, 'get_extractor')
        assert hasattr(opencsi, 'list_extractors')
        assert hasattr(opencsi, 'SanitizationLevel')
        assert hasattr(opencsi, 'CalibratedStream')
        assert hasattr(opencsi, 'get_config')
        # --- __all__ should be small (~14 symbols) ---
        assert len(opencsi.__all__) < 20

        # --- Symbols that should NOT be at top level anymore ---
        assert not hasattr(opencsi, 'FEATURE_MIMO')
        assert not hasattr(opencsi, 'WAVELENGTH_5GHZ_M')
        assert not hasattr(opencsi, 'HardwareCapabilities')
        assert not hasattr(opencsi, 'HardwareRegistry')
        assert not hasattr(opencsi, 'ESP32Extractor')
        assert not hasattr(opencsi, 'FrequencyAwareBaseline')
        assert not hasattr(opencsi, 'circular_mean')

    def test_removed_symbols_importable_from_submodules(self):
        """Test that symbols removed from top level are still importable."""
        from opencsi.hardware.capabilities import FEATURE_MIMO, HardwareCapabilities
        from opencsi.hardware.registry import HardwareRegistry
        from opencsi.hardware.esp32 import ESP32Extractor
        from opencsi.core.config import WAVELENGTH_5GHZ_M, channel_to_wavelength
        from opencsi.stats.circular import circular_mean
        from opencsi.baseline.frequency_baseline import FrequencyAwareBaseline
        from opencsi.core.env import config, get_array_module

        # Quick sanity checks
        assert FEATURE_MIMO == "mimo"
        assert isinstance(WAVELENGTH_5GHZ_M, float)
        assert callable(circular_mean)
        assert callable(get_array_module)


class TestEnvironmentConfig:
    """Tests for environment configuration."""

    def test_default_hardware_config(self):
        """Test default hardware configuration."""
        from opencsi.core.env import config

        # Default should be esp32
        assert config.default_hardware == "esp32"

    def test_get_extractor_uses_default(self):
        """Test get_extractor() uses environment default."""
        from opencsi.hardware.registry import get_extractor, HardwareRegistry

        # Reset registry for clean state
        HardwareRegistry.reset()

        # Call without arguments should use default
        extractor = get_extractor()
        assert extractor.hardware_type == "esp32"

    def test_config_override(self):
        """Test runtime config override."""
        from opencsi.core.env import config

        original = config.default_hardware

        config.set("default_hardware", "intel_5300")
        assert config.default_hardware == "intel_5300"

        # Reset
        config.set("default_hardware", original)

    def test_array_module_default(self):
        """Test default array module is numpy."""
        from opencsi.core.env import get_array_module
        import numpy as np

        xp = get_array_module()
        assert xp is np

    def test_config_to_dict(self):
        """Test config serialization."""
        from opencsi.core.env import config

        d = config.to_dict()
        assert "default_hardware" in d
        assert "array_backend" in d
        assert d["array_backend"] == "numpy"


class TestFeatureDetection:
    """Tests for feature detection API."""

    def test_esp32_features(self):
        """Test ESP32 feature detection."""
        from opencsi.hardware.capabilities import (
            create_esp32_capabilities,
            FEATURE_WIFI4,
            FEATURE_BAND_2_4GHZ,
            FEATURE_BW_40MHZ,
            FEATURE_MIMO,
        )

        caps = create_esp32_capabilities()

        # ESP32 has WiFi 4 and 2.4 GHz
        assert caps.has_feature(FEATURE_WIFI4)
        assert caps.has_feature(FEATURE_BAND_2_4GHZ)
        assert caps.has_feature(FEATURE_BW_40MHZ)

        # ESP32 does not have MIMO
        assert not caps.has_feature(FEATURE_MIMO)

    def test_intel5300_features(self):
        """Test Intel 5300 feature detection."""
        from opencsi.hardware.capabilities import (
            create_intel5300_capabilities,
            FEATURE_MIMO,
            FEATURE_MULTI_ANTENNA_TX,
            FEATURE_BAND_5GHZ,
            FEATURE_RELIABLE_AGC,
        )

        caps = create_intel5300_capabilities()

        # Intel 5300 has MIMO and 5 GHz
        assert caps.has_feature(FEATURE_MIMO)
        assert caps.has_feature(FEATURE_MULTI_ANTENNA_TX)
        assert caps.has_feature(FEATURE_BAND_5GHZ)
        assert caps.has_feature(FEATURE_RELIABLE_AGC)

    def test_nexmon_features(self):
        """Test Nexmon feature detection."""
        from opencsi.hardware.capabilities import (
            create_nexmon_capabilities,
            FEATURE_BW_80MHZ,
            FEATURE_WIFI5,
            FEATURE_HIGH_SUBCARRIER_COUNT,
        )

        caps = create_nexmon_capabilities()

        # Nexmon supports 80 MHz (WiFi 5)
        assert caps.has_feature(FEATURE_BW_80MHZ)
        assert caps.has_feature(FEATURE_WIFI5)
        assert caps.has_feature(FEATURE_HIGH_SUBCARRIER_COUNT)

    def test_list_features(self):
        """Test listing all features."""
        from opencsi.hardware.capabilities import create_esp32_capabilities

        caps = create_esp32_capabilities()
        features = caps.list_features()

        assert isinstance(features, frozenset)
        assert "wifi4" in features
        assert "band_2.4ghz" in features

    def test_require_features_success(self):
        """Test require_features passes when features exist."""
        from opencsi.hardware.capabilities import (
            create_intel5300_capabilities,
            FEATURE_MIMO,
            FEATURE_BAND_5GHZ,
        )

        caps = create_intel5300_capabilities()

        # Should not raise
        caps.require_features(FEATURE_MIMO, FEATURE_BAND_5GHZ)

    def test_require_features_failure(self):
        """Test require_features raises when features missing."""
        from opencsi.hardware.capabilities import (
            create_esp32_capabilities,
            FEATURE_MIMO,
        )

        caps = create_esp32_capabilities()

        with pytest.raises(ValueError, match="missing required features"):
            caps.require_features(FEATURE_MIMO)

    def test_add_custom_feature(self):
        """Test adding custom features."""
        from opencsi.hardware.capabilities import create_esp32_capabilities

        caps = create_esp32_capabilities()

        # Add custom feature
        caps.add_feature("custom_sensor")

        assert caps.has_feature("custom_sensor")
        assert "custom_sensor" in caps.list_features()

    def test_wifi6_detection(self):
        """Test WiFi 6 detection from subcarrier count."""
        from opencsi.hardware.esp32 import ESP32Extractor
        from opencsi.hardware.capabilities import FEATURE_WIFI6

        # WiFi 4 ESP32
        ext_wifi4 = ESP32Extractor(wifi6=False)
        assert not ext_wifi4.capabilities.has_feature(FEATURE_WIFI6)

        # WiFi 6 ESP32-C6
        ext_wifi6 = ESP32Extractor(wifi6=True)
        assert ext_wifi6.capabilities.has_feature(FEATURE_WIFI6)


class TestIwlWifiExtractor:
    """Tests for Intel iwlwifi extractor (AX210, BE200, etc.)."""

    def test_ax210_capabilities(self):
        """Test AX210 capabilities."""
        from opencsi.hardware.iwlwifi import IwlWifiExtractor
        from opencsi.hardware.capabilities import (
            FEATURE_WIFI6,
            FEATURE_BAND_6GHZ,
            FEATURE_BW_160MHZ,
            FEATURE_MIMO,
        )

        extractor = IwlWifiExtractor(device="ax210")
        caps = extractor.capabilities

        # AX210 is WiFi 6E
        assert caps.has_feature(FEATURE_WIFI6)
        assert caps.has_feature(FEATURE_BAND_6GHZ)
        assert caps.has_feature(FEATURE_BW_160MHZ)
        assert caps.has_feature(FEATURE_MIMO)

        # Check bandwidths
        assert 160 in caps.supported_bandwidths
        assert "6GHz" in caps.supported_bands

    def test_be200_capabilities(self):
        """Test BE200 (WiFi 7) capabilities."""
        from opencsi.hardware.iwlwifi import IwlWifiExtractor
        from opencsi.hardware.capabilities import (
            FEATURE_WIFI7,
            FEATURE_BW_320MHZ,
        )

        extractor = IwlWifiExtractor(device="be200")
        caps = extractor.capabilities

        # BE200 is WiFi 7
        assert caps.has_feature(FEATURE_WIFI7)
        assert caps.has_feature(FEATURE_BW_320MHZ)

        # Check 320 MHz support
        assert 320 in caps.supported_bandwidths

    def test_ax200_vs_ax210(self):
        """Test AX200 (WiFi 6) vs AX210 (WiFi 6E) differences."""
        from opencsi.hardware.iwlwifi import IwlWifiExtractor
        from opencsi.hardware.capabilities import FEATURE_BAND_6GHZ

        ax200 = IwlWifiExtractor(device="ax200")
        ax210 = IwlWifiExtractor(device="ax210")

        # AX200 does NOT have 6GHz
        assert not ax200.capabilities.has_feature(FEATURE_BAND_6GHZ)

        # AX210 DOES have 6GHz
        assert ax210.capabilities.has_feature(FEATURE_BAND_6GHZ)

    def test_subcarrier_counts(self):
        """Test subcarrier counts for different bandwidths."""
        from opencsi.hardware.iwlwifi import IwlWifiExtractor

        extractor = IwlWifiExtractor(device="ax210")
        caps = extractor.capabilities

        # 802.11ax HE subcarrier counts
        assert caps.get_num_subcarriers(20) == 242
        assert caps.get_num_subcarriers(40) == 484
        assert caps.get_num_subcarriers(80) == 996
        assert caps.get_num_subcarriers(160) == 1992

    def test_be200_subcarrier_counts(self):
        """Test WiFi 7 320 MHz subcarrier count."""
        from opencsi.hardware.iwlwifi import IwlWifiExtractor

        extractor = IwlWifiExtractor(device="be200")
        caps = extractor.capabilities

        # WiFi 7 320 MHz
        assert caps.get_num_subcarriers(320) == 3984

    def test_registry_integration(self):
        """Test iwlwifi extractor via registry."""
        from opencsi.hardware.registry import get_extractor, HardwareRegistry

        HardwareRegistry.reset()

        # Get AX210 via registry
        extractor = get_extractor("ax210")
        assert extractor.device == "ax210"

        # Get BE200 via registry
        extractor = get_extractor("be200")
        assert extractor.device == "be200"

    def test_parse_raises_not_implemented(self):
        """Test that parse() raises NotImplementedError."""
        from opencsi.hardware.iwlwifi import IwlWifiExtractor

        extractor = IwlWifiExtractor(device="ax210")

        with pytest.raises(NotImplementedError):
            extractor.parse(b"test data")

    def test_supported_devices_list(self):
        """Test listing supported devices."""
        from opencsi.hardware.iwlwifi import IwlWifiExtractor

        devices = IwlWifiExtractor.list_supported_devices()

        assert "ax200" in devices
        assert "ax210" in devices
        assert "be200" in devices

    def test_wifi7_device_detection(self):
        """Test WiFi 7 device detection."""
        from opencsi.hardware.iwlwifi import IwlWifiExtractor

        assert IwlWifiExtractor.is_wifi7_device("be200")
        assert IwlWifiExtractor.is_wifi7_device("be202")
        assert not IwlWifiExtractor.is_wifi7_device("ax210")
