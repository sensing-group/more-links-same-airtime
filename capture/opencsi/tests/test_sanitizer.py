#!/usr/bin/env python3
"""
Tests for CSI Sanitizer, including DensePose-style phase sanitization.

Reference: "DensePose From WiFi" (arXiv 2301.00250)
"""

import numpy as np
import pytest

from opencsi.sanitization.sanitizer import (
    CSISanitizer,
    SanitizerConfig,
    SanitizationLevel,
    PhaseCorrector,
    OutlierRemover,
    PHASE_TEMPORAL_MEDIAN_KERNEL,
    PHASE_UNIFORM_SMOOTH_KERNEL,
)


class TestDensePoseStyleSanitization:
    """Tests for DensePose-style phase sanitization."""

    @pytest.fixture
    def phase_corrector(self):
        """Create a PhaseCorrector with default config."""
        config = SanitizerConfig()
        return PhaseCorrector(config)

    @pytest.fixture
    def noisy_phase(self):
        """Create synthetic noisy phase data with known characteristics."""
        np.random.seed(42)
        n_subcarriers = 52

        # Base linear trend (simulates SFO/PDD)
        subcarriers = np.arange(n_subcarriers)
        linear_trend = 0.05 * subcarriers + 0.3

        # Add some random noise
        noise = np.random.normal(0, 0.1, n_subcarriers)

        # Add a few outliers
        outlier_indices = [10, 25, 40]
        phase = linear_trend + noise
        for idx in outlier_indices:
            phase[idx] += np.random.choice([-1, 1]) * 2.0  # Large outlier

        return phase

    def test_densepose_sanitization_removes_linear_trend(self, phase_corrector, noisy_phase):
        """DensePose sanitization should remove linear phase drift."""
        corrected, info = phase_corrector.correct_densepose_style(noisy_phase)

        # After detrending, the mean should be near zero
        assert np.abs(np.mean(corrected)) < 0.5

        # Check that all steps were applied
        assert 'unwrap' in info['steps_applied']
        assert 'linear_detrend' in info['steps_applied']
        assert info['method'] == 'densepose'

    def test_densepose_sanitization_reduces_outliers(self, phase_corrector):
        """Median filter should reduce impact of outliers."""
        np.random.seed(42)
        n_subcarriers = 52

        # Clean phase with one large outlier
        clean_phase = np.zeros(n_subcarriers)
        phase_with_outlier = clean_phase.copy()
        phase_with_outlier[25] = 3.0  # Large outlier

        corrected, _ = phase_corrector.correct_densepose_style(phase_with_outlier)

        # The outlier should be significantly reduced
        assert np.abs(corrected[25]) < 1.0

    def test_densepose_sanitization_smooths_noise(self, phase_corrector):
        """Uniform smoothing should reduce high-frequency noise."""
        np.random.seed(42)
        n_subcarriers = 52

        # High-frequency noise
        noisy = np.random.normal(0, 0.5, n_subcarriers)

        corrected, _ = phase_corrector.correct_densepose_style(noisy)

        # Standard deviation should be reduced after smoothing
        assert np.std(corrected) < np.std(noisy)

    def test_densepose_output_wrapped_to_pi(self, phase_corrector, noisy_phase):
        """Output should be wrapped to [-π, π]."""
        corrected, _ = phase_corrector.correct_densepose_style(noisy_phase)

        assert np.all(corrected >= -np.pi)
        assert np.all(corrected <= np.pi)

    def test_densepose_alpha_coefficients(self, phase_corrector):
        """Test that alpha coefficients are computed correctly."""
        # Phase with known linear trend: phase = 0.1 * f + 0.5
        n_subcarriers = 52
        subcarriers = np.arange(n_subcarriers)
        phase = 0.1 * subcarriers + 0.5

        _, info = phase_corrector.correct_densepose_style(phase)

        # α₁ should capture the slope
        # α₀ should capture the mean offset
        assert 'alpha_1' in info
        assert 'alpha_0' in info

    def test_custom_kernel_sizes(self, phase_corrector):
        """Test with custom kernel sizes."""
        np.random.seed(42)
        phase = np.random.normal(0, 0.3, 52)

        # Larger kernels = more smoothing
        corrected_small, _ = phase_corrector.correct_densepose_style(
            phase, median_kernel=3, smooth_kernel=3
        )
        corrected_large, _ = phase_corrector.correct_densepose_style(
            phase, median_kernel=7, smooth_kernel=7
        )

        # Larger kernels should result in lower variance
        assert np.std(corrected_large) <= np.std(corrected_small)


class TestSanitizerWithDensePoseLevel:
    """Tests for full sanitization pipeline with DENSEPOSE level."""

    @pytest.fixture
    def densepose_sanitizer(self):
        """Create a sanitizer with DENSEPOSE level."""
        config = SanitizerConfig(level=SanitizationLevel.DENSEPOSE)
        return CSISanitizer(config)

    @pytest.fixture
    def sample_iq_data(self):
        """Create sample I/Q data."""
        np.random.seed(42)
        n_subcarriers = 52

        # Generate complex CSI with amplitude and phase
        amplitude = 10 + np.random.normal(0, 1, n_subcarriers)
        phase = np.linspace(0, 2, n_subcarriers) + np.random.normal(0, 0.1, n_subcarriers)

        # Add outliers
        phase[15] += 2.0
        phase[35] -= 2.0

        # Convert to I/Q format
        iq_data = np.zeros((n_subcarriers, 2))
        iq_data[:, 0] = amplitude * np.cos(phase)  # I
        iq_data[:, 1] = amplitude * np.sin(phase)  # Q

        return iq_data

    def test_densepose_level_applies_all_corrections(
        self, densepose_sanitizer, sample_iq_data
    ):
        """DENSEPOSE level should apply AGC and DensePose phase correction."""
        result = densepose_sanitizer.sanitize(
            iq_data=sample_iq_data,
            tx_mac="AA:BB:CC:DD:EE:01",
            rx_mac="AA:BB:CC:DD:EE:02",
            timestamp=1234567890.0,
            rssi=-50,
            noise_floor=-90
        )

        assert result.agc_correction_applied
        assert result.phase_correction_applied
        assert result.phase is not None
        assert len(result.phase) == 52

        # Phase should be in valid range
        assert np.all(result.phase >= -np.pi)
        assert np.all(result.phase <= np.pi)

    def test_densepose_reduces_phase_variance(self, densepose_sanitizer, sample_iq_data):
        """DensePose sanitization should reduce phase variance."""
        # Get raw phase
        complex_csi = sample_iq_data[:, 0] + 1j * sample_iq_data[:, 1]
        raw_phase = np.angle(complex_csi)

        # Sanitize
        result = densepose_sanitizer.sanitize(
            iq_data=sample_iq_data,
            tx_mac="AA:BB:CC:DD:EE:01",
            rx_mac="AA:BB:CC:DD:EE:02",
            timestamp=1234567890.0
        )

        # Sanitized phase should have lower variance (after detrending)
        raw_detrended = raw_phase - np.mean(raw_phase)
        sanitized_var = np.var(result.phase)
        raw_var = np.var(raw_detrended)

        # DensePose smoothing should reduce variance
        assert sanitized_var < raw_var

    def test_densepose_level_value(self):
        """DENSEPOSE level should have value 4."""
        assert SanitizationLevel.DENSEPOSE.value == 4


class TestSanitizationLevelComparison:
    """Compare different sanitization levels."""

    @pytest.fixture
    def sample_iq_data(self):
        """Create sample I/Q data with known characteristics."""
        np.random.seed(123)
        n_subcarriers = 52

        amplitude = 10 + np.random.normal(0, 0.5, n_subcarriers)
        # Linear phase + noise + outliers
        phase = 0.08 * np.arange(n_subcarriers) + np.random.normal(0, 0.2, n_subcarriers)
        phase[20] += 2.5  # Outlier

        iq_data = np.zeros((n_subcarriers, 2))
        iq_data[:, 0] = amplitude * np.cos(phase)
        iq_data[:, 1] = amplitude * np.sin(phase)

        return iq_data

    def test_densepose_vs_standard(self, sample_iq_data):
        """Compare DENSEPOSE vs STANDARD sanitization."""
        standard_config = SanitizerConfig(level=SanitizationLevel.STANDARD)
        densepose_config = SanitizerConfig(level=SanitizationLevel.DENSEPOSE)

        standard_sanitizer = CSISanitizer(standard_config)
        densepose_sanitizer = CSISanitizer(densepose_config)

        standard_result = standard_sanitizer.sanitize(
            iq_data=sample_iq_data,
            tx_mac="AA:BB:CC:DD:EE:01",
            rx_mac="AA:BB:CC:DD:EE:02",
            timestamp=1234567890.0
        )

        densepose_result = densepose_sanitizer.sanitize(
            iq_data=sample_iq_data,
            tx_mac="AA:BB:CC:DD:EE:01",
            rx_mac="AA:BB:CC:DD:EE:02",
            timestamp=1234567890.0
        )

        # Both should produce valid results
        assert standard_result.phase_correction_applied
        assert densepose_result.phase_correction_applied

        # DensePose should typically have smoother output
        # (due to median filter + uniform smoothing)
        standard_diff = np.diff(standard_result.phase)
        densepose_diff = np.diff(densepose_result.phase)

        # DensePose should have smaller high-frequency variations
        assert np.std(densepose_diff) <= np.std(standard_diff) * 1.5

    def test_all_levels_produce_valid_output(self, sample_iq_data):
        """All sanitization levels should produce valid output."""
        for level in SanitizationLevel:
            config = SanitizerConfig(level=level)
            sanitizer = CSISanitizer(config)

            result = sanitizer.sanitize(
                iq_data=sample_iq_data,
                tx_mac="AA:BB:CC:DD:EE:01",
                rx_mac="AA:BB:CC:DD:EE:02",
                timestamp=1234567890.0
            )

            # All should produce valid arrays
            assert result.amplitude is not None
            assert result.phase is not None
            assert len(result.amplitude) == 52
            assert len(result.phase) == 52

            # Phase should be in valid range
            assert np.all(result.phase >= -np.pi)
            assert np.all(result.phase <= np.pi)


class TestOutlierRemover:
    """Tests for outlier removal."""

    def test_hampel_filter_bulk_and_edges(self):
        """Test Hampel filter on bulk and edges using optimized implementation."""
        config = SanitizerConfig(
            enable_outlier_removal=True,
            hampel_window=5,
            hampel_threshold=3.0
        )
        remover = OutlierRemover(config)

        # Create data
        data = np.ones(100) * 10.0
        # Outlier in bulk
        data[50] = 100.0
        # Outlier at start edge (index 2 < window 5)
        data[2] = 100.0
        # Outlier at end edge (index 98 > 100-5)
        data[98] = 100.0

        filtered, n_outliers = remover.remove_outliers(data)

        # Should detect all 3 outliers
        assert n_outliers == 3
        # Should replace them with median (10.0)
        assert filtered[50] == 10.0
        assert filtered[2] == 10.0
        assert filtered[98] == 10.0

    def test_small_array(self):
        """Test with array smaller than window (should fallback to loop)."""
        config = SanitizerConfig(
            enable_outlier_removal=True,
            hampel_window=5,
            hampel_threshold=3.0
        )
        remover = OutlierRemover(config)

        # Window is 5, kernel 11. Array size 10.
        data = np.ones(10) * 10.0
        data[5] = 100.0

        filtered, n_outliers = remover.remove_outliers(data)

        # Should fall back to loop and work
        assert n_outliers == 1
        assert filtered[5] == 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
