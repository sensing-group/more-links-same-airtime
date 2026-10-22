#!/usr/bin/env python3
"""
Tests for WelfordAccumulator.

Tests the numerically stable online mean/variance calculation using
Welford's algorithm.
"""

import math
import pytest

from opencsi.core.welford import WelfordAccumulator


class TestWelfordBasics:
    """Test basic WelfordAccumulator operations."""

    def test_empty_accumulator(self):
        """Empty accumulator has zero count and zero mean."""
        acc = WelfordAccumulator()
        assert acc.n == 0
        assert acc.mean == 0.0
        assert acc.m2 == 0.0
        assert acc.variance() == 0.0
        assert acc.std() == 0.0

    def test_single_sample(self):
        """Single sample gives that value as mean."""
        acc = WelfordAccumulator()
        acc.update(5.0)

        assert acc.n == 1
        assert acc.mean == 5.0
        assert acc.variance() == 0.0  # Need 2 samples for sample variance

    def test_two_samples(self):
        """Two samples give correct mean and variance."""
        acc = WelfordAccumulator()
        acc.update(2.0)
        acc.update(4.0)

        assert acc.n == 2
        assert acc.mean == 3.0
        # Sample variance: sum((x - mean)^2) / (n-1) = (1 + 1) / 1 = 2
        assert acc.variance() == 2.0
        assert math.isclose(acc.std(), math.sqrt(2.0))

    def test_multiple_samples(self):
        """Multiple samples give correct statistics."""
        acc = WelfordAccumulator()
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        for x in samples:
            acc.update(x)

        assert acc.n == 5
        assert acc.mean == 3.0

        # Sample variance: sum of squared deviations / (n-1)
        # deviations: -2, -1, 0, 1, 2 -> squared: 4, 1, 0, 1, 4 -> sum: 10
        # variance = 10 / 4 = 2.5
        assert math.isclose(acc.variance(), 2.5)

    def test_population_variance(self):
        """Population variance (ddof=0) works correctly."""
        acc = WelfordAccumulator()
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        for x in samples:
            acc.update(x)

        # Population variance: sum of squared deviations / n = 10 / 5 = 2.0
        assert acc.variance(ddof=0) == 2.0


class TestWelfordConfidenceInterval:
    """Test confidence interval calculation."""

    def test_ci_insufficient_samples(self):
        """CI returns (mean, mean) with insufficient samples."""
        acc = WelfordAccumulator()
        acc.update(5.0)

        ci = acc.confidence_interval()
        assert ci == (5.0, 5.0)

    def test_ci_basic(self):
        """CI is computed correctly for sufficient samples."""
        acc = WelfordAccumulator()
        # Add enough samples for meaningful CI
        samples = [10.0] * 100  # All same value = zero variance
        for x in samples:
            acc.update(x)

        ci = acc.confidence_interval(0.95)
        # With zero variance, CI should be very tight around mean
        assert ci[0] <= acc.mean <= ci[1]
        assert math.isclose(ci[0], ci[1], abs_tol=0.001)

    def test_ci_wider_for_higher_variance(self):
        """CI is wider with higher variance."""
        # Low variance
        acc_low = WelfordAccumulator()
        for x in [10.0, 10.1, 9.9, 10.0, 10.0] * 20:
            acc_low.update(x)

        # High variance
        acc_high = WelfordAccumulator()
        for x in [5.0, 15.0, 5.0, 15.0, 5.0] * 20:
            acc_high.update(x)

        ci_low = acc_low.confidence_interval(0.95)
        ci_high = acc_high.confidence_interval(0.95)

        width_low = ci_low[1] - ci_low[0]
        width_high = ci_high[1] - ci_high[0]

        assert width_high > width_low


class TestWelfordCombine:
    """Test parallel combination of accumulators."""

    def test_combine_empty(self):
        """Combining empty list gives empty accumulator."""
        combined = WelfordAccumulator.combine()
        assert combined.n == 0
        assert combined.mean == 0.0

    def test_combine_single(self):
        """Combining single accumulator gives copy."""
        acc = WelfordAccumulator()
        acc.update(5.0)
        acc.update(10.0)

        combined = WelfordAccumulator.combine(acc)
        assert combined.n == acc.n
        assert combined.mean == acc.mean
        assert combined.m2 == acc.m2

    def test_combine_two_accumulators(self):
        """Combining two accumulators gives correct result."""
        # First half
        acc1 = WelfordAccumulator()
        for x in [1.0, 2.0, 3.0, 4.0, 5.0]:
            acc1.update(x)

        # Second half
        acc2 = WelfordAccumulator()
        for x in [6.0, 7.0, 8.0, 9.0, 10.0]:
            acc2.update(x)

        # Combined
        combined = WelfordAccumulator.combine(acc1, acc2)

        # Should match single accumulator with all values
        reference = WelfordAccumulator()
        for x in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
            reference.update(x)

        assert combined.n == reference.n
        assert math.isclose(combined.mean, reference.mean)
        assert math.isclose(combined.variance(), reference.variance(), rel_tol=1e-10)

    def test_combine_with_empty(self):
        """Combining with empty accumulator works correctly."""
        acc = WelfordAccumulator()
        for x in [1.0, 2.0, 3.0]:
            acc.update(x)

        empty = WelfordAccumulator()
        combined = WelfordAccumulator.combine(acc, empty)

        assert combined.n == acc.n
        assert combined.mean == acc.mean

    def test_combine_multiple(self):
        """Combining multiple accumulators works correctly."""
        accumulators = []
        all_values = []

        for i in range(5):
            acc = WelfordAccumulator()
            values = [float(i * 10 + j) for j in range(10)]
            for x in values:
                acc.update(x)
                all_values.append(x)
            accumulators.append(acc)

        combined = WelfordAccumulator.combine(*accumulators)

        reference = WelfordAccumulator()
        for x in all_values:
            reference.update(x)

        assert combined.n == reference.n
        assert math.isclose(combined.mean, reference.mean, rel_tol=1e-10)
        assert math.isclose(combined.variance(), reference.variance(), rel_tol=1e-10)


class TestWelfordSerialization:
    """Test serialization to/from dict."""

    def test_to_dict(self):
        """to_dict produces correct dictionary."""
        acc = WelfordAccumulator(n=10, mean=5.5, m2=22.5)
        d = acc.to_dict()

        assert d == {'n': 10, 'mean': 5.5, 'm2': 22.5}

    def test_from_dict(self):
        """from_dict restores state correctly."""
        d = {'n': 10, 'mean': 5.5, 'm2': 22.5}
        acc = WelfordAccumulator.from_dict(d)

        assert acc.n == 10
        assert acc.mean == 5.5
        assert acc.m2 == 22.5

    def test_roundtrip(self):
        """Serialization roundtrip preserves state."""
        acc = WelfordAccumulator()
        for x in [1.0, 2.0, 3.0, 4.0, 5.0]:
            acc.update(x)

        restored = WelfordAccumulator.from_dict(acc.to_dict())

        assert restored.n == acc.n
        assert restored.mean == acc.mean
        assert restored.m2 == acc.m2
        assert restored.variance() == acc.variance()

    def test_from_dict_with_missing_keys(self):
        """from_dict handles missing keys gracefully."""
        acc = WelfordAccumulator.from_dict({})
        assert acc.n == 0
        assert acc.mean == 0.0
        assert acc.m2 == 0.0


class TestWelfordNumericalStability:
    """Test numerical stability with edge cases."""

    def test_large_values(self):
        """Handles large values without overflow."""
        acc = WelfordAccumulator()
        for x in [1e10, 1e10 + 1, 1e10 + 2]:
            acc.update(x)

        assert math.isclose(acc.mean, 1e10 + 1, rel_tol=1e-10)

    def test_small_differences(self):
        """Handles small differences accurately."""
        acc = WelfordAccumulator()
        base = 1e10
        for x in [base, base + 1e-6, base + 2e-6, base + 3e-6]:
            acc.update(x)

        # Mean should be base + 1.5e-6
        expected_mean = base + 1.5e-6
        assert math.isclose(acc.mean, expected_mean, rel_tol=1e-10)

    def test_repr(self):
        """String representation works."""
        acc = WelfordAccumulator()
        acc.update(5.0)
        s = repr(acc)
        assert "WelfordAccumulator" in s
        assert "n=1" in s
