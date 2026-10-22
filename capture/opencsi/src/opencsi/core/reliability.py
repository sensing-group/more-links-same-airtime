"""
Reliability Scoring Module for OpenCSI.

Computes reliability scores for CSI links based on multiple quality factors.
The reliability score indicates how much to trust a link's measurements.

Formula:
    reliability = Q_cv * Q_age * Q_rate * Q_rssi * Q_sample

Where:
    Q_cv:     Baseline stability (1 / (1 + CV)), CV = std/mean
    Q_age:    Baseline freshness (exponential decay with 24h half-life)
    Q_rate:   Frequency bucket calibration confidence (from PassiveFrequencyCollector)
    Q_rssi:   Signal quality from RSSI (sigmoid function)
    Q_sample: Sample count confidence (ramp-up for incomplete baselines)

Usage:
    from opencsi.core.reliability import ReliabilityCalculator

    calculator = ReliabilityCalculator()

    reliability = calculator.compute(
        rssi_dbm=-65,
        amplitude_std=0.3,
        amplitude_mean=5.0,
        baseline_age_s=3600.0,
        q_rate=0.9,
        quiet_samples=50,
        min_baseline_samples=10,
    )
    print(f"Link reliability: {reliability:.2f}")
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

# =============================================================================
# Constants
# =============================================================================

# Q_rssi calculation
NOISE_FLOOR_DBM = -95.0
GOOD_RSSI_DBM = -60.0
RSSI_SIGMOID_SCALE = 10.0

# Q_cv calculation
CV_EPSILON = 1e-6
CV_CAP = 3.0

# Q_age calculation
DEFAULT_DECAY_LAMBDA = 1.0 / 86400.0  # 24-hour half-life

# Uninitialised baseline
UNINITIALIZED_BASELINE_RELIABILITY = 0.3

# Overall reliability
MIN_RELIABILITY = 0.1
MAX_RELIABILITY = 1.0


# =============================================================================
# Reliability Result
# =============================================================================

@dataclass
class ReliabilityBreakdown:
    """Detailed breakdown of reliability score components."""
    overall: float
    q_cv: float
    q_age: float
    q_rate: float
    q_rssi: float
    q_sample: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'overall': self.overall,
            'q_cv': self.q_cv,
            'q_age': self.q_age,
            'q_rate': self.q_rate,
            'q_rssi': self.q_rssi,
            'q_sample': self.q_sample,
        }


# =============================================================================
# Abstract Base Class
# =============================================================================

class ReliabilityScorer(ABC):
    """
    Abstract base class for reliability scoring.

    Subclass to implement custom reliability metrics.
    """

    @abstractmethod
    def compute(
        self,
        rssi_dbm: float,
        amplitude_std: float = 0.0,
        amplitude_mean: float = 1.0,
        baseline_age_s: float = 0.0,
        decay_lambda: float = DEFAULT_DECAY_LAMBDA,
        q_rate: float = 1.0,
        quiet_samples: int = 0,
        min_baseline_samples: int = 10,
        baseline_initialized: bool = True,
    ) -> float:
        """
        Compute overall reliability score.

        Args:
            rssi_dbm: Received signal strength in dBm
            amplitude_std: Baseline amplitude standard deviation
            amplitude_mean: Baseline amplitude mean
            baseline_age_s: Seconds since baseline was last updated
            decay_lambda: Exponential decay rate (1/half_life_s)
            q_rate: Frequency bucket calibration confidence [0, 1]
            quiet_samples: Number of quiet samples in baseline
            min_baseline_samples: Minimum samples for full confidence
            baseline_initialized: Whether baseline has been learned

        Returns:
            Reliability score between 0 and 1
        """
        ...


# =============================================================================
# Default Reliability Calculator
# =============================================================================

class ReliabilityCalculator(ReliabilityScorer):
    """
    Default reliability calculator using multiplicative formula.

    Combines five quality factors into a single reliability score:
        R = Q_cv * Q_age * Q_rate * Q_rssi * Q_sample

    Args:
        min_reliability: Floor for reliability score (default: 0.1)
    """

    def __init__(
        self,
        min_reliability: float = MIN_RELIABILITY,
    ):
        self.min_reliability = min_reliability

    def compute(
        self,
        rssi_dbm: float,
        amplitude_std: float = 0.0,
        amplitude_mean: float = 1.0,
        baseline_age_s: float = 0.0,
        decay_lambda: float = DEFAULT_DECAY_LAMBDA,
        q_rate: float = 1.0,
        quiet_samples: int = 0,
        min_baseline_samples: int = 10,
        baseline_initialized: bool = True,
    ) -> float:
        """
        Compute overall reliability as product of quality factors.

        Returns:
            Reliability score between min_reliability and 1.0
        """
        if not baseline_initialized:
            return UNINITIALIZED_BASELINE_RELIABILITY

        q_cv = self.compute_q_cv(amplitude_std, amplitude_mean)
        q_age = self.compute_q_age(baseline_age_s, decay_lambda)
        q_rssi = self.compute_q_rssi(rssi_dbm)
        q_sample = self.compute_q_sample(quiet_samples, min_baseline_samples)

        reliability = q_cv * q_age * q_rate * q_rssi * q_sample

        return max(self.min_reliability, min(MAX_RELIABILITY, reliability))

    def compute_with_breakdown(
        self,
        rssi_dbm: float,
        amplitude_std: float = 0.0,
        amplitude_mean: float = 1.0,
        baseline_age_s: float = 0.0,
        decay_lambda: float = DEFAULT_DECAY_LAMBDA,
        q_rate: float = 1.0,
        quiet_samples: int = 0,
        min_baseline_samples: int = 10,
        baseline_initialized: bool = True,
    ) -> ReliabilityBreakdown:
        """
        Compute reliability with detailed breakdown of each factor.

        Returns:
            ReliabilityBreakdown with all components
        """
        if not baseline_initialized:
            return ReliabilityBreakdown(
                overall=UNINITIALIZED_BASELINE_RELIABILITY,
                q_cv=0.0,
                q_age=0.0,
                q_rate=0.0,
                q_rssi=0.0,
                q_sample=0.0,
            )

        q_cv = self.compute_q_cv(amplitude_std, amplitude_mean)
        q_age = self.compute_q_age(baseline_age_s, decay_lambda)
        q_rssi = self.compute_q_rssi(rssi_dbm)
        q_sample = self.compute_q_sample(quiet_samples, min_baseline_samples)

        overall = q_cv * q_age * q_rate * q_rssi * q_sample
        overall = max(self.min_reliability, min(MAX_RELIABILITY, overall))

        return ReliabilityBreakdown(
            overall=overall,
            q_cv=q_cv,
            q_age=q_age,
            q_rate=q_rate,
            q_rssi=q_rssi,
            q_sample=q_sample,
        )

    def compute_q_rssi(self, rssi_dbm: float) -> float:
        """
        Compute RSSI-based quality factor using sigmoid function.

        Higher RSSI (stronger signal) = higher quality.
        Sigmoid centered at GOOD_RSSI_DBM, where Q_rssi = 0.5.

        Args:
            rssi_dbm: Received signal strength in dBm

        Returns:
            Quality factor between 0 and 1 (0.5 at GOOD_RSSI_DBM)
        """
        normalized = (rssi_dbm - GOOD_RSSI_DBM) / RSSI_SIGMOID_SCALE
        return 1.0 / (1.0 + math.exp(-normalized))

    def compute_q_cv(self, amplitude_std: float, amplitude_mean: float) -> float:
        """
        Compute coefficient-of-variation quality factor.

        Stable baselines (low CV) yield higher quality. CV is capped at
        CV_CAP to prevent collapse for low-amplitude links.

        Args:
            amplitude_std: Baseline amplitude standard deviation
            amplitude_mean: Baseline amplitude mean

        Returns:
            Quality factor between 0 and 1
        """
        cv = amplitude_std / max(amplitude_mean, CV_EPSILON)
        cv = min(cv, CV_CAP)
        return 1.0 / (1.0 + cv)

    def compute_q_age(
        self,
        age_s: float,
        decay_lambda: float = DEFAULT_DECAY_LAMBDA,
    ) -> float:
        """
        Compute age-based quality factor using exponential decay.

        Fresher baselines are more trustworthy. Uses e^(-lambda * age)
        where lambda defaults to a 24-hour half-life.

        Args:
            age_s: Seconds since baseline was last updated
            decay_lambda: Exponential decay rate

        Returns:
            Quality factor between 0 and 1
        """
        return math.exp(-decay_lambda * age_s)

    def compute_q_sample(
        self,
        quiet_samples: int,
        min_baseline_samples: int,
    ) -> float:
        """
        Compute sample-count confidence factor.

        Ramps linearly from 0 to 1 as quiet_samples approaches
        min_baseline_samples. At or above min_baseline_samples, returns 1.0.

        Args:
            quiet_samples: Number of quiet samples collected
            min_baseline_samples: Minimum samples for full confidence

        Returns:
            Quality factor between 0 and 1
        """
        if min_baseline_samples <= 0:
            return 1.0
        if quiet_samples >= min_baseline_samples:
            return 1.0
        return quiet_samples / min_baseline_samples


# =============================================================================
# Factory Function
# =============================================================================

def get_reliability_calculator(
    calculator_type: str = "default",
    **kwargs
) -> ReliabilityScorer:
    """
    Factory function for creating reliability calculators.

    Args:
        calculator_type: "default"
        **kwargs: Arguments passed to calculator constructor

    Returns:
        ReliabilityScorer instance
    """
    calculators = {
        "default": ReliabilityCalculator,
    }

    if calculator_type not in calculators:
        raise ValueError(
            f"Unknown calculator type: {calculator_type}. "
            f"Available: {list(calculators.keys())}"
        )

    return calculators[calculator_type](**kwargs)
