
from dataclasses import dataclass
import numpy as np


@dataclass
class EstimationResult:
    estimate: float
    sample_variance: float
    standard_error: float
    n_observations: int

    @property
    def ci_lower(self):
        return self.estimate - 1.96 * self.standard_error

    @property
    def ci_upper(self):
        return self.estimate + 1.96 * self.standard_error

    @property
    def confidence_interval(self):
        return self.ci_lower, self.ci_upper


@dataclass
class PricingResult(EstimationResult):

    @property
    def price(self):
        return self.estimate


def sample_statistics(observations) -> EstimationResult:
    """Calculate statistics of independent iid observations."""

    observations = np.asarray(observations, dtype=float)

    if observations.ndim != 1:
        raise ValueError("Observations must be one-dimensional.")

    n = len(observations)

    if n < 2:
        raise ValueError("At least two observations are required.")

    if not np.all(np.isfinite(observations)):
        raise ValueError("Observations must be finite.")

    # Monte Carlo statistics
    estimate = np.mean(observations)
    sample_variance = np.var(observations, ddof=1)
    standard_error = np.sqrt(sample_variance / n)

    return EstimationResult(
        estimate=estimate,
        sample_variance=sample_variance,
        standard_error=standard_error,
        n_observations=n,
    )


# Temporary wrapper for backward compatibility
# def mc_estimator(
#         payoff,
#         discount_factor,
#         weights=None,
# ) -> PricingResult:

#     samples = discount_factor * np.asarray(payoff, dtype=float)

#     # Importance sampling correction
#     if weights is not None:
#         weights = np.asarray(weights, dtype=float)

#         if weights.shape != samples.shape:
#             raise ValueError("Weights must match the payoff shape.")

#         samples = samples * weights

#     # Statistical evaluation
#     stats = sample_statistics(samples)

#     return PricingResult(
#         estimate=stats.estimate,
#         sample_variance=stats.sample_variance,
#         standard_error=stats.standard_error,
#         n_observations=stats.n_observations,
#     )
