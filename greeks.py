
from dataclasses import dataclass
import numpy as np

from pricing_engine import discounted_payoff_samples
from estimators import sample_statistics, EstimationResult
from stoch_models import GBM


# ============================================================
# Greek result
# ============================================================

@dataclass
class GreekResult(EstimationResult):
    greek_name: str
    method: str
    bump_size: float
    common_random_numbers: bool


# ============================================================
# Internal helper functions
# ============================================================

def _relative_bump(value, dx):
    """Calculate and validate a relative bump size."""

    if not np.isfinite(value) or value <= 0:
        raise ValueError("Parameter must be positive and finite.")

    if not 0 < dx < 1:
        raise ValueError("Require 0 < dx < 1.")

    return dx * value


def _common_normal_draws(
        N,
        times,
        rng=None,
        normal_draws=None,
        weights=None,
):
    """Generate common random numbers if not supplied."""

    if normal_draws is None:

        if weights is not None:
            raise ValueError(
                "Weights must be supplied with their corresponding draws."
            )

        if rng is None:
            rng = np.random.default_rng()

        M = len(times) - 1

        normal_draws = rng.standard_normal(size=(N, M))

    return normal_draws


def _greek_statistics(
        samples,
        greek_name,
        method,
        bump_size,
        common_random_numbers=True,
) -> GreekResult:
    """Calculate statistics and construct a GreekResult."""

    stats = sample_statistics(samples)

    return GreekResult(
        estimate=stats.estimate,
        sample_variance=stats.sample_variance,
        standard_error=stats.standard_error,
        n_observations=stats.n_observations,
        greek_name=greek_name,
        method=method,
        bump_size=bump_size,
        common_random_numbers=common_random_numbers,
    )


# ============================================================
# Central difference Delta
# ============================================================

def central_diff_delta(
        model,
        option_contract,
        scheme,
        S,
        N,
        times,
        dx=0.001,
        rng=None,
        normal_draws=None,
        weights=None,
) -> GreekResult:
    """Estimate Delta using central differences and CRN."""

    h = _relative_bump(S, dx)

    normal_draws = _common_normal_draws(
        N, times, rng, normal_draws, weights
    )

    # Upper spot perturbation
    samples_up = discounted_payoff_samples(
        model=model,
        option_contract=option_contract,
        scheme=scheme,
        S=S + h,
        N=N,
        times=times,
        normal_draws=normal_draws,
        weights=weights,
    )

    # Lower spot perturbation
    samples_down = discounted_payoff_samples(
        model=model,
        option_contract=option_contract,
        scheme=scheme,
        S=S - h,
        N=N,
        times=times,
        normal_draws=normal_draws,
        weights=weights,
    )

    # Individual Delta observations
    delta_samples = (samples_up - samples_down) / (2 * h)

    return _greek_statistics(
        samples=delta_samples,
        greek_name="delta",
        method="central_difference",
        bump_size=h,
    )


# ============================================================
# Central difference Vega
# ============================================================

def central_diff_vega(
        model,
        option_contract,
        scheme,
        S,
        N,
        times,
        dx=0.001,
        rng=None,
        normal_draws=None,
        weights=None,
) -> GreekResult:
    """Estimate GBM Vega using central differences and CRN."""

    if not isinstance(model, GBM):
        raise TypeError("This Vega implementation requires GBM.")

    h = _relative_bump(model.sigma, dx)

    normal_draws = _common_normal_draws(
        N, times, rng, normal_draws, weights
    )

    # Upper volatility perturbation
    model_up = GBM(
        r=model.r,
        sigma=model.sigma + h,
    )

    # Lower volatility perturbation
    model_down = GBM(
        r=model.r,
        sigma=model.sigma - h,
    )

    samples_up = discounted_payoff_samples(
        model=model_up,
        option_contract=option_contract,
        scheme=scheme,
        S=S,
        N=N,
        times=times,
        normal_draws=normal_draws,
        weights=weights,
    )

    samples_down = discounted_payoff_samples(
        model=model_down,
        option_contract=option_contract,
        scheme=scheme,
        S=S,
        N=N,
        times=times,
        normal_draws=normal_draws,
        weights=weights,
    )

    # Individual Vega observations
    vega_samples = (samples_up - samples_down) / (2 * h)

    return _greek_statistics(
        samples=vega_samples,
        greek_name="vega",
        method="central_difference",
        bump_size=h,
    )


# ============================================================
# Central difference Gamma
# ============================================================

def central_diff_gamma(
        model,
        option_contract,
        scheme,
        S,
        N,
        times,
        dx=0.001,
        rng=None,
        normal_draws=None,
        weights=None,
) -> GreekResult:
    """Estimate Gamma using central differences and CRN."""

    h = _relative_bump(S, dx)

    normal_draws = _common_normal_draws(
        N, times, rng, normal_draws, weights
    )

    # Upper spot perturbation
    samples_up = discounted_payoff_samples(
        model=model,
        option_contract=option_contract,
        scheme=scheme,
        S=S + h,
        N=N,
        times=times,
        normal_draws=normal_draws,
        weights=weights,
    )

    # Unperturbed valuation
    samples_base = discounted_payoff_samples(
        model=model,
        option_contract=option_contract,
        scheme=scheme,
        S=S,
        N=N,
        times=times,
        normal_draws=normal_draws,
        weights=weights,
    )

    # Lower spot perturbation
    samples_down = discounted_payoff_samples(
        model=model,
        option_contract=option_contract,
        scheme=scheme,
        S=S - h,
        N=N,
        times=times,
        normal_draws=normal_draws,
        weights=weights,
    )

    # Individual Gamma observations
    gamma_samples = (
        samples_up - 2 * samples_base + samples_down
    ) / h**2

    return _greek_statistics(
        samples=gamma_samples,
        greek_name="gamma",
        method="central_difference",
        bump_size=h,
    )
