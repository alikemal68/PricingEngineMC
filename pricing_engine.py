
import numpy as np

from path_simulation import simulate
from estimators import sample_statistics, PricingResult


def discounted_payoff_samples(
        model,
        product,
        scheme,
        S_0,
        N,
        times,
        rng=None,
        normal_draws=None,
        weights=None,
) -> np.ndarray:
    """Generate discounted and optionally weighted payoff samples."""

    # 1. Validate the time grid
    times = np.asarray(times, dtype=float)

    if times.ndim != 1 or len(times) < 2:
        raise ValueError("Invalid time grid.")

    if not np.isclose(times[-1], product.maturity):
        raise ValueError(
            "The time grid must end at the option maturity."
        )

    M = len(times) - 1

    # 2. Generate random numbers if not provided
    if normal_draws is None:

        if weights is not None:
            raise ValueError(
                "Weights must be supplied with their corresponding draws."
            )

        if rng is None:
            rng = np.random.default_rng()

        normal_draws = rng.standard_normal(size=(N, M))

    else:
        normal_draws = np.asarray(normal_draws, dtype=float)

        if normal_draws.shape != (N, M):
            raise ValueError(
                "normal_draws must have shape (N, M)."
            )

    # 3. Simulate paths
    result = simulate(
        model=model,
        S_0=S_0,
        times=times,
        normal_draws=normal_draws,
        scheme=scheme,
    )

    # 4. Evaluate payoff
    payoffs = product.payoff(
        result.paths,
        result.times,
    )

    # 5. Discount payoffs
    discount_factor = np.exp(
        -model.r * product.maturity
    )

    samples = discount_factor * payoffs

    # 6. Importance sampling correction
    if weights is not None:

        weights = np.asarray(weights, dtype=float)

        if weights.shape != samples.shape:
            raise ValueError(
                "Weights must match the payoff shape."
            )

        samples = samples * weights

    return samples


def price_option(
        model,
        product,
        scheme,
        S_0,
        N,
        times,
        rng=None,
        normal_draws=None,
        weights=None,
) -> PricingResult:
    """Price an option using independent Monte Carlo observations."""

    # 1. Generate Monte Carlo contributions
    samples = discounted_payoff_samples(
        model=model,
        product=product,
        scheme=scheme,
        S_0=S_0,
        N=N,
        times=times,
        rng=rng,
        normal_draws=normal_draws,
        weights=weights,
    )

    # 2. Calculate statistical properties
    stats = sample_statistics(samples)

    # 3. Return pricing result
    return PricingResult(
        estimate=stats.estimate,
        sample_variance=stats.sample_variance,
        standard_error=stats.standard_error,
        n_observations=stats.n_observations,
    )
