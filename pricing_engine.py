
import numpy as np

from path_simulation import simulate
from estimators import sample_statistics, PricingResult


def discounted_payoff_samples(
        model,
        option_contract,
        scheme,
        S,
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

    if not np.all(np.isfinite(times)):
        raise ValueError("Time grid must contain finite values.")

    if np.any(np.diff(times) <= 0):
        raise ValueError("Time grid must be strictly increasing.")

    t = times[0]
    T = option_contract.maturity
    tau = T - t

    if t < 0 or tau <= 0:
        raise ValueError("Valuation time must satisfy 0 <= t < T.")

    if not np.isclose(times[-1], T):
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
        S=S,
        times=times,
        normal_draws=normal_draws,
        scheme=scheme,
    )

    # 4. Evaluate payoff
    payoffs = option_contract.payoff(
        result.paths,
        result.times,
    )

    # 5. Discount payoffs
    discount_factor = np.exp(-model.r * tau)

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
        option_contract,
        scheme,
        S,
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
        option_contract=option_contract,
        scheme=scheme,
        S=S,
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
