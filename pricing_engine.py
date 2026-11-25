

import numpy as np

from path_simulation import simulate
from estimators import mc_estimator

# NOTES TO MYSELF
# We have implemented importance sampling but we need to incorparate it into our pricing engine here! 


def price_option(
        model,
        product,
        scheme,
        S_0,
        N,
        times,
        rng=None,
        normal_draws=None,
        estimator=mc_estimator,
        weights=None,
):
    """Price an option using Monte Carlo simulation."""

    # 1. Validate the time grid
    times = np.asarray(times, dtype=float)

    if not np.isclose(times[-1], product.maturity):
        raise ValueError(
            "The time grid must end at the option maturity."
        )

    M = len(times) - 1

    # 2. Generate random numbers if not provided
    if normal_draws is None:

        if rng is None:
            rng = np.random.default_rng()

        normal_draws = rng.standard_normal(size=(N, M))

    elif normal_draws.shape != (N, M):
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

    # 5. Calculate discount factor
    discount_factor = np.exp(
        -model.r * product.maturity
    )

    # 6. Estimate price and statistical uncertainty
    return estimator(
        payoff=payoffs,
        discount_factor=discount_factor,
        weights=weights,
    )
