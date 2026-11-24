

import numpy as np


def mc_estimator(
        payoff,
        discount_factor,
        weights=None,
):
    """Estimate price and statistical uncertainty."""

    N = len(payoff)

    if N < 2:
        raise ValueError("At least two samples are required.")

    discounted_payoff = discount_factor * payoff

    # Importance sampling correction
    if weights is not None:
        discounted_payoff = discounted_payoff * weights

    # Monte Carlo statistics
    price = np.mean(discounted_payoff)
    payoff_variance = np.var(discounted_payoff, ddof=1)
    standard_error = np.sqrt(payoff_variance / N)

    return {
        "price": price,
        "payoff_variance": payoff_variance,
        "standard_error": standard_error,
        "ci_lower": price - 1.96 * standard_error,
        "ci_upper": price + 1.96 * standard_error,
    }

# One thing we will need to revisit later is the standard-error calculation for methods that introduce dependence between samples, such as moment matching, antithetic variates and quasi-Monte Carlo.