
import numpy as np
import scipy.stats as st

from option_contracts import EuropeanCall, EuropeanPut
from stoch_models import GBM


def _black_scholes_terms(
        model,
        option_contract,
        S,
        t=0.0,
):
    """Calculate common Black-Scholes quantities."""

    # Validate model and option
    if not isinstance(model, GBM):
        raise TypeError("Black-Scholes pricing requires a GBM model.")

    if not isinstance(option_contract, (EuropeanCall, EuropeanPut)):
        raise TypeError("Unsupported option type.")

    # Model parameters
    r = model.r
    sigma = model.sigma

    # Contract parameters
    K = option_contract.strike
    T = option_contract.maturity

    # Remaining time to maturity
    tau = T - t

    if t < 0 or tau <= 0:
        raise ValueError("Valuation time must satisfy 0 <= t < T.")

    if np.any(np.asarray(S) <= 0) or K <= 0 or sigma <= 0:
        raise ValueError("Spot, strike and volatility must be positive.")

    # Black-Scholes quantities
    d1 = (
        np.log(S / K)
        + (r + 0.5 * sigma**2) * tau
    ) / (sigma * np.sqrt(tau))

    d2 = d1 - sigma * np.sqrt(tau)

    return d1, d2, tau


def black_scholes_price(
        model,
        option_contract,
        S,
        t=0.0,
):
    """Analytical Black-Scholes price."""

    d1, d2, tau = _black_scholes_terms(
        model, option_contract, S, t
    )

    K = option_contract.strike
    discount_factor = np.exp(-model.r * tau)

    if isinstance(option_contract, EuropeanCall):
        return (
            S * st.norm.cdf(d1)
            - K * discount_factor * st.norm.cdf(d2)
        )

    elif isinstance(option_contract, EuropeanPut):
        return (
            K * discount_factor * st.norm.cdf(-d2)
            - S * st.norm.cdf(-d1)
        )


def black_scholes_delta(
        model,
        option_contract,
        S,
        t=0.0,
):
    """First derivative with respect to spot."""

    d1, _, _ = _black_scholes_terms(
        model, option_contract, S, t
    )

    if isinstance(option_contract, EuropeanCall):
        return st.norm.cdf(d1)

    elif isinstance(option_contract, EuropeanPut):
        return st.norm.cdf(d1) - 1


def black_scholes_vega(
        model,
        option_contract,
        S,
        t=0.0,
):
    """First derivative with respect to volatility."""

    d1, _, tau = _black_scholes_terms(
        model, option_contract, S, t
    )

    return S * st.norm.pdf(d1) * np.sqrt(tau)


def black_scholes_gamma(
        model,
        option_contract,
        S,
        t=0.0,
):
    """Second derivative with respect to spot."""

    d1, _, tau = _black_scholes_terms(
        model, option_contract, S, t
    )

    return (
        st.norm.pdf(d1)
        / (S * model.sigma * np.sqrt(tau))
    )
