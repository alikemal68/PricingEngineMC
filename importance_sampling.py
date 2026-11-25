
import numpy as np

from stoch_models import GBM
from option_contracts import EuropeanCall, EuropeanPut


def gbm_strike_shift(
        model,
        option_contract,
        S,
        t=0.0,
):
    """Strike-based importance sampling shift for GBM."""

    if not isinstance(model, GBM):
        raise TypeError("This shift requires a GBM model.")

    r = model.r
    sigma = model.sigma

    K = option_contract.strike
    T = option_contract.maturity

    tau = T - t

    if t < 0 or tau <= 0:
        raise ValueError("Valuation time must satisfy 0 <= t < T.")

    if S <= 0 or K <= 0 or sigma <= 0:
        raise ValueError("Spot, strike and volatility must be positive.")

    z_K = (
        np.log(K / S)
        - (r - 0.5 * sigma**2) * tau
    ) / (sigma * np.sqrt(tau))

    if isinstance(option_contract, EuropeanCall):
        return max(z_K, 0.0)

    elif isinstance(option_contract, EuropeanPut):
        return min(z_K, 0.0)

    else:
        raise TypeError("Unsupported option type.")
