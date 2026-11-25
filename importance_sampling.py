
import numpy as np

from stoch_models import GBM
from option_contracts import EuropeanCall, EuropeanPut


def gbm_strike_shift(
        model,
        product,
        S_0,
):
    """Strike-based importance sampling shift for GBM."""

    if not isinstance(model, GBM):
        raise TypeError("This shift requires a GBM model.")

    r = model.r
    sigma = model.sigma

    K = product.strike
    T = product.maturity

    z_K = (
        np.log(K / S_0)
        - (r - 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    if isinstance(product, EuropeanCall):
        return max(z_K, 0.0)

    elif isinstance(product, EuropeanPut):
        return min(z_K, 0.0)

    else:
        raise TypeError("Unsupported product type.")
