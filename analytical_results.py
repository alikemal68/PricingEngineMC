
import numpy as np
import scipy.stats as st

from option_contracts import EuropeanCall, EuropeanPut
from stoch_models import GBM


def black_scholes_price(
        model,
        product,
        S_0,
):
    """Analytical Black-Scholes price of a European option."""

    if not isinstance(model, GBM):
        raise TypeError("Black-Scholes pricing requires a GBM model.")

    if not isinstance(product, (EuropeanCall, EuropeanPut)):
        raise TypeError("Unsupported option type.")

    # Extract model parameters
    r = model.r
    sigma = model.sigma

    # Extract product parameters
    K = product.strike
    T = product.maturity

    d1 = (
        np.log(S_0 / K) + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    discount_factor = np.exp(-r * T)

    if isinstance(product, EuropeanCall):

        value = (
            S_0 * st.norm.cdf(d1)
            - K * discount_factor * st.norm.cdf(d2)
        )

    elif isinstance(product, EuropeanPut):

        value = (
            K * discount_factor * st.norm.cdf(-d2)
            - S_0 * st.norm.cdf(-d1)
        )

    return value


# import numpy as np
# import scipy.stats as st

# def analytical_BS_call_put_price(
#         option_type,
#         S_0,
#         K,
#         T,
#         r,
#         sigma,
# ):

#     # K = np.array(K).reshape([len(K),1])
#     d1 = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0))
#     * T) / float(sigma * np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)
#     if option_type == "call":
#         value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * T)
#     elif option_type == "put":
#         value = st.norm.cdf(-d2) * K * np.exp(-r * T) - st.norm.cdf(-d1)*S_0
#     else:
#         raise ValueError("option_type must be 'call' or 'put'")
#     return value
