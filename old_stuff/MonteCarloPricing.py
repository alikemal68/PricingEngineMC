
import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats as st


def generate_GBM_paths(
        N, M, T, r, sigma, S_0,
        scheme="euler",
        moment_matching=False,
):
    """Generate GBM paths on M time steps, optionally with a coupled approximation."""
    if scheme not in ("exact", "euler", "milstein"):
        raise ValueError("scheme must be 'exact', 'euler', or 'milstein'")

    if N < 1 or M < 1:
        raise ValueError("N and M must be positive.")

    if moment_matching and N < 2:
        raise ValueError("Moment matching requires at least two paths.")

    dt = T / M
    sqrt_dt = np.sqrt(dt)
    time = np.linspace(0.0, T, M + 1)
    Z = np.random.normal(size=(N, M))

    S_exact = np.empty((N, M + 1))
    S_exact[:, 0] = S_0

    S_approx = None
    if scheme != "exact":
        S_approx = np.empty_like(S_exact)
        S_approx[:, 0] = S_0

    for i in range(M):
        z = Z[:, i]
        if moment_matching:
            z = (z - np.mean(z)) / np.std(z)

        dW = sqrt_dt * z
        S_exact[:, i + 1] = S_exact[:, i] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * dW
        )

        if S_approx is None:
            continue

        # Both schemes use the same increment as the exact reference.
        S_current = S_approx[:, i]
        S_next = S_current + r * S_current * dt + sigma * S_current * dW

        if scheme == "milstein":
            S_next += 0.5 * sigma**2 * S_current * (dW**2 - dt)

        S_approx[:, i + 1] = S_next

    return {
        "time": time,
        "S_exact": S_exact,
        "S_approx": S_approx,
        "scheme": scheme,
        "moment_matching": moment_matching,
    }

def analytical_BS_call_put_price(
        option_type,
        S_0,
        K,
        T,
        r,
        sigma,
):

    # K = np.array(K).reshape([len(K),1])
    d1 = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0))
    * T) / float(sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * T)
    elif option_type == "put":
        value = st.norm.cdf(-d2) * K * np.exp(-r * T) - st.norm.cdf(-d1)*S_0
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return value

def mc_call_put_price(
        option_type,
        S_T,
        K,
        T,
        r
):
    if option_type == "call":
        payoff = np.maximum(S_T - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - S_T, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted_payoff = np.exp(-r * T) * payoff

    N = S_T.size
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

#############################################   
