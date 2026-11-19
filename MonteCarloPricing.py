
import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats as st


def generate_GBM_paths(
        N,      # number of paths
        M,      # number of time steps
        T,      # maturity
        r,      # interest rate
        sigma,  # volatility
        S_0,    # initial stock value
        scheme="euler",
        moment_matching=False,

):
    if scheme not in ("euler", "milstein"):
        raise ValueError("scheme must be 'euler' or 'milstein'")

    if moment_matching and N < 2:
        raise ValueError("Moment matching requires at least two paths.")

    dt = T / M
    time = np.linspace(0.0, T, M + 1)

    Z = np.random.normal(size=(N, M))

    S_approx = np.zeros((N, M + 1))
    S_exact = np.zeros((N, M + 1))

    S_approx[:, 0] = S_0
    S_exact[:, 0] = S_0

    for i in range(M):
        z = Z[:, i]

        if moment_matching:
            z = (z - np.mean(z)) / np.std(z)

        dW = np.sqrt(dt) * z
        S_current = S_approx[:, i]

        # Euler update, also the base of the Milstein update.
        S_approx[:, i + 1] = (
            S_current
            + r * S_current * dt
            + sigma * S_current * dW
        )

        if scheme == "milstein":
            S_approx[:, i + 1] += (
                0.5 * sigma**2 * S_current * (dW**2 - dt)
            )

        # Coupled exact-form update.
        S_exact[:, i + 1] = S_exact[:, i] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * dW
        )

    return {
        "time": time,
        "S_approx": S_approx,
        "S_exact": S_exact,
        "scheme": scheme,
        "moment_matching": moment_matching,
    }

# def generate_GBM_paths_euler(
#         N,
#         M,
#         T,
#         r,
#         sigma,
#         S_0,
#         moment_matching=False
# ):
#     # random innovations 
#     Z = np.random.normal(0.0,1.0,[N, M])

#     # Wiener process
#     W = np.zeros([N,M+1])

#     # Monte Carlo euler process
#     S_euler_approx = np.zeros([N,M+1])
#     S_euler_approx[:,0] = S_0

#     # exact process
#     S_exact = np.zeros([N,M+1])
#     S_exact[:,0] = S_0

#     time = np.zeros([M+1])
#     dt = T / M

#     for i in range(M):
#         if moment_matching and N > 1:
#             Z[:, i] = (
#                 (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
#             )

#         # update Wiener process
#         W[:,i+1] = W[:,i] + np.sqrt(dt) * Z[:,i]

#         # update monte carlo euler process and exact process
#         S_euler_approx[:,i+1] = S_euler_approx[:,i] + r * S_euler_approx[:,i] * dt + sigma * S_euler_approx[:,i] * (W[:,i+1]-W[:,i])
#         S_exact[:,i+1] = S_exact[:,i] * np.exp((r-0.5*sigma**2)*dt + sigma*(W[:,i+1]-W[:,i]))

#         # update time
#         time[i+1] = time[i] + dt 


#     paths = {"time":time, "S_euler_approx": S_euler_approx, "S_exact":S_exact}
#     return paths

# def generate_GBM_paths_milstein(
#         N,
#         M,
#         T,
#         r,
#         sigma,
#         S_0,
#         moment_matching=False
# ):
#     # random innovations 
#     Z = np.random.normal(0.0,1.0,[N, M])

#     # Wiener process
#     W = np.zeros([N,M+1])

#     # Monte Carlo euler process
#     S_milstein_approx = np.zeros([N,M+1])
#     S_milstein_approx[:,0] = S_0

#     # exact process
#     S_exact = np.zeros([N,M+1])
#     S_exact[:,0] = S_0

#     time = np.zeros([M+1])
#     dt = T / M

#     for i in range(M):
#         if moment_matching and N > 1:
#             Z[:, i] = (
#                 (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
#             )

#         # update Wiener process
#         W[:,i+1] = W[:,i] + np.sqrt(dt) * Z[:,i]

#         # update monte carlo euler process and exact process
#         S_milstein_approx[:,i+1] = S_milstein_approx[:,i] + r * S_milstein_approx[:,i] * dt + sigma * S_milstein_approx[:,i] * (W[:,i+1]-W[:,i]) + 0.5 * sigma**2 * S_milstein_approx[:,i] * (np.power((W[:,i+1] -
# W[:,i]),2) - dt)
#         S_exact[:,i+1] = S_exact[:,i] * np.exp((r-0.5*sigma**2)*dt + sigma*(W[:,i+1]-W[:,i]))

#         # update time
#         time[i+1] = time[i] + dt 


#     paths = {"time":time, "S_milstein_approx": S_milstein_approx, "S_exact":S_exact}
#     return paths

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

def weak_and_strong_convergence(
        N,
        step_counts,
        T,
        r,
        sigma,
        S_0,
        scheme="euler",
        moment_matching=False
):
    # if scheme == "euler":
    #     generator = generate_GBM_paths_euler
    #     price_key = "S_euler_approx"
    # elif scheme == "milstein":
    #     generator = generate_GBM_paths_milstein
    #     price_key = "S_milstein_approx"
    # else:
    #     raise ValueError("scheme must be 'euler' or 'milstein'")

    dt_values = T / np.asarray(step_counts, dtype=float)
    error_weak = np.zeros(len(step_counts))
    error_strong = np.zeros(len(step_counts))

    for k, M in enumerate(step_counts):
        paths = generate_GBM_paths(
            N, int(M), T, r, sigma, S_0, scheme, moment_matching 
        )

        S_approx_last = paths["S_approx"][:, -1]
        S_exact_last = paths["S_exact"][:, -1]

        difference = S_approx_last - S_exact_last

        error_weak[k] = np.abs(np.mean(difference))
        error_strong[k] = np.mean(np.abs(difference))

    return {
        "scheme": scheme,
        "dt": dt_values,
        "error_weak": error_weak,
        "error_strong": error_strong,
    }

def mc_convergence(
        path_counts,
        M,
        option_type,
        S_0,
        K,
        T,
        r,
        sigma,
        seed,
):
    bs_price = analytical_BS_call_put_price(
        option_type, S_0, K, T, r, sigma
    )

    results = []

    for N in path_counts:
        # Same innovations for Euler and Milstein.
        np.random.seed(seed)
        paths_euler = generate_GBM_paths(
            N, M, T, r, sigma, S_0, scheme="euler"
        )

        np.random.seed(seed)
        paths_milstein = generate_GBM_paths(
            N, M, T, r, sigma, S_0,  scheme="milstein"
        )

        terminal_prices = {
            "exact": paths_euler["S_exact"][:, -1],
            "euler": paths_euler["S_approx"][:, -1],
            "milstein": paths_milstein["S_approx"][:, -1],
        }

        row = {"N": N, "bs_price": bs_price}

        for scheme, S_T in terminal_prices.items():
            stats = mc_call_put_price(
                option_type, S_T, K, T, r
            )

            for name, value in stats.items():
                row[f"{scheme}_{name}"] = value

            row[f"{scheme}_error"] = stats["price"] - bs_price

        results.append(row)

    return results



