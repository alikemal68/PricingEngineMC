
import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats as st

import MonteCarloPricing


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
        paths = MonteCarloPricing.generate_GBM_paths(
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
    bs_price = MonteCarloPricing.analytical_BS_call_put_price(
        option_type, S_0, K, T, r, sigma
    )

    results = []

    for N in path_counts:
        # Same innovations for Euler and Milstein.
        np.random.seed(seed)
        paths_euler = MonteCarloPricing.generate_GBM_paths(
            N, M, T, r, sigma, S_0, scheme="euler"
        )

        np.random.seed(seed)
        paths_milstein = MonteCarloPricing.generate_GBM_paths(
            N, M, T, r, sigma, S_0,  scheme="milstein"
        )

        terminal_prices = {
            "exact": paths_euler["S_exact"][:, -1],
            "euler": paths_euler["S_approx"][:, -1],
            "milstein": paths_milstein["S_approx"][:, -1],
        }

        row = {"N": N, "bs_price": bs_price}

        for scheme, S_T in terminal_prices.items():
            stats = MonteCarloPricing.mc_call_put_price(
                option_type, S_T, K, T, r
            )

            for name, value in stats.items():
                row[f"{scheme}_{name}"] = value

            row[f"{scheme}_error"] = stats["price"] - bs_price

        results.append(row)

    return results



