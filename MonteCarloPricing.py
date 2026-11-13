
import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats as st

import enum


def generate_GBM_paths_euler(
        paths_no,
        step_no,
        end_time,
        interest_rate,
        vol,
        S_0
):
    # random innovations 
    Z = np.random.normal(0.0,1.0,[paths_no, step_no])

    # Wiener process
    W = np.zeros([paths_no,step_no+1])

    # Monte Carlo euler process
    S_euler_approx = np.zeros([paths_no,step_no+1])
    S_euler_approx[:,0] = S_0

    # exact process
    S_exact = np.zeros([paths_no,step_no+1])
    S_exact[:,0] = S_0

    time = np.zeros([step_no+1])
    dt = end_time / step_no

    for i in range(step_no):
        # if paths_no > 1:
        #     Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])

        # update Wiener process
        W[:,i+1] = W[:,i] + np.sqrt(dt) * Z[:,i]

        # update monte carlo euler process and exact process
        S_euler_approx[:,i+1] = S_euler_approx[:,i] + interest_rate * S_euler_approx[:,i] * dt + vol * S_euler_approx[:,i] * (W[:,i+1]-W[:,i])
        S_exact[:,i+1] = S_exact[:,i] * np.exp((interest_rate-0.5*vol**2)*dt + vol*(W[:,i+1]-W[:,i]))

        # update time
        time[i+1] = time[i] + dt 


    paths = {"time":time, "S_euler_approx": S_euler_approx, "S_exact":S_exact}
    return paths

def generate_GBM_paths_milstein(
        paths_no,
        step_no,
        end_time,
        interest_rate,
        vol,
        S_0
):
    # random innovations 
    Z = np.random.normal(0.0,1.0,[paths_no, step_no])

    # Wiener process
    W = np.zeros([paths_no,step_no+1])

    # Monte Carlo euler process
    S_milstein_approx = np.zeros([paths_no,step_no+1])
    S_milstein_approx[:,0] = S_0

    # exact process
    S_exact = np.zeros([paths_no,step_no+1])
    S_exact[:,0] = S_0

    time = np.zeros([step_no+1])
    dt = end_time / step_no

    for i in range(step_no):
        # if paths_no > 1:
        #     Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])

        # update Wiener process
        W[:,i+1] = W[:,i] + np.sqrt(dt) * Z[:,i]

        # update monte carlo euler process and exact process
        S_milstein_approx[:,i+1] = S_milstein_approx[:,i] + interest_rate * S_milstein_approx[:,i] * dt + vol * S_milstein_approx[:,i] * (W[:,i+1]-W[:,i]) + 0.5 * vol**2 * S_milstein_approx[:,i] * (np.power((W[:,i+1] -
W[:,i]),2) - dt)
        S_exact[:,i+1] = S_exact[:,i] * np.exp((interest_rate-0.5*vol**2)*dt + vol*(W[:,i+1]-W[:,i]))

        # update time
        time[i+1] = time[i] + dt 


    paths = {"time":time, "S_milstein_approx": S_milstein_approx, "S_exact":S_exact}
    return paths




def analytical_BS_call_put_price(option_type,S_0,K,sigma,tau,r):

    # K = np.array(K).reshape([len(K),1])
    d1 = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0))
    * tau) / float(sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    if option_type == "call":
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * tau)
    elif option_type == "put":
        value = st.norm.cdf(-d2) * K * np.exp(-r * tau) - st.norm.cdf(-d1)*S_0
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return value

def mc_call_put_price(option_type,S,K,T,r):
    if option_type == "call":
        payoff = np.maximum(S - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - S, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted_payoff = np.exp(-r * T) * payoff

    N = S.size
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
        paths_no,
        step_counts,
        end_time,
        interest_rate,
        vol,
        S_0):

    dt_values = end_time / np.asarray(step_counts, dtype=float)
    error_weak = np.zeros(len(step_counts))
    error_strong = np.zeros(len(step_counts))

    for k, step_no in enumerate(step_counts):
        paths = generate_GBM_paths_euler(
            paths_no, int(step_no), end_time, interest_rate, vol, S_0
        )

        S_euler_last = paths["S_euler_approx"][:, -1]
        S_exact_last = paths["S_exact"][:, -1]

        difference = S_euler_last - S_exact_last

        error_weak[k] = np.abs(np.mean(difference))
        error_strong[k] = np.mean(np.abs(difference))

    return {
        "dt": dt_values,
        "error_weak": error_weak,
        "error_strong": error_strong,
    }




# def main_calc():


    paths_no = 2000
    step_no = 1000

    end_time=1
    interest_rate = 0.05

    vol = 0.3
    S_0 = 50

    # paths = generate_GBM_paths_euler(paths_no, step_no, end_time, interest_rate, vol, S_0)

    # time = paths["time"]
    # S_euler_approx = paths["S_euler_approx"]
    # S_exact = paths["S_exact"]

    # plt.figure(1)
    # plt.plot(time, np.transpose(S_euler_approx),'k')
    # plt.plot(time, np.transpose(S_exact),'--r')
    # plt.grid()
    # plt.xlabel("time")
    # plt.ylabel("S(t)")

    # plt.show()
    # plt.close()

   
#     step_counts = 2 ** np.arange(10)  # 1, 2, 4, ..., 512

#     errors = weak_and_strong_convergence(
#         paths_no, step_counts, end_time, interest_rate, vol, S_0
#     )

#     # Sort into increasing time-step size.
#     order = np.argsort(errors["dt"])
#     dt = errors["dt"][order]
#     weak = errors["error_weak"][order]
#     strong = errors["error_strong"][order]

#     fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

#     ax.loglog(
#         dt, weak, "o-",
#         color="tab:blue", linewidth=1.8, markersize=5,
#         label="Weak error estimate",
#     )
#     ax.loglog(
#         dt, strong, "s-",
#         color="tab:orange", linewidth=1.8, markersize=5,
#         label="Strong error estimate",
#     )

#     # Reference slopes, scaled to the strong error at the largest dt.
#     ax.loglog(
#         dt, strong[-1] * (dt / dt[-1]) ** 0.5,
#         "--", color="0.45", linewidth=1.2,
#         label=r"Reference: $(\Delta t)^{1/2}$",
#     )

#     ax.set(
#         title="Euler–Maruyama convergence for GBM",
#         xlabel=r"Time-step size $\Delta t$",
#         ylabel="Terminal error",
#     )

#     ax.grid(which="major", alpha=0.3)
#     ax.grid(which="minor", alpha=0.1)
#     ax.spines[["top", "right"]].set_visible(False)

#     weak_bias = S_0 * np.abs(
#     (1 + interest_rate * dt) ** (end_time / dt)
#     - np.exp(interest_rate * end_time)
# )

#     ax.loglog(
#     dt, weak_bias, "--",
#     color="tab:blue", alpha=0.65,
#     label="Exact weak bias for f(S) = S",
# )

#     ax.legend(frameon=False)

#     plt.show()

