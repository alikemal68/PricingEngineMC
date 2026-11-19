
import numpy as np
import matplotlib.pyplot as plt 
# import scipy.stats as st

def print_terminal_dist_moments_table(
        paths,
        S_0,
        interest_rate,
        vol,
        terminal_time,
        decimals=4,
):
    from IPython.display import Markdown, display

    theoretical_mean = S_0 * np.exp(interest_rate * terminal_time)
    theoretical_variance = (
        S_0**2
        * np.exp(2 * interest_rate * terminal_time)
        * np.expm1(vol**2 * terminal_time)
    )

    rows = {
        "Theoretical GBM": (
            theoretical_mean,
            theoretical_variance,
            np.sqrt(theoretical_variance),
        )
    }

    samples = {
        "Exact GBM sample": paths["euler"]["S_exact"][:, -1],
        "Euler": paths["euler"]["S_approx"][:, -1],
        "Milstein": paths["milstein"]["S_approx"][:, -1],
    }

    for name, values in samples.items():
        rows[name] = (
            np.mean(values),
            np.var(values, ddof=1),
            np.std(values, ddof=1),
        )

    table = (
        "| Method | Mean | Variance | Standard deviation |\n"
        "|:---|---:|---:|---:|\n"
    )

    for name, (mean, variance, std) in rows.items():
        table += (
            f"| {name} "
            f"| {mean:.{decimals}f} "
            f"| {variance:.{decimals}f} "
            f"| {std:.{decimals}f} |\n"
        )

    display(Markdown(table))

def plot_gbm_simulation(
        paths,
        scheme="euler",
        paths_to_show=5,
        bins=40
):
    # if scheme == "euler":
    #     price_key = "S_euler_approx"
    # elif scheme == "milstein":
    #     price_key = "S_milstein_approx"
    # else:
    #     raise ValueError("scheme must be 'euler' or 'milstein'")

    time = paths["time"]
    S_approx = paths["S_approx"]
    S_exact = paths["S_exact"]

    terminal_approx = S_approx[:, -1]
    terminal_exact = S_exact[:, -1]
    difference = terminal_approx - terminal_exact

    fig, axes = plt.subplots(
        1, 3,
        figsize=(16, 4),
        constrained_layout=True,
    )

    # 1. Paired sample paths: same color for each approximate/exact pair.
    n_show = min(paths_to_show, S_approx.shape[0])

    for j in range(n_show):
        line, = axes[0].plot(
            time,
            S_approx[j],
            linewidth=1.3,
            label=scheme.capitalize() if j == 0 else None,
        )

        axes[0].plot(
            time,
            S_exact[j],
            "--",
            color=line.get_color(),
            linewidth=1.3,
            label="Exact GBM" if j == 0 else None,
        )

    axes[0].set(
        title=f"Sample paths ({n_show} shown)",
        xlabel="Time",
        ylabel="Stock price",
    )
    axes[0].legend()

    # 2. Terminal distributions, using identical bin edges.
    bin_edges = np.histogram_bin_edges(
        np.concatenate([terminal_approx, terminal_exact]),
        bins=bins,
    )

    axes[1].hist(
        terminal_approx,
        bins=bin_edges,
        density=True,
        alpha=0.5,
        color="tab:blue",
        label=scheme.capitalize(),
    )
    axes[1].hist(
        terminal_exact,
        bins=bin_edges,
        density=True,
        histtype="step",
        linewidth=1.5,
        color="black",
        label="Exact GBM",
    )

    axes[1].set(
        title="Terminal-price distributions",
        xlabel=r"$S_T$",
        ylabel="Density",
    )
    axes[1].legend()

    # 3. Signed, paired terminal errors.
    axes[2].hist(
        difference,
        bins=bins,
        color="tab:orange",
        alpha=0.7,
    )
    axes[2].axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Zero error",
    )
    axes[2].axvline(
        np.mean(difference),
        color="tab:red",
        linewidth=1.5,
        label="Mean error",
    )

    axes[2].set(
        title="Paired terminal errors",
        xlabel=r"$S_T^{\mathrm{scheme}}-S_T^{\mathrm{exact}}$",
        ylabel="Number of paths",
    )
    axes[2].legend()

    for ax in axes:
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)

    fig.suptitle(f"{scheme.capitalize()} versus exact GBM")

    return fig, axes

def plot_gbm_comparison(
        paths_euler,
        paths_milstein,
        paths_to_show=5,
        bins=40,
):
    simulations = [
        ("Euler", paths_euler, "S_approx"),
        ("Milstein", paths_milstein, "S_approx"),
    ]

    terminal_values = []
    errors = []

    for _, paths, price_key in simulations:
        terminal_approx = paths[price_key][:, -1]
        terminal_exact = paths["S_exact"][:, -1]

        terminal_values.extend([terminal_approx, terminal_exact])
        errors.append(terminal_approx - terminal_exact)

    # Shared bins for both terminal-price histograms.
    terminal_bins = np.histogram_bin_edges(
        np.concatenate(terminal_values), bins=bins
    )

    # Shared error bins, symmetric around zero.
    max_error = np.max(np.abs(np.concatenate(errors)))
    if max_error == 0:
        max_error = 1e-12

    error_bins = np.linspace(-max_error, max_error, bins + 1)

    fig, axes = plt.subplots(
        2, 3,
        figsize=(16, 8),
        sharex="col",
        sharey="col",
        constrained_layout=True,
    )

    for row, (scheme, paths, price_key) in enumerate(simulations):
        time = paths["time"]
        S_approx = paths[price_key]
        S_exact = paths["S_exact"]
        difference = errors[row]

        ax_paths, ax_terminal, ax_error = axes[row]

        # 1. Sample paths.
        n_show = min(paths_to_show, S_approx.shape[0])

        for j in range(n_show):
            line, = ax_paths.plot(
                time,
                S_approx[j],
                linewidth=1.3,
                label=scheme if j == 0 else None,
            )
            ax_paths.plot(
                time,
                S_exact[j],
                "--",
                color=line.get_color(),
                linewidth=1.3,
                label="Exact GBM" if j == 0 else None,
            )

        ax_paths.set_title(f"{scheme}: sample paths")
        ax_paths.set_ylabel("Stock price")
        ax_paths.legend()

        # 2. Terminal-price distributions.
        ax_terminal.hist(
            S_approx[:, -1],
            bins=terminal_bins,
            density=True,
            alpha=0.5,
            color="tab:blue",
            label=scheme,
        )
        ax_terminal.hist(
            S_exact[:, -1],
            bins=terminal_bins,
            density=True,
            histtype="step",
            linewidth=1.5,
            color="black",
            label="Exact GBM",
        )

        ax_terminal.set_title(f"{scheme}: terminal distribution")
        ax_terminal.set_ylabel("Density")
        ax_terminal.legend()

        # 3. Paired errors: identical bins and scales for both schemes.
        ax_error.hist(
            difference,
            bins=error_bins,
            density=True,
            alpha=0.7,
            color="tab:orange",
        )
        ax_error.axvline(
            0,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Zero error",
        )
        ax_error.axvline(
            np.mean(difference),
            color="tab:red",
            linewidth=1.5,
            label="Mean error",
        )

        ax_error.set_xlim(error_bins[0], error_bins[-1])
        ax_error.set_title(f"{scheme}: paired terminal errors")
        ax_error.set_ylabel("Density")
        ax_error.legend()

    axes[1, 0].set_xlabel("Time")
    axes[1, 1].set_xlabel(r"$S_T$")
    axes[1, 2].set_xlabel(
        r"$S_T^{\mathrm{scheme}}-S_T^{\mathrm{exact}}$"
    )

    for ax in axes.flat:
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)

    return fig, axes

def plot_weak_and_strong_convergence(
        errors,
        S_0,
        interest_rate,
        end_time,
        scheme="euler",
):
    if scheme not in ("euler", "milstein"):
        raise ValueError("scheme must be 'euler' or 'milstein'")

    order = np.argsort(errors["dt"])
    dt = np.asarray(errors["dt"])[order]
    weak = np.asarray(errors["error_weak"])[order]
    strong = np.asarray(errors["error_strong"])[order]

    strong_order = 0.5 if scheme == "euler" else 1.0

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    ax.loglog(
        dt, weak, "o-",
        color="tab:blue",
        label="Weak error estimate",
    )
    ax.loglog(
        dt, strong, "s-",
        color="tab:orange",
        label="Strong error estimate",
    )

    # Reference slope anchored at the largest time step.
    reference = strong[-1] * (dt / dt[-1]) ** strong_order

    ax.loglog(
        dt, reference, "--",
        color="gray",
        alpha=0.6,
        zorder=1,
        label=f"Strong reference slope: {strong_order:g}",
    )

    # Valid for both schemes when the weak test is f(S) = S.
    weak_bias = S_0 * np.abs(
        (1 + interest_rate * dt) ** (end_time / dt)
        - np.exp(interest_rate * end_time)
    )

    ax.loglog(
        dt, weak_bias, "--",
        color="tab:blue",
        alpha=0.65,
        label="Exact weak bias for f(S) = S",
    )

    ax.set(
        title=f"{scheme.capitalize()} convergence for GBM",
        xlabel=r"Time-step size $\Delta t$",
        ylabel="Terminal error",
    )

    ax.grid(which="major", alpha=0.3)
    ax.grid(which="minor", alpha=0.1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()

    return fig, ax

def plot_convergence_matching_comparison(
        results,
        S_0,
        interest_rate,
        end_time,
):
    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 9),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for row, scheme in enumerate(("euler", "milstein")):
        strong_order = 0.5 if scheme == "euler" else 1.0

        # Use the same reference-line height in both columns.
        baseline = results[(scheme, False)]
        anchor = np.argmax(baseline["dt"])
        anchor_dt = baseline["dt"][anchor]
        anchor_error = baseline["error_strong"][anchor]

        for col, matching in enumerate((False, True)):
            ax = axes[row, col]
            errors = results[(scheme, matching)]

            order = np.argsort(errors["dt"])
            dt = np.asarray(errors["dt"])[order]
            weak = np.asarray(errors["error_weak"])[order]
            strong = np.asarray(errors["error_strong"])[order]

            # Zero estimates cannot be displayed on logarithmic axes.
            ax.loglog(
                dt, np.where(weak > 0, weak, np.nan),
                "o-",
                color="tab:blue",
                label="Weak error estimate",
            )
            ax.loglog(
                dt, np.where(strong > 0, strong, np.nan),
                "s-",
                color="tab:orange",
                label="Strong error estimate",
            )

            reference = (
                anchor_error * (dt / anchor_dt) ** strong_order
            )

            ax.loglog(
                dt, reference,
                "--",
                color="gray",
                alpha=0.7,
                zorder=1,
                label=f"Strong reference slope: {strong_order:g}",
            )

            # Theoretical bias for ordinary independent-normal sampling.
            weak_bias = S_0 * np.abs(
                (1 + interest_rate * dt) ** (end_time / dt)
                - np.exp(interest_rate * end_time)
            )

            ax.loglog(
                dt, np.where(weak_bias > 0, weak_bias, np.nan),
                ":",
                color="tab:blue",
                linewidth=2,
                label="Weak bias: ordinary-sampling benchmark",
            )

            matching_label = (
                "WITH moment matching"
                if matching
                else "WITHOUT moment matching"
            )

            # ax.set_title(
            #     f"{scheme.capitalize()} — {matching_label}",
            #     fontweight="bold",
            # )

            ax.set_title(
            f"{scheme.capitalize()}\n{matching_label}",
            fontweight="bold",
)

            # Subtle background distinction between the columns.
            # if matching:
            #     ax.set_facecolor("#f0f7f5")

            ax.grid(which="major", alpha=0.3)
            ax.grid(which="minor", alpha=0.1)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(fontsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel(r"Time-step size $\Delta t$")

    for ax in axes[:, 0]:
        ax.set_ylabel("Terminal error")

    fig.suptitle(
        "GBM convergence: effect of moment matching",
        fontsize=15,
    )

    return fig, axes

def plot_mc_convergence(results):
    results = sorted(results, key=lambda row: row["N"])

    N = [row["N"] for row in results]
    bs_price = results[0]["bs_price"]

    schemes = [
        ("exact", "Exact GBM"),
        ("euler", "Euler"),
        ("milstein", "Milstein"),
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for ax, (scheme, title) in zip(axes, schemes):
        prices = [row[f"{scheme}_price"] for row in results]
        lower = [row[f"{scheme}_ci_lower"] for row in results]
        upper = [row[f"{scheme}_ci_upper"] for row in results]

        ax.plot(N, prices, "o-", label="MC price")
        ax.fill_between(
            N, lower, upper,
            alpha=0.2,
            label="Approximate 95% CI",
        )
        ax.axhline(
            bs_price,
            color="black",
            linestyle="--",
            label="Black–Scholes",
        )

        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("Number of paths N")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Option price")
    axes[0].legend()

    return fig, axes

def plot_mc_standard_error(results):
    results = sorted(results, key=lambda row: row["N"])

    N = np.array([row["N"] for row in results], dtype=float)

    schemes = [
        ("exact", "Exact GBM"),
        ("euler", "Euler"),
        ("milstein", "Milstein"),
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for ax, (scheme, title) in zip(axes, schemes):
        standard_errors = np.array([
            row[f"{scheme}_standard_error"] for row in results
        ])

        # Match the reference line to the SE at the largest N.
        reference = standard_errors[-1] * (N / N[-1]) ** (-0.5)

        ax.loglog(
            N, standard_errors,
            "o-",
            label="Estimated standard error",
        )
        ax.loglog(
            N, reference,
            "--",
            color="black",
            label=r"Reference: $N^{-1/2}$",
        )

        ax.set_title(title)
        ax.set_xlabel("Number of paths N")
        ax.grid(which="major", alpha=0.3)
        ax.grid(which="minor", alpha=0.1)

    axes[0].set_ylabel("Standard error")
    axes[0].legend()

    return fig, axes

def print_mc_convergence_table(results, decimals=4):
    if not results:
        print("No results to display.")
        return

    results = sorted(results, key=lambda row: row["N"])
    width = max(16, decimals + 10)

    print("EUROPEAN OPTION PRICING")
    print(f"Black-Scholes price = {results[0]['bs_price']:.{decimals}f}")
    print("Errors = MC price - Black-Scholes price\n")

    headers = [
        "Exact GBM",
        "Euler",
        "Milstein",
        "Exact error",
        "Euler error",
        "Milstein error",
    ]

    print(f"{'N':>10}" + "".join(f"{h:>{width}}" for h in headers))
    print("-" * (10 + width * len(headers)))

    for row in results:
        values = [
            row["exact_price"],
            row["euler_price"],
            row["milstein_price"],
            row["exact_error"],
            row["euler_error"],
            row["milstein_error"],
        ]

        print(
            f"{row['N']:>10,d}"
            + "".join(
                f"{value:>{width}.{decimals}f}"
                for value in values
            )
        )