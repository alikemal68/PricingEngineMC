
from dataclasses import dataclass
import numpy as np


@dataclass
class SimulationResult:
    times: np.ndarray
    paths: np.ndarray

    @property
    def terminal_prices(self):
        return self.paths[:, -1]

    @property
    def N(self):
        return self.paths.shape[0]

    @property
    def M(self):
        return self.paths.shape[1] - 1


def simulate(
        model,
        S_0,
        times,
        normal_draws,
        scheme,
):
    """Simulate paths of a scalar stochastic process."""

    N, M = normal_draws.shape

    if len(times) != M + 1:
        raise ValueError(
            "The time grid must contain M + 1 time points."
        )

    if np.any(np.diff(times) <= 0):
        raise ValueError(
            "Time points must be strictly increasing."
        )

    # Initialize paths
    paths = np.empty((N, M + 1))
    paths[:, 0] = S_0

    # Loop over time steps
    for i in range(M):

        t = times[i]
        dt = times[i + 1] - times[i]

        # Brownian increments
        dW = np.sqrt(dt) * normal_draws[:, i]

        # Current state
        S_current = paths[:, i]

        # Numerical step
        paths[:, i + 1] = scheme(
            model, t, S_current, dt, dW
        )

    return SimulationResult(
        times=times,
        paths=paths,
    )
