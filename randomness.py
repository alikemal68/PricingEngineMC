
from dataclasses import dataclass
import numpy as np


@dataclass
class SamplingResult:
    normal_draws: np.ndarray
    weights: np.ndarray


def gaussian_shift(
        normal_draws,
        shifts,
) -> SamplingResult:
    """Apply a deterministic Gaussian mean shift."""

    Z = np.asarray(normal_draws, dtype=float)

    M = Z.shape[1]

    shifts = np.broadcast_to(
        np.asarray(shifts, dtype=float),
        (M,),
    )

    # Shift Gaussian draws
    shifted_draws = Z + shifts

    # Calculate log likelihood-ratio weights
    log_weights = (
        -Z @ shifts
        - 0.5 * np.sum(shifts**2)
    )

    weights = np.exp(log_weights)

    return SamplingResult(
        normal_draws=shifted_draws,
        weights=weights,
    )


def constant_drift_shift(
        normal_draws,
        times,
        theta,
) -> SamplingResult:
    """Distribute a terminal Gaussian shift over the grid."""

    dt = np.diff(times)
    T = times[-1] - times[0]

    shifts = theta * np.sqrt(dt / T)

    return gaussian_shift(
        normal_draws=normal_draws,
        shifts=shifts,
    )
