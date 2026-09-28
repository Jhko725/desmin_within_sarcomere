import numpy as np
from jaxtyping import Array, Float


AbstractArray = Array | np.ndarray


def powspace(
    start: float, stop: float, num: int, exponent: int | float
) -> Float[np.ndarray, " num"]:
    """Return an array of values whose powers to the exponent are equispaced."""

    powers = np.linspace(start**exponent, stop**exponent, num)
    return powers ** (1 / exponent)


def midpoints(x: Float[AbstractArray, " N"]) -> Float[AbstractArray, " N-1"]:
    """Given an array of values, return an array of midpoints.

    Particularly useful when computing the centers of histogram bins given an array
    of bin edges.
    This function works with all ArrayLike types."""
    return (x[1:] + x[:-1]) / 2


def internal_division_point(x1, x2, y1, y2):
    return (y1 * x2 + y2 * x1) / (y1 + y2)


def find_crossing_points(x, y1, y2):
    y_diff = y1 - y2
    x_exact_crossing = x[np.flatnonzero(y_diff == 0)]

    ind_sign_change = np.flatnonzero(y_diff[:-1] * y_diff[1:] <= 0)

    x_interped_crossing = internal_division_point(
        x[ind_sign_change],
        x[ind_sign_change + 1],
        y_diff[ind_sign_change],
        -y_diff[ind_sign_change + 1],
    )
    x_crossing = np.concatenate((x_exact_crossing, x_interped_crossing), axis=0)
    return np.sort(x_crossing), ind_sign_change