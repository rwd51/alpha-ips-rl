"""Tensor-product hypergrids and finite-group atlas helpers.

The numerical experiments repeatedly evaluate smooth (or piecewise smooth)
quantities on Cartesian products of parameters.  This module keeps the grid
bookkeeping and interpolation independent of any plotting code, and provides
a vectorized K=2 finite-group stationary solver for large parameter sweeps.

The interpolation coordinates are deliberately supplied by the caller.  For
positive parameters spanning decades, callers should normally construct a
grid in log-parameter coordinates and transform back only when evaluating the
model.  Multilinear interpolation then operates in those log coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Mapping, Sequence

import numpy as np

from .finite_group_general import effective_weight


Array = np.ndarray


def _validated_axis(name: str, values: Sequence[float]) -> Array:
    axis = np.asarray(values, dtype=np.float64)
    if axis.ndim != 1 or axis.size == 0:
        raise ValueError(f"axis {name!r} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"axis {name!r} contains a non-finite value")
    if axis.size > 1 and not np.all(np.diff(axis) > 0.0):
        raise ValueError(f"axis {name!r} must be strictly increasing")
    axis = axis.copy()
    axis.setflags(write=False)
    return axis


@dataclass(frozen=True)
class TensorGrid:
    """A named, validated tensor-product grid.

    Parameters
    ----------
    axes
        Ordered mapping from axis name to strictly increasing coordinates.
        Python dictionaries preserve insertion order, which determines the
        dimension order of arrays stored on the grid.
    """

    names: tuple[str, ...]
    axes: tuple[Array, ...]

    def __init__(self, axes: Mapping[str, Sequence[float]]):
        if not isinstance(axes, Mapping) or not axes:
            raise ValueError("axes must be a non-empty ordered mapping")
        names = tuple(axes.keys())
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("every axis name must be a non-empty string")
        if len(set(names)) != len(names):
            raise ValueError("axis names must be unique")
        arrays = tuple(_validated_axis(name, axes[name]) for name in names)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "axes", arrays)

    @property
    def ndim(self) -> int:
        return len(self.axes)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(axis) for axis in self.axes)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    def points(self, flat: bool = True) -> Array:
        """Return Cartesian points in array-index order.

        With ``flat=True`` the shape is ``(size, ndim)``.  Otherwise it is
        ``shape + (ndim,)``.
        """
        mesh = np.meshgrid(*self.axes, indexing="ij")
        out = np.stack(mesh, axis=-1)
        return out.reshape(-1, self.ndim) if flat else out

    def evaluate(self, function: Callable[..., object], dtype=np.float64) -> Array:
        """Evaluate ``function(*coordinates)`` at every grid point.

        The function may return a scalar or a fixed-shape array.  Evaluation
        is intentionally explicit: it works for numerical solvers that are
        not vectorized and gives deterministic row-major call order.
        """
        first_index = tuple(0 for _ in self.axes)
        first = np.asarray(function(*(axis[0] for axis in self.axes)), dtype=dtype)
        out = np.empty(self.shape + first.shape, dtype=first.dtype)
        out[first_index] = first
        for index in np.ndindex(self.shape):
            if index == first_index:
                continue
            value = function(*(axis[i] for axis, i in zip(self.axes, index)))
            value = np.asarray(value, dtype=out.dtype)
            if value.shape != first.shape:
                raise ValueError(
                    f"function returned shape {value.shape} at index {index}; "
                    f"expected {first.shape}"
                )
            out[index] = value
        return out

    def interpolate(self, values: Array, points: Array, bounds_error: bool = True) -> Array:
        """Multilinearly interpolate values at one or more query points."""
        return multilinear_interpolate(self.axes, values, points, bounds_error=bounds_error)


def multilinear_interpolate(
    axes: Sequence[Sequence[float]],
    values: Array,
    points: Array,
    *,
    bounds_error: bool = True,
) -> Array:
    """Interpolate a scalar- or vector-valued tensor grid.

    ``values`` must start with the tensor shape; any trailing dimensions are
    treated as output dimensions.  ``points`` has shape ``(n, ndim)`` or
    ``(ndim,)``.  A single point returns one output value, while a batch
    returns a leading query dimension.

    When ``bounds_error=False``, out-of-range coordinates are clipped to the
    closest grid boundary; this function never silently extrapolates.
    """
    checked = tuple(_validated_axis(str(i), axis) for i, axis in enumerate(axes))
    ndim = len(checked)
    if ndim == 0:
        raise ValueError("at least one axis is required")
    if ndim > 16:
        raise ValueError("multilinear interpolation is limited to 16 dimensions")

    data = np.asarray(values)
    expected = tuple(len(axis) for axis in checked)
    if data.shape[:ndim] != expected:
        raise ValueError(f"values starts with shape {data.shape[:ndim]}, expected {expected}")

    query = np.asarray(points, dtype=np.float64)
    single = query.ndim == 1
    if single:
        query = query[None, :]
    if query.ndim != 2 or query.shape[1] != ndim:
        raise ValueError(f"points must have shape ({ndim},) or (n, {ndim})")
    if not np.all(np.isfinite(query)):
        raise ValueError("points contains a non-finite coordinate")

    lower: list[Array] = []
    upper: list[Array] = []
    fraction: list[Array] = []
    for dim, axis in enumerate(checked):
        q = query[:, dim]
        outside = (q < axis[0]) | (q > axis[-1])
        if bounds_error and np.any(outside):
            bad = q[np.flatnonzero(outside)[0]]
            raise ValueError(
                f"point coordinate {bad:g} lies outside axis {dim} "
                f"range [{axis[0]:g}, {axis[-1]:g}]"
            )
        q = np.clip(q, axis[0], axis[-1])
        if len(axis) == 1:
            lo = np.zeros(len(q), dtype=np.int64)
            hi = lo.copy()
            t = np.zeros(len(q), dtype=np.float64)
        else:
            hi = np.searchsorted(axis, q, side="right")
            hi = np.clip(hi, 1, len(axis) - 1)
            lo = hi - 1
            t = (q - axis[lo]) / (axis[hi] - axis[lo])
        lower.append(lo)
        upper.append(hi)
        fraction.append(t)

    output_shape = data.shape[ndim:]
    result_dtype = np.result_type(data.dtype, np.float64)
    result = np.zeros((len(query),) + output_shape, dtype=result_dtype)
    weight_shape = (len(query),) + (1,) * len(output_shape)

    # Singleton dimensions have identical lower/upper indices.  Fix their
    # corner bit at zero to avoid adding the same value twice.
    corner_choices = [(0,) if len(axis) == 1 else (0, 1) for axis in checked]
    for corner in product(*corner_choices):
        indices = []
        weight = np.ones(len(query), dtype=np.float64)
        for dim, bit in enumerate(corner):
            if bit:
                indices.append(upper[dim])
                weight *= fraction[dim]
            else:
                indices.append(lower[dim])
                weight *= 1.0 - fraction[dim]
        result += data[tuple(indices)] * weight.reshape(weight_shape)

    return result[0] if single else result


def finite_group_margin(reward_ratio: Array, group_size: int, alpha: Array, eps: Array) -> Array:
    """Signed K=2 survival margin.

    Positive values mean that the lower-reward outcome has enough maximum
    inverse-probability weight to invade the collapsed boundary.  Zero is the
    exact finite-group phase boundary.
    """
    ratio = np.asarray(reward_ratio, dtype=np.float64)
    if not np.all(np.isfinite(ratio)) or np.any(ratio < 1.0):
        raise ValueError("reward_ratio must contain finite values >= 1")
    if not isinstance(group_size, (int, np.integer)) or group_size < 2:
        raise ValueError("group_size must be an integer >= 2")
    alpha_array = np.asarray(alpha, dtype=np.float64)
    eps_array = np.asarray(eps, dtype=np.float64)
    if not np.all(np.isfinite(alpha_array)) or np.any(alpha_array <= 0.0):
        raise ValueError("alpha must contain finite values > 0")
    if (not np.all(np.isfinite(eps_array)) or np.any(eps_array <= 0.0)
            or np.any(eps_array > 1.0)):
        raise ValueError("eps must contain finite values in (0, 1]")
    try:
        ratio, alpha_array, eps_array = np.broadcast_arrays(ratio, alpha_array, eps_array)
    except ValueError as exc:
        raise ValueError("reward_ratio, alpha, and eps must be broadcast-compatible") from exc
    ceiling_base = np.minimum(float(group_size), 1.0 / eps_array)
    return alpha_array * np.log(ceiling_base) - np.log(ratio)


def stationary_k2_batch(
    reward_ratio: Array,
    group_size: int,
    alpha: float,
    eps: float = 1e-3,
    *,
    iterations: int = 64,
) -> Array:
    """Vectorized exact finite-group K=2 majority probability.

    Rewards are represented by their ratio ``r1/r2 >= 1``.  Interior roots
    solve ``ratio*w_G(p1) = w_G(1-p1)`` on ``[1/2, 1]``.  When the finite
    correction ceiling cannot overcome the reward ratio, the stationary
    point is the collapsed boundary ``p1=1``.

    A fixed number of bisection iterations makes the result deterministic and
    gives an absolute probability error below ``2**(-iterations-1)``.
    """
    ratio = np.asarray(reward_ratio, dtype=np.float64)
    original_shape = ratio.shape
    flat = ratio.reshape(-1)
    if flat.size == 0:
        raise ValueError("reward_ratio must contain at least one value")
    alpha_array = np.asarray(alpha, dtype=np.float64)
    eps_array = np.asarray(eps, dtype=np.float64)
    if alpha_array.ndim != 0 or eps_array.ndim != 0:
        raise ValueError("stationary_k2_batch requires scalar alpha and eps")
    alpha_value = float(alpha_array)
    eps_value = float(eps_array)
    # Centralized validation, including the scalar hyperparameters.
    margin = finite_group_margin(flat, group_size, alpha_value, eps_value)
    if not isinstance(iterations, (int, np.integer)) or iterations < 1:
        raise ValueError("iterations must be a positive integer")

    result = np.ones_like(flat)
    tied = flat == 1.0
    result[tied] = 0.5
    active = (margin > 0.0) & ~tied
    if np.any(active):
        rho = flat[active]
        lo = np.full(len(rho), 0.5, dtype=np.float64)
        hi = np.ones(len(rho), dtype=np.float64)
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            balance = (
                rho * effective_weight(mid, group_size, alpha_value, eps_value)
                - effective_weight(1.0 - mid, group_size, alpha_value, eps_value)
            )
            lo = np.where(balance > 0.0, mid, lo)
            hi = np.where(balance > 0.0, hi, mid)
        result[active] = 0.5 * (lo + hi)
    return result.reshape(original_shape)
