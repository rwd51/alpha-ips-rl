"""Tensor-product hypergrids and finite-group atlas helpers.

The numerical experiments repeatedly evaluate smooth (or piecewise smooth)
quantities on Cartesian products of parameters.  This module keeps the grid
bookkeeping and interpolation independent of any plotting code, provides a
vectorized K=2 finite-group stationary solver for large parameter sweeps, and
locates the derivative kinks of that solution (the extinction surface m=0 and
the clipping thresholds eps=k/G) so interpolation error can be attributed to
the cells that contain them.

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

    def cell_bounds(self, points: Array, bounds_error: bool = True) -> tuple[Array, Array]:
        """Corner coordinates of the cell used to interpolate each point."""
        return cell_bounds(self.axes, points, bounds_error=bounds_error)


def _checked_query(checked: tuple[Array, ...], points: Array, bounds_error: bool) -> tuple[Array, bool]:
    """Validate query points against validated axes; return (n, ndim) points."""
    ndim = len(checked)
    query = np.asarray(points, dtype=np.float64)
    single = query.ndim == 1
    if single:
        query = query[None, :]
    if query.ndim != 2 or query.shape[1] != ndim:
        raise ValueError(f"points must have shape ({ndim},) or (n, {ndim})")
    if not np.all(np.isfinite(query)):
        raise ValueError("points contains a non-finite coordinate")
    if bounds_error:
        for dim, axis in enumerate(checked):
            q = query[:, dim]
            outside = (q < axis[0]) | (q > axis[-1])
            if np.any(outside):
                bad = q[np.flatnonzero(outside)[0]]
                raise ValueError(
                    f"point coordinate {bad:g} lies outside axis {dim} "
                    f"range [{axis[0]:g}, {axis[-1]:g}]"
                )
    return query, single


def _locate(axis: Array, q: Array) -> tuple[Array, Array, Array]:
    """Cell [axis[lo], axis[hi]] holding each in-range coordinate, and the
    fractional position t in it.  An interior node belongs to the cell on its
    right; the last node belongs to the last cell."""
    if len(axis) == 1:
        lo = np.zeros(len(q), dtype=np.int64)
        return lo, lo.copy(), np.zeros(len(q), dtype=np.float64)
    hi = np.searchsorted(axis, q, side="right")
    hi = np.clip(hi, 1, len(axis) - 1)
    lo = hi - 1
    return lo, hi, (q - axis[lo]) / (axis[hi] - axis[lo])


def cell_bounds(
    axes: Sequence[Sequence[float]],
    points: Array,
    *,
    bounds_error: bool = True,
) -> tuple[Array, Array]:
    """Lower and upper corners of the cell that ``multilinear_interpolate``
    uses for each query point.

    Returns two arrays of shape ``(n, ndim)`` (``(ndim,)`` for a single
    point).  With ``bounds_error=False`` coordinates are clipped to the grid,
    exactly as in interpolation.
    """
    checked = tuple(_validated_axis(str(i), axis) for i, axis in enumerate(axes))
    if not checked:
        raise ValueError("at least one axis is required")
    query, single = _checked_query(checked, points, bounds_error)
    lower = np.empty_like(query)
    upper = np.empty_like(query)
    for dim, axis in enumerate(checked):
        lo, hi, _ = _locate(axis, np.clip(query[:, dim], axis[0], axis[-1]))
        lower[:, dim] = axis[lo]
        upper[:, dim] = axis[hi]
    return (lower[0], upper[0]) if single else (lower, upper)


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

    query, single = _checked_query(checked, points, bounds_error)

    lower: list[Array] = []
    upper: list[Array] = []
    fraction: list[Array] = []
    for axis, q in zip(checked, query.T):
        lo, hi, t = _locate(axis, np.clip(q, axis[0], axis[-1]))
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


def clip_kink_epsilons(group_size: int, eps_low: float = 0.0, eps_high: float = 1.0) -> Array:
    """Clipping thresholds ``eps = k/G`` strictly inside ``(eps_low, eps_high)``.

    ``w_G(p) = sum_n Binom(n; G-1, p) max((1+n)/G, eps)^-alpha``, and the n-th
    term changes from a constant to ``eps^-alpha`` at its own ``eps = (1+n)/G``.
    Every ``k/G`` (k = 1, ..., G-1) is therefore a derivative kink of ``w_G``,
    and of the stationary point, as a function of eps -- not only ``1/G``.
    """
    if not isinstance(group_size, (int, np.integer)) or group_size < 2:
        raise ValueError("group_size must be an integer >= 2")
    if not (np.isfinite(eps_low) and np.isfinite(eps_high)) or eps_low >= eps_high:
        raise ValueError("eps_low and eps_high must be finite with eps_low < eps_high")
    kinks = np.arange(1, group_size) / float(group_size)
    return kinks[(kinks > eps_low) & (kinks < eps_high)]


def k2_cell_kinks(lower: Array, upper: Array, group_size: int) -> dict[str, Array]:
    """Which derivative kinks of the K=2 finite-group minority mass a cell contains.

    ``lower`` and ``upper`` are ``(n, 3)`` cell corners in the parameter
    coordinates ``(alpha, reward_ratio, eps)``; exponentiate log-coordinate
    corners first.  The margin m increases with alpha, decreases with the
    reward ratio and does not increase with eps, so over a box its extremes
    sit at two opposite corners.

    Returns boolean arrays:
      ``extinct``     m <= 0 on the whole cell: the minority mass is exactly
                      zero there, so interpolating the cell is exact;
      ``extinction``  the surface m = 0 passes through the cell;
      ``clip``        some eps = k/G lies strictly inside the cell's eps range
                      and the minority survives somewhere in the cell (inside
                      an extinct cell the response is identically zero);
      ``clean``       neither kink passes through the cell.
    """
    lower = np.atleast_2d(np.asarray(lower, dtype=np.float64))
    upper = np.atleast_2d(np.asarray(upper, dtype=np.float64))
    if lower.ndim != 2 or lower.shape[1] != 3 or lower.shape != upper.shape:
        raise ValueError("lower and upper must both have shape (n, 3)")
    if np.any(upper < lower):
        raise ValueError("every upper corner must be >= its lower corner")
    m_max = finite_group_margin(lower[:, 1], group_size, upper[:, 0], lower[:, 2])
    m_min = finite_group_margin(upper[:, 1], group_size, lower[:, 0], upper[:, 2])
    extinct = m_max <= 0.0
    extinction = (m_min < 0.0) & (m_max > 0.0)
    kinks = clip_kink_epsilons(group_size)
    inside = (kinks[None, :] > lower[:, 2:3]) & (kinks[None, :] < upper[:, 2:3])
    clip = np.any(inside, axis=1) & ~extinct
    return {
        "extinct": extinct,
        "extinction": extinction,
        "clip": clip,
        "clean": ~extinction & ~clip,
    }
