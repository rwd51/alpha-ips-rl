"""Experiment 4 unit and numerical-consistency tests."""

from __future__ import annotations

import unittest
from math import comb

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from experiments.exp4_hypergrid import _normalized_entropy, _scalar_solver_check, _solve_general_checked
from src.finite_group import meanfield_stationary_K2
from src.finite_group_general import effective_weight
from src.hypergrid import (TensorGrid, clip_kink_epsilons, finite_group_margin, k2_cell_kinks,
                           multilinear_interpolate, stationary_k2_batch)


class TensorGridTests(unittest.TestCase):
    def test_grid_points_follow_array_index_order(self):
        grid = TensorGrid({"x": [0.0, 1.0], "y": [10.0, 20.0, 30.0]})
        expected = np.array([
            [0.0, 10.0], [0.0, 20.0], [0.0, 30.0],
            [1.0, 10.0], [1.0, 20.0], [1.0, 30.0],
        ])
        np.testing.assert_array_equal(grid.points(), expected)
        self.assertEqual(grid.shape, (2, 3))
        self.assertEqual(grid.size, 6)

    def test_evaluate_supports_vector_outputs(self):
        grid = TensorGrid({"x": [-1.0, 2.0], "y": [0.0, 3.0]})
        values = grid.evaluate(lambda x, y: np.array([x + y, x - y]))
        self.assertEqual(values.shape, (2, 2, 2))
        np.testing.assert_allclose(values[1, 1], [5.0, -1.0])

    def test_multilinear_interpolation_is_exact_for_multiaffine_data(self):
        grid = TensorGrid({"x": [-2.0, 0.0, 3.0], "y": [1.0, 4.0], "z": [7.0]})

        def function(x, y, z):
            return 1.5 + 2.0 * x - 0.25 * y + 0.5 * x * y + z

        values = grid.evaluate(function)
        points = np.array([[-1.2, 2.7, 7.0], [1.3, 3.5, 7.0], [3.0, 1.0, 7.0]])
        expected = np.array([function(*point) for point in points])
        np.testing.assert_allclose(grid.interpolate(values, points), expected, rtol=0.0, atol=2e-14)

    def test_single_query_and_vector_valued_interpolation(self):
        axes = ([0.0, 1.0], [0.0, 2.0])
        scalar = np.array([[0.0, 2.0], [1.0, 3.0]])
        vector = np.stack([scalar, 2.0 * scalar], axis=-1)
        self.assertAlmostEqual(float(multilinear_interpolate(axes, scalar, [0.25, 0.5])), 0.75)
        np.testing.assert_allclose(multilinear_interpolate(axes, vector, [0.25, 0.5]), [0.75, 1.5])

    def test_bounds_policy_never_extrapolates(self):
        grid = TensorGrid({"x": [0.0, 1.0]})
        values = np.array([2.0, 4.0])
        with self.assertRaises(ValueError):
            grid.interpolate(values, [-0.1])
        self.assertEqual(float(grid.interpolate(values, [-0.1], bounds_error=False)), 2.0)
        self.assertEqual(float(grid.interpolate(values, [1.1], bounds_error=False)), 4.0)

    def test_invalid_axes_and_shapes_are_rejected(self):
        for bad in ([], [0.0, 0.0], [1.0, 0.0], [0.0, np.nan]):
            with self.assertRaises(ValueError):
                TensorGrid({"bad": bad})
        grid = TensorGrid({"x": [0.0, 1.0], "y": [0.0, 1.0]})
        with self.assertRaises(ValueError):
            grid.interpolate(np.zeros((2, 3)), [0.5, 0.5])
        with self.assertRaises(ValueError):
            grid.evaluate(lambda x, y: np.zeros(1) if x == 0.0 else np.zeros(2))

    def test_matches_scipy_in_one_through_five_dimensions(self):
        rng = np.random.default_rng(4404)
        for ndim in range(1, 6):
            axes = tuple(np.cumsum(rng.uniform(0.1, 1.0, size=n)) for n in range(2, 2 + ndim))
            values = rng.normal(size=tuple(map(len, axes)) + (3,))
            points = np.column_stack([rng.uniform(axis[0], axis[-1], size=100) for axis in axes])
            grid = TensorGrid({f"x{i}": axis for i, axis in enumerate(axes)})
            expected = RegularGridInterpolator(axes, values, method="linear")(points)
            np.testing.assert_allclose(grid.interpolate(values, points), expected, rtol=0.0, atol=2e-15)
            scalar = rng.normal(size=tuple(map(len, axes)))
            expected = RegularGridInterpolator(axes, scalar, method="linear")(points)
            np.testing.assert_allclose(grid.interpolate(scalar, points), expected, rtol=0.0, atol=2e-15)

    def test_cell_bounds_follow_the_interpolation_cells(self):
        grid = TensorGrid({"x": [0.0, 1.0, 3.0], "y": [5.0]})
        # An interior node belongs to the cell on its right, the last node to the last cell.
        lower, upper = grid.cell_bounds(np.array([[0.5, 5.0], [1.0, 5.0], [3.0, 5.0], [0.0, 5.0]]))
        np.testing.assert_array_equal(lower, [[0.0, 5.0], [1.0, 5.0], [1.0, 5.0], [0.0, 5.0]])
        np.testing.assert_array_equal(upper, [[1.0, 5.0], [3.0, 5.0], [3.0, 5.0], [1.0, 5.0]])
        single_lower, single_upper = grid.cell_bounds([2.0, 5.0])
        np.testing.assert_array_equal(single_lower, [1.0, 5.0])
        np.testing.assert_array_equal(single_upper, [3.0, 5.0])
        with self.assertRaises(ValueError):
            grid.cell_bounds([3.5, 5.0])
        clipped = grid.cell_bounds([3.5, 5.0], bounds_error=False)
        np.testing.assert_array_equal(clipped[0], [1.0, 5.0])
        np.testing.assert_array_equal(clipped[1], [3.0, 5.0])


class FiniteGroupAtlasTests(unittest.TestCase):
    def test_margin_broadcast_and_exact_boundary(self):
        rho = np.array([2.0, 4.0, 8.0])
        alpha = np.log(rho) / np.log(16.0)
        margin = finite_group_margin(rho, 16, alpha, 1e-3)
        np.testing.assert_allclose(margin, 0.0, atol=5e-16)
        clipped = finite_group_margin(4.0, 64, 1.0, np.array([1e-3, 0.25]))
        np.testing.assert_allclose(clipped, [np.log(16.0), 0.0], atol=5e-15)

    def test_k2_solver_handles_ties_extinction_and_interior(self):
        values = stationary_k2_batch(np.array([1.0, 4.0, 4.0]), 16,
                                     np.array(1.0), np.array(1e-3))
        self.assertEqual(values[0], 0.5)
        self.assertLess(values[1], 1.0)
        # epsilon=0.25 caps the weight at 4, exactly the collapse boundary.
        self.assertEqual(float(stationary_k2_batch(np.array([4.0]), 16, 1.0, 0.25)[0]), 1.0)

    def test_batch_solver_matches_original_scalar_solver(self):
        ratios = np.array([1.2, 2.0, 4.0, 10.0])
        for G in (2, 4, 16):
            for alpha in (0.4, 1.0, 2.0):
                for eps in (1e-3, 0.2):
                    batch = stationary_k2_batch(ratios, G, alpha, eps)
                    scalar = np.array([
                        meanfield_stationary_K2(np.array([rho, 1.0]), G, alpha, eps)
                        for rho in ratios
                    ])
                    np.testing.assert_allclose(batch, scalar, rtol=0.0, atol=2e-10)

    def test_k2_stationarity_balance(self):
        ratios = np.array([1.1, 1.7, 4.0, 12.0])
        p1 = stationary_k2_batch(ratios, 32, 1.3, 0.01)
        balance = ratios * self._weight(p1, 32, 1.3, 0.01) - self._weight(1.0 - p1, 32, 1.3, 0.01)
        active = finite_group_margin(ratios, 32, 1.3, 0.01) > 0.0
        np.testing.assert_allclose(balance[active], 0.0, atol=5e-11)

    @staticmethod
    def _weight(p, G, alpha, eps):
        from src.finite_group_general import effective_weight
        return effective_weight(p, G, alpha, eps)

    def test_general_solver_is_reward_scale_invariant_in_exp4_wrapper(self):
        rewards = np.array([2.0, 1.0, 0.3])
        p1, _, _, sum1, residual1, _ = _solve_general_checked(rewards, 4, 1.0)
        p2, _, _, sum2, residual2, _ = _solve_general_checked(1e-20 * rewards, 4, 1.0)
        np.testing.assert_allclose(p1, p2, rtol=0.0, atol=2e-12)
        self.assertLess(max(sum1, sum2), 2e-12)
        self.assertLess(max(residual1, residual2), 2e-10)

    def test_public_parameter_validation(self):
        with self.assertRaises(ValueError):
            stationary_k2_batch(np.array([]), 4, 1.0)
        with self.assertRaises(ValueError):
            stationary_k2_batch(np.array([2.0, 4.0]), 4, np.array([1.0, 2.0]))
        for ratio in (0.9, np.nan):
            with self.assertRaises(ValueError):
                stationary_k2_batch(np.array([ratio]), 4, 1.0)
        for G in (0, 1, 2.5):
            with self.assertRaises(ValueError):
                stationary_k2_batch(np.array([2.0]), G, 1.0)
        for alpha in (0.0, -1.0, np.inf):
            with self.assertRaises(ValueError):
                stationary_k2_batch(np.array([2.0]), 4, alpha)
        for eps in (0.0, -0.1, 1.1, np.nan):
            with self.assertRaises(ValueError):
                stationary_k2_batch(np.array([2.0]), 4, 1.0, eps)

    def test_weight_has_a_slope_jump_at_every_clip_threshold(self):
        # d w_G / d log(eps) jumps by -alpha (k/G)^-alpha Pr[B = k-1] at eps = k/G,
        # B ~ Binomial(G-1, p): every k/G is a kink, not only eps = 1/G.
        G, alpha, p, h = 8, 1.3, 0.3, 1e-7

        def w(eps):
            return effective_weight(p, G, alpha, eps)[0]

        def slope_jump(eps):
            left = (w(eps) - w(eps * np.exp(-h))) / h
            right = (w(eps * np.exp(h)) - w(eps)) / h
            return right - left

        for k in range(1, 5):
            probability = comb(G - 1, k - 1) * p ** (k - 1) * (1.0 - p) ** (G - k)
            expected = -alpha * (k / G) ** (-alpha) * probability
            self.assertAlmostEqual(slope_jump(k / G), expected, delta=1e-4)
        self.assertLess(abs(slope_jump(1.5 / G)), 1e-4)

    def test_stationary_minority_mass_inherits_the_clip_kinks(self):
        def minority(eps):
            return 1.0 - stationary_k2_batch(np.array([2.0]), 16, 1.0, eps)[0]

        def slope_jump(eps, h=1e-6):
            left = (minority(eps) - minority(eps * np.exp(-h))) / h
            right = (minority(eps * np.exp(h)) - minority(eps)) / h
            return right - left

        self.assertLess(slope_jump(2.0 / 16), -5e-3)     # a kink at 2/G, stronger than ...
        self.assertLess(slope_jump(1.0 / 16), -1e-3)     # ... the one at 1/G
        self.assertLess(abs(slope_jump(0.09)), 1e-6)     # and none between thresholds

    def test_clip_kink_epsilons(self):
        np.testing.assert_allclose(clip_kink_epsilons(16, 1e-3, 0.4), np.arange(1, 7) / 16)
        np.testing.assert_allclose(clip_kink_epsilons(4), [0.25, 0.5, 0.75])
        self.assertEqual(len(clip_kink_epsilons(16, 0.125, 0.1875)), 0)    # endpoints excluded
        for bad in ((1, 0.0, 1.0), (4, 0.5, 0.5), (4, np.nan, 1.0)):
            with self.assertRaises(ValueError):
                clip_kink_epsilons(*bad)

    def test_k2_cell_kinks_on_hand_built_cells(self):
        # (alpha, reward ratio, eps) corners at G=16.
        lower = np.array([
            [0.20, 1.8, 1e-3],     # m changes sign inside
            [1.00, 1.1, 0.10],     # contains eps = 2/16, minority alive throughout
            [0.10, 15.0, 0.10],    # contains eps = 2/16 but extinct throughout
            [1.00, 1.1, 1e-3],     # smooth and alive
            [1.00, 1.1, 0.125],    # eps = 2/16 exactly on the cell boundary
        ])
        upper = np.array([
            [0.30, 2.0, 2e-3],
            [1.10, 1.2, 0.14],
            [0.12, 16.0, 0.14],
            [1.10, 1.2, 2e-3],
            [1.10, 1.2, 0.14],
        ])
        flags = k2_cell_kinks(lower, upper, 16)
        np.testing.assert_array_equal(flags["extinction"], [True, False, False, False, False])
        np.testing.assert_array_equal(flags["clip"], [False, True, False, False, False])
        np.testing.assert_array_equal(flags["extinct"], [False, False, True, False, False])
        np.testing.assert_array_equal(flags["clean"], [False, False, True, True, True])
        with self.assertRaises(ValueError):
            k2_cell_kinks(lower[:, :2], upper[:, :2], 16)
        with self.assertRaises(ValueError):
            k2_cell_kinks(upper, lower, 16)

    def test_scalar_solver_check_is_independent_of_the_margin(self):
        alphas, group_sizes = np.array([0.5, 2.0]), np.array([4])
        epsilons, ratios = np.array([1e-3]), np.array([2.0, 16.0])
        p1 = np.empty((2, 1, 1, 2))
        margin = np.empty_like(p1)
        for ai, alpha in enumerate(alphas):
            p1[ai, 0, 0] = stationary_k2_batch(ratios, 4, float(alpha), 1e-3)
            margin[ai, 0, 0] = finite_group_margin(ratios, 4, float(alpha), 1e-3)
        _, check = _scalar_solver_check(alphas, group_sizes, epsilons, ratios, p1, margin)
        # 0.5 ln 4 = ln 2 and 2 ln 4 = ln 16 exactly: two points sit on m = 0.
        self.assertEqual(check["points_on_boundary"], 2)
        self.assertEqual(check["points_on_boundary_collapsed"], 2)
        self.assertEqual(check["classification_points_compared"], 2)
        self.assertEqual(check["boundary_classification_mismatches"], 0)
        self.assertLess(check["max_abs_batch_vs_scalar_solver"], 2e-10)
        # The check does not trust its inputs: a wrong boundary and a wrong solver are both caught.
        _, bad = _scalar_solver_check(alphas, group_sizes, epsilons, ratios, np.ones_like(p1), -np.abs(margin))
        self.assertEqual(bad["boundary_classification_mismatches"], 1)
        self.assertGreater(bad["max_abs_batch_vs_scalar_solver"], 0.1)

    def test_normalized_entropy_of_a_point_mass_is_positive_zero(self):
        value = _normalized_entropy([1.0, 0.0, 0.0])
        self.assertEqual(value, 0.0)
        self.assertFalse(np.signbit(value))


if __name__ == "__main__":
    unittest.main()
