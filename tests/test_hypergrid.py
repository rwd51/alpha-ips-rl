"""Experiment 4 unit and numerical-consistency tests."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from experiments.exp4_hypergrid import _solve_general_checked
from src.finite_group import meanfield_stationary_K2
from src.hypergrid import TensorGrid, finite_group_margin, multilinear_interpolate, stationary_k2_batch


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


if __name__ == "__main__":
    unittest.main()
