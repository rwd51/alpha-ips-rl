"""Experiment 5 unit and numerical-consistency tests.

Run from the repository root:

    python -m unittest tests.test_estimators -v
"""

from __future__ import annotations

import unittest

import numpy as np
from scipy.stats import binom

from src.dynamics import linear_rates, rhs_sampled, stationary_p
from src.eigen import (deflated_power_method, gram_schmidt_qr, power_method,
                       qr_algorithm)
from src.ema import (coupled_jacobian, coupled_rhs, critical_kappa, decay_rate,
                     effective_group_size, is_oscillatory, lag_bias, mode_roots,
                     variance_factor)
from src.estimators import (AddLambdaRule, ClipRule, IdealRule, OffsetRule,
                            RichardsonRule, delta_relative_bias,
                            delta_relative_sd, effective_weight_alpha1,
                            empty_group_bias, sampled_step)
from src.finite_group import meanfield_stationary_K2
from src.finite_group_general import effective_weight as clip_effective_weight
from src.finite_group_general import meanfield_stationary
from src.fitting import least_squares_line
from src.meanfield_rule import (check_monotone, extinction_alpha_K2, stationary,
                                stationary_K2)
from src.optimize1d import (golden_section, golden_section_log, grid_minimum,
                            is_unimodal)


R5 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
R41 = np.array([4.0, 1.0])


def _phi_over_p_count(rule, p, G):
    """Direct definition phi_G(p)/p = E[p_hat omega(n)] / p, for a count rule."""
    n = np.arange(G + 1)
    v = rule.values(G)
    return np.array([float((binom.pmf(n, G, q) * (n / G) * v).sum() / q) for q in p])


def _phi_over_p_split(rule, p, G):
    """The same for a split rule, summing over both half-counts."""
    M = G // 2
    a = np.arange(M + 1)
    W = rule.values(G)
    out = []
    for q in p:
        P = binom.pmf(a, M, q)
        out.append(float((P[:, None] * P[None, :] * ((a[:, None] + a[None, :]) / G) * W).sum() / q))
    return np.array(out)


class WeightRuleTests(unittest.TestCase):
    def test_clip_rule_values_and_paper_form(self):
        rule = ClipRule(1.3, 0.05)
        G = 12
        expected = np.maximum(np.arange(G + 1) / G, 0.05) ** (-1.3)
        np.testing.assert_allclose(rule.values(G), expected, rtol=1e-14)

    def test_moments_match_a_brute_force_binomial_sum(self):
        for rule, G in [(ClipRule(1.0, 1e-3), 16), (AddLambdaRule(0.7, 1.0, 4), 10),
                        (OffsetRule(2.0, 1e-3), 14)]:
            p = np.linspace(0.05, 0.95, 13)
            n = np.arange(G + 1)
            v = rule.values(G)
            for i, q in enumerate(p):
                P = binom.pmf(n, G, q)
                mean = float((P * v).sum())
                var = float((P * (v - mean) ** 2).sum())
                m1r, varr = rule.moments_relative(np.array([q]), G)
                self.assertAlmostEqual(float(m1r[0]), mean * q ** rule.alpha, delta=1e-10)
                self.assertAlmostEqual(float(varr[0]), var * q ** (2 * rule.alpha),
                                       delta=1e-10 * max(1.0, var * q ** (2 * rule.alpha)))

    def test_absolute_and_relative_moments_agree(self):
        rule = ClipRule(1.5, 1e-2)
        p = np.linspace(0.05, 0.95, 9)
        m1, var = rule.moments(p, 16)
        m1r, varr = rule.moments_relative(p, 16)
        np.testing.assert_allclose(m1r, m1 * p ** rule.alpha, rtol=1e-12)
        np.testing.assert_allclose(varr, var * p ** (2 * rule.alpha), rtol=1e-12)

    def test_second_central_moment_equals_the_variance(self):
        for rule, G in [(ClipRule(1.0, 1e-3), 16), (RichardsonRule(1.0, 1e-3, 1), 16)]:
            p = np.linspace(0.1, 0.9, 7)
            _, var = rule.moments_relative(p, G)
            np.testing.assert_allclose(rule.relative_central_moment(p, G, 2), var,
                                       rtol=1e-10, atol=1e-14)

    def test_fourth_central_moment_is_positive_and_bounds_the_variance(self):
        rule = ClipRule(1.0, 1e-3)
        p = np.linspace(0.1, 0.9, 7)
        _, var = rule.moments_relative(p, 16)
        mu4 = rule.relative_central_moment(p, 16, 4)
        self.assertTrue(np.all(mu4 > 0.0))
        self.assertTrue(np.all(mu4 >= var ** 2 - 1e-12))   # kurtosis >= 1

    def test_size_biasing_identity_for_count_rules(self):
        p = np.linspace(0.05, 0.95, 11)
        for rule, G in [(ClipRule(1.0, 1e-3), 16), (AddLambdaRule(1.7, 0.5, 3), 9),
                        (OffsetRule(0.6, 1e-4), 12)]:
            np.testing.assert_allclose(rule.effective_weight(p, G),
                                       _phi_over_p_count(rule, p, G), rtol=1e-12)

    def test_size_biasing_identity_for_split_rules(self):
        p = np.linspace(0.05, 0.95, 11)
        for guard in (None, 1, 2):
            for G in (8, 16):
                rule = RichardsonRule(1.0, 1e-3, guard=guard)
                np.testing.assert_allclose(rule.effective_weight(p, G),
                                           _phi_over_p_split(rule, p, G), rtol=1e-11)

    def test_effective_weight_matches_experiment_3_implementation(self):
        p = np.linspace(0.0, 1.0, 101)
        for alpha in (0.5, 1.0, 1.3, 2.0):
            for G, eps in [(4, 1e-3), (16, 1e-3), (16, 0.2), (64, 1e-2)]:
                np.testing.assert_allclose(
                    ClipRule(alpha, eps).effective_weight(p, G),
                    clip_effective_weight(p, G, alpha, eps), rtol=0.0, atol=1e-11)

    def test_alpha_one_closed_form(self):
        p = np.linspace(1e-6, 1.0, 501)
        for G in (2, 3, 8, 16, 64, 256):
            np.testing.assert_allclose(ClipRule(1.0, 1.0 / G).effective_weight(p, G),
                                       effective_weight_alpha1(p, G), rtol=2e-10)
        # endpoints: w(0) = G, w(1) = 1
        ends = effective_weight_alpha1(np.array([0.0, 1.0]), 32)
        self.assertAlmostEqual(float(ends[0]), 32.0, places=9)
        self.assertAlmostEqual(float(ends[1]), 1.0, places=12)

    def test_dynamic_ranges_match_their_closed_forms(self):
        for G in (8, 16, 32):
            for alpha in (0.5, 1.0, 2.0):
                self.assertAlmostEqual(ClipRule(alpha, 1e-3).dynamic_range(G),
                                       min(G, 1e3) ** alpha, delta=1e-9 * G ** alpha)
                # a clip above 1/G caps the range at (1/eps)^alpha
                self.assertAlmostEqual(ClipRule(alpha, 0.25).dynamic_range(G),
                                       min(G, 4.0) ** alpha, delta=1e-9 * G ** alpha)
                for lam in (0.5, 1.0, 2.0):
                    for K in (2, 5):
                        self.assertAlmostEqual(
                            AddLambdaRule(alpha, lam, K).dynamic_range(G),
                            ((G + lam) / (1.0 + lam)) ** alpha,
                            delta=1e-9 * G ** alpha)

    def test_add_lambda_always_shrinks_the_dynamic_range(self):
        for G in (4, 16, 64):
            base = ClipRule(1.0, 1e-3).dynamic_range(G)
            for lam in (0.1, 0.5, 1.0, 5.0):
                self.assertLess(AddLambdaRule(1.0, lam, 2).dynamic_range(G), base)

    def test_ideal_rule_is_exact(self):
        rule = IdealRule(1.4)
        p = np.linspace(0.05, 0.95, 7)
        t = rule.error_table(p, 16)
        np.testing.assert_allclose(t["rel_bias"], 0.0, atol=0.0)
        np.testing.assert_allclose(t["rel_sd"], 0.0, atol=0.0)
        np.testing.assert_allclose(rule.effective_weight(p, 16), p ** -1.4, rtol=1e-14)
        self.assertEqual(rule.dynamic_range(16), np.inf)

    def test_parameter_validation(self):
        for eps in (0.0, -1.0, 1.5):
            with self.assertRaises(ValueError):
                ClipRule(1.0, eps)
        for alpha in (0.0, -1.0, np.inf):
            with self.assertRaises(ValueError):
                ClipRule(alpha, 1e-3)
        with self.assertRaises(ValueError):
            AddLambdaRule(1.0, 0.0, 2)
        with self.assertRaises(ValueError):
            AddLambdaRule(1.0, 1.0, 1)
        with self.assertRaises(ValueError):
            RichardsonRule(1.0, 1e-3, guard=0)
        with self.assertRaises(ValueError):
            RichardsonRule(1.0, 1e-3).values(15)          # odd group size
        for G in (1, 2.5):
            with self.assertRaises(ValueError):
                ClipRule(1.0, 1e-3).values(G)
        for bad in (-0.1, 1.1, np.nan):
            with self.assertRaises(ValueError):
                ClipRule(1.0, 1e-3).moments_relative(np.array([bad]), 8)


class RichardsonTests(unittest.TestCase):
    def test_naive_rule_gives_a_negative_singleton_weight(self):
        for G in (8, 16, 64, 128):
            naive = RichardsonRule(1.0, 1e-3, guard=None)
            self.assertLess(naive.values(G)[1, 0], 0.0)
            self.assertLess(naive.dynamic_range(G), 0.0)

    def test_guard_removes_every_negative_weight_and_bounds_the_rule(self):
        for alpha in (0.5, 1.0, 2.0):
            for eps in (1e-4, 1e-3, 0.1):
                for G in (4, 8, 16, 32):
                    rule = RichardsonRule(alpha, eps, guard=1)
                    W = rule.values(G)
                    M = G // 2
                    a = np.arange(M + 1)[:, None]
                    b = np.arange(M + 1)[None, :]
                    plain = np.maximum((a + b) / G, eps) ** (-alpha)
                    self.assertTrue(np.all(W > 0.0))
                    self.assertTrue(np.all(W <= 2.0 * plain + 1e-9))
                    p = np.geomspace(1e-4, 0.999, 60)
                    np.testing.assert_allclose(
                        rule.negative_weight_probability(p, G), 0.0, atol=0.0)

    def test_min_only_guard_would_not_have_been_enough(self):
        """The documented failure the positivity check exists to catch: with
        alpha = 1 the raw rule is negative whenever 8ab < (a+b)^2."""
        G, M = 32, 16
        a, b = 1, 12                                   # b/a = 12 > 3 + 2 sqrt 2
        self.assertLess(8 * a * b, (a + b) ** 2)
        f = lambda x: max(x, 1e-3) ** -1.0
        raw = 2 * f((a + b) / G) - 0.5 * (f(a / M) + f(b / M))
        self.assertLess(raw, 0.0)
        self.assertGreater(RichardsonRule(1.0, 1e-3, guard=1).values(G)[a, b], 0.0)

    def test_guarded_rule_keeps_the_clipped_rule_endpoints(self):
        for alpha in (0.5, 1.0, 2.0):
            for G in (8, 16, 32):
                guarded = RichardsonRule(alpha, 1e-3, guard=1)
                clip = ClipRule(alpha, 1e-3)
                self.assertAlmostEqual(guarded.dynamic_range(G), clip.dynamic_range(G),
                                       delta=1e-9 * G ** alpha)

    def test_bias_order_rises_from_one_to_two(self):
        p, Gs = 0.3, np.array([128, 256, 512, 1024])
        plain = [abs(float(ClipRule(1.0, 1e-3).error_table(np.array([p]), int(G))["rel_bias"][0]))
                 for G in Gs]
        rich = [abs(float(RichardsonRule(1.0, 1e-3, 1)
                          .error_table(np.array([p]), int(G))["rel_bias"][0])) for G in Gs]
        order_plain = -least_squares_line(np.log(Gs), np.log(plain))[0]
        order_rich = -least_squares_line(np.log(Gs), np.log(rich))[0]
        self.assertAlmostEqual(order_plain, 1.0, delta=0.15)
        self.assertAlmostEqual(order_rich, 2.0, delta=0.25)
        self.assertLess(rich[-1], plain[-1] / 20.0)

    def test_leading_variance_is_essentially_unchanged(self):
        p = 0.3
        for G in (512, 1024):
            s_plain = float(ClipRule(1.0, 1e-3).error_table(np.array([p]), G)["rel_sd"][0])
            s_rich = float(RichardsonRule(1.0, 1e-3, 1)
                           .error_table(np.array([p]), G)["rel_sd"][0])
            self.assertAlmostEqual(s_rich / s_plain, 1.0, delta=0.05)


class OffsetRuleTests(unittest.TestCase):
    def test_default_offset_and_the_alpha_one_special_case(self):
        self.assertAlmostEqual(OffsetRule(2.0).c, -0.5)
        self.assertAlmostEqual(OffsetRule(0.5).c, 0.25)
        self.assertAlmostEqual(OffsetRule(1.0).c, 0.0)
        np.testing.assert_allclose(OffsetRule(1.0, 1e-3).values(16),
                                   ClipRule(1.0, 1e-3).values(16), rtol=1e-14)

    def test_dynamics_level_distortion_order_rises_to_two(self):
        p, Gs = 0.3, np.array([128, 256, 512, 1024])

        def distortion(rule, G):
            return abs(float(rule.effective_weight(np.array([p]), G)[0]) * p ** rule.alpha - 1.0)

        for alpha in (0.5, 2.0, 3.0):
            d_clip = [distortion(ClipRule(alpha, 1e-9), int(G)) for G in Gs]
            d_off = [distortion(OffsetRule(alpha, 1e-9), int(G)) for G in Gs]
            order_clip = -least_squares_line(np.log(Gs), np.log(d_clip))[0]
            order_off = -least_squares_line(np.log(Gs), np.log(d_off))[0]
            self.assertAlmostEqual(order_clip, 1.0, delta=0.15)
            self.assertAlmostEqual(order_off, 2.0, delta=0.25)

    def test_clip_distortion_coefficient_matches_alpha_times_alpha_minus_one(self):
        p, G = 0.3, 4096
        for alpha in (0.5, 1.5, 2.0, 2.5):
            d = float(ClipRule(alpha, 1e-9).effective_weight(np.array([p]), G)[0]) * p ** alpha - 1.0
            coeff = d * G * p / (1.0 - p)
            self.assertAlmostEqual(coeff, alpha * (alpha - 1.0) / 2.0, delta=0.02)

    def test_alpha_one_distortion_is_exactly_minus_the_miss_probability(self):
        p = 0.1
        for G in (16, 32, 64, 128):
            d = float(ClipRule(1.0, 1e-9).effective_weight(np.array([p]), G)[0]) * p - 1.0
            self.assertAlmostEqual(d, -(1.0 - p) ** G, delta=1e-9 * (1.0 - p) ** G + 1e-13)


class AsymptoticsTests(unittest.TestCase):
    def test_delta_method_predicts_the_exact_bias_when_Gp_is_large(self):
        for alpha in (0.5, 1.0, 2.0):
            for p, G in [(0.3, 4096), (0.5, 2048)]:
                exact = float(ClipRule(alpha, 1e-12)
                              .error_table(np.array([p]), G)["rel_bias"][0])
                self.assertAlmostEqual(exact / delta_relative_bias(p, G, alpha), 1.0,
                                       delta=0.02)

    def test_delta_method_predicts_the_exact_sd_when_Gp_is_large(self):
        for alpha in (0.5, 1.0, 2.0):
            exact = float(ClipRule(alpha, 1e-12)
                          .error_table(np.array([0.3]), 4096)["rel_sd"][0])
            self.assertAlmostEqual(exact / delta_relative_sd(0.3, 4096, alpha), 1.0,
                                   delta=0.02)

    def test_empty_group_term_explains_the_small_Gp_breakdown(self):
        """Where G p is small the delta method alone is wrong by an order of
        magnitude, and adding the single empty-group term (5.3) restores it."""
        alpha, eps = 1.0, 1e-3
        for p, G in [(0.2, 16), (0.1, 16), (0.3, 16)]:
            exact = float(ClipRule(alpha, eps).error_table(np.array([p]), G)["rel_bias"][0])
            delta = delta_relative_bias(p, G, alpha)
            both = delta + empty_group_bias(p, G, alpha, eps)
            self.assertGreater(exact / delta, 5.0, msg=f"p={p}, G={G}")
            self.assertAlmostEqual(exact / both, 1.0, delta=0.12, msg=f"p={p}, G={G}")


class SampledUpdateTests(unittest.TestCase):
    def test_sampled_step_reproduces_rhs_sampled(self):
        z = np.random.default_rng(4).normal(size=(11, 5))
        for G in (4, 16):
            for alpha in (0.5, 1.0, 2.0):
                g1, h1 = sampled_step(z, R5, ClipRule(alpha, 1e-3), G,
                                      np.random.default_rng(77))
                g2, h2 = rhs_sampled(z, R5, alpha, G, np.random.default_rng(77), 1e-3)
                np.testing.assert_allclose(g1, g2, rtol=0.0, atol=1e-11)
                np.testing.assert_array_equal(h1, h2)

    def test_weights_from_samples_agree_with_the_value_tables(self):
        rng = np.random.default_rng(9)
        K, G = 4, 8
        idx = rng.integers(0, K, size=(20, G))
        counts = np.stack([(idx == k).sum(axis=1) for k in range(K)], axis=1)
        rule = ClipRule(1.0, 1e-3)
        np.testing.assert_allclose(rule.weights_from_samples(idx, K),
                                   rule.values(G)[counts], rtol=1e-14)
        split = RichardsonRule(1.0, 1e-3, guard=1)
        a = np.stack([(idx[:, :G // 2] == k).sum(axis=1) for k in range(K)], axis=1)
        b = np.stack([(idx[:, G // 2:] == k).sum(axis=1) for k in range(K)], axis=1)
        np.testing.assert_allclose(split.weights_from_samples(idx, K),
                                   split.values(G)[a, b], rtol=1e-14)

    def test_the_update_direction_sums_to_zero_in_the_softmax_gauge(self):
        rng = np.random.default_rng(3)
        z = rng.normal(size=(6, 5))
        ghat, p_hat = sampled_step(z, R5, ClipRule(1.0, 1e-3), 16, rng)
        # sum_i ghat_i = sum_i term_i - (sum_i p_i) sum_k term_k = 0
        np.testing.assert_allclose(ghat.sum(axis=1), 0.0, atol=1e-12)
        np.testing.assert_allclose(p_hat.sum(axis=1), 1.0, rtol=1e-14)


class Optimize1DTests(unittest.TestCase):
    def test_golden_section_finds_a_smooth_minimum(self):
        f = lambda x: (x - 0.37) ** 2 + 1.0
        x, fx, _, _ = golden_section(f, -2.0, 3.0, tol=1e-12)
        self.assertAlmostEqual(x, 0.37, delta=1e-7)     # sqrt(eps) is the floor
        self.assertAlmostEqual(fx, 1.0, delta=1e-13)

    def test_golden_section_reproduces_the_closed_form_optimal_lambda(self):
        for G, K, p in [(16, 5, 0.1), (32, 3, 0.4), (64, 10, 0.02), (8, 2, 0.25)]:
            closed = K * p * (1.0 - p) / (1.0 - K * p) ** 2

            def mse(lam):
                D = G + K * lam
                return (lam * (1.0 - K * p) / D) ** 2 + G * p * (1.0 - p) / D ** 2

            got, _, _, _ = golden_section_log(mse, 1e-6, 1e4, rel_tol=1e-14)
            self.assertAlmostEqual(got / closed, 1.0, delta=1e-6)

    def test_the_mse_optimal_clip_is_exactly_p(self):
        for alpha in (0.5, 1.0, 2.0, 3.0):
            for G in (4, 16, 100):
                for p in (0.007, 0.13, 0.77):
                    obj = lambda e: float(ClipRule(alpha, e)
                                          .error_table(np.array([p]), G)["rel_mse"][0])
                    e_star, _, _, _ = golden_section_log(obj, 1e-6, 0.999, rel_tol=1e-14)
                    self.assertAlmostEqual(e_star / p, 1.0, delta=1e-6)

    def test_grid_minimum_and_unimodality_flag(self):
        f = lambda x: (np.log(x) + 1.0) ** 2
        x, fx, values, grid = grid_minimum(f, 1e-3, 1e3, n=4001, geometric=True)
        self.assertAlmostEqual(x, np.exp(-1.0), delta=0.01)
        self.assertTrue(is_unimodal(values))
        self.assertFalse(is_unimodal(np.array([1.0, 0.0, 1.0, 0.0, 1.0])))

    def test_invalid_brackets_are_rejected(self):
        with self.assertRaises(ValueError):
            golden_section(lambda x: x, 1.0, 1.0)
        with self.assertRaises(ValueError):
            golden_section_log(lambda x: x, 0.0, 1.0)


class EigenTests(unittest.TestCase):
    @staticmethod
    def _cov(alpha=1.0):
        ps = stationary_p(R5, alpha)
        return np.diag(ps) - np.outer(ps, ps)

    def test_power_method_matches_numpy(self):
        C = self._cov()
        ref = np.linalg.eigvalsh(C)
        lam, vec, _, converged = power_method(C)
        self.assertTrue(converged)
        self.assertAlmostEqual(lam, ref[-1], delta=1e-11)
        np.testing.assert_allclose(C @ vec, lam * vec, atol=1e-9)

    def test_deflated_power_method_recovers_the_whole_spectrum(self):
        C = self._cov()
        vals, vecs = deflated_power_method(C)
        np.testing.assert_allclose(np.sort(vals), np.linalg.eigvalsh(C), atol=1e-10)
        np.testing.assert_allclose(vecs.T @ vecs, np.eye(len(C)), atol=1e-8)

    def test_gram_schmidt_handles_a_singular_matrix(self):
        C = self._cov()
        Q, R = gram_schmidt_qr(C)
        np.testing.assert_allclose(Q.T @ Q, np.eye(len(C)), atol=1e-12)
        np.testing.assert_allclose(Q @ R, C, atol=1e-12)
        np.testing.assert_allclose(np.tril(R, -1), 0.0, atol=0.0)

    def test_qr_algorithm_matches_numpy_including_singular_input(self):
        rng = np.random.default_rng(31)
        mats = [self._cov(), self._cov(2.0)]
        for n in (2, 3, 5, 8):
            B = rng.normal(size=(n, n))
            mats.append(B + B.T)
        for n in (4, 6):
            B = rng.normal(size=(n, n - 2))
            mats.append(B @ B.T)                        # rank deficient
        for A in mats:
            vals, _, _ = qr_algorithm(A)
            np.testing.assert_allclose(vals, np.linalg.eigvalsh(A), atol=1e-10)

    def test_non_symmetric_input_is_rejected(self):
        with self.assertRaises(ValueError):
            qr_algorithm(np.array([[1.0, 2.0], [0.0, 1.0]]))
        with self.assertRaises(ValueError):
            deflated_power_method(np.array([[1.0, 2.0], [0.0, 1.0]]))


class EmaTests(unittest.TestCase):
    def test_variance_factor_and_effective_group_size(self):
        for beta in (0.05, 0.2, 0.5, 1.0):
            self.assertAlmostEqual(float(variance_factor(beta)), beta / (2 - beta))
            self.assertAlmostEqual(float(effective_group_size(32, beta)),
                                   32 * (2 - beta) / beta)
        self.assertAlmostEqual(float(variance_factor(1.0)), 1.0)
        for bad in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                variance_factor(bad)

    def test_lag_formula_against_the_exact_deterministic_recursion(self):
        for beta in (0.02, 0.1, 0.5, 1.0):
            delta = 1e-3
            pbar, p = 0.0, 0.0
            for _ in range(int(40 / beta)):
                p += delta
                pbar = (1 - beta) * pbar + beta * p
            self.assertAlmostEqual(p - pbar, float(lag_bias(beta, delta)), delta=1e-12)

    def test_coupled_jacobian_against_finite_differences(self):
        for kappa in (0.5, 3.0, 50.0):
            for alpha in (0.7, 1.0, 1.5):
                J = coupled_jacobian(R5, alpha, kappa)
                ps = stationary_p(R5, alpha)
                s0 = np.concatenate([np.log(ps), ps])
                h = 1e-6
                Jn = np.zeros_like(J)
                for j in range(len(s0)):
                    e = np.zeros_like(s0)
                    e[j] = h
                    Jn[:, j] = (coupled_rhs((s0 + e)[None, :], R5, alpha, kappa)[0]
                                - coupled_rhs((s0 - e)[None, :], R5, alpha, kappa)[0]) / (2 * h)
                np.testing.assert_allclose(J, Jn, atol=2e-6)
                # the z-block vanishes: that is what makes the system second order
                np.testing.assert_allclose(J[:5, :5], 0.0, atol=0.0)

    def test_mode_roots_are_the_jacobian_eigenvalues(self):
        for kappa in (0.7, 4.6, 40.0):
            J = coupled_jacobian(R5, 1.0, kappa)
            ev = np.linalg.eigvals(J)
            for lam in linear_rates(R5, 1.0):
                for root in mode_roots(lam, kappa):
                    self.assertLess(float(np.min(np.abs(ev - root))), 1e-9)

    def test_decay_rate_peaks_at_four_lambda_with_twice_the_rate(self):
        for lam in (0.3, 1.1497936639994455, 4.5):
            kap = np.geomspace(1e-3, 1e5, 20001)
            rates = decay_rate(lam, kap)
            self.assertAlmostEqual(float(rates.max()) / lam, 2.0, delta=2e-3)
            self.assertAlmostEqual(float(kap[int(np.argmax(rates))]) / critical_kappa(lam),
                                   1.0, delta=2e-3)
            # kappa -> infinity recovers Experiment 2's rate
            self.assertAlmostEqual(float(decay_rate(lam, 1e8)) / lam, 1.0, delta=1e-6)
            self.assertTrue(bool(is_oscillatory(lam, 3.9 * lam)))
            self.assertFalse(bool(is_oscillatory(lam, 4.1 * lam)))

    def test_ema_does_not_move_the_fixed_point(self):
        for alpha in (0.6, 1.0, 2.0):
            ps = stationary_p(R5, alpha)
            state = np.concatenate([np.log(ps), ps])[None, :]
            f = coupled_rhs(state, R5, alpha, 3.0)
            np.testing.assert_allclose(f, 0.0, atol=1e-12)


class MeanFieldRuleTests(unittest.TestCase):
    def test_K2_solver_matches_experiment_2(self):
        for G in (2, 4, 5, 8, 16, 64):
            for alpha in (0.5, 1.0, 2.0):
                for eps in (1e-3, 0.2):
                    mine = stationary_K2(R41, ClipRule(alpha, eps), G)
                    theirs = meanfield_stationary_K2(R41, G, alpha, eps)
                    self.assertAlmostEqual(mine, theirs, delta=5e-11)

    def test_general_solver_matches_experiment_3(self):
        for G in (4, 16, 64):
            for alpha in (0.5, 1.0, 2.0):
                mine, _, stats = stationary(R5, ClipRule(alpha, 1e-3), G)
                theirs, _, _ = meanfield_stationary(R5, G, alpha, 1e-3, method="newton")
                np.testing.assert_allclose(mine, theirs, atol=1e-10)
                self.assertLess(stats["scaled_residual"], 1e-11)
                self.assertTrue(stats["extinct_condition_ok"])
                self.assertAlmostEqual(float(mine.sum()), 1.0, places=12)

    def test_survival_matches_the_dynamic_range_criterion(self):
        rho = float(R41[0] / R41[1])
        for G in range(2, 33):
            for alpha in (0.5, 1.0, 1.5):
                rule = ClipRule(alpha, 1e-3)
                survives = rule.dynamic_range(G) > rho * (1.0 + 1e-12)
                p1 = stationary_K2(R41, rule, G)
                self.assertEqual(survives, p1 < 1.0 - 1e-9,
                                 msg=f"G={G}, alpha={alpha}, p1={p1}")

    def test_extinction_alpha_matches_the_closed_form(self):
        rho = float(R41[0] / R41[1])
        for G in (4, 8, 16, 32, 64):
            a_c, _ = extinction_alpha_K2(lambda a: ClipRule(a, 1e-3), R41, G)
            self.assertAlmostEqual(a_c, np.log(rho) / np.log(G), delta=1e-8)
            # the guarded Richardson rule inherits it exactly
            a_r, _ = extinction_alpha_K2(lambda a: RichardsonRule(a, 1e-3, 1), R41, G)
            self.assertAlmostEqual(a_r, a_c, delta=1e-8)
            # add-lambda needs a strictly larger alpha
            a_l, _ = extinction_alpha_K2(lambda a: AddLambdaRule(a, 1.0, 2), R41, G)
            self.assertGreater(a_l, a_c)

    def test_monotonicity_check(self):
        for rule in (ClipRule(1.0, 1e-3), AddLambdaRule(1.0, 1.0, 5),
                     RichardsonRule(1.0, 1e-3, guard=1), OffsetRule(2.0, 1e-3)):
            ok, rise = check_monotone(rule, 16)
            self.assertTrue(ok, msg=f"{rule.name} gave a rise of {rise}")
        ok, rise = check_monotone(RichardsonRule(1.0, 1e-3, guard=None), 16)
        self.assertFalse(ok)
        self.assertGreater(rise, 0.0)

    def test_stationary_is_reward_scale_invariant(self):
        a, _, _ = stationary(R5, ClipRule(1.0, 1e-3), 16)
        b, _, _ = stationary(1e-9 * R5, ClipRule(1.0, 1e-3), 16)
        c, _, _ = stationary(1e9 * R5, ClipRule(1.0, 1e-3), 16)
        np.testing.assert_allclose(a, b, atol=1e-11)
        np.testing.assert_allclose(a, c, atol=1e-11)

    def test_naive_richardson_has_no_interior_stationary_point(self):
        self.assertEqual(stationary_K2(R41, RichardsonRule(1.0, 1e-3, guard=None), 16), 1.0)
        with self.assertRaises(ValueError):
            stationary(R5, RichardsonRule(1.0, 1e-3, guard=None), 16)

    def test_large_groups_approach_the_ideal_distribution(self):
        for alpha in (0.5, 1.0, 2.0):
            ideal = stationary_p(R5, alpha)
            previous = np.inf
            for G in (32, 64, 128, 256):
                p_mf, _, _ = stationary(R5, ClipRule(alpha, 1e-3), G)
                gap = float(np.sum(np.abs(p_mf - ideal)))
                self.assertLess(gap, previous)
                previous = gap
            self.assertLess(previous, 0.02)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            stationary_K2(np.array([1.0, 4.0]), ClipRule(1.0, 1e-3), 16)   # r1 < r2
        with self.assertRaises(ValueError):
            stationary(np.array([1.0, -1.0]), ClipRule(1.0, 1e-3), 16)


class ExperimentModuleTests(unittest.TestCase):
    def test_experiment_module_imports_and_self_checks_pass(self):
        from experiments.exp5_bias_variance import _run_self_checks
        checks = _run_self_checks()
        self.assertLess(checks["size_biasing_max_rel_error"], 1e-12)
        self.assertLess(checks["clip_vs_finite_group_general"], 1e-12)
        self.assertLess(checks["sampled_step_vs_rhs_sampled"], 1e-12)
        self.assertLess(checks["golden_section_vs_closed_form_lambda"], 1e-6)
        self.assertLess(checks["qr_algorithm"]["max_error"], 1e-10)


if __name__ == "__main__":
    unittest.main()
