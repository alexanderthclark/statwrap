"""Numerical and composition checks for the regularization teaching helpers."""

import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from statwrap.visualization import LossSurface, plot_l1_ball, plot_l2_ball


class TestRegularization(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(777)
        self.X = rng.normal(size=(80, 2)) + [2, 5]
        self.X[:, 1] += 0.8 * self.X[:, 0]
        self.y = 4 + self.X @ [1, 2] + rng.normal(size=80)
        self.surface = LossSurface(LinearRegression(), self.X, self.y)

    def tearDown(self):
        plt.close("all")

    def test_surface_matches_direct_profiled_predictions(self):
        pairs = np.array([[0, 0], [1, 2], [-3, 7]])
        expected = [
            np.mean(
                (self.y - self.X @ w - (self.y.mean() - self.X.mean(axis=0) @ w)) ** 2
            )
            for w in pairs
        ]
        np.testing.assert_allclose(self.surface.surface_loss(pairs), expected)
        np.testing.assert_allclose(
            self.surface.surface_loss(self.surface.w_ols_), self.surface.minimum_loss_
        )
        self.assertEqual(
            self.surface.surface_loss(self.surface.w_ols_, relative=True), 0
        )

    def test_grid_and_fixed_intercept_semantics_agree(self):
        surface = LossSurface(LinearRegression(fit_intercept=False), self.X, self.y)
        W1, W2, Z = surface._mse_grid([-1, 1], [0, 2, 4])
        for i, j in np.ndindex(Z.shape):
            expected = np.mean((self.y - self.X @ [W1[i, j], W2[i, j]]) ** 2)
            self.assertAlmostEqual(Z[i, j], expected, places=10)
        np.testing.assert_allclose(surface.gram_, self.X.T @ self.X / len(self.y))
        self.assertEqual(surface.plot().get_title(), "MSE (intercept = 0)")

    def test_fixed_loss_keeps_legacy_default(self):
        expected = np.mean((self.y - self.surface.intercept_opt_) ** 2)
        self.assertAlmostEqual(self.surface.evaluate_loss([0, 0]), expected)
        self.assertNotAlmostEqual(expected, self.surface.surface_loss([0, 0]))

    def test_orthogonal_noise_only_changes_loss_offset(self):
        design = np.column_stack((np.ones(len(self.y)), self.X))
        noise = np.random.default_rng(42).normal(size=len(self.y)) * 10
        noise -= design @ np.linalg.lstsq(design, noise, rcond=None)[0]
        noisy = LossSurface(LinearRegression(), self.X, self.y + noise)
        pairs = np.array([[0, 0], [1, 2], [-3, 7]])
        np.testing.assert_allclose(noisy.w_ols_, self.surface.w_ols_, atol=1e-12)
        np.testing.assert_allclose(
            noisy.surface_loss(pairs, relative=True),
            self.surface.surface_loss(pairs, relative=True),
            atol=1e-11,
        )
        np.testing.assert_allclose(
            noisy.surface_loss(pairs) - self.surface.surface_loss(pairs),
            noisy.minimum_loss_ - self.surface.minimum_loss_,
        )
        for kind in ("ridge", "lasso"):
            np.testing.assert_allclose(
                noisy.regularization_path(kind, [0.1, 1]).coefficients,
                self.surface.regularization_path(kind, [0.1, 1]).coefficients,
                atol=1e-6,
            )

    def test_paths_match_sklearn_including_intercepts_and_mse(self):
        for fit_intercept in (True, False):
            surface = LossSurface(
                LinearRegression(fit_intercept=fit_intercept), self.X, self.y
            )
            for kind, cls in (("ridge", Ridge), ("lasso", Lasso)):
                with self.subTest(kind=kind, fit_intercept=fit_intercept):
                    result = surface.regularization_path(kind, [10, 0.1, 1, 1])
                    np.testing.assert_array_equal(result.alphas, [0, 0.1, 1, 10])
                    np.testing.assert_allclose(result.coefficients[0], surface.w_ols_)
                    for index, alpha in enumerate(result.alphas[1:], 1):
                        kwargs = (
                            {"max_iter": 100000, "tol": 1e-8} if kind == "lasso" else {}
                        )
                        model = cls(
                            alpha=alpha, fit_intercept=fit_intercept, **kwargs
                        ).fit(self.X, self.y)
                        np.testing.assert_allclose(
                            result.coefficients[index], model.coef_, atol=1e-10
                        )
                        self.assertAlmostEqual(
                            result.intercepts[index], model.intercept_
                        )
                        self.assertAlmostEqual(
                            result.mse[index],
                            np.mean((self.y - model.predict(self.X)) ** 2),
                        )

    def test_data_scaled_defaults_reach_strong_shrinkage(self):
        lasso = self.surface.regularization_path("lasso")
        np.testing.assert_allclose(lasso.coefficients[-1], 0, atol=1e-10)
        ridge = self.surface.regularization_path("ridge")
        self.assertLess(
            np.linalg.norm(ridge.coefficients[-1]),
            0.001 * np.linalg.norm(ridge.coefficients[0]),
        )
        result = self.surface.regularization_path("ridge", [1], include_ols=False)
        np.testing.assert_array_equal(result.alphas, [1])
        self.assertEqual(result.coefficients.shape, (1, 2))

    def test_prefit_and_unfitted_inputs_are_not_mutated(self):
        model = Ridge(alpha=2).fit(self.X, self.y[:, None])
        coef, intercept = model.coef_.copy(), model.intercept_.copy()
        surface = LossSurface(model, self.X, self.y[:, None])
        surface.plot_regularization("lasso", 0.5)
        surface.plot_eigenvectors()
        np.testing.assert_array_equal(model.coef_, coef)
        np.testing.assert_array_equal(model.intercept_, intercept)
        fresh = LinearRegression(copy_X=False)
        X = self.X.copy()
        LossSurface(fresh, X, self.y)
        np.testing.assert_array_equal(X, self.X)
        self.assertFalse(hasattr(fresh, "coef_"))
        self.assertIs(surface.model, model)

    def test_dataframe_and_custom_labels(self):
        frame = pd.DataFrame(self.X, columns=["Wealth", "Income"])
        surface = LossSurface(LinearRegression(), frame, self.y)
        ax = surface.plot()
        self.assertEqual((ax.get_xlabel(), ax.get_ylabel()), ("Wealth", "Income"))
        ax = LossSurface(
            LinearRegression(), frame, self.y, feature_names=["A", "B"]
        ).plot_ridge_coef_path([1, 10])
        self.assertEqual([line.get_label() for line in ax.lines[:2]], ["A", "B"])

    def test_existing_axes_are_composable_without_duplicate_contours(self):
        ax = self.surface.plot(
            relative_loss=True, levels=[0.1, 1, 5], label_contours=False
        )
        ax.set_xlim(-1, 3)
        ax.set_ylim(-2, 4)
        initial_contours = sum(
            type(c).__name__ == "QuadContourSet" for c in ax.collections
        )
        with patch.object(ax, "contour", wraps=ax.contour) as contour:
            self.assertIs(self.surface.plot_lasso_path_on_surface(ax=ax), ax)
            self.assertIs(self.surface.plot_ridge_path_on_surface(ax=ax), ax)
            self.surface.plot_constraint(2, ax=ax)
            self.surface.plot_eigenvectors(ax=ax)
            self.surface.plot_regularization("ridge", 10, ax=ax)
            contour.assert_not_called()
        self.assertEqual(
            sum(type(c).__name__ == "QuadContourSet" for c in ax.collections),
            initial_contours,
        )
        np.testing.assert_array_equal(ax.get_xlim(), [-1, 3])
        np.testing.assert_array_equal(ax.get_ylim(), [-2, 4])
        self.assertEqual(ax.get_aspect(), 1)

    def test_ball_geometry_and_zero_radius(self):
        for func, order in ((plot_l1_ball, 1), (plot_l2_ball, 2)):
            ax = func(2.5, lw=3, c="purple")
            points = ax.lines[-1].get_xydata()
            np.testing.assert_allclose(np.linalg.norm(points, ord=order, axis=1), 2.5)
            self.assertEqual(ax.lines[-1].get_color(), "purple")
            func(0, ax=ax)
            np.testing.assert_allclose(ax.lines[-1].get_xydata(), 0)

    def test_selected_constraints_pass_through_correct_fits(self):
        for kind, cls, order, alpha in (
            ("ridge", Ridge, 2, 30),
            ("lasso", Lasso, 1, 0.5),
        ):
            ax = self.surface.plot_regularization(kind, alpha, show_path=False)
            model = cls(alpha=alpha, tol=1e-8, max_iter=100000).fit(self.X, self.y)
            selected = np.asarray(ax.collections[-1].get_offsets())[0]
            np.testing.assert_allclose(selected, model.coef_, atol=1e-7)
            radius = np.linalg.norm(model.coef_, ord=order)
            boundary = ax.lines[-1].get_xydata()
            np.testing.assert_allclose(
                np.linalg.norm(boundary, ord=order, axis=1), radius, atol=1e-7
            )
            self.assertLessEqual(ax.get_xlim()[0], -radius)
            self.assertGreaterEqual(ax.get_xlim()[1], radius)
            self.assertLessEqual(ax.get_ylim()[0], -radius)
            self.assertGreaterEqual(ax.get_ylim()[1], radius)
            # KKT stationarity ties the boundary to the plotted MSE gradient.
            gradient = 2 * self.surface.gram_ @ (selected - self.surface.w_ols_)
            if kind == "ridge":
                np.testing.assert_allclose(
                    gradient + 2 * alpha / len(self.y) * selected, 0, atol=1e-7
                )
            else:
                active = np.abs(selected) > 1e-8
                np.testing.assert_allclose(
                    gradient[active] + 2 * alpha * np.sign(selected[active]),
                    0,
                    atol=1e-7,
                )
                self.assertTrue(np.all(np.abs(gradient[~active]) <= 2 * alpha + 1e-7))

    def test_zero_alpha_and_zero_solution(self):
        ax = self.surface.plot_regularization("lasso", 0, show_path=False)
        np.testing.assert_allclose(
            ax.collections[-1].get_offsets()[0], self.surface.w_ols_
        )
        ax = self.surface.plot_regularization("lasso", 1e8, show_path=False)
        np.testing.assert_allclose(ax.collections[-1].get_offsets()[0], 0)
        np.testing.assert_allclose(ax.lines[-1].get_xydata(), 0)

    def test_highlighted_loss_contours_match_selected_solution(self):
        singular = LossSurface(
            LinearRegression(), [[0, 0], [1, 1], [2, 2]], [100, 101, 102]
        )
        for surface in (self.surface, singular):
            ax = surface.plot_regularization("ridge", 1, show_path=False)
            selected = ax.collections[-1].get_offsets()[0]
            # All points of the highlighted ellipse (or parallel lines at
            # rank one) have the selected fit's MSE, even on a coarse grid.
            for line in ax.lines[:-1]:
                np.testing.assert_allclose(
                    surface.surface_loss(line.get_xydata()),
                    surface.surface_loss(selected),
                    atol=1e-10,
                )

    def test_cleared_axes_get_a_new_surface(self):
        ax = self.surface.plot()
        ax.clear()
        with patch.object(ax, "contour", wraps=ax.contour) as contour:
            self.surface.plot_ridge_path_on_surface(ax=ax, alphas=[1])
            contour.assert_called_once()

    def test_overlay_rejects_3d_surface_options(self):
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            self.surface.plot_regularization("ridge", 1, plot_type="3d")

    def test_rank_deficient_lasso_does_not_connect_arbitrary_ols_point(self):
        X = np.array([[0, 0], [1, 1], [2, 2]])
        surface = LossSurface(LinearRegression(), X, [100, 101, 102])
        pairs = np.array([[1, 0], [0, 1], [0.5, 0.5]])
        np.testing.assert_allclose(surface.surface_loss(pairs), 0, atol=1e-25)
        ax = surface.plot_lasso_path_on_surface([0.01, 0.1])
        self.assertEqual(len(ax.lines[0].get_xdata()), 2)
        np.testing.assert_allclose(
            ax.lines[0].get_xydata(),
            surface.regularization_path("lasso", [0.01, 0.1]).coefficients[1:],
        )
        self.assertFalse(np.allclose(ax.lines[0].get_xydata()[0], surface.w_ols_))
        surface.plot_eigenvectors(ax=ax)

    def test_coefficient_plot_uses_real_zero_alpha(self):
        ax = self.surface.plot_ridge_coef_path([0.1, 1])
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [0, 0.1, 1])
        self.assertEqual(ax.get_xscale(), "symlog")

    def test_eigenvectors_use_ols_center_and_correct_curvature(self):
        surface = LossSurface(Ridge(alpha=50), self.X, self.y)
        ax = surface.plot_eigenvectors(length=2, annotate=False)
        arrows = ax.texts[-2:]
        self.assertFalse(np.allclose(surface.w_opt_, surface.w_ols_))
        for arrow in arrows:
            np.testing.assert_allclose(arrow.get_position(), surface.w_ols_)
            direction = np.asarray(arrow.xy) - surface.w_ols_
            self.assertAlmostEqual(np.linalg.norm(direction), 2)
            transformed = surface.gram_ @ direction
            self.assertAlmostEqual(
                direction[0] * transformed[1] - direction[1] * transformed[0], 0
            )

    def test_legacy_range_alias_preserves_original_notebook_calls(self):
        for method in (
            self.surface.plot,
            self.surface.plot_ridge_path_on_surface,
            self.surface.plot_lasso_path_on_surface,
        ):
            with self.assertWarns(DeprecationWarning):
                ax = method(loss_range=15)
            np.testing.assert_allclose(
                ax.get_xlim(), self.surface.w_opt_[0] + [-15, 15]
            )
            with self.assertRaises(ValueError):
                method(loss_range=15, coefficient_range=4)

    def test_3d_marker_lies_on_the_same_surface(self):
        ax = self.surface.plot("3d", relative_loss=True)
        x, y, z = ax.collections[-1]._offsets3d
        self.assertAlmostEqual(
            float(z[0]), self.surface.surface_loss([x[0], y[0]], relative=True)
        )
        with self.assertRaises(ValueError):
            self.surface.plot(ax=ax)
        with self.assertRaises(ValueError):
            self.surface.plot("heatmap")

    def test_invalid_inputs_fail_early(self):
        for kwargs in (
            {"coefficient_range": 0},
            {"coefficient_range": np.nan},
            {"coefficient_range": np.inf},
            {"grid_size": 0},
            {"grid_size": 2.5},
            {"grid_size": True},
            {"feature_names": ["one"]},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                LossSurface(LinearRegression(), self.X, self.y, **kwargs)
        for X, y in (
            (np.empty((0, 2)), []),
            (self.X, self.y.reshape(2, 40)),
            (self.X * np.nan, self.y),
            (self.X, self.y[:-1]),
        ):
            with self.assertRaises(ValueError):
                LossSurface(LinearRegression(), X, y)
        for alphas in ([], [0], [-1], [np.nan], [np.inf], [[1, 2]]):
            with self.assertRaises(ValueError):
                self.surface.regularization_path("lasso", alphas)
        for radius in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                plot_l1_ball(radius)
        with self.assertRaises(ValueError):
            self.surface.plot_regularization("elasticnet")
        with self.assertRaises(ValueError):
            self.surface.plot(xlim=(1, -1))


if __name__ == "__main__":
    unittest.main()
