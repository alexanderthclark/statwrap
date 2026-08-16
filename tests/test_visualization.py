import unittest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statwrap.visualization import LossSurface


class TestLossSurface(unittest.TestCase):

    def setUp(self):
        # Simple, deterministic test data
        self.X = np.array([[1.0, 2.0],
                          [3.0, 4.0]])
        self.y = np.array([5.0, 11.0])

        # Fit a model
        self.model = LinearRegression()
        self.model.fit(self.X, self.y)

        # Create loss surface
        self.loss_surface = LossSurface(self.model, self.X, self.y)

    def test_evaluate_loss_simple_case(self):
        """Test evaluate_loss with simple hardcoded values."""
        # Test with coefficients [1, 2] and intercept 0
        # X = [[1, 2], [3, 4]], y = [5, 11]
        # predictions = [1*1 + 2*2 + 0, 1*3 + 2*4 + 0] = [5, 11]
        # residuals = [5-5, 11-11] = [0, 0]
        # MSE = (0^2 + 0^2) / 2 = 0

        loss = self.loss_surface.evaluate_loss([1.0, 2.0], intercept=0.0)
        self.assertEqual(loss, 0.0)

    def test_evaluate_loss_nonzero_case(self):
        """Test evaluate_loss with nonzero residuals."""
        # Test with coefficients [1, 1] and intercept 0
        # predictions = [1*1 + 1*2 + 0, 1*3 + 1*4 + 0] = [3, 7]
        # residuals = [5-3, 11-7] = [2, 4]
        # MSE = (2^2 + 4^2) / 2 = (4 + 16) / 2 = 10

        loss = self.loss_surface.evaluate_loss([1.0, 1.0], intercept=0.0)
        self.assertEqual(loss, 10.0)

    def test_evaluate_loss_input_validation(self):
        """Test that evaluate_loss validates coefficient input."""
        # Test wrong shape
        with self.assertRaises(ValueError):
            self.loss_surface.evaluate_loss([1.0])  # Too few coefficients

        with self.assertRaises(ValueError):
            self.loss_surface.evaluate_loss([1.0, 2.0, 3.0])  # Too many coefficients

    def test_evaluate_loss_returns_float(self):
        """Test that evaluate_loss returns a float."""
        loss = self.loss_surface.evaluate_loss([1.0, 2.0])
        self.assertIsInstance(loss, float)

    def test_resolve_coefficient_range_validates_overrides(self):
        """Test coefficient-range defaults, coercion, and validation."""
        self.assertEqual(
            self.loss_surface._resolve_coefficient_range(None),
            self.loss_surface.coefficient_range,
        )
        self.assertEqual(
            self.loss_surface._resolve_coefficient_range("1.5"),
            1.5,
        )

        for invalid_range in (0, -1):
            with self.subTest(coefficient_range=invalid_range):
                with self.assertRaisesRegex(
                    ValueError,
                    "coefficient_range must be a positive number",
                ):
                    self.loss_surface._resolve_coefficient_range(invalid_range)

    def test_ridge_surface_respects_coefficient_range_limits(self):
        """Test that path artists do not expand the requested surface window."""
        coefficient_range = 0.25
        center = self.loss_surface.w_opt_
        expected_x_limits = (
            center[0] - coefficient_range,
            center[0] + coefficient_range,
        )
        expected_y_limits = (
            center[1] - coefficient_range,
            center[1] + coefficient_range,
        )

        ax = self.loss_surface.plot_ridge_path_on_surface(
            alphas=[1e6],
            coefficient_range=coefficient_range,
        )
        try:
            path_x = ax.lines[0].get_xdata()
            path_y = ax.lines[0].get_ydata()
            self.assertTrue(
                np.any((path_x < expected_x_limits[0]) | (path_x > expected_x_limits[1]))
                or np.any((path_y < expected_y_limits[0]) | (path_y > expected_y_limits[1]))
            )
            np.testing.assert_allclose(ax.get_xlim(), expected_x_limits)
            np.testing.assert_allclose(ax.get_ylim(), expected_y_limits)
        finally:
            plt.close(ax.figure)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
