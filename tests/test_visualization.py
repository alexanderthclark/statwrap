import unittest
import numpy as np
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

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()