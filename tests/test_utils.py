import unittest
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statwrap.utils import Hyperplane, RegressionLine


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class TestImportOrder(unittest.TestCase):

    def test_utils_and_fpp_import_independently(self):
        import_orders = (
            (
                "import sys; import statwrap.utils; "
                "assert 'statwrap.fpp' not in sys.modules; import statwrap.fpp"
            ),
            "import statwrap.fpp; import statwrap.utils",
        )

        for statement in import_orders:
            with self.subTest(statement=statement):
                result = subprocess.run(
                    [sys.executable, "-c", statement],
                    cwd=REPOSITORY_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


class DummyResults:
    def __init__(self, params, resid):
        self.params = params
        self.resid = np.array(resid)


class TestHyperplane(unittest.TestCase):

    def test_call_uses_linear_combination(self):
        plane = Hyperplane(2, 3, -1)
        result = plane(4, 5)
        self.assertEqual(result, 2 + 3 * 4 - 1 * 5)

    def test_predict_with_dataframe_adds_constant(self):
        plane = Hyperplane(1, 2)
        data = pd.DataFrame({"x": [0, 1, 2]})
        predictions = plane.predict(data)
        np.testing.assert_array_equal(predictions, np.array([1, 3, 5]))


class TestRegressionLine(unittest.TestCase):

    def setUp(self):
        self.x = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
        self.y = np.array([1.0, 3.1, 4.9])
        params = pd.Series([1.0, 2.0])
        resid = np.array([0.0, 0.1, -0.1])
        self.results = DummyResults(params=params, resid=resid)
        self.line = RegressionLine(self.y, self.x, self.results)

    def test_predictions_follow_underlying_hyperplane(self):
        expected = np.array([1.0, 3.0, 5.0])
        np.testing.assert_allclose(self.line.predictions, expected)

    def test_call_returns_scalar_prediction(self):
        self.assertEqual(self.line(3), 7)

    def test_residuals_and_rms_error(self):
        np.testing.assert_allclose(self.line.residuals, self.results.resid.round(5))
        expected_rms = np.sqrt(np.mean(self.results.resid ** 2))
        self.assertAlmostEqual(self.line.rms_error, expected_rms)

    def test_scatter_plot_forwards_style_to_shared_helper(self):
        self.addCleanup(plt.close, "all")
        with patch("statwrap.utils._scatter_plot") as scatter_plot:
            with patch("statwrap.utils.plt.show"):
                self.line.scatter_plot(color="purple")

        self.assertEqual(scatter_plot.call_count, 1)
        self.assertEqual(scatter_plot.call_args.kwargs["color"], "purple")
        self.assertFalse(scatter_plot.call_args.kwargs["show"])


if __name__ == "__main__":
    unittest.main()
