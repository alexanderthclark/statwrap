import unittest
import pandas as pd

# Ensure modules import in order to avoid circular import
import statwrap.fpp
from statwrap.sheets import linest
from statwrap.exceptions import StatwrapError, SimplePlotError


class TestPlotExceptions(unittest.TestCase):
    def test_plot_multivariate_raises(self):
        y = [1, 2, 3]
        X = pd.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4], "c": [3, 4, 5]})
        model = linest(y, X)
        with self.assertRaises(SimplePlotError):
            model.plot(show=False)

    def test_exception_str(self):
        err = SimplePlotError("bad stuff")
        self.assertEqual(str(err), "bad stuff")
        self.assertIn("SimplePlotError", repr(err))


if __name__ == "__main__":
    unittest.main()
