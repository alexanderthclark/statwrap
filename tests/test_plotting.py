import unittest

import matplotlib.pyplot as plt
import numpy as np

from statwrap.fpp import scatter_plot as fpp_scatter_plot
from statwrap.plotting import scatter_plot


class TestScatterPlot(unittest.TestCase):

    def test_fpp_reexports_shared_helper(self):
        self.assertIs(fpp_scatter_plot, scatter_plot)

    def test_returns_the_figure_that_owns_supplied_axes(self):
        figure, axes = plt.subplots()
        other_figure = plt.figure()

        try:
            returned_figure, returned_axes = scatter_plot(
                [0, 1, 2],
                [1, 3, 5],
                ax=axes,
                show=False,
                xlim=(-1, 3),
                ylim=(0, 6),
            )

            self.assertIs(returned_figure, figure)
            self.assertIs(returned_axes, axes)
            np.testing.assert_allclose(axes.get_xlim(), (-1, 3))
            np.testing.assert_allclose(axes.get_ylim(), (0, 6))
        finally:
            plt.close(other_figure)
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
