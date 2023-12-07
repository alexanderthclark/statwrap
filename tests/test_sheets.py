import unittest
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statwrap.sheets import linest

class TestLinest(unittest.TestCase):

	def setUp(self):
		self.x1 = np.array([-1, 0, 1, 2])
		self.x2 = np.array([5, 4, 3, 10])
		self.X = np.array([self.x1, self.x2]).T
		self.y = 2 + self.x1 - self.x2

	def single_linest(self):
		reg_line1 = linest(self.y, self.x1)
		reg_line12 = linest(self.y, self.X)
		params12 = reg_line12.results.params
		self.assertTrue(math.isclose(params12[0], 2, abs_tol=10**-6))

	def double_call(self):
		y = np.random.normal(size = 10)
		X = pd.DataFrame(np.random.rand(10,2))
		X['ones'] = 1
		r, s = linest(y, X, verbose=True)
		r2, s2 = linest(y, X, verbose=True)

		self.assertTrue('ones' in X.columns)
		self.assertTrue(r.results.params['const'] == r2.results.params['const'])

	def tearDown(self):
		pass
