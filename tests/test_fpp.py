import unittest
import pandas as pd
import matplotlib.pyplot as plt
from statwrap.fpp import apply_pd_changes, sd, standard_units, average, r, rms_size, var, sd_plus, var_plus


class TestCorrelation(unittest.TestCase):

    def setUp(self):
        self.x1 = [0, 1, 1]
        self.y1 = [2, -9, 2]
        self.x2 = [1, 2, 3, 4, 5]
        self.y2 = [5, 4, 3, 2, 1]
        self.x3 = [1, 2, 3]
        self.y3 = [4, 5, 6]

    def test_correlation_basic(self):
        result = r(self.x1, self.y1)
        expected = -0.5
        self.assertAlmostEqual(result, expected)

    def test_correlation_inverse(self):
        result = r(self.x2, self.y2)
        expected = -1.0
        self.assertAlmostEqual(result, expected)

    def test_correlation_perfect(self):
        result = r(self.x3, self.y3)
        expected = 1.0
        self.assertAlmostEqual(result, expected)

    def tearDown(self):
        pass

class TestAverage(unittest.TestCase):

    def setUp(self):
        self.numbers_array = [0, 5, -8, 7, -3]
        self.numbers_multiple = [1, 2, 3, 4, 5]
        self.single_value = [42]
        self.mixed_types = [1, 2.5, 3, 4.5]

    #This passes in a list of numbers
    def test_average_array(self):
        result = average(self.numbers_array)
        expected = 0.2
        self.assertAlmostEqual(result, expected)

    #This unpacks the list of numbers first
    def test_average_multiple_args_unpacked(self):
        result = average(*self.numbers_multiple)
        expected = 3.0
        self.assertAlmostEqual(result, expected)

    def test_average_single_value(self):
        result = average(*self.single_value)
        expected = 42.0
        self.assertAlmostEqual(result, expected)

    def test_average_mixed_types(self):
        result = average(*self.mixed_types)
        expected = 2.75
        self.assertAlmostEqual(result, expected)

    def tearDown(self):
        pass

class TestRMSSize(unittest.TestCase):

    def setUp(self):
        self.numbers_array = [0, 5, -8, 7, -3]
        self.numbers_multiple = [1, 2, 3, 4, 5]
        self.single_value = [42]
        self.mixed_types = [1, 2.5, 3, 4.5]
        self.zeroes = [0, 0, 0, 0]

    def test_rms_size_array(self):
        result = rms_size(self.numbers_array)
        expected = 5.422176684690384
        self.assertAlmostEqual(result, expected)

    def test_rms_size_multiple_args(self):
        result = rms_size(*self.numbers_multiple)
        expected = 3.3166247903554
        self.assertAlmostEqual(result, expected)

    def test_rms_size_single_value(self):
        result = rms_size(*self.single_value)
        expected = 42.0
        self.assertAlmostEqual(result, expected)

    def test_rms_size_mixed_types(self):
        result = rms_size(*self.mixed_types)
        expected = 3.020761493398643
        self.assertAlmostEqual(result, expected)

    def test_rms_size_zeroes(self):
        result = rms_size(*self.zeroes)
        expected = 0.0
        self.assertAlmostEqual(result, expected)

    def tearDown(self):
        pass
    
class TestStd(unittest.TestCase):

    def setUp(self):
        self.numbers = [-1, 0, 1]
        self.df = pd.DataFrame({'col1': self.numbers})

    def test_sd(self):
        sd_ = sd(self.numbers)
        self.assertFalse(sd_ == 1)
        apply_pd_changes()
        self.assertTrue(self.df.col1.sd() == sd_)
        self.assertTrue(sd_ == sd(*self.numbers))

    def tearDown(self):
        pass

class TestVariance(unittest.TestCase):

    def setUp(self):
        self.numbers_array = [-1, 0, 1]
        self.numbers_multiple = [1, 2, 3, 4, 5]
        self.single_value = [42]
        self.mixed_types = [1, 2.5, 3, 4.5]
        self.zeroes = [0, 0, 0, 0]

    def test_variance_array(self):
        result = var(self.numbers_array)
        expected = 0.6666666666666666
        self.assertAlmostEqual(result, expected)

    def test_variance_multiple_args(self):
        result = var(*self.numbers_multiple)
        expected = 2.0
        self.assertAlmostEqual(result, expected)

    def test_variance_single_value(self):
        result = var(*self.single_value)
        expected = 0.0
        self.assertAlmostEqual(result, expected)

    def test_variance_mixed_types(self):
        result = var(*self.mixed_types)
        expected = 1.5625
        self.assertAlmostEqual(result, expected)

    def test_variance_zeroes(self):
        result = var(*self.zeroes)
        expected = 0.0
        self.assertAlmostEqual(result, expected)

    def tearDown(self):
        pass

class TestSDPlus(unittest.TestCase):

    def setUp(self):
        self.numbers_array = [-1, 0, 1]
        self.numbers_multiple = [1, 2, 3, 4, 5]
        self.single_value = [42]
        self.mixed_types = [1, 2.5, 3, 4.5]
        self.zeroes = [0, 0, 0, 0]

    def test_sd_plus_division_error(self):
        with self.assertRaises(ValueError):
            sd_plus(self.single_value)

    def test_sd_plus_array(self):
        result = sd_plus(self.numbers_array)
        expected = 1.0
        self.assertAlmostEqual(result, expected)

    def test_sd_plus_multiple_args(self):
        result = sd_plus(*self.numbers_multiple)
        expected = 1.5811388300841898
        self.assertAlmostEqual(result, expected)

    def test_sd_plus_mixed_types(self):
        result = sd_plus(*self.mixed_types)
        expected = 1.4433756729740645
        self.assertAlmostEqual(result, expected)

    def tearDown(self):
        pass
    
class TestStandardUnits(unittest.TestCase):

    def setUp(self):
        self.numbers_array = [-1, 0, 1]
        self.numbers_multiple = [-100, 0, 1000, 2, 17]
        self.identical_numbers = [1, 1, 1]
        self.numbers_sd_plus = [1, 6, 100]

    def test_standard_units_array(self):
        result = standard_units(self.numbers_array)
        expected = [-1.224744871391589, 0.0, 1.224744871391589]
        self.assertEqual(result, expected)

    def test_standard_units_multiple_args(self):
        result = standard_units(*self.numbers_multiple)
        expected = [-0.6918327146385096, -0.4480579737510855, 1.9896894351231555, -0.44318247893333707, -0.4066162678002234]
        self.assertEqual(result, expected)

    def test_standard_units_zero_sd(self):
        with self.assertRaises(ValueError) as context:
            standard_units(self.identical_numbers)
        self.assertEqual(str(context.exception), "Standard deviation is zero. Standard units are undefined.")

    def test_standard_units_sd_plus(self):
        result = standard_units(self.numbers_sd_plus, sd_plus=True)
        expected = [-0.6215965812833754, -0.5319432282136578, 1.1535398094970335]
        self.assertEqual(result, expected)

    def tearDown(self):
        pass

class TestVariancePlus(unittest.TestCase):

    def setUp(self):
        self.numbers_array = [-1, 0, 1]
        self.numbers_multiple = [1, 2, 3, 4, 5]
        self.single_value = [42]
        self.mixed_types = [1, 2.5, 3, 4.5]
        self.zeroes = [0, 0, 0, 0]

    def test_var_plus_array(self):
        result = var_plus(self.numbers_array)
        expected = 1.0
        self.assertAlmostEqual(result, expected)
        
    def test_var_plus_division_error(self):
        with self.assertRaises(ValueError):
            sd_plus(self.single_value)

    def test_var_plus_multiple_args(self):
        result = var_plus(*self.numbers_multiple)
        expected = 2.5
        self.assertAlmostEqual(result, expected)

    def test_var_plus_mixed_types(self):
        result = var_plus(*self.mixed_types)
        expected = 2.0833333333333335
        self.assertAlmostEqual(result, expected)

    def tearDown(self):
        pass

if __name__ == "__main__":
    unittest.main()
