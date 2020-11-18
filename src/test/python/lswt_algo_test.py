import unittest
import numpy
from datetime import date

import snappy

from src.main.python.lswt_algo import SplitWindowAlgo, MonoWindowAlgo
from snappy import jpy

Float = jpy.get_type('java.lang.Float')
Date = jpy.get_type('java.util.Date')


class TestLSWTAlgo(unittest.TestCase):
    def setUp(self):
        a0 = 0.5
        a1 = 1
        a2 = 1.5
        a3 = 2
        self.algo = SplitWindowAlgo(a0, a1, a2, a3, False)

    def test_algo(self):
        lswt = self.algo.compute_lswt(lower_data=1, upper_data=3, zenith_data=0.5)
        self.assertEqual(-1.499847686457052, lswt)

    def test_algo_shape(self):
        lower_samples = [[1.5, 2.4, 7.3], [1.5, 2.4, 7.3]]
        upper_samples = [[4.7, 5.9, 6.5], [4.7, 5.9, 6.5]]
        zenith_samples = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]

        lower_data = numpy.array(lower_samples, dtype=numpy.float32)
        upper_data = numpy.array(upper_samples, dtype=numpy.float32)
        zenith_data = numpy.array(zenith_samples, dtype=numpy.float32)
        original_shape = lower_data.shape

        lswt = self.algo.compute_lswt(lower_data, upper_data, zenith_data)
        shape = lswt.shape

        self.assertEqual(original_shape, shape)
        self.assertAlmostEqual(-2.79975557, lswt[0][0])
        self.assertAlmostEqual(-2.34973288, lswt[0][1])
        self.assertAlmostEqual(8.99993896, lswt[0][2])

    def test_get_season(self):
        algo = MonoWindowAlgo()
        self.assertEqual(0, algo.get_season(Date(117, 4, 15)))
        self.assertEqual(2, algo.get_season(Date(114, 10, 12)))

    def test_get_season_product(self):
        algo = MonoWindowAlgo()
        source_product = snappy.ProductIO.readProduct(
            "..\\resources\\SEN3_resampled.dim")
        date_java = source_product.getStartTime().getAsDate()
        season = algo.get_season(date_java)
        self.assertEqual(0, season)

    def test_mono_lut_value(self):
        algo = MonoWindowAlgo()
        season = 3  # winter
        height = 800
        zenith = 40
        result = algo.lut.getValue([season, height, zenith])
        self.assertAlmostEqual(1.0936078, result[0], 2)
        self.assertAlmostEqual(-23.638379, result[1], 2)

    def test_mono_lut_values(self):
        algo = MonoWindowAlgo()
        season = numpy.array([3, 3, 3])  # winter
        height = numpy.array([800, 800, 800])
        zenith = numpy.array([40, 45, 50])
        result = algo.lut.getValues([season, height, zenith])
        self.assertAlmostEqual(1.0936078, result[0, 0], 2)
        self.assertAlmostEqual(-23.638379, result[1, 0], 2)
        self.assertAlmostEqual(1.1013171, result[0, 1], 2)
        self.assertAlmostEqual(-25.503619, result[1, 1], 2)
        self.assertAlmostEqual(1.1128374, result[0, 2], 2)
        self.assertAlmostEqual(-28.273845, result[1, 2], 2)

    def test_mono_lut_intermediate(self):
        algo = MonoWindowAlgo()
        season = 3  # winter
        height = 380
        zenith = 40
        result = algo.lut.getValue([season, height, zenith])
        self.assertAlmostEqual(1.1091340, result[0], 2)
        self.assertAlmostEqual(-27.831785, result[1], 2)

    def test_mono_some_nan(self):
        algo = MonoWindowAlgo()
        season = numpy.ones((5, 5))
        height = numpy.ones((5, 5)) * 400
        zenith = numpy.ones((5, 5)) * 40

        data = numpy.array([[1, 2, 3, 4, 5],
                            [Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN],
                            [1, 2, 3, 4, 5],
                            [Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN],
                            [1, 2, 3, 4, 5]])

        result = algo.compute_lswt(data, season, height, zenith)
        self.assertTrue(numpy.isnan(result[1][0]))
        self.assertFalse(numpy.isnan(result[0][0]))

    def test_mono_no_nan(self):
        algo = MonoWindowAlgo()
        season = numpy.ones((5, 5))
        height = numpy.ones((5, 5)) * 400
        zenith = numpy.ones((5, 5)) * 40

        data = numpy.array([[1, 2, 3, 4, 5],
                            [1, 2, 3, 4, 5],
                            [1, 2, 3, 4, 5],
                            [1, 2, 3, 4, 5],
                            [1, 2, 3, 4, 5]])

        result = algo.compute_lswt(data, season, height, zenith)
        self.assertFalse(numpy.isnan(result[1][0]))
        self.assertFalse(numpy.isnan(result[0][0]))
        self.assertTrue(result[0][0] == result[1][0])

    def test_mono_nan(self):
        algo = MonoWindowAlgo()
        season = numpy.ones((5, 5))
        height = numpy.ones((5, 5)) * 400
        zenith = numpy.ones((5, 5)) * 40

        data = numpy.array([[Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN],
                            [Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN],
                            [Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN],
                            [Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN],
                            [Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN]])

        result = algo.compute_lswt(data, season, height, zenith)
        self.assertTrue(numpy.isnan(result[0][0]))

    def test_split_lut_nan(self):
        algo = SplitWindowAlgo(0, 0, 0, 0, True)
        season = numpy.ones((5, 5))
        height = numpy.ones((5, 5)) * 400
        zenith = numpy.ones((5, 5)) * 40

        data = numpy.array([[Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN],
                            [Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN],
                            [Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN],
                            [Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN],
                            [Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN]])

        result = algo.compute_lswt_lut(data, data, zenith, season, height)
        self.assertTrue(numpy.isnan(result[0][0]))

    def test_split_lut_some_nan(self):
        algo = SplitWindowAlgo(0, 0, 0, 0, True)
        season = numpy.ones((5, 5))
        height = numpy.ones((5, 5)) * 400
        zenith = numpy.ones((5, 5)) * 40


        data = numpy.array([[1, 2, 3, 4, 5],
                            [Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN],
                            [1, 2, 3, 4, 5],
                            [Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN],
                            [1, 2, 3, 4, 5]])
        result = algo.compute_lswt_lut(data, data, zenith, season, height)
        self.assertTrue(numpy.isnan(result[1][0]))
        self.assertFalse(numpy.isnan(result[0][0]))

    def test_split_nan(self):
        algo = SplitWindowAlgo(1, 2, 3, 4, False)
        zenith = numpy.ones((5, 5)) * 40

        data = numpy.array([[Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN],
                            [Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN],
                            [Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN],
                            [Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN],
                            [Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN]])

        result = algo.compute_lswt(data, data, zenith)
        self.assertTrue(numpy.isnan(result[0][0]))


if __name__ == '__main__':
    unittest.main()
