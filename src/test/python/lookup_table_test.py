import unittest

import numpy

from lookup_table import LookupTable


class TestLookupTable(unittest.TestCase):
    def setUp(self):
        dimensions = [numpy.array(range(1, 5), dtype=numpy.float64), numpy.array(range(1, 6), dtype=numpy.float64),
                      numpy.array(range(1, 4), dtype=numpy.float64)]
        values = numpy.ones(60)
        self.lut = LookupTable(values, dimensions)

    def test_get_value(self):
        self.assertEqual(1, self.lut.getValue([4, 3, 2]))

    def test_get_values(self):
        coordinates = [numpy.array(range(2, 5), dtype=numpy.float64), numpy.array(range(2, 5), dtype=numpy.float64),
                       numpy.array(range(1, 4), dtype=numpy.float64)]
        result = self.lut.getValues(coordinates)
        self.assertEqual(coordinates[0].shape, result.shape)
        self.assertEqual(1, result[0])

    def test_get_values_mulit_index(self):
        coordinates = [numpy.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]), numpy.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]),
                       numpy.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])]
        result = self.lut.getValues(coordinates)
        self.assertEqual(coordinates[0].shape, result.shape)
        self.assertEqual(1, result[0][0])

    def test_get_value_length_2(self):
        dimensions = [numpy.array(range(1, 5), dtype=numpy.float64), numpy.array(range(1, 6), dtype=numpy.float64),
                      numpy.array(range(1, 4), dtype=numpy.float64)]
        values = numpy.array(range(0, 120))
        self.lut = LookupTable(values, dimensions, 2)
        result = self.lut.getValue([1, 1, 1])
        self.assertEqual((2,), result.shape)
        [a, b] = result
        self.assertEqual(0, a)
        self.assertEqual(1, b)

    def test_get_values_length_2(self):
        dimensions = [numpy.array(range(1, 5), dtype=numpy.float64), numpy.array(range(1, 6), dtype=numpy.float64),
                      numpy.array(range(1, 4), dtype=numpy.float64)]
        values = numpy.array(range(0, 120))
        self.lut = LookupTable(values, dimensions, 2)
        result = self.lut.getValues([numpy.array([1, 1, 1]), numpy.array([1, 1, 1]), numpy.array([1, 1, 1])])
        self.assertEqual((2, 3), result.shape)
        [[a, b, c], [d, e, f]] = result
        self.assertEqual(0, a)
        self.assertEqual(0, b)
        self.assertEqual(0, c)
        self.assertEqual(1, d)
        self.assertEqual(1, e)
        self.assertEqual(1, f)

    def test_get_values_big(self):
        dimensions = [numpy.array(range(1, 5), dtype=numpy.float64), numpy.array(range(1, 6), dtype=numpy.float64),
                      numpy.array(range(1, 4), dtype=numpy.float64)]
        values = numpy.array(range(0, 120))
        self.lut = LookupTable(values, dimensions, 2)
        result = self.lut.getValues([numpy.ones(300000), numpy.ones(300000), numpy.ones(300000)])
        self.assertEqual((2, 300000), result.shape)


if __name__ == '__main__':
    unittest.main()
