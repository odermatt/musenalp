import unittest
import numpy
from lswt_quality_check import QualityCheck
from numpy import testing
import snappy

from utils import Utils


class TestLSWTQualityCheck(unittest.TestCase):
    def setUp(self):
        lswt = numpy.ones((4, 4)) * 293
        visible = numpy.ones((4, 4)) * 3.0
        nir = numpy.ones((4, 4)) * 0.25
        bt3 = numpy.ones((4, 4)) * 294
        lower_bt = numpy.ones((4, 4)) * 295
        upper_bt = numpy.ones((4, 4)) * 294
        sat_za = numpy.ones((4, 4)) * 20
        sun_za = numpy.ones((4, 4)) * 40
        rel_az = numpy.ones((4, 4)) * 100
        lwm = numpy.ones((4, 4))
        cmsk = numpy.zeros((4, 4))
        self.check = QualityCheck(lswt, visible, nir, bt3, lower_bt, upper_bt, sat_za, sun_za, rel_az, lwm, cmsk, 4, 4,
                                  'AVHRR')

    def test_spatial_stddev(self):
        in_data = [[5, 2, 3, 1], [5, 2, 3, 1], [5, 2, 3, 1], [5, 2, 3, 1]]
        in_data = numpy.array(in_data, dtype=numpy.float32)
        in_data[0, 0:2] = numpy.nan
        stdv = QualityCheck.spatial_stddev(in_data, 3)
        testing.assert_allclose(numpy.array(
            [[numpy.nan, numpy.nan, numpy.nan, 0.99999994], [numpy.nan, numpy.nan, numpy.nan, 0.99999994],
             [1.5, 1.32287566, 0.8660254, 0.99999994], [1.5, 1.32287566, 0.8660254, 0.99999994]]), stdv.tolist())

    def test_get_quality_flag_Q8(self):
        # 0: 0000 0000 0000
        flag = numpy.ones(1) * 0
        result = self.check.get_quality_mask(flag)
        self.assertEqual(8, result[0], 'Quality level should be 8 for 0')

    def test_get_quality_flag_Q7(self):
        # 1024: 0100 0000 0000
        flag = numpy.ones(1) * 1024
        result = self.check.get_quality_mask(flag)
        self.assertEqual(7, result[0], 'Quality level should be 7 for 1024')

    def test_get_quality_flag_Q6(self):
        # 512: 0010 0000 0000
        flag = numpy.ones(1) * 512
        result = self.check.get_quality_mask(flag)
        self.assertEqual(6, result[0], 'Quality level should be 6 for 512')

    def test_get_quality_flag_Q5(self):
        # 2048: 1000 0000 0000
        flag = numpy.ones(1) * 2048
        result = self.check.get_quality_mask(flag)
        self.assertEqual(5, result[0], 'Quality level should be 5 for 2048')

    def test_get_quality_flag_Q4(self):
        # 3072: 1100 0000 0000
        flag = numpy.ones(1) * 3072
        result = self.check.get_quality_mask(flag)
        self.assertEqual(4, result[0], 'Quality level should be 4 for 3072')

    def test_get_quality_flag_Q3(self):
        # 2560: 1010 0000 0000
        flag = numpy.ones(1) * 2560
        result = self.check.get_quality_mask(flag)
        self.assertEqual(3, result[0], 'Quality level should be 3 for 2560')

    def test_get_quality_flag_Q2(self):
        # 1536: 0110 0000 0000
        # 3584: 1110 0000 0000
        for i in [1536, 3584]:
            flag = numpy.ones(1) * i
            result = self.check.get_quality_mask(flag)
            self.assertEqual(2, result[0], 'Quality level should be 2 for ' + str(i))

    def test_get_quality_flag_Q1(self):
        #  256: 0001 0000 0000
        #  768: 0011 0000 0000
        # 1280: 0101 0000 0000
        # 1792: 0111 0000 0000
        # 2304: 1001 0000 0000
        # 2816: 1011 0000 0000
        # 3328: 1101 0000 0000
        # 3840: 1111 0000 0000
        for i in [256, 768, 1280, 1792, 2304, 2816, 3328, 3840]:
            flag = numpy.ones(1) * i
            result = self.check.get_quality_mask(flag)
            self.assertEqual(1, result[0], 'Quality level should be 1 for ' + str(i))

        # 3584: 1110 0000 0000
        #    0: 0000 0000 0000
        # 4095: 1111 1111 1111
        for i in [3584, 0, 4095]:
            flag = numpy.ones(1) * i
            result = self.check.get_quality_mask(flag)
            self.assertNotEqual(1, result[0], 'Quality level should NOT be 1 for ' + str(i))

    def test_get_quality_flag_Q0(self):
        # 255: 00000 1111 1111
        # everything below should be quality level 0
        for i in range(1, 256):
            flag = numpy.ones(1) * i
            result = self.check.get_quality_mask(flag)
            self.assertEqual(0, result[0], 'Quality level should be 0 for ' + str(i))
        # 3855: 1111 0000 1111 -> also Q0
        #  257: 0001 0000 0001
        flag = numpy.ones(1) * 3855
        result = self.check.get_quality_mask(flag)
        self.assertEqual(0, result[0], 'Quality level should be 0 for 3855')

        # 256: 0001 0000 0000 -> not Q0
        flag = numpy.ones(1) * 256
        result = self.check.get_quality_mask(flag)
        self.assertNotEqual(0, result[0], 'Quality level should NOT be 0 for 256')


if __name__ == '__main__':
    unittest.main()
