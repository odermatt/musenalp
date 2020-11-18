import unittest

import snappy

from config import Config
from slstr_utils import SlstrUtils
from tirs_utils import TirsUtils
from utils import Utils
import numpy
import os


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.utils = Utils()

    def test_read_coef_file(self):
        # example file
        # # SST Coefficients for M01 from 02.07.2014 to 02.07.2014
        # # Equation: SST = a0 + a1*T4 + a2*(T4 - T5) + a3*(T4-T5)*(1-SEC(ViewZenithAngle))
        # #        DateJul     DateStr            a0            a1            a2            a3          STDV          BIAS             n
        #   2456841.000000  2014070212      1.439901      0.995912      1.951689     -0.706211      0.133513      0.008240         48354
        a0 = 1.439901
        a1 = 0.995912
        a2 = 1.951689
        a3 = -0.706211

        # create file for testing
        coef_file = 'coef.dat'
        f = open(coef_file, 'w')
        f.write('# SST Coefficients for M01 from 02.07.2014 to 02.07.2014\n')
        f.write('# Equation: SST = a0 + a1*T4 + a2*(T4 - T5) + a3*(T4-T5)*(1-SEC(ViewZenithAngle))\n')
        f.write(
            '#        DateJul     DateStr            a0            a1            a2            a3          STDV          BIAS             n\n')
        f.write(
            '  2456841.000000  2014070212      1.439901      0.995912      1.951689     -0.706211      0.133513      0.008240         48354\n')
        f.close()

        coef_list = self.utils.read_coef_file(coef_file)
        self.assertEqual(4, len(coef_list))
        self.assertEqual(a0, coef_list[0])
        self.assertEqual(a1, coef_list[1])
        self.assertEqual(a2, coef_list[2])
        self.assertEqual(a3, coef_list[3])

        os.remove('coef.dat')

    def test_create_wrs_lut(self):
        tirs_utils = TirsUtils()
        lut = tirs_utils.create_wrs_lut()
        ul_long = lut.getValue([196, 28, 3])
        ul_lat = lut.getValue([196, 28, 2])
        ll_long = lut.getValue([196, 28, 7])
        ll_lat = lut.getValue([196, 28, 6])
        self.assertAlmostEqual(4.897, ul_long, 2)
        self.assertAlmostEqual(46.997, ul_lat, 2)
        self.assertAlmostEqual(4.342, ll_long, 2)
        self.assertAlmostEqual(45.406, ll_lat, 2)

    def test_sat_azimuth(self):
        LL = (4.342, 45.406)
        UL = (4.897, 46.997)

        LR = (6.759, 45.053)
        UR = (7.381, 46.635)
        tirs_utils = TirsUtils()
        angle1 = tirs_utils.calculate_sat_azimuth([LL, UL])
        self.assertAlmostEqual(44.286, angle1, places=2)
        angle2 = tirs_utils.calculate_sat_azimuth([LR, UR])
        self.assertAlmostEqual(angle1, angle2, places=1)

    def test_get_corners(self):
        LL = (4.342, 45.406)
        UL = (4.897, 46.997)
        tirs_utils = TirsUtils()
        lut = tirs_utils.create_wrs_lut()
        corners = tirs_utils.get_corners(lut, 196, 28)
        self.assertAlmostEqual(UL[0], corners[0][0], 2)
        self.assertAlmostEqual(UL[1], corners[0][1], 2)
        self.assertAlmostEqual(LL[0], corners[1][0], 2)
        self.assertAlmostEqual(LL[1], corners[1][1], 2)

    def test_get_reference_coordinates(self):
        tiff = "..\\resources\\rsgb_full_ROI_typePoint.tif"
        poi_x = -50841
        poi_y = 5823054
        padding = 45
        (y1, y2, x1, x2) = self.utils.get_reference_coordinates(tiff, poi_x, poi_y, padding)
        self.assertEqual(y1, 2732)
        self.assertEqual(y2, 2822)
        self.assertEqual(x1, 1604)
        self.assertEqual(x2, 1694)

    def test_land_water_mask(self):
        utils = Utils()
        source_product = snappy.ProductIO.readProduct(
            '..\\resources\\SEN3_resampled.dim')
        target_product = utils.getLandWaterMask(source_product)
        self.assertIsNotNone(target_product)


if __name__ == '__main__':
    unittest.main()
