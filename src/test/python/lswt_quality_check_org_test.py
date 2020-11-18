import unittest
import numpy
import lswt_algo
import snappy

from config import Config
from utils import Utils


class TestLSWTQualityOrgCheck(unittest.TestCase):
    def test_check_quality_org_data_avhrr(self):
        utils = Utils()
        area_range = (2732, 2822, 1605, 1695)
        lwm = snappy.ProductIO.readProduct(
            '..\\resources\\LWM_gen-bod_OSM.tiff')
        cmsk = snappy.ProductIO.readProduct(
            '..\\resources\\20140927_1257_noaa19_avhrr3_pcmmask-_v02.tif')
        lwm = utils.cut_product(lwm, area_range)
        cmsk = utils.cut_product(cmsk, area_range)
        lwm_band = lwm.getBand('band_1')
        lwm = numpy.zeros((90, 90), numpy.float32)
        lwm_band.readPixels(0, 0, 90, 90, lwm)
        cmsk_band = cmsk.getBand('band_1')
        cmsk = numpy.zeros((90, 90), numpy.float32)
        cmsk_band.readPixels(0, 0, 90, 90, cmsk)

        lswt = snappy.ProductIO.readProduct(
            "..\\resources\\20140927_1257_noaa19_avhrr3_lswtBOD-_v03.tiff")
        lswt_band = lswt.getBand('band_1')
        lswt = numpy.zeros((90, 90), numpy.float32)
        lswt_band.readPixels(0, 0, 90, 90, lswt)
        lswt *= 0.01

        geom = snappy.ProductIO.readProduct(
            "..\\resources\\20140927_1257_noaa19_avhrr3_viewgeom_v03.tif")
        geom = utils.cut_product(geom, area_range)
        sat_za_band = geom.getBand('band_1')
        sat_za = numpy.zeros((90, 90), numpy.float32)
        sat_za_band.readPixels(0, 0, 90, 90, sat_za)
        sat_za *= 0.01
        sun_za_band = geom.getBand('band_2')
        sun_za = numpy.zeros((90, 90), numpy.float32)
        sun_za_band.readPixels(0, 0, 90, 90, sun_za)
        sun_za *= 0.01
        rel_az_band = geom.getBand('band_3')
        rel_az = numpy.zeros((90, 90), numpy.float32)
        rel_az_band.readPixels(0, 0, 90, 90, rel_az)
        rel_az *= 0.01

        source_product = snappy.ProductIO.readProduct(
            "..\\resources\\20140927_1257_noaa19_avhrr3_rtoabtmp_v03.tif")
        source_product = utils.cut_product(source_product, area_range)
        lower_band = source_product.getBand('band_4')
        lower = numpy.zeros((90, 90), numpy.float32)
        lower_band.readPixels(0, 0, 90, 90, lower)
        lower *= 0.1
        upper_band = source_product.getBand('band_5')
        upper = numpy.zeros((90, 90), numpy.float32)
        upper_band.readPixels(0, 0, 90, 90, upper)
        upper *= 0.1
        visible_band = source_product.getBand('band_1')
        visible = numpy.zeros((90, 90), numpy.float32)
        visible_band.readPixels(0, 0, 90, 90, visible)
        visible *= 0.001
        nir_band = source_product.getBand('band_2')
        nir = numpy.zeros((90, 90), numpy.float32)
        nir_band.readPixels(0, 0, 90, 90, nir)
        nir *= 0.001
        bt3_band = source_product.getBand('band_3')
        bt3 = numpy.zeros((90, 90), numpy.float32)
        bt3_band.readPixels(0, 0, 90, 90, bt3)
        bt3 *= 0.1

        self.algo = lswt_algo.SplitWindowAlgo(0.0, 0.0, 0.0, 0.0)
        lswt_flags = self.algo.compute_flags(numpy.array(lswt, copy=True), visible, nir, bt3, lower, upper, sat_za,
                                             sun_za, rel_az, lwm, cmsk, 90, 90, 'AVHRR')

        self.assertFalse((lswt_flags == 1).all(), 'quality should not be one (Q0) everywhere')

    def test_check_quality_org_data_slstr(self):
        utils = Utils()
        area_range = (297, 356, 1124, 1204)
        config = Config('SLSTR').get_conf()
        source_product = snappy.ProductIO.readProduct(
            '..\\resources\\S3A_SL_1_RBT____20170407T100909_20170407T101209_20170411T083956_0179_016' +
            '_179_2160_LN2_O_NT_002.SEN3\\xfdumanifest.xml')
        source_product = utils.resample(source_product, 'S8_BT_in')
        source_product = utils.cut_product(source_product, area_range)

        lower_band = source_product.getBand(config['bands']['lower_band'])
        lower = numpy.zeros((81, 60), numpy.float32)
        lower_band.readPixels(0, 0, 81, 60, lower)
        upper_band = source_product.getBand(config['bands']['upper_band'])
        upper = numpy.zeros((81, 60), numpy.float32)
        upper_band.readPixels(0, 0, 81, 60, upper)
        visible_band = source_product.getBand(config['bands']['visible_band'])
        visible = numpy.zeros((81, 60), numpy.float32)
        visible_band.readPixels(0, 0, 81, 60, visible)
        nir_band = source_product.getBand(config['bands']['nir_band'])
        nir = numpy.zeros((81, 60), numpy.float32)
        nir_band.readPixels(0, 0, 81, 60, nir)
        bt3_band = source_product.getBand(config['bands']['lowest_band'])
        bt3 = numpy.zeros((81, 60), numpy.float32)
        bt3_band.readPixels(0, 0, 81, 60, bt3)

        cmsk = numpy.zeros((81, 60), numpy.float32)

        lwm = numpy.zeros((81, 60), numpy.float32)

        sat_za_band = source_product.getTiePointGrid(config['angles']['sat_za'])
        sat_za = numpy.zeros((81, 60), numpy.float32)
        sat_za_band.readPixels(0, 0, 81, 60, sat_za)
        sun_za_band = source_product.getTiePointGrid(config['angles']['sun_za'])
        sun_za = numpy.zeros((81, 60), numpy.float32)
        sun_za_band.readPixels(0, 0, 81, 60, sun_za)
        sat_az_band = source_product.getTiePointGrid(config['angles']['sat_az'])
        sat_az = numpy.zeros((81, 60), numpy.float32)
        sat_az_band.readPixels(0, 0, 81, 60, sat_az)
        sun_az_band = source_product.getTiePointGrid(config['angles']['sun_az'])
        sun_az = numpy.zeros((81, 60), numpy.float32)
        sun_az_band.readPixels(0, 0, 81, 60, sun_az)
        rel_az = numpy.absolute(sat_az - sun_az)

        algo = lswt_algo.SplitWindowAlgo(4.783855, 0.982082, 1.760306, -0.414053)
        lswt = algo.compute_lswt(lower, upper, sat_za)

        self.algo = lswt_algo.SplitWindowAlgo(0.0, 0.0, 0.0, 0.0)
        lswt_flags = self.algo.compute_flags(numpy.array(lswt, copy=True), visible, nir, bt3, lower, upper, sat_za,
                                             sun_za, rel_az, lwm, cmsk, 81, 60, 'SLSTR')

        self.assertFalse((lswt_flags == 1).all(), 'quality should not be one (Q0) everywhere')


if __name__ == '__main__':
    unittest.main()
