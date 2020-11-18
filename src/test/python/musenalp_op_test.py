import unittest
import snappy
import numpy
from src.main.python.musenalp_op import MuSenALPOp
from snappy import jpy


class TestMuSenALPOp(unittest.TestCase):
    def setUp(self):
        self.op = MuSenALPOp()

    def test_op_avhrr(self):
        context = FakeContext(
            "..\\resources\\20140927_1257_noaa19_avhrr3_rtoabtmp_v03.tif",
            "AVHRR")
        self.op.initialize(context)
        self.assertEqual('py_MuSenALP', context.getTargetProduct().getName())

    def test_op_slstr(self):
        context = FakeContext(
            "..\\resources\\SEN3_resampled.dim",
            "SLSTR")
        self.op.initialize(context)
        self.assertEqual('py_MuSenALP', context.getTargetProduct().getName())

    @unittest.skip("skipping")
    def test_op_tirs(self):
        context = FakeContext(
            "\\\\SYNOLOGYNAS\\data\\Landsat-8\\Switzerland\\LC81960282017046LGN00.tgz", "TIRS")
        self.op.initialize(context)
        self.assertEqual('py_MuSenALP', context.getTargetProduct().getName())

    def test_read_add_metadata(self):
        product = snappy.ProductIO.readProduct("..\\resources\\S2_quality_an.nc")
        metadata_elements = product.getMetadataRoot().getElementNames()


class FakeContext:
    def __init__(self, source, sensor):
        self.source = source
        self.sensor = sensor

    def getSourceProduct(self, source):
        return snappy.ProductIO.readProduct(self.source)

    def getParameter(self, param):
        if (self.sensor == "AVHRR"):
            if len(param) == 2:
                return 1.0
            if (param == 'coef_file'):
                return ''
            if param == 'sattype':
                return 'AVHRR'
            if param == 'algorithm':
                return 'split-window'
            if param == 'refImage':
                return '..\\resources\\rsgb_full_ROI_typePoint.tif'
            if param == 'productLandWaterMask':
                return '..\\resources\\LWM_gen-bod_OSM.tiff'
            if param == 'bandLandWaterMask':
                return 'band_1'
            if param == 'productCloudMask':
                return '..\\resources\\20140927_1257_noaa19_avhrr3_pcmmask-_v02.tif'
            if param == 'bandCloudMask':
                return 'band_1'
            if param == 'coef-file':
                return '..\\resources\\BOD_RTTOV-coefficients_M01_9999_alt0370.dat'
            if param == 'poix':
                return -50841
            if param == 'poiy':
                return 5823054
            if param == 'padding':
                return 45
            if param == 'geometricData':
                return '..\\resources\\20140927_1257_noaa19_avhrr3_viewgeom_v03.tif'
            if param == 'validPixelExpression':
                return ''
        if (self.sensor == "SLSTR"):
            if len(param) == 2:
                return 1.0
            if (param == 'coef_file'):
                return ''
            if param == 'sattype':
                return 'SLSTR'
            if param == 'algorithm':
                return 'split-window'
            if param == 'refImage':
                return ''
            if param == 'productLandWaterMask':
                return ''
            if param == 'bandLandWaterMask':
                return 'confidence_an_inland_water'
            if param == 'productCloudMask':
                return ''
            if param == 'bandCloudMask':
                return 'confidence_an_summary_cloud'
            if param == 'coef-file':
                return '..\\resources\\BOD_RTTOV-coefficients_M01_9999_alt0370.dat'
            if param == 'sourceAncillaryData':
                return ''
            if param == 'cut':
                return False
        if (self.sensor == "TIRS"):
            if len(param) == 2:
                return 1.0
            if param == 'sattype':
                return 'TIRS'
            if param == 'algorithm':
                return 'split-window'
            if param == 'refImage':
                return ''
            if param == 'sourceLandWaterMask':
                return 'water_confidence_mid'
            if param == 'sourceCloudMask':
                return 'cloud_condifence_high'
            if param == 'coef-file':
                return '..\\resources\\BOD_RTTOV-coefficients_M01_9999_alt0370.dat'
            if param == 'sourceAncillaryData':
                return ''
            if param == 'cut':
                return False
        return param

    def setTargetProduct(self, product):
        self.targetProduct = product

    def getTargetProduct(self):
        return self.targetProduct

    def getSourceTile(self, band, rectangle):
        arr = numpy.ones((rectangle.getWidth(), rectangle.getHeight()))
        return band.readPixels(rectangle.getX(), rectangle.getY(), rectangle.getWidth(), rectangle.getHeight(), arr)


if __name__ == '__main__':
    unittest.main()
