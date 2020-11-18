import numpy
import snappy
from snappy import jpy
from datetime import date, datetime


class Utils:
    def get_reference_coordinates(self, tiff_file, poi_x, poi_y, padding):
        tiff = snappy.ProductIO.readProduct(tiff_file)
        tiff_xw = float(tiff.getMetadataRoot().getElement('TIFF Metadata').getElement(
            'GeoTIFF Metadata').getAttributeString('ModelTiePointTag').split(', ')[3])
        tiff_yn = float(tiff.getMetadataRoot().getElement('TIFF Metadata').getElement(
            'GeoTIFF Metadata').getAttributeString('ModelTiePointTag').split(', ')[4])
        tiff_dx = float(tiff.getMetadataRoot().getElement('TIFF Metadata').getElement(
            'GeoTIFF Metadata').getAttributeString('ModelPixelScaleTag').split(', ')[0])
        tiff_dy = float(tiff.getMetadataRoot().getElement('TIFF Metadata').getElement(
            'GeoTIFF Metadata').getAttributeString('ModelPixelScaleTag').split(', ')[1])

        col_lakes = numpy.around((poi_x - tiff_xw) / tiff_dx)
        row_lakes = numpy.around((tiff_yn - poi_y) / tiff_dy)

        y1 = int(max(row_lakes - padding, 0))
        y2 = int(min(row_lakes + padding, tiff.getSceneRasterWidth() - 1))
        x1 = int(max(col_lakes - padding, 0))
        x2 = int(min(col_lakes + padding, tiff.getSceneRasterHeight() - 1))

        return y1, y2, x1, x2

    def cut_product(self, source_product, ranges):
        subset_op = snappy.jpy.get_type('org.esa.snap.core.gpf.common.SubsetOp')
        op = subset_op()
        op.setSourceProduct(source_product)
        op.setRegion(snappy.Rectangle(ranges[2], ranges[0], ranges[3] - ranges[2] + 1, ranges[1] - ranges[0] + 1))
        sub_product = op.getTargetProduct()
        return sub_product

    def calculate_tiff(self, tiff, col_lake, geotag, new_geotag, p, row_lake, tmp_tiff, xw, yn):
        x0_temp = (col_lake - p) * 1000 + xw
        y0_temp = yn - (row_lake - p) * 1000
        new_geotag.append((x0_temp,) + geotag[1:3] + (y0_temp,) + geotag[4:])
        shape = tmp_tiff.shape
        tiff.append(tmp_tiff[int(max(row_lake - p, 0)): int(min(row_lake + p, shape[1] - 1)),
                    int(max(col_lake - p, 0)): int(min(col_lake + p, shape[0] - 1))])

    def read_coef_file(self, filename):
        coef_list = []
        with open(filename) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        for line in content:
            if not line.startswith('#'):
                line.strip()
                whole_list = line.split()
                coef_list = [float(i) for i in whole_list[2:6]]
                break
        return coef_list

    def resample(self, product, reference_band):
        resampling_op = jpy.get_type('org.esa.snap.core.gpf.common.resample.ResamplingOp')
        op = resampling_op()
        op.setSourceProduct(product)
        op.setParameter('referenceBand', reference_band)
        op.setParameter('upsampling', 'Nearest')
        op.setParameter('downsampling', 'Mean')
        op.setParameter('flagDownsampling', 'First')
        op.setParameter('resampleOnPyramidLevels', 'true')
        return op.getTargetProduct()

    def getLandWaterMask(self, product):

        HashMap = snappy.jpy.get_type('java.util.HashMap')
        snappy.GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()

        parameters = HashMap()
        parameters.put('resolution', 150)
        parameters.put('subSamplingFactorX',1)
        parameters.put('subSamplingFactorY',1)
        return snappy.GPF.createProduct('LandWaterMask', parameters, product)


    def get_season(self, start_date):
        dummy_year = 2000  # dummy leap year to allow input X-02-29 (leap day)
        # 0: spring, 1: summer, 2: autumn, 3: winter
        seasons = [(3, (date(dummy_year, 1, 1), date(dummy_year, 3, 20))),
                   (0, (date(dummy_year, 3, 21), date(dummy_year, 6, 20))),
                   (1, (date(dummy_year, 6, 21), date(dummy_year, 9, 22))),
                   (2, (date(dummy_year, 9, 23), date(dummy_year, 12, 20))),
                   (3, (date(dummy_year, 12, 21), date(dummy_year, 12, 31)))]
        start_date = date(start_date.getYear() + 1900, start_date.getMonth() + 1, start_date.getDate())
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        start_date = start_date.replace(year=dummy_year)
        season = next(season for season, (start, end) in seasons
                      if start <= start_date <= end)
        return season


class OperatorException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
