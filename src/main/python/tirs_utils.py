import numpy
from snappy import jpy
from wrs import Wrs

# LookupTable = jpy.get_type('org.esa.snap.core.util.math.LookupTable')
from lookup_table import LookupTable

File = jpy.get_type('java.io.File')
Properties = jpy.get_type('java.util.Properties')
IntervalPartition = jpy.get_type('org.esa.snap.core.util.math.IntervalPartition')


def isclose(a, b):
    epsilon = 0.000001
    return abs(a - b) < epsilon


class TirsUtils:
    def radiance_to_kelvin(self, radiance, K1_CONSTANT, K2_CONSTANT):
        kelvin = K2_CONSTANT / numpy.log(K1_CONSTANT / radiance + 1)
        return kelvin

    def radiance_to_reflectance(self, radiance, RADIANCE_MULT, RADIANCE_ADD, REFLECTANCEW_MULT, REFLECTANCE_ADD,
                                solar_elevation):
        reflectance = REFLECTANCEW_MULT * (radiance + RADIANCE_ADD) / RADIANCE_MULT + REFLECTANCE_ADD
        sin_sun_elevation = numpy.sin(numpy.radians(solar_elevation))
        return reflectance / sin_sun_elevation

    def create_wrs_lut(self):
        paths = numpy.array(range(1, 234), dtype=numpy.float64)
        rows = numpy.array(range(1, 249), dtype=numpy.float64)
        # points=['CTR LAT', 'CTR LON', 'UL LAT', 'UL LON', 'UR LAT', 'UR LON', 'LL LAT', 'LL LON', 'LR LAT', 'LR LON']
        points = numpy.array(range(0, 10), dtype=numpy.float64)
        dimensions = [paths, rows, points]
        # file = File('wrs.py')
        # props = Properties()
        # inStream = file.toURI().toURL().openStream()
        # props.clear()
        # props.load(inStream)
        # values = numpy.array(props.getProperty('values').split(','), dtype=numpy.float64)
        wrs = Wrs()
        values = numpy.array(wrs.values, dtype=numpy.float64)
        return LookupTable(values, dimensions)

    def get_corners(self, lut, path, row):
        UL = (lut.getValue([path, row, 3]), lut.getValue([path, row, 2]))
        LL = (lut.getValue([path, row, 7]), lut.getValue([path, row, 6]))

        return [UL, LL]

    def calculate_sat_azimuth(self, corners):
        dx = corners[0][0] - corners[0][1]
        dy = corners[1][0] - corners[1][1]
        m = numpy.sqrt((dx * dx) + (dy * dy))
        azimuth = numpy.rad2deg(numpy.arctan2(dx / m, dy / m))
        if azimuth < 0:
            azimuth = 180 + azimuth
        return azimuth
