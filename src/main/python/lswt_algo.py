import numpy

from lookup_table import LookupTable
from lswt_quality_check import QualityCheck
from lswt_quality_check_mono import QualityCheckMono
from utils import Utils
from mono import Mono
from snappy import jpy

Float = jpy.get_type('java.lang.Float')


class SplitWindowAlgo:
    def __init__(self, a0, a1, a2, a3, needs_lut):
        if needs_lut:
            self.has_lut = True
            season = numpy.array(range(0, 4), dtype=numpy.float64)  # spring, summer, autumn, winter
            height = numpy.array([400, 800, 1000])  # height in meter
            zenith = numpy.array(range(5, 56, 5), dtype=numpy.float64)  # 5 - 55deg in 5deg steps
            dimensions = [season, height, zenith]
            #  TODO needs to be loaded from file!
            values = numpy.tile([3.079654, 0.988938, 1.867209, -0.434251],
                                len(season) * len(zenith) * len(height))
            self.lut = LookupTable(values, dimensions, 4)
        else:
            self.has_lut = False
            self.a0 = a0
            self.a1 = a1
            self.a2 = a2
            self.a3 = a3
        self.check = None

    def get_season(self, start_date):
        utils = Utils()
        return utils.get_season(start_date)

    def compute_lswt_lut(self, lower_data, upper_data, zenith_data, season, height):
        lswt = numpy.zeros(numpy.shape(lower_data))
        lswt[numpy.where(numpy.logical_or(numpy.isnan(lower_data), numpy.isnan(upper_data)))] = Float.NaN

        season = season[numpy.where(numpy.logical_and(~numpy.isnan(lower_data), ~numpy.isnan(upper_data)))]
        zenith_data = zenith_data[numpy.where(numpy.logical_and(~numpy.isnan(lower_data), ~numpy.isnan(upper_data)))]
        height = height[numpy.where(numpy.logical_and(~numpy.isnan(lower_data), ~numpy.isnan(upper_data)))]

        size = numpy.shape(season)
        if size != (0,):
            [self.a0, self.a1, self.a2, self.a3] = self.lut.getValues([season, height, zenith_data])
            lswt[
                numpy.where(numpy.logical_and(~numpy.isnan(lower_data), ~numpy.isnan(upper_data)))] = self.compute_lswt(
                lower_data[numpy.where(numpy.logical_and(~numpy.isnan(lower_data), ~numpy.isnan(upper_data)))],
                upper_data[numpy.where(numpy.logical_and(~numpy.isnan(lower_data), ~numpy.isnan(upper_data)))],
                zenith_data)
        return lswt

    def compute_lswt(self, lower_data, upper_data, zenith_data):
        lswt = self.a0 + self.a1 * lower_data + self.a2 * (lower_data - upper_data) + \
               self.a3 * (1 - 1 / (numpy.cos(zenith_data * (numpy.math.pi / 180.0)))) * (lower_data - upper_data)
        return lswt

    def compute_flags(self, lswt, visible, nir, bt3, lower_bt, upper_bt, sat_za, sun_za, rel_az, lwm,
                      cmsk, height, width, sattype):
        self.check = QualityCheck(lswt, visible, nir, bt3, lower_bt, upper_bt, sat_za, sun_za, rel_az, lwm,
                                  cmsk, height, width, sattype)
        return self.check.check_quality()

    def compute_quality_index(self, flags):
        return self.check.get_quality_mask(flags)


class MonoWindowAlgo:
    def __init__(self, a0, a1, needs_lut):
        self.needs_lut = needs_lut
        if not needs_lut:
            self.a0 = a0
            self.a1 = a1
        else:
            season = numpy.array(range(0, 4), dtype=numpy.float64)  # spring, summer, autumn, winter
            height = numpy.array([400, 800, 1000])  # height in meter
            zenith = numpy.array(range(5, 56, 5), dtype=numpy.float64)  # 5 - 55deg in 5deg steps
            dimensions = [season, height, zenith]
            mono_coeff = Mono()
            values = numpy.array(mono_coeff.values, dtype=numpy.float64)
            self.lut = LookupTable(values, dimensions, 2)
            self.check = None

    def get_season(self, start_date):
        utils = Utils()
        return utils.get_season(start_date)

    def compute_lswt(self, data, season, height, zenith_data):
        if not self.needs_lut:
            return self.a0 * data + self.a1
        else:
            lswt = numpy.zeros(numpy.shape(data))
            lswt[numpy.where(numpy.isnan(data))] = Float.NaN

            season = season[numpy.where(~numpy.isnan(data))]
            height = height[numpy.where(~numpy.isnan(data))]
            zenith_data = zenith_data[numpy.where(~numpy.isnan(data))]

            size = numpy.shape(season)
            if size != (0,):
                if len(size) == 2:
                    (shape_x, shape_y) = size
                    result = self.lut.getValues([season[::2, ::2], height[::2, ::2], zenith_data[::2, ::2]])
                    [a0, a1] = result
                    a0 = numpy.repeat(numpy.repeat(a0, 2, axis=0), 2, axis=1)
                    a1 = numpy.repeat(numpy.repeat(a1, 2, axis=0), 2, axis=1)
                    a0 = a0[:shape_x, :shape_y]
                    a1 = a1[:shape_x, :shape_y]
                elif len(size) == 1:
                    (shape_x,) = size
                    result = self.lut.getValues([season[::2], height[::2], zenith_data[::2]])
                    [a0, a1] = result
                    a0 = numpy.repeat(a0, 2)
                    a1 = numpy.repeat(a1, 2)
                    a0 = a0[:shape_x]
                    a1 = a1[:shape_x]

                lswt[numpy.where(~numpy.isnan(data))] = a0 * data[numpy.where(~numpy.isnan(data))] + a1
            return lswt

    def compute_flags(self, lswt, visible, nir, bt, sat_za, sun_za, rel_az, lwm,
                      cmsk, height, width, sattype):
        self.check = QualityCheck(lswt, visible, nir, None, bt, None, sat_za, sun_za, rel_az, lwm,
                                  cmsk, height, width, sattype)
        return self.check.check_quality()

    def compute_quality_index(self, flags):
        return self.check.get_quality_mask(flags)
