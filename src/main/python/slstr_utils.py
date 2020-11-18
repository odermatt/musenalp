import numpy
import math


class SlstrUtils:
    def radiance_to_reflectance(self, radiance, sun_zenith, irradiance):
        cos_sun_zenith = numpy.cos(numpy.radians(sun_zenith))
        reflectance = math.pi * radiance / (irradiance * cos_sun_zenith)
        return reflectance
