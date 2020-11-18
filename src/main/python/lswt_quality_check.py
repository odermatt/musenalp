###################################################################################################
# this class checks the quality of the calculated SST according to several tests and returns an array with quality flags
# description              ¦  bit ¦ value ¦
#
# cloud test               ¦      ¦       ¦
#    cloud mask            ¦  01  ¦     1 ¦
#    VIS cloud (day)       ¦  02  ¦     2 ¦
#    NIR/VIS ratio (day)   ¦  03  ¦     4 ¦
# land/sea mask            ¦  04  ¦     8 ¦
# GROSS IR/Valid VIS (day) ¦  05  ¦    16 ¦
# SST LE -5°C OR GE 35°C   ¦  06  ¦    32 ¦
# Exclude singel pixels    ¦  07  ¦    64 ¦
# SZA > 55°                ¦  08  ¦   128 ¦
# Spatial STDV > 3°K       ¦  09  ¦   256 ¦
# SZA > 45°                ¦  10  ¦   512 ¦
# Spatial STDV > 1.5°K     ¦  11  ¦  1024 ¦
# Glint Angle < 36°        ¦  12  ¦  2048 ¦
###################################################################################################


import numpy
from scipy import ndimage


class QualityCheck:
    def __init__(self, lswt, rf1, rf2, bt3, bt4, bt5, sat_za, sun_za, rel_az, lwm, cmsk, height, width, sattype):
        self.qfilter = {'cmask': 7, 'r1lim': 0.005, 'r2lim': 0.1, 'r21lim': 1.0, 'T4lim': 263.15,
                        'lswtLim': [268.15, 308.15], 'vzaLim': 55., 'georefRel': 1.8, 'georefRel2': 1.8, 'std1': 3.,
                        'std2': 1.5, 'VZA2': 45., 'sunglint': 36., 'ircldtshld': 1.0, 'strattshld': -0.6,
                        'cloudTEST': 0}
        self.org_dim = lswt.shape
        height = int(height)
        width = int(width)
        self.lswt = numpy.reshape(lswt, (height, width))
        self.rf1 = numpy.reshape(rf1, (height, width))
        self.rf2 = numpy.reshape(rf2, (height, width))
        self.bt4 = numpy.reshape(bt4, (height, width))
        if bt5 is not None:
            self.bt5 = numpy.reshape(bt5, (height, width))

        self.sat_za = numpy.reshape(sat_za, (height, width))
        self.sun_za = numpy.reshape(sun_za, (height, width))
        # Relative azimuth angle at the point for the view angle between the ray to the satellite and the ray to the Sun
        self.rel_az = numpy.reshape(rel_az, (height, width))
        if lwm is not None:
            self.lwm = numpy.reshape(lwm, (height, width))
        else:
            self.lwm = None
        if cmsk is not None:
            self.cmsk = numpy.reshape(cmsk, (height, width))
        else:
            self.cmsk = None

        self.height = height
        self.width = width

        self.sattype = sattype

    def check_quality(self):
        quality_flag = numpy.zeros(self.lswt.shape)

        ###############################################################
        # Applying Cloud Mask
        ###############################################################
        if self.cmsk is not None:
            if self.sattype == 'AVHRR':
                quality_flag += numpy.logical_or(self.cmsk >= self.qfilter['cmask'], self.cmsk == 0) * 1
            else:
                quality_flag += (self.cmsk != 0) * 1
        else:
            quality_flag += 1

        ###############################################################
        # VISIBLE CLOUD THRESHOLD TEST # reflectance higher than 0.1 then
        # do not compute a SST (BoM)
        ###############################################################
        quality_flag += numpy.logical_and(self.rf2 >= self.qfilter['r2lim'], self.sun_za < 90) * 2

        ###############################################################
        # NIR/VISIBLE RATIO TEST # ration higher than 1.0 then
        # do not compute a SST (BoM)
        ###############################################################
        if numpy.nanmax(self.sun_za) < 90:
            r21 = self.rf2 / self.rf1
            quality_flag += (r21 > self.qfilter['r21lim']) * 4

        ###############################################################
        # Applying Sea and Lake Mask
        ###############################################################
        if self.lwm is not None:
            quality_flag += (self.lwm == 0) * 8
        else:
            quality_flag += 8

        ###############################################################
        # GROSS IR TEST / VALID VIS TEST # if the channel 4 temperature is less than  #10° then
        # do not compute a SST (BOM)
        ###############################################################
        quality_flag += numpy.logical_and(
            numpy.logical_or(self.bt4 <= self.qfilter['T4lim'], self.rf1 < self.qfilter['r1lim']),
            self.sun_za < 90) * 16

        ###############################################################
        # Creating final product, limiting data to -5°C to +35°C
        ###############################################################
        quality_flag += numpy.logical_or(self.lswt <= self.qfilter['lswtLim'][0],
                                         self.lswt >= self.qfilter['lswtLim'][1]) * 32

        ###############################################################
        # Valid pixels only those which are not completely surrounded by nonvaliddata
        # according to Schwab et al 1999
        ###############################################################
        # the nxn window total not including central pixel
        n = 3
        if quality_flag.ndim == 1:
            raise ArrayDimensionException(quality_flag.ndim)
        else:
            kernel = numpy.ones((n, n))
        # check which pix are good so far
        good_pix = (quality_flag == 0).astype(int)
        # count good neighbors
        neighbors = ndimage.convolve(good_pix, kernel)
        # use only those pix with two or more neighbors
        quality_flag += (neighbors < 2) * 64

        ###############################################################
        # Use only pixels for VZA LE 45 deg.
        ###############################################################
        quality_flag += (self.sat_za > self.qfilter['vzaLim']) * 128
        quality_flag += (self.sat_za > self.qfilter['VZA2']) * 1024

        ###############################################################
        # Valid pixels only those which sourrounding pixels have a SDEV from lower 3°C
        # according to Schwab et al 1999
        ###############################################################
        tmp_lswt = self.lswt
        bad_pix = (good_pix == 0).nonzero()
        if bad_pix[0].size != 0:
            tmp_lswt[bad_pix] = 0
        lswt_stdv = self.spatial_stddev(tmp_lswt, 3)

        quality_flag += numpy.logical_and(numpy.isfinite(lswt_stdv), lswt_stdv > self.qfilter['std1']) * 256
        quality_flag += numpy.logical_and(numpy.isfinite(lswt_stdv), lswt_stdv > self.qfilter['std2']) * 512

        ###############################################################
        # Calculate glint angle
        ###############################################################
        glint = numpy.arcsin(numpy.sin(numpy.radians(self.sat_za)) * numpy.sin(numpy.radians(self.sun_za)) * numpy.cos(
            numpy.radians(self.rel_az)) + numpy.cos(numpy.radians(self.sat_za)) * numpy.cos(numpy.radians(self.sun_za)))
        glint = numpy.degrees(glint)
        quality_flag += (glint < self.qfilter['sunglint']) * 2048
        return quality_flag

    def get_quality_mask(self, flag):
        quality_flags = numpy.zeros(flag.shape)
        flag = numpy.uint16(flag)
        init_id = numpy.logical_not(numpy.bitwise_and(flag, 255))
        if numpy.sum(init_id) > 0:
            quality_flags[numpy.where(numpy.bitwise_and(flag, 511) == 256)] = 1
            quality_flags[numpy.where(numpy.bitwise_and(flag, 2047) == 1536)] = 2
            quality_flags[numpy.where(flag == 2560)] = 3
            quality_flags[numpy.where(flag == 3072)] = 4
            quality_flags[numpy.where(flag == 2048)] = 5
            quality_flags[numpy.where(flag == 512)] = 6
            quality_flags[numpy.where(flag == 1024)] = 7
            quality_flags[numpy.where(flag==0)]=8
        return quality_flags

    @staticmethod
    def bit(value, shift, mask):
        return numpy.bitwise_and(mask, numpy.right_shift(value, shift))

    @staticmethod
    def spatial_stddev(in_data, n):
        numpy.set_printoptions(threshold=numpy.nan)
        # Initialize kernel for convolution operation
        if in_data.ndim == 1:
            raise ArrayDimensionException(in_data.ndim)
        else:
            kernel = numpy.ones((n, n))
        # Compute the square of each data point
        data_sq = ndimage.convolve(numpy.power(in_data, 2), kernel)  # square of each data point
        sub_sum = ndimage.convolve(in_data, kernel)  # sum of each n x n pixel window
        n_valid = ndimage.convolve((in_data != 0).astype(int), kernel)
        # number of valid points in n x n pixel window
        data_me = sub_sum / n_valid  # mean of n x n pixel around each pixel, invalid pixels ignored

        # Compute variance
        data_var = data_sq - 2 * data_me * sub_sum + n_valid * numpy.power(data_me, 2)
        data_var /= (n_valid - 1)

        # compute standarddeviation
        return numpy.sqrt(numpy.absolute(data_var))


class ArrayDimensionException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
