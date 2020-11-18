###################################################################################################
# this class checks the quality of the calculated SST according to several tests and returns an array with quality flags
# description              ¦  bit ¦ value ¦
#                          ¦      ¦       ¦
# GROSS IR/Valid VIS (day) ¦  01  ¦     1 ¦
# VIS cloud (day)          ¦  02  ¦     2 ¦
# NIR/VIS ratio (day)      ¦  03  ¦     4 ¦
# IR cloud      (night)    ¦  04  ¦     8 ¦
# Low Stratus   (night)    ¦  05  ¦    16 ¦
# SST LE -5°C OR GE 35°C   ¦  06  ¦    32 ¦
# land/sea mask            ¦  07  ¦    64 ¦
# cloud mask               ¦  08  ¦   128 ¦
# Exclude singel pixels    ¦  09  ¦   256 ¦
# Spatial STDV > 3°K       ¦  10  ¦   512 ¦
# Spatial STDV > 1.5°K     ¦  11  ¦  1024 ¦
# SZA > 55°                ¦  12  ¦  2048 ¦
# SZA > 45°                ¦  13  ¦  4096 ¦
# Glint Angle < 36°        ¦  14  ¦  8192 ¦
# Error in QFlag proc.     ¦  15  ¦ 32767 ¦
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
        self.lswt = numpy.reshape(lswt, (height, width))
        self.rf1 = numpy.reshape(rf1, (height, width))
        self.rf2 = numpy.reshape(rf2, (height, width))
        self.bt3 = numpy.reshape(bt3, (height, width))
        self.bt4 = numpy.reshape(bt4, (height, width))
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
        # GROSS IR TEST / VALID VIS TEST # if the channel 4 temperature is less than  #10° then
        # do not compute a SST (BOM)
        ###############################################################
        quality_flag += numpy.logical_and(
            numpy.logical_or(self.bt4 <= self.qfilter['T4lim'], self.rf1 < self.qfilter['r1lim']), self.sun_za < 90) * 1

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
        # NIGHTTIME IR CLOUD TEST # if a calculated channel 4 temperature based on the channel 5 value
        # (channel 5 temp * 1.0439 # 11.49) differs from the actual channel 4 temperature by more than
        # 1.0° C then do not compute a SST (BoM)
        ###############################################################

        # TODO, GL, this is masking out the LAKES !!!
        syn_ch4 = (self.bt5 * 1.0439) - 11.49
        quality_flag += numpy.logical_and(numpy.absolute(syn_ch4 - self.bt4) >= self.qfilter['ircldtshld'],
                                          self.sun_za > 85) * 8

        ###############################################################
        # NIGHTTIME LOW STRATUS TEST # the difference obtained when subtracting channel 3
        # temperature from channel 5 temperature must be LE #0.6° C. (BoM)
        ###############################################################
        quality_flag += numpy.logical_and((self.bt5 - self.bt3) > self.qfilter['strattshld'], self.bt3 > 100) * 16

        ###############################################################
        # Creating final product, limiting data to #5°C to +35°C
        ###############################################################
        quality_flag += numpy.logical_or(self.lswt <= self.qfilter['lswtLim'][0],
                                         self.lswt >= self.qfilter['lswtLim'][1]) * 32

        ###############################################################
        # Applying Sea and Lake Mask
        ###############################################################
        if self.lwm is not None:
            quality_flag += (self.lwm == 0) * 64
        else:
            quality_flag += 64

        ###############################################################
        # Applying Cloud Mask
        ###############################################################

        # if self.qfilter['cloudTEST'] == 0:
        if self.cmsk is not None:
            if self.sattype == 'AVHRR':
                quality_flag += numpy.logical_or(self.cmsk >= self.qfilter['cmask'], self.cmsk == 0) * 128
            else:
                quality_flag += (self.cmsk != 0) * 128
        else:
            quality_flag += 128

        #####################################
        # EXPERIMENTEL Cloud Mask alternative TEST
        #####################################
        # if self.qfilter['cloudTEST'] == 1:
        #     if numpy.nanmax(self.sun_za) < 80:  # daytime check
        #         cmsk_alt = numpy.logical_not(numpy.logical_or(
        #             numpy.logical_and(numpy.logical_and((self.rf1 + self.rf2) <= 1.2, self.bt5 >= 265),
        #                               (self.rf1 + self.rf2) <= 0.8), self.bt5 >= 285))
        #         cmsk_alt = numpy.logical_or(numpy.logical_or(cmsk_alt, self.cmsk >= self.qfilter['cmask']),
        #                                     self.cmsk == 0)
        #         quality_flag += cmsk_alt * 128
        #     else:
        #         cmsk_alt = numpy.logical_or(
        #             numpy.logical_or(self.bt4 < 243.15, (self.bt5 - self.bt3) > self.qfilter['strattshld']),
        #             numpy.absolute(((self.bt5 * 1.0439) - 11.49) - self.bt4) <= self.qfilter['ircldtshld'])
        #         cmsk_alt = numpy.logical_or(numpy.logical_or(cmsk_alt, self.cmsk >= self.qfilter['cmask']),
        #                                     self.cmsk == 0)
        #         quality_flag += cmsk_alt * 128
        #
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
        quality_flag += (neighbors < 2) * 256

        ###############################################################
        # Valid pixels only those which sourrounding pixels have a SDEV from lower 3°C
        # according to Schwab et al 1999
        ###############################################################
        tmp_lswt = self.lswt
        bad_pix = (good_pix == 0).nonzero()
        if bad_pix[0].size != 0:
            tmp_lswt[bad_pix] = 0
        lswt_stdv = self.spatial_stddev(tmp_lswt, 3)

        quality_flag += numpy.logical_and(numpy.isfinite(lswt_stdv), lswt_stdv > self.qfilter['std1']) * 512
        quality_flag += numpy.logical_and(numpy.isfinite(lswt_stdv), lswt_stdv > self.qfilter['std2']) * 1024

        ###############################################################
        # Use only pixels for VZA LE 45 deg.
        ###############################################################
        quality_flag += (self.sat_za > self.qfilter['vzaLim']) * 2048
        quality_flag += (self.sat_za > self.qfilter['VZA2']) * 4096

        ###############################################################
        # Calculate glint angle
        ###############################################################
        glint = numpy.arcsin(numpy.sin(numpy.radians(self.sat_za)) * numpy.sin(numpy.radians(self.sun_za)) * numpy.cos(
            numpy.radians(self.rel_az)) + numpy.cos(numpy.radians(self.sat_za)) * numpy.cos(numpy.radians(self.sun_za)))
        glint = numpy.degrees(glint)
        quality_flag += (glint < self.qfilter['sunglint']) * 8192
        return quality_flag

    def get_quality_flags(self, flag):
        quality_flags = numpy.ones(flag.shape)
        flag = numpy.uint16(flag)
        init_id = numpy.logical_not(numpy.bitwise_and(flag, 511))
        if numpy.sum(init_id) > 0:
            quality_flags[init_id.nonzero()] = 2  # Q1
            quality_flags += init_id * 2  # georef1 Q2
            q3_id = numpy.bitwise_and(init_id, numpy.logical_not(self.bit(flag, 11)))  # VZA1
            if numpy.sum(q3_id) > 0:
                quality_flags[q3_id.nonzero()] = 8  # Q3
            q4_id = numpy.bitwise_and(q3_id, numpy.logical_not(self.bit(flag, 12)))  # VZA2
            if numpy.sum(q4_id) > 0:
                quality_flags[q4_id.nonzero()] = 16  # Q4
            q5_id = numpy.bitwise_and(q4_id, numpy.logical_not(self.bit(flag, 13)))  # glint
            if numpy.sum(q5_id) > 0:
                quality_flags[q5_id.nonzero()] = 32  # Q5
            q6_id = numpy.bitwise_and(q5_id, numpy.logical_not(self.bit(flag, 10)))  # std2
            if numpy.sum(q6_id) > 0:
                quality_flags[q6_id.nonzero()] = 64  # Q6
        return quality_flags

    # Different version: quality levels have value 1 to 6, like in old processor
    # def get_quality_flags(self, flag):
    #     quality_flags = numpy.zeros(flag.shape)
    #     flag = numpy.uint16(flag)
    #     init_id = numpy.logical_not(numpy.bitwise_and(flag, 511))
    #     if numpy.sum(init_id) > 0:
    #         quality_flags[init_id.nonzero()] = 1  # Q1
    #         quality_flags += init_id  # georef1 Q2
    #         q3_id = numpy.bitwise_and(init_id, numpy.logical_not(self.bit(flag, 11)))  # VZA1
    #         if numpy.sum(q3_id) > 0:
    #             quality_flags[q3_id.nonzero()] += 1  # Q3
    #         q4_id = numpy.bitwise_and(q3_id, numpy.logical_not(self.bit(flag, 12)))  # VZA2
    #         if numpy.sum(q4_id) > 0:
    #             quality_flags[q4_id.nonzero()] += 1  # Q4
    #         q5_id = numpy.bitwise_and(q4_id, numpy.logical_not(self.bit(flag, 13)))  # glint
    #         if numpy.sum(q5_id) > 0:
    #             quality_flags[q5_id.nonzero()] += 1  # Q5
    #         q6_id = numpy.bitwise_and(q5_id, numpy.logical_not(self.bit(flag, 10)))  # std2
    #         if numpy.sum(q6_id) > 0:
    #             quality_flags[q6_id.nonzero()] += 1  # Q6
    #     return quality_flags

    @staticmethod
    def bit(value, shift):
        return numpy.bitwise_and(7, numpy.right_shift(value, shift))

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
