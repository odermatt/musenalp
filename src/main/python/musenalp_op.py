import lswt_algo
import numpy
import snappy
from snappy import FlagCoding
from datetime import datetime

from config import Config
from tirs_utils import TirsUtils
from utils import Utils
import configparser
import os
from slstr_utils import SlstrUtils

from snappy import jpy

Float = jpy.get_type('java.lang.Float')
Color = jpy.get_type('java.awt.Color')
IndexCoding = jpy.get_type('org.esa.snap.core.datamodel.IndexCoding')
DateFormat = jpy.get_type('java.text.DateFormat')
Date = jpy.get_type('java.util.Date')
SimpleDateFormat = jpy.get_type('java.text.SimpleDateFormat')


class MuSenALPOp:
    def __init__(self):
        self.source_product = None
        self.lower_band = None
        self.upper_band = None
        self.lswt_flags_band = None
        self.lswt_quality_index = None
        self.lswt_band = None
        self.visible_band = None
        self.nir_band = None
        self.bt3_band = None
        self.sat_za = None
        self.sun_za = None
        self.rel_az = None
        self.sat_az = None
        self.sun_az = None
        self.lwm_band = None
        self.ref_band_org = None
        self.cmsk_band = None
        self.ref_band = None
        self.mono_band = None
        self.valid_pixel_band = None

        self.algorithm_parameter = ''
        self.algo = None
        self.a0 = 0.0
        self.a1 = 0.0
        self.a2 = 0.0
        self.a3 = 0.0
        self.a0_mono = 0.0
        self.a1_mono = 0.0
        self.needs_lut = False
        self.season = None
        self.lat_band = None
        self.lon_band = None

        self.ref_image = None
        self.cut = False
        self.poi_x = 0
        self.poi_y = 0
        self.padding = 0
        self.range = (0, 0, 0, 0)

        self.solar_irr_s2 = 0.0
        self.solar_irr_s3 = 0.0
        self.azimuth = 0.0

        self.utils = Utils()
        self.width = 0
        self.height = 0

        self.sattype = ''
        self.file_location = None

        self.config = configparser.ConfigParser()

    def initialize(self, context):
        self.sattype = context.getParameter('sattype')
        self.get_config()
        self.cut = context.getParameter('cut')
        self.get_cut_param(context)

        self.algorithm_parameter = context.getParameter('algorithm')
        self.check_correct_algoithm(self.config['algorithms']['lswt-algorithm'])

        if self.algorithm_parameter == 'split-window':
            self.get_split_window_coeff(context)
        elif self.algorithm_parameter == 'mono-window':
            self.get_mono_window_coeff(context)
        self.source_product = context.getSourceProduct('source')
        if self.needs_lut:
            self.get_lut_info(context)
        self.file_location = self.source_product.getFileLocation()
        if self.file_location is not None:
            self.file_location = self.file_location.getAbsolutePath()
        if self.sattype == 'SLSTR' or self.sattype == 'TIRS':
            self.source_product = self.utils.resample(self.source_product, self.config['bands']['lower_band'])
        if self.cut:
            self.source_product = self.utils.cut_product(self.source_product, self.range)
        self.width = self.source_product.getSceneRasterWidth()
        self.height = self.source_product.getSceneRasterHeight()
        if self.algorithm_parameter == 'split-window':
            self.get_split_window_bands(context)
        else:
            self.get_mono_window_bands(context)

        ref_image = self.get_reg_image(context)
        self.get_land_water_mask(context)
        self.get_cloud_mask(context)
        self.get_valid_pixel_band(context)
        self.get_geo_data(context)

        self.configure_target_product(context, ref_image)

    def computeTileStack(self, context, target_tiles, target_rectangle):
        lswt = None
        lswt_flags = None
        lswt_index = None
        lwm_data = self.get_lwm_data(context, target_rectangle)
        cmsk_data = self.get_cmsk_data(context, target_rectangle)
        valid_data = self.get_valid_data(context, target_rectangle)

        if self.algorithm_parameter == 'split-window':
            bt3_data, data, nir_data, upper_data, visible_data = self.get_split_window_band_data(context,
                                                                                                 target_rectangle)
            if valid_data is not None:
                data[numpy.where(valid_data == 0)] = Float.NaN
                upper_data[numpy.where(valid_data == 0)] = Float.NaN
            if context.getParameter('maskBeforeCalculation'):
                if lwm_data is not None:
                    data[numpy.where(lwm_data == 0)] = Float.NaN
                    upper_data[numpy.where(lwm_data == 0)] = Float.NaN
                if cmsk_data is not None:
                    if self.sattype == 'AVHRR':
                        data[numpy.where(numpy.logical_or(cmsk_data >= 7, cmsk_data == 0))] = Float.NaN
                        upper_data[numpy.where(numpy.logical_or(cmsk_data >= 7, cmsk_data == 0))] = Float.NaN
                    else:
                        data[numpy.where(cmsk_data != 0)] = Float.NaN
                        upper_data[numpy.where(cmsk_data != 0)] = Float.NaN

            rel_az_data = None
            sat_za_data = None
            sun_za_data = None
            data, nir_data, rel_az_data, sat_za_data, sun_za_data, upper_data, visible_data = self.get_geom_data(
                context, data, nir_data, rel_az_data, sat_za_data, sun_za_data, target_rectangle, upper_data,
                visible_data)
            if self.needs_lut:
                height_data, season_data = self.get_lut_data(context, data, target_rectangle)
                lswt = self.algo.compute_lswt_lut(data, upper_data, sat_za_data, season_data, height_data)
            else:
                lswt = self.algo.compute_lswt(data, upper_data, sat_za_data)

            lswt_flags = self.algo.compute_flags(numpy.array(lswt, copy=True), visible_data, nir_data, bt3_data,
                                                 data, upper_data, sat_za_data, sun_za_data, rel_az_data, lwm_data,
                                                 cmsk_data, target_rectangle.getHeight(), target_rectangle.getWidth(),
                                                 self.sattype)

            lswt_index = self.algo.compute_quality_index(lswt_flags)

        elif self.algorithm_parameter == 'mono-window':
            data, visible_data, nir_data = self.get_mono_window_band_data(context, target_rectangle)
            rel_az_data = None
            sat_za_data = None
            sun_za_data = None
            if self.sattype == 'AVHRR':
                data, rel_az_data, sat_za_data, sun_za_data, upper_data = self.get_geom_data_avhrr(context, data,
                                                                                                   rel_az_data,
                                                                                                   sun_za_data,
                                                                                                   target_rectangle,
                                                                                                   None)
            if self.sattype == 'SLSTR':
                nir_data, rel_az_data, sat_za_data, sun_za_data, visible_data = self.get_geom_data_slstr(context,
                                                                                                         nir_data,
                                                                                                         target_rectangle,
                                                                                                         visible_data)
            if self.sattype == 'AATSR':
                rel_az_data, sat_za_data, sun_za_data = self.get_geom_data_aatsr(context,
                                                                                 target_rectangle)
            if self.sattype == 'VIIRS':
                rel_az_data, sat_za_data, sun_za_data = self.get_geom_data_viirs(context, target_rectangle)
            height_data = 0.0
            season_data = 0
            if self.needs_lut:
                height_data, season_data = self.get_lut_data(context, data, target_rectangle)
            if valid_data is not None:
                data[numpy.where(valid_data == 0)] = Float.NaN
            if context.getParameter('maskBeforeCalculation'):
                if lwm_data is not None:
                    data[numpy.where(lwm_data == 0)] = Float.NaN
                if cmsk_data is not None:
                    if self.sattype == 'AVHRR':
                        data[numpy.where(numpy.logical_or(cmsk_data >= 7, cmsk_data == 0))] = Float.NaN
                    else:
                        data[numpy.where(cmsk_data != 0)] = Float.NaN
            lswt = self.algo.compute_lswt(data, season_data, height_data, sat_za_data)

            lswt_flags = self.algo.compute_flags(numpy.array(lswt, copy=True), visible_data, nir_data, data,
                                                 sat_za_data, sun_za_data, rel_az_data, lwm_data, cmsk_data,
                                                 target_rectangle.getHeight(), target_rectangle.getWidth(),
                                                 self.sattype)
            lswt_index = self.algo.compute_quality_index(lswt_flags)

        self.set_target_tiles(lswt, lswt_flags, lswt_index, target_tiles)

    def check_correct_algoithm(self, valid_algorithms):
        if valid_algorithms != 'both' and valid_algorithms != self.algorithm_parameter:
            raise AlgorithmException(
                'the sensor ' + self.sattype + ' cannot be run with the algorithm ' + self.algorithm_parameter)

    def set_target_tiles(self, lswt, lswt_flags, lswt_index, target_tiles):
        lswt_tile = target_tiles.get(self.lswt_band)
        lswt_flags_tile = target_tiles.get(self.lswt_flags_band)
        lswt_quality_tile = target_tiles.get(self.lswt_quality_index)
        # Set the result to the target tiles
        lswt_tile.setSamples(lswt)
        lswt_flags_tile.setSamples(lswt_flags)
        lswt_quality_tile.setSamples(lswt_index)

    def get_mono_window_band_data(self, context, target_rectangle):
        data_tile = context.getSourceTile(self.mono_band, target_rectangle)
        data_samples = data_tile.getSamplesFloat()
        data = numpy.array(data_samples, dtype=numpy.float32) * float(
            self.config['scale_factors']['scale_factor_K'])
        visible_tile = context.getSourceTile(self.visible_band, target_rectangle)
        nir_tile = context.getSourceTile(self.nir_band, target_rectangle)
        visible_sample = visible_tile.getSamplesFloat()
        nir_sample = nir_tile.getSamplesFloat()
        visible_data = numpy.array(visible_sample, dtype=numpy.float32) * float(
            self.config['scale_factors']['scale_rf'])
        nir_data = numpy.array(nir_sample, dtype=numpy.float32) * float(self.config['scale_factors']['scale_rf'])
        return data, visible_data, nir_data

    def get_geom_data(self, context, data, nir_data, rel_az_data, sat_za_data, sun_za_data, target_rectangle,
                      upper_data, visible_data):
        if self.sattype == 'AVHRR':
            data, rel_az_data, sat_za_data, sun_za_data, upper_data = self.get_geom_data_avhrr(context, data,
                                                                                               rel_az_data,
                                                                                               sun_za_data,
                                                                                               target_rectangle,
                                                                                               upper_data)
        if self.sattype == 'SLSTR':
            nir_data, rel_az_data, sat_za_data, sun_za_data, visible_data = self.get_geom_data_slstr(context,
                                                                                                     nir_data,
                                                                                                     target_rectangle,
                                                                                                     visible_data)
        if self.sattype == 'TIRS':
            data, nir_data, rel_az_data, sat_za_data, sun_za_data, upper_data, visible_data = self.get_geom_data_tiirs(
                data, nir_data, sat_za_data, sun_za_data, upper_data, visible_data)

        if self.sattype == 'AATSR':
            rel_az_data, sat_za_data, sun_za_data = self.get_geom_data_aatsr(context,
                                                                             target_rectangle)

        if self.sattype == 'VIIRS':
            rel_az_data, sat_za_data, sun_za_data = self.get_geom_data_viirs(context,
                                                                             target_rectangle)

        return data, nir_data, rel_az_data, sat_za_data, sun_za_data, upper_data, visible_data

    def get_lut_data(self, context, data, target_rectangle):
        if self.sattype == 'SLSTR':
            height_tile = context.getSourceTile(self.height_band, target_rectangle)
            height_samples = height_tile.getSamplesFloat()
            height_data = numpy.array(height_samples, dtype=numpy.float32)
        else:
            height_data = numpy.ones(data.shape) * self.elevation
        season_data = numpy.ones(data.shape) * self.season
        return height_data, season_data

    def get_geom_data_tiirs(self, data, nir_data, sat_za_data, sun_za_data, upper_data, visible_data):
        tirs_utils = TirsUtils()
        data = tirs_utils.radiance_to_kelvin(data, self.source_product.getMetadataRoot().getElement(
            'L1_METADATA_FILE').getElement('TIRS_THERMAL_CONSTANTS').getAttributeDouble(
            'K1_CONSTANT_BAND_10'),
                                             self.source_product.getMetadataRoot().getElement(
                                                 'L1_METADATA_FILE').getElement(
                                                 'TIRS_THERMAL_CONSTANTS').getAttributeDouble(
                                                 'K2_CONSTANT_BAND_10'))
        upper_data = tirs_utils.radiance_to_kelvin(upper_data,
                                                   self.source_product.getMetadataRoot().getElement(
                                                       'L1_METADATA_FILE').getElement(
                                                       'TIRS_THERMAL_CONSTANTS').getAttributeDouble(
                                                       'K1_CONSTANT_BAND_10'),
                                                   self.source_product.getMetadataRoot().getElement(
                                                       'L1_METADATA_FILE').getElement(
                                                       'TIRS_THERMAL_CONSTANTS').getAttributeDouble(
                                                       'K2_CONSTANT_BAND_10'))
        sat_za_data = numpy.ones(data.shape) * self.source_product.getMetadataRoot().getElement(
            'L1_METADATA_FILE').getElement('IMAGE_ATTRIBUTES').getAttributeDouble(
            self.config['angles']['sat_za'])
        sun_za_data = 90 - numpy.ones(data.shape) * self.source_product.getMetadataRoot().getElement(
            'L1_METADATA_FILE').getElement('IMAGE_ATTRIBUTES').getAttributeDouble(
            self.config['angles']['sun_za'])
        sat_az_data = numpy.ones(data.shape) * self.azimuth
        sun_az_data = numpy.ones(data.shape) * self.source_product.getMetadataRoot().getElement(
            'L1_METADATA_FILE').getElement('IMAGE_ATTRIBUTES').getAttributeDouble(
            self.config['angles']['sun_az'])
        rel_az_data = numpy.absolute(sat_az_data - sun_az_data
                                     * float(self.config['scale_factors']['scale_factor_deg']))
        rel_az_data = rel_az_data * (rel_az_data <= 180) + (360 - rel_az_data) * (rel_az_data > 180)
        visible_data = tirs_utils.radiance_to_reflectance(visible_data,
                                                          self.source_product.getMetadataRoot().getElement(
                                                              'L1_METADATA_FILE').getElement(
                                                              'RADIOMETRIC_RESCALING').getAttributeDouble(
                                                              'RADIANCE_MULT_BAND_4'),
                                                          self.source_product.getMetadataRoot().getElement(
                                                              'L1_METADATA_FILE').getElement(
                                                              'RADIOMETRIC_RESCALING').getAttributeDouble(
                                                              'RADIANCE_ADD_BAND_4'),
                                                          self.source_product.getMetadataRoot().getElement(
                                                              'L1_METADATA_FILE').getElement(
                                                              'RADIOMETRIC_RESCALING').getAttributeDouble(
                                                              'REFLECTANCE_MULT_BAND_4'),
                                                          self.source_product.getMetadataRoot().getElement(
                                                              'L1_METADATA_FILE').getElement(
                                                              'RADIOMETRIC_RESCALING').getAttributeDouble(
                                                              'REFLECTANCE_ADD_BAND_4'),
                                                          self.source_product.getMetadataRoot().getElement(
                                                              'L1_METADATA_FILE').getElement(
                                                              'IMAGE_ATTRIBUTES').getAttributeDouble(
                                                              self.config['angles']['sun_za']))
        nir_data = tirs_utils.radiance_to_reflectance(nir_data,
                                                      self.source_product.getMetadataRoot().getElement(
                                                          'L1_METADATA_FILE').getElement(
                                                          'RADIOMETRIC_RESCALING').getAttributeDouble(
                                                          'RADIANCE_MULT_BAND_5'),
                                                      self.source_product.getMetadataRoot().getElement(
                                                          'L1_METADATA_FILE').getElement(
                                                          'RADIOMETRIC_RESCALING').getAttributeDouble(
                                                          'RADIANCE_ADD_BAND_5'),
                                                      self.source_product.getMetadataRoot().getElement(
                                                          'L1_METADATA_FILE').getElement(
                                                          'RADIOMETRIC_RESCALING').getAttributeDouble(
                                                          'REFLECTANCE_MULT_BAND_5'),
                                                      self.source_product.getMetadataRoot().getElement(
                                                          'L1_METADATA_FILE').getElement(
                                                          'RADIOMETRIC_RESCALING').getAttributeDouble(
                                                          'REFLECTANCE_ADD_BAND_5'),
                                                      self.source_product.getMetadataRoot().getElement(
                                                          'L1_METADATA_FILE').getElement(
                                                          'IMAGE_ATTRIBUTES').getAttributeDouble(
                                                          self.config['angles']['sun_za']))
        return data, nir_data, rel_az_data, sat_za_data, sun_za_data, upper_data, visible_data

    def get_geom_data_slstr(self, context, nir_data, target_rectangle, visible_data):
        sat_za_tile = context.getSourceTile(self.sat_za, target_rectangle)
        sun_za_tile = context.getSourceTile(self.sun_za, target_rectangle)
        sat_za_samples = sat_za_tile.getSamplesFloat()
        sun_za_samples = sun_za_tile.getSamplesFloat()
        sat_za_data = numpy.array(sat_za_samples, dtype=numpy.float32) * float(
            self.config['scale_factors']['scale_factor_deg'])
        sun_za_data = numpy.array(sun_za_samples, dtype=numpy.float32) * float(
            self.config['scale_factors']['scale_factor_deg'])
        sat_az_tile = context.getSourceTile(self.sat_az, target_rectangle)
        sat_az_samples = sat_az_tile.getSamplesFloat()
        sun_az_tile = context.getSourceTile(self.sun_az, target_rectangle)
        sun_az_samples = sun_az_tile.getSamplesFloat()
        rel_az_data = numpy.absolute(numpy.array(sat_az_samples, dtype=numpy.float32)
                                     * float(self.config['scale_factors']['scale_factor_deg']) -
                                     numpy.array(sun_az_samples, dtype=numpy.float32)
                                     * float(self.config['scale_factors']['scale_factor_deg']))
        rel_az_data = rel_az_data * (rel_az_data <= 180) + (360 - rel_az_data) * (rel_az_data > 180)
        slstr_utils = SlstrUtils()
        visible_data = slstr_utils.radiance_to_reflectance(visible_data, sun_za_data, self.solar_irr_s2)
        nir_data = slstr_utils.radiance_to_reflectance(nir_data, sun_za_data, self.solar_irr_s3)
        return nir_data, rel_az_data, sat_za_data, sun_za_data, visible_data

    def get_geom_data_aatsr(self, context, target_rectangle):
        sat_za_tile = context.getSourceTile(self.sat_za, target_rectangle)
        sun_za_tile = context.getSourceTile(self.sun_za, target_rectangle)
        sat_za_samples = sat_za_tile.getSamplesFloat()
        sun_za_samples = sun_za_tile.getSamplesFloat()
        sat_za_data = 90 - numpy.array(sat_za_samples, dtype=numpy.float32) * float(
            self.config['scale_factors']['scale_factor_deg'])
        sun_za_data = 90 - numpy.array(sun_za_samples, dtype=numpy.float32) * float(
            self.config['scale_factors']['scale_factor_deg'])
        sat_az_tile = context.getSourceTile(self.sat_az, target_rectangle)
        sat_az_samples = sat_az_tile.getSamplesFloat()
        sun_az_tile = context.getSourceTile(self.sun_az, target_rectangle)
        sun_az_samples = sun_az_tile.getSamplesFloat()
        rel_az_data = numpy.absolute(numpy.array(sat_az_samples, dtype=numpy.float32)
                                     * float(self.config['scale_factors']['scale_factor_deg']) -
                                     numpy.array(sun_az_samples, dtype=numpy.float32)
                                     * float(self.config['scale_factors']['scale_factor_deg']))
        rel_az_data = rel_az_data * (rel_az_data <= 180) + (360 - rel_az_data) * (rel_az_data > 180)
        return rel_az_data, sat_za_data, sun_za_data

    def get_geom_data_viirs(self, context, target_rectangle):
        sat_za_tile = context.getSourceTile(self.sat_za, target_rectangle)
        sun_za_tile = context.getSourceTile(self.sun_za, target_rectangle)
        sat_za_samples = sat_za_tile.getSamplesFloat()
        sun_za_samples = sun_za_tile.getSamplesFloat()
        sat_za_data = numpy.array(sat_za_samples, dtype=numpy.float32) * float(
            self.config['scale_factors']['scale_factor_deg'])
        sun_za_data = numpy.array(sun_za_samples, dtype=numpy.float32) * float(
            self.config['scale_factors']['scale_factor_deg'])
        sat_az_tile = context.getSourceTile(self.sat_az, target_rectangle)
        sat_az_samples = sat_az_tile.getSamplesFloat()
        sun_az_tile = context.getSourceTile(self.sun_az, target_rectangle)
        sun_az_samples = sun_az_tile.getSamplesFloat()
        rel_az_data = numpy.absolute(numpy.array(sat_az_samples, dtype=numpy.float32)
                                     * float(self.config['scale_factors']['scale_factor_deg']) -
                                     numpy.array(sun_az_samples, dtype=numpy.float32)
                                     * float(self.config['scale_factors']['scale_factor_deg']))
        rel_az_data = rel_az_data * (rel_az_data <= 180) + (360 - rel_az_data) * (rel_az_data > 180)
        return rel_az_data, sat_za_data, sun_za_data

    def get_geom_data_avhrr(self, context, data, rel_az_data, sun_za_data, target_rectangle, upper_data):
        nan_array = numpy.ones(data.shape)
        nan_array[:] = Float.NaN
        data = numpy.where(data == 6553.5, nan_array, data)
        if upper_data is not None:
            upper_data = numpy.where(data == 6553.5, nan_array, upper_data)
        sat_za_tile = context.getSourceTile(self.sat_za, target_rectangle)
        sun_za_tile = context.getSourceTile(self.sun_za, target_rectangle)
        sat_za_samples = sat_za_tile.getSamplesFloat()
        sun_za_samples = sun_za_tile.getSamplesFloat()
        sat_za_data = numpy.array(sat_za_samples, dtype=numpy.float32) * float(
            self.config['scale_factors']['scale_factor_deg'])
        sun_za_data = numpy.array(sun_za_samples, dtype=numpy.float32) * float(
            self.config['scale_factors']['scale_factor_deg'])
        sat_za_data = numpy.where(data == -327.68, nan_array, sat_za_data)
        rel_az_tile = context.getSourceTile(self.rel_az, target_rectangle)
        rel_az_samples = rel_az_tile.getSamplesFloat()
        rel_az_data = numpy.array(rel_az_samples, dtype=numpy.float32) * float(
            self.config['scale_factors']['scale_factor_deg'])
        return data, rel_az_data, sat_za_data, sun_za_data, upper_data

    def get_cmsk_data(self, context, target_rectangle):
        if self.cmsk_band is not None:
            cmsk_tile = context.getSourceTile(self.cmsk_band, target_rectangle)
            cmsk_samples = cmsk_tile.getSamplesFloat()
            cmsk_data = numpy.array(cmsk_samples, dtype=numpy.float32)
        else:
            cmsk_data = None
        return cmsk_data

    def get_valid_data(self, context, target_rectangle):
        if self.valid_pixel_band is not None:
            valid_tile = context.getSourceTile(self.valid_pixel_band, target_rectangle)
            valid_samples = valid_tile.getSamplesFloat()
            valid_data = numpy.array(valid_samples, dtype=numpy.float32)
        else:
            valid_data = None
        return valid_data

    def get_lwm_data(self, context, target_rectangle):
        if self.lwm_band is not None:
            lwm_tile = context.getSourceTile(self.lwm_band, target_rectangle)
            lwm_samples = lwm_tile.getSamplesFloat()
            lwm_data = numpy.array(lwm_samples, dtype=numpy.float32)
        else:
            lwm_data = None
        return lwm_data

    def get_split_window_band_data(self, context, target_rectangle):
        data_tile = context.getSourceTile(self.lower_band, target_rectangle)
        upper_tile = context.getSourceTile(self.upper_band, target_rectangle)
        visible_tile = context.getSourceTile(self.visible_band, target_rectangle)
        nir_tile = context.getSourceTile(self.nir_band, target_rectangle)
        bt3_tile = context.getSourceTile(self.bt3_band, target_rectangle)
        data_samples = data_tile.getSamplesFloat()
        upper_samples = upper_tile.getSamplesFloat()
        visible_sample = visible_tile.getSamplesFloat()
        nir_sample = nir_tile.getSamplesFloat()
        bt3_sample = bt3_tile.getSamplesFloat()
        # Convert the data into numpy data. It is easier and faster to work with
        data = numpy.array(data_samples, dtype=numpy.float32) * float(
            self.config['scale_factors']['scale_factor_K'])
        upper_data = numpy.array(upper_samples, dtype=numpy.float32) * float(
            self.config['scale_factors']['scale_factor_K'])
        visible_data = numpy.array(visible_sample, dtype=numpy.float32) * float(
            self.config['scale_factors']['scale_rf'])
        nir_data = numpy.array(nir_sample, dtype=numpy.float32) * float(
            self.config['scale_factors']['scale_rf'])
        bt3_data = numpy.array(bt3_sample, dtype=numpy.float32) * float(
            self.config['scale_factors']['scale_factor_K'])
        return bt3_data, data, nir_data, upper_data, visible_data

    def configure_target_product(self, context, ref_image):
        musenalp_product = snappy.Product('py_MuSenALP', 'py_MuSenALP', self.width, self.height)
        # geocoding from reference picture?
        self.add_geocoding(musenalp_product, ref_image)
        self.add_metadata(context, musenalp_product)
        self.add_lswt_band(musenalp_product)
        if context.getParameter('optional'):
            self.add_optional_bands(context, musenalp_product)
        self.add_quality_flags(musenalp_product)
        self.create_quality_masks(musenalp_product)
        self.add_quality_index(musenalp_product, self.algorithm_parameter)
        context.setTargetProduct(musenalp_product)

    def add_metadata(self, context, musenalp_product):
        snappy.ProductUtils.copyMetadata(self.source_product, musenalp_product)
        snappy.ProductUtils.copyTimeInformation(self.source_product, musenalp_product)
        self.add_processor_metadata(context, musenalp_product, self.source_product)

    def add_geocoding(self, musenalp_product, ref_image):
        if ref_image:
            snappy.ProductUtils.copyGeoCoding(ref_image, musenalp_product)
        else:
            snappy.ProductUtils.copyGeoCoding(self.source_product, musenalp_product)

    def add_lswt_band(self, musenalp_product):
        self.lswt_band = musenalp_product.addBand('lswt', snappy.ProductData.TYPE_FLOAT32)
        self.lswt_band.setDescription('The Lake Surface Water Temperature')
        self.lswt_band.setNoDataValue(Float.NaN)
        self.lswt_band.setNoDataValueUsed(True)
        self.lswt_band.setUnit('K')

    def add_quality_flags(self, musenalp_product):
        self.lswt_flags_band = musenalp_product.addBand('lswt_flags', snappy.ProductData.TYPE_UINT16)
        self.lswt_flags_band.setDescription('The quality flag information')

    def add_quality_index(self, musenalp_product, algorithm):
        self.lswt_quality_index = musenalp_product.addBand('lswt_quality_index', snappy.ProductData.TYPE_UINT16)
        self.lswt_quality_index.setDescription('Index Band for quality levels')
        lswt_quality_coding = IndexCoding('quality_index')
        level_number = 9
        for i in range(0, level_number):
            lswt_quality_coding.addIndex('Q' + str(i), i, 'quality level ' + str(i))
        self.lswt_quality_index.setSampleCoding(lswt_quality_coding)
        musenalp_product.getIndexCodingGroup().add(lswt_quality_coding)

    def add_optional_bands(self, context, musenalp_product):
        band_string = context.getParameter('optionalBands')
        if band_string:
            bands = band_string.split(';')
            for band in bands:
                snappy.ProductUtils.copyBand(band, self.source_product, musenalp_product, True)

    def create_quality_masks(self, musenalp_product):
        # Levels as bitmap (1 to 2048)
        lswt_flag_coding = FlagCoding('lswt_flags')
        flag_q0 = lswt_flag_coding.addFlag("cloud_mask", 1, "Application of cloud mask")
        flag_q1 = lswt_flag_coding.addFlag("VIS_cloud", 2,
                                           "remove non detetable cloud cover by using threshold (NIR< 0.08)")
        flag_q2 = lswt_flag_coding.addFlag("NIR_VIS_ratio", 4,
                                           "ratio of reflectance for NIR and VIS (threshold 1.0)")
        flag_q3 = lswt_flag_coding.addFlag("land_water_mask", 8, "application of land/water mask")
        flag_q4 = lswt_flag_coding.addFlag("GROSS_IR_Valid_VIS", 16,
                                           "the radiation temperature of channel T4 (AVHRR) needs" +
                                           " to be bigger than -10°C")
        flag_q5 = lswt_flag_coding.addFlag("LSWT_range", 32,
                                           "Creating final product, limiting data to -5°C to +35°C")
        flag_q6 = lswt_flag_coding.addFlag("Excluded_singel_pixels", 64,
                                           "Valid pixels only those which are not completely surrounded by" +
                                           " nonvaliddata according to Schwab et al 1999")
        flag_q7 = lswt_flag_coding.addFlag("SZA_gr_55", 128, "Use only pixels for satelite zenith angle <= 55°")
        flag_q8 = lswt_flag_coding.addFlag("Spatial_STDV_gr_3", 256,
                                           "Valid pixels only those which sourrounding pixels have a SDEV from " +
                                           "lower 3°C according to Schwab et al 1999")
        flag_q9 = lswt_flag_coding.addFlag("SZA_gr_45", 512, "Use only pixels for satelite zenith angle <= 45°")
        flag_q10 = lswt_flag_coding.addFlag("Spatial_STDV_gr_1.5", 1024,
                                            "Valid pixels only those which sourrounding pixels have a SDEV from" +
                                            " lower 1.5°C according to Schwab et al 1999")
        flag_q11 = lswt_flag_coding.addFlag("Glint_Angle_le_36", 2048, "Use only pixels with glint angle > 36°")
        # for levels 0 to 6, not working with the masks!
        # flag_q0 = lswt_flag_coding.addFlag("Q0", 0, "Quality level 0")
        # flag_q1 = lswt_flag_coding.addFlag("Q1", 1, "Quality level 1")
        # flag_q2 = lswt_flag_coding.addFlag("Q2", 2, "Quality level 2")
        # flag_q3 = lswt_flag_coding.addFlag("Q3", 3, "Quality level 3")
        # flag_q4 = lswt_flag_coding.addFlag("Q4", 4, "Quality level 4")
        # flag_q5 = lswt_flag_coding.addFlag("Q5", 5, "Quality level 5")
        # flag_q6 = lswt_flag_coding.addFlag("Q6", 6, "Quality level 6")
        musenalp_product.getFlagCodingGroup().add(lswt_flag_coding)
        self.lswt_flags_band.setSampleCoding(lswt_flag_coding)
        # Also for each flag a layer should be created
        musenalp_product.addMask('mask_' + flag_q0.getName(), 'lswt_flags.' + flag_q0.getName(),
                                 flag_q0.getDescription(), Color(255, 0, 0), 0.3)
        musenalp_product.addMask('mask_' + flag_q1.getName(), 'lswt_flags.' + flag_q1.getName(),
                                 flag_q1.getDescription(), Color(255, 128, 0), 0.3)
        musenalp_product.addMask('mask_' + flag_q2.getName(), 'lswt_flags.' + flag_q2.getName(),
                                 flag_q2.getDescription(), Color(255, 255, 0), 0.3)
        musenalp_product.addMask('mask_' + flag_q3.getName(), 'lswt_flags.' + flag_q3.getName(),
                                 flag_q3.getDescription(), Color(128, 255, 0), 0.3)
        musenalp_product.addMask('mask_' + flag_q4.getName(), 'lswt_flags.' + flag_q4.getName(),
                                 flag_q4.getDescription(), Color(0, 255, 0), 0.3)
        musenalp_product.addMask('mask_' + flag_q5.getName(), 'lswt_flags.' + flag_q5.getName(),
                                 flag_q5.getDescription(), Color(0, 255, 128), 0.3)
        musenalp_product.addMask('mask_' + flag_q6.getName(), 'lswt_flags.' + flag_q6.getName(),
                                 flag_q6.getDescription(), Color(0, 255, 255), 0.3)
        musenalp_product.addMask('mask_' + flag_q7.getName(), 'lswt_flags.' + flag_q7.getName(),
                                 flag_q7.getDescription(), Color(0, 128, 255), 0.3)
        musenalp_product.addMask('mask_' + flag_q8.getName(), 'lswt_flags.' + flag_q8.getName(),
                                 flag_q8.getDescription(), Color(0, 0, 255), 0.3)
        musenalp_product.addMask('mask_' + flag_q9.getName(), 'lswt_flags.' + flag_q9.getName(),
                                 flag_q9.getDescription(), Color(127, 0, 255), 0.3)
        musenalp_product.addMask('mask_' + flag_q10.getName(), 'lswt_flags.' + flag_q10.getName(),
                                 flag_q10.getDescription(), Color(255, 0, 255), 0.3)
        musenalp_product.addMask('mask_' + flag_q11.getName(), 'lswt_flags.' + flag_q11.getName(),
                                 flag_q11.getDescription(), Color(255, 0, 127), 0.3)

    def get_geo_data(self, context):
        self.get_geo_data_avhrr(context)
        self.get_geo_data_slstr()
        self.get_geo_data_tirs()
        self.get_geo_data_viirs()
        self.get_geo_data_aatsr()

    def get_geo_data_viirs(self):
        if self.sattype == 'VIIRS':
            self.sat_za = self._get_band(self.source_product, self.config['angles']['sat_za'])
            self.sun_za = self._get_band(self.source_product, self.config['angles']['sun_za'])
            self.sat_az = self._get_band(self.source_product, self.config['angles']['sat_az'])
            self.sun_az = self._get_band(self.source_product, self.config['angles']['sun_az'])

    def get_geo_data_tirs(self):
        if self.sattype == 'TIRS':
            tirs_utils = TirsUtils()
            lut = tirs_utils.create_wrs_lut()
            path = self.source_product.getMetadataRoot().getElement('L1_METADATA_FILE').getElement(
                'PRODUCT_METADATA').getAttributeDouble('WRS_PATH')
            row = self.source_product.getMetadataRoot().getElement('L1_METADATA_FILE').getElement(
                'PRODUCT_METADATA').getAttributeDouble('WRS_ROW')
            corners = tirs_utils.get_corners(lut, path, row)
            self.azimuth = tirs_utils.calculate_sat_azimuth(corners)

    def get_geo_data_slstr(self):
        if self.sattype == 'SLSTR':
            self.sat_za = self.source_product.getTiePointGrid(self.config['angles']['sat_za'])
            self.sun_za = self.source_product.getTiePointGrid(self.config['angles']['sun_za'])
            self.sat_az = self.source_product.getTiePointGrid(self.config['angles']['sat_az'])
            self.sun_az = self.source_product.getTiePointGrid(self.config['angles']['sun_az'])

            product_s2 = None
            product_s3 = None
            if self.file_location is not None and 'xfdumanifest.xml' in self.file_location:
                product_s2 = snappy.ProductIO.readProduct(
                    self.file_location.replace('xfdumanifest.xml', 'S2_quality_an.nc'))
                product_s3 = snappy.ProductIO.readProduct(
                    self.file_location.replace('xfdumanifest.xml', 'S3_quality_an.nc'))
            if product_s2 is not None:
                self.solar_irr_s2 = product_s2.getMetadataRoot().getElement('Variable_Attributes').getElement(
                    'S2_solar_irradiance_an').getElement('Values').getAttributeDouble('data')
            else:
                self.solar_irr_s2 = 1525.94
            if product_s3 is not None:
                self.solar_irr_s3 = product_s3.getMetadataRoot().getElement('Variable_Attributes').getElement(
                    'S3_solar_irradiance_an').getElement('Values').getAttributeDouble('data')
            else:
                self.solar_irr_s3 = 956.17

    def get_geo_data_aatsr(self):
        if self.sattype == 'AATSR':
            self.sat_za = self.source_product.getTiePointGrid(self.config['angles']['sat_za'])
            self.sun_za = self.source_product.getTiePointGrid(self.config['angles']['sun_za'])
            self.sat_az = self.source_product.getTiePointGrid(self.config['angles']['sat_az'])
            self.sun_az = self.source_product.getTiePointGrid(self.config['angles']['sun_az'])

    def get_geo_data_avhrr(self, context):
        if self.sattype == 'AVHRR':
            geom = snappy.ProductIO.readProduct(str(context.getParameter('geometricData')))
            if self.cut:
                geom = self.utils.cut_product(geom, self.range)
            self.sat_za = self._get_band(geom, self.config['angles']['sat_za'])
            self.sun_za = self._get_band(geom, self.config['angles']['sun_za'])
            self.rel_az = self._get_band(geom, self.config['angles']['rel_az'])

    def get_valid_pixel_band(self, context):
        expression = str(context.getParameter('validPixelExpression'))
        if expression != 'None' and expression != '':
            self.valid_pixel_band = snappy.VirtualBand("valid", snappy.ProductData.TYPE_FLOAT32,
                                                       self.source_product.getSceneRasterWidth(),
                                                       self.source_product.getSceneRasterHeight(), expression)
            self.valid_pixel_band.setOwner(self.source_product)

    def get_cloud_mask(self, context):
        cmsk_product = str(context.getParameter('productCloudMask'))
        cmsk_band_param = str(context.getParameter('bandCloudMask'))
        if cmsk_product != 'None' and cmsk_product != '':
            cmsk = snappy.ProductIO.readProduct(cmsk_product)
            if self.cut:
                cmsk = self.utils.cut_product(cmsk, self.range)
            self.cmsk_band = snappy.VirtualBand("cmsk", snappy.ProductData.TYPE_FLOAT32,
                                                self.source_product.getSceneRasterWidth(),
                                                self.source_product.getSceneRasterHeight(), cmsk_band_param)
            self.cmsk_band.setOwner(cmsk)
        else:
            if cmsk_band_param != 'None' and cmsk_band_param != '':
                self.cmsk_band = snappy.VirtualBand("cmsk", snappy.ProductData.TYPE_FLOAT32,
                                                    self.source_product.getSceneRasterWidth(),
                                                    self.source_product.getSceneRasterHeight(), cmsk_band_param)
                self.cmsk_band.setOwner(self.source_product)

    def get_land_water_mask(self, context):
        calculate_lwm = context.getParameter('calculateLwm')
        if calculate_lwm:
            utils = Utils()
            lwm = utils.getLandWaterMask(self.source_product)
            if self.cut:
                lwm = self.utils.cut_product(lwm, self.range)
            self.lwm_band = snappy.VirtualBand("lwm", snappy.ProductData.TYPE_FLOAT32,
                                               self.source_product.getSceneRasterWidth(),
                                               self.source_product.getSceneRasterHeight(), 'land_water_fraction>50')
            self.lwm_band.setOwner(lwm)
        else:
            lwm_product = str(context.getParameter('productLandWaterMask'))
            lwm_band_param = str(context.getParameter('bandLandWaterMask'))
            if lwm_product != 'None' and lwm_product != '':
                lwm = snappy.ProductIO.readProduct(lwm_product)
                if self.cut:
                    lwm = self.utils.cut_product(lwm, self.range)
                self.lwm_band = snappy.VirtualBand("lwm", snappy.ProductData.TYPE_FLOAT32,
                                                   self.source_product.getSceneRasterWidth(),
                                                   self.source_product.getSceneRasterHeight(), lwm_band_param)
                self.lwm_band.setOwner(lwm)
            else:
                if lwm_band_param != 'None' and lwm_band_param != '':
                    self.lwm_band = snappy.VirtualBand("lwm", snappy.ProductData.TYPE_FLOAT32,
                                                       self.source_product.getSceneRasterWidth(),
                                                       self.source_product.getSceneRasterHeight(), lwm_band_param)
                    self.lwm_band.setOwner(self.source_product)

    def get_reg_image(self, context):
        ref_image = None
        if self.sattype == 'AVHRR':
            ref_image = snappy.ProductIO.readProduct(str(context.getParameter('refImage')))
            if self.cut:
                ref_image = self.utils.cut_product(ref_image, self.range)
            self.ref_band_org = self._get_band(ref_image, 'band_1')

        return ref_image

    def get_mono_window_bands(self, context):
        self.mono_band = self.source_product.getBand(self.config['bands']['mono_band'])
        self.visible_band = self._get_band(self.source_product, self.config['bands']['visible_band'])
        self.nir_band = self._get_band(self.source_product, self.config['bands']['nir_band'])

    def get_split_window_bands(self, context):
        self.lower_band = self.source_product.getBand(self.config['bands']['lower_band'])
        self.upper_band = self.source_product.getBand(self.config['bands']['upper_band'])
        self.visible_band = self._get_band(self.source_product, self.config['bands']['visible_band'])
        self.nir_band = self._get_band(self.source_product, self.config['bands']['nir_band'])
        self.bt3_band = self._get_band(self.source_product, self.config['bands']['lowest_band'])

    def get_lut_info(self, context):
        start_time = self.source_product.getStartTime()
        if start_time is None:
            if self.sattype == 'AVHRR':
                date_string = str(self.source_product.getName().split('_')[0])
                date_format = SimpleDateFormat("yyyyMMdd")
                date = date_format.parse(date_string)
                self.season = self.algo.get_season(date)
        else:
            self.season = self.algo.get_season(start_time.getAsDate())
        if self.sattype == 'SLSTR':
            self.height_band = self._get_band(self.source_product, "elevation_an")
        else:
            self.elevation = context.getParameter('height')

    def get_mono_window_coeff(self, context):
        self.a0_mono = context.getParameter('a0-mono')
        self.a1_mono = context.getParameter('a1-mono')
        if self.a0_mono == 0.0:
            self.needs_lut = True
        self.algo = lswt_algo.MonoWindowAlgo(self.a0_mono, self.a1_mono, self.needs_lut)

    def get_split_window_coeff(self, context):
        # algorithm coefficients
        coef_file = context.getParameter('coef-file')
        lut = context.getParameter('lut')
        if coef_file:
            coef_list = self.utils.read_coef_file(str(coef_file))
            self.a0 = coef_list[0]
            self.a1 = coef_list[1]
            self.a2 = coef_list[2]
            self.a3 = coef_list[3]
        elif lut:
            self.needs_lut = True
        else:
            self.a0 = context.getParameter('a0')
            self.a1 = context.getParameter('a1')
            self.a2 = context.getParameter('a2')
            self.a3 = context.getParameter('a3')
        self.algo = lswt_algo.SplitWindowAlgo(self.a0, self.a1, self.a2, self.a3, self.needs_lut)

    def get_cut_param(self, context):
        if self.cut:
            self.poi_x = context.getParameter('poix')
            self.poi_y = context.getParameter('poiy')
            self.padding = context.getParameter('padding')
            if self.sattype == 'AVHRR':
                ref = str(context.getParameter('refImage'))
                self.range = self.utils.get_reference_coordinates(ref, self.poi_x, self.poi_y, self.padding)

    def get_config(self):
        cwd = os.getcwd()
        self.config.read(
            [self.sattype + '.ini', cwd + self.sattype + '.ini', os.path.expanduser('~/' + self.sattype + '.ini')])
        if len(self.config.sections()) == 0:
            self.config = Config(self.sattype).get_conf()

    def dispose(self, context):
        pass

    @staticmethod
    def _get_band(product, name):
        # Retrieve the band from the product
        # Some times data is not stored in a band but in a tie-point grid or a mask or a vector data.
        # To get access to this information other methods are exposed by the product class. Like
        # getTiePointGridGroup().get('name'), getVectorDataGroup().get('name') or getMaskGroup().get('name')
        # For bands and tie-point grids a short cut exists. Simply use getBand('name') or getTiePointGrid('name')
        band = product.getBandGroup().get(name)
        if not band:
            raise RuntimeError('Product does not contain a band named', name)
        return band

    @staticmethod
    def add_processor_metadata(context, target_product, source_product):
        metadata_element = jpy.get_type('org.esa.snap.core.datamodel.MetadataElement')
        meta_elem = metadata_element('MuSenALP')
        meta_elem.setAttributeString('source_product', source_product.getName())
        meta_elem.setAttributeString('processing_time', datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
        # Parameters
        meta_elem.setAttributeString('sattype', context.getParameter('sattype'))
        if context.getParameter('refImage'):
            meta_elem.setAttributeString('refImage', str(context.getParameter('refImage')))
        if context.getParameter('geometricData'):
            meta_elem.setAttributeString('geometricData', str(context.getParameter('geometricData')))
        if context.getParameter('productCloudMask'):
            meta_elem.setAttributeString('productCloudMask', str(context.getParameter('productCloudMask')))
        if context.getParameter('bandCloudMask'):
            meta_elem.setAttributeString('bandCloudMask', context.getParameter('bandCloudMask'))
        else:
            meta_elem.setAttributeString('bandCloudMask', 'False')
        if context.getParameter('productLandWaterMask'):
            meta_elem.setAttributeString('productLandWaterMask', str(context.getParameter('productLandWaterMask')))
        if context.getParameter('bandLandWaterMask'):
            meta_elem.setAttributeString('bandLandWaterMask', context.getParameter('bandLandWaterMask'))
        else:
            meta_elem.setAttributeString('bandLandWaterMask', 'False')
        if context.getParameter('validPixelExpression'):
            meta_elem.setAttributeString('validPixelExpression', context.getParameter('validPixelExpression'))
        if context.getParameter('maskBeforeCalculation'):
            meta_elem.setAttributeString('maskBeforeCalculation', 'True')
        meta_elem.setAttributeString('algorithm', context.getParameter('algorithm'))
        if context.getParameter('coef-file'):
            meta_elem.setAttributeString('coef-file', str(context.getParameter('coef-file')))
        if context.getParameter('a0'):
            meta_elem.setAttributeDouble('a0', context.getParameter('a0'))
        if context.getParameter('a1'):
            meta_elem.setAttributeDouble('a1', context.getParameter('a1'))
        if context.getParameter('a2'):
            meta_elem.setAttributeDouble('a2', context.getParameter('a2'))
        if context.getParameter('a3'):
            meta_elem.setAttributeDouble('a3', context.getParameter('a3'))
        if context.getParameter('cut'):
            meta_elem.setAttributeString('cut', 'True')
            meta_elem.setAttributeDouble('poix', context.getParameter('poix'))
            meta_elem.setAttributeDouble('poiy', context.getParameter('poiy'))
            meta_elem.setAttributeDouble('padding', context.getParameter('padding'))
        else:
            meta_elem.setAttributeString('cut', 'False')
        if context.getParameter('optional'):
            meta_elem.setAttributeString('optional', 'True')
            meta_elem.setAttributeString('optionalBands', context.getParameter('optionalBands'))
        else:
            meta_elem.setAttributeString('optional', 'False')

        # TODO add metadata for warnings
        root = target_product.getMetadataRoot()
        root.addElement(meta_elem)


class AlgorithmException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
