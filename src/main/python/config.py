class Config:
    def __init__(self, sensor):
        if sensor == 'AVHRR':
            self.config = {'bands': {'lower_band': 'Band_4_BT____[K_x_10]', 'upper_band': 'Band_5_BT____[K_x_10]',
                                     'lowest_band': 'Band_3B_BT___[K_x_10]',
                                     'visible_band': 'Band_1_RTOA__[x_1000]', 'nir_band': 'Band_2_RTOA__[x_1000]',
                                     'mono_band': 'Band_4_BT____[K_x_10]'},
                           'angles': {'sat_za': 'Satellite_Zenith___[°_x_100]',
                                      'sun_za': 'Sun_Zenith_________[°_x_100]',
                                      'rel_az': 'Relative_Azimuth___[°_x_100]'},
                           'scale_factors': {'scale_factor_deg': '0.01', 'scale_factor_K': '0.1', 'scale_rf': '0.001'},
                           'algorithms': {'lswt-algorithm': 'both'}}
        if sensor == 'SLSTR':
            self.config = {'bands': {'lower_band': 'S8_BT_in', 'upper_band': 'S9_BT_in', 'lowest_band': 'S7_BT_in',
                                     'visible_band': 'S2_radiance_an', 'nir_band': 'S3_radiance_an',
                                     'mono_band': 'S8_BT_in'},
                           'angles': {'sat_za': 'sat_zenith_tn', 'sun_za': 'solar_zenith_tn',
                                      'sun_az': 'solar_azimuth_tn', 'sat_az': 'sat_azimuth_tn'},
                           'scale_factors': {'scale_factor_deg': '1', 'scale_factor_K': '1', 'scale_rf': '1'},
                           'algorithms': {'lswt-algorithm': 'both'}}
        if sensor == 'TIRS':
            self.config = {
                'bands': {'lower_band': 'thermal_infrared_(tirs)_1', 'upper_band': 'thermal_infrared_(tirs)_2',
                          'lowest_band': 'swir_1',
                          'visible_band': 'red', 'nir_band': 'near_infrared'},
                'angles': {'sat_za': 'ROLL_ANGLE', 'sun_za': 'SUN_ELEVATION', 'sun_az': 'SUN_AZIMUTH', 'sat_az': ''},
                'scale_factors': {'scale_factor_deg': '1', 'scale_factor_K': '1', 'scale_rf': '1'},
                'algorithms': {'lswt-algorithm': 'split-window'}}
        if sensor == 'VIIRS':
            self.config = {
                'bands': {'visible_band': 'VIIRS-I1-SDR_All.Reflectance', 'nir_band': 'VIIRS-I2-SDR_All.Reflectance',
                          'mono_band': 'VIIRS-I5-SDR_All.BrightnessTemperature'},
                'angles': {'sat_za': 'VIIRS-IMG-GEO_All.SatelliteZenithAngle',
                           'sun_za': 'VIIRS-IMG-GEO_All.SolarZenithAngle',
                           'sun_az': 'VIIRS-IMG-GEO_All.SolarAzimuthAngle',
                           'sat_az': 'VIIRS-IMG-GEO_All.SatelliteAzimuthAngle'},
                'scale_factors': {'scale_factor_deg': '1', 'scale_factor_K': '0.01', 'scale_rf': '1'},
                'algorithms': {'lswt-algorithm': 'mono-window'}}
        if sensor == 'AATSR':
            self.config = {'bands': {'lower_band': 'btemp_nadir_1100', 'upper_band': 'btemp_nadir_1200',
                                     'lowest_band': 'btemp_nadir_0370',
                                     'visible_band': 'reflec_nadir_0670', 'nir_band': 'reflec_nadir_0870',
                                     'mono_band': 'btemp_nadir_1200'},
                           'angles': {'sat_za': 'view_elev_nadir', 'sun_za': 'sun_elev_nadir',
                                      'sun_az': 'sun_azimuth_nadir', 'sat_az': 'view_azimuth_nadir'},
                           'scale_factors': {'scale_factor_deg': '1', 'scale_factor_K': '1', 'scale_rf': '1'},
                           'algorithms': {'lswt-algorithm': 'both'}}

    def get_conf(self):
        return self.config
