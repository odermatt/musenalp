import unittest

from config import Config


class TestConfig(unittest.TestCase):
    def test_config_avhrr(self):
        config = Config('AVHRR').get_conf()
        self.assertEqual('Band_4_BT____[K_x_10]', config['bands']['lower_band'])
        self.assertEqual('Satellite_Zenith___[°_x_100]', config['angles']['sat_za'])
        self.assertEqual(0.1, float(config['scale_factors']['scale_factor_K']))

    def test_config_slstr(self):
        config = Config('SLSTR').get_conf()
        self.assertEqual('S8_BT_in', config['bands']['lower_band'])
        self.assertEqual('sat_zenith_tn', config['angles']['sat_za'])
        self.assertEqual(1, float(config['scale_factors']['scale_factor_K']))

if __name__ == '__main__':
    unittest.main()

