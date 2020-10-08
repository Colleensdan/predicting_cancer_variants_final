import unittest
import os

from predicting_cancer_variants.Code.common import configuration
from predicting_cancer_variants.Code.common.configuration import Configuration


class ConfigurationUnitTests(unittest.TestCase):
    def setUp(self):
        # prevent singleton Configuration object hanging on from a previous test
        configuration.reset_singleton()

    def tearDown(self):
        # prevent singleton Configuration object from affecting any further usage
        configuration.reset_singleton()

    def test_application_root_is_ok(self):
        c = Configuration()
        self.assertIn('predicting_cancer_variants', c.application_root)

    def test_default_config_file_is_correct(self):
        c = Configuration()
        self.assertIn(r'Configuration Files\predicting_cancer_variants.yaml',
                      c._Configuration__get_default_configuration_filename())

    def test_error_is_raised_if_config_file_does_not_exist(self):
        non_existent_configuration_file = r"c:\this\file\does\not-exist.yaml"
        c = Configuration(non_existent_configuration_file)
        try:
            # this should raise an error because the config file does not exist
            c._Configuration__get_configuration_filename()

            # should not reach here
            self.fail("An exception was not raised")
        except FileNotFoundError as e:
            # expected to reach here
            self.assertEqual(str(e), "This configuration file does not exist: '{}'".
                             format(non_existent_configuration_file))
            pass
        except:
            self.fail("The wrong exception was raised")

    def test_error_is_not_raised_if_config_file_does_exist(self):
        c = Configuration(r"Test Configuration Files\test_configuration_file_absolute_path.yaml")
        self.assertIn(r'test_configuration_file_absolute_path.yaml', c._Configuration__get_configuration_filename())

    def test_can_fetch_database_location_with_absolute_path(self):
        c = Configuration(r"Test Configuration Files\test_configuration_file_absolute_path.yaml")
        self.assertEqual(r"c:\a-test\location\file.yaml", c.database_location)

    def test_can_fetch_database_location_with_relative_path_resolved(self):
        c = Configuration(r"Test Configuration Files\test_configuration_file_relative_path.yaml")
        database_location = c.database_location
        drive, path = os.path.splitdrive(database_location)
        self.assertTrue(drive)
        self.assertIsNotNone(drive)
        self.assertIn(r'test_folder\test_file.db', path)

    def test_can_fetch_cancer_table(self):
        c = Configuration(r"Test Configuration Files\test_configuration_file_absolute_path.yaml")
        self.assertEqual("test_cancer_study", c.database_table_study)

    def test_can_fetch_study_url(self):
        c = Configuration()
        self.assertIn("www.cbioportal.org", c.study_url)


if __name__ == '__main__':
    unittest.main()
