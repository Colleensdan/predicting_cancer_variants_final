import os
import sqlite3
import unittest

from predicting_cancer_variants.Code.common import configuration
from predicting_cancer_variants.Code.common.configuration import Configuration
from predicting_cancer_variants.Code.download_and_prepare_mutation_data import create_database_controller


class CreateDatabaseControllerSystemTests(unittest.TestCase):

    __test_configuration_file = r"Test Configuration Files\create_database_test_configuration.yaml"

    # run before every test
    def setUp(self):
        # Clean up before running each test, in case quited test before tearDown() was run, eg when debugging
        self.__cleanup()

        # Use test YAML configuration file
        Configuration(self.__test_configuration_file)

        # Delete test database if it exists
        if os.path.exists(Configuration().database_location):
            os.remove(Configuration().database_location)

    # run after every test
    def tearDown(self):
        # Clean up after running each test
        self.__cleanup()

    @staticmethod
    def __cleanup():
        # Prevent singleton Configuration object hanging on from a previous test, or accidentally used in the future
        configuration.reset_singleton()

    def test_database_file_exists(self):
        # Precondition - database must not exist
        self.assertFalse(os.path.exists(Configuration().database_location))

        # Create the database
        create_database_controller.run()

        # Database should now exist
        self.assertTrue(os.path.exists(Configuration().database_location))

    def test_tables_have_been_created(self):
        # Create the database
        create_database_controller.run()

        # Connect to the database, and get all of the table names into a list
        db = sqlite3.connect(Configuration().database_location)
        cursor = db.cursor()
        results = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()

        # At least 5 tables should have been created (more detailed tests are in a separate test class)
        self.assertGreaterEqual(len(results), 5)
