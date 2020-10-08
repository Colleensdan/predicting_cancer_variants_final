import os
import sqlite3
import unittest

from predicting_cancer_variants.Code.common import configuration
from predicting_cancer_variants.Code.common.configuration import Configuration
from predicting_cancer_variants.Code.download_and_prepare_mutation_data import create_database_controller, \
    download_cancer_genomics_controller, prepare_machine_learning_data_controller


class DownloadAndPrepareControllerSystemTests(unittest.TestCase):

    __test_configuration_file = r"Test Configuration Files\create_database_test_configuration.yaml"

    # run once, before any test scenarios start to run
    @classmethod
    def setUpClass(cls):
        # Clean up before running each test, in case quited test before tearDown() was run, eg when debugging
        cls.__cleanup()

        # Use test YAML configuration file
        Configuration(cls.__test_configuration_file)

        # Delete test database if it exists
        if os.path.exists(Configuration().database_location):
            os.remove(Configuration().database_location)

        # Create the database
        create_database_controller.run()

        # Populate the database
        download_cancer_genomics_controller.run()

        # Pre-prepare the downloaded data by converting it into a form that can be used for machine learning
        prepare_machine_learning_data_controller.run()

    # run once, after all test scenarios have run
    @classmethod
    def tearDownClass(cls):
        # Clean up after running each test
        cls.__cleanup()

    @staticmethod
    def __cleanup():
        # Prevent singleton Configuration object hanging on from a previous test, or accidentally used in the future
        configuration.reset_singleton()

    @staticmethod
    def row_count(table_name):
        db = sqlite3.connect(Configuration().database_location)
        cursor = db.cursor()
        results = cursor.execute(f'SELECT COUNT(*) FROM {table_name};')
        value = results.fetchone()

        return value[0]

    @staticmethod
    def table_exists(table_name):
        sql_command = f'SELECT name FROM sqlite_master WHERE type="table" AND name = "{table_name}";'
        db = sqlite3.connect(Configuration().database_location)
        cursor = db.cursor()
        matching_table_count = len(cursor.execute(sql_command).fetchall())

        return matching_table_count == 1

    def test_database_file_exists(self):
        # Database should now exist
        self.assertTrue(os.path.exists(Configuration().database_location))

    def test_cancer_table_exists(self):
        self.assertTrue(self.table_exists("test_import_cancers"))

    def test_cancer_table_is_populated(self):
        # Connect to the database, and get a count of the number of rows in the cancer table
        row_count = self.row_count("test_import_cancers")

        # There should be at least two rows in the table
        self.assertGreaterEqual(row_count, 2)

    def test_study_table_exists(self):
        self.assertTrue(self.table_exists("test_import_studies"))

    def test_study_table_is_populated(self):
        # Connect to the database, and get a count of the number of rows in the studies table
        row_count = self.row_count("test_import_studies")

        # There should be at least two rows in the table
        self.assertGreaterEqual(row_count, 2)

    def test_molecular_profile_table_exists(self):
        self.assertTrue(self.table_exists("test_import_molecular_profiles"))

    def test_molecular_profiles_table_is_populated(self):
        # Connect to the database, and get a count of the number of rows in the molecular profiles table
        row_count = self.row_count("test_import_molecular_profiles")

        # There should be at least two rows in the table
        self.assertGreaterEqual(row_count, 2)

    def test_genes_table_exists(self):
        self.assertTrue(self.table_exists("test_import_genes"))

    def test_genes_table_is_populated(self):
        # Connect to the database, and get a count of the number of rows in the genes table
        row_count = self.row_count("test_import_genes")

        # There should be at least two rows in the table
        self.assertGreaterEqual(row_count, 2)

    def test_mutation_table_exists(self):
        self.assertTrue(self.table_exists("test_import_mutations"))

    def test_mutation_table_is_populated(self):
        # Connect to the database, and get a count of the number of rows in the mutations table
        row_count = self.row_count("test_import_mutations")

        # There should be at least two rows in the table
        self.assertGreaterEqual(row_count, 2)

    def test_gene_mutations_table_exists(self):
        self.assertTrue(self.table_exists("test_prepare_gene_mutations"))

    def test_gene_mutations_table_is_populated(self):
        # Connect to the database, and get a count of the number of rows in the gene mutations table
        row_count = self.row_count("test_prepare_gene_mutations")

        # There should be at least two rows in the table
        self.assertGreaterEqual(row_count, 2)

    def test_patients_table_exists(self):
        self.assertTrue(self.table_exists("test_prepare_patients"))

    def test_patients_table_is_populated(self):
        # Connect to the database, and get a count of the number of rows in the patients table
        row_count = self.row_count("test_prepare_patients")

        # There should be at least two rows in the table
        self.assertGreaterEqual(row_count, 2)

    def test_net_data_table_exists(self):
        self.assertTrue(self.table_exists("test_prepare_net_data"))

