import unittest

from predicting_cancer_variants.Code.download_and_prepare_mutation_data.create_database.cancer_table import CancerTable
from predicting_cancer_variants.Code.download_and_prepare_mutation_data.create_database.gene_table import GeneTable
from predicting_cancer_variants.Code.download_and_prepare_mutation_data.create_database.molecular_table import MolecularTable
from predicting_cancer_variants.Code.download_and_prepare_mutation_data.create_database.mutation_table import MutationTable
from predicting_cancer_variants.Code.download_and_prepare_mutation_data.create_database.studies_table import StudiesTable
from predicting_cancer_variants.Code.download_and_prepare_mutation_data.create_database.table import Table
from predicting_cancer_variants.Code.download_and_prepare_mutation_data_tests.example_table import ExampleTable


class TableUnitTests(unittest.TestCase):
    def test_cannot_instantiate_abstract_table_class(self):
        self.assertRaises(TypeError, Table)

    def test_mutation_table_is_subclass_of_table(self):
        self.assertTrue(issubclass(MutationTable, Table))

    def test_cancer_table_is_subclass_of_table(self):
        self.assertTrue(issubclass(CancerTable, Table))

    def test_gene_table_is_subclass_of_table(self):
        self.assertTrue(issubclass(GeneTable, Table))

    def test_molecular_table_is_subclass_of_table(self):
        self.assertTrue(issubclass(MolecularTable, Table))

    def test_studies_table_is_subclass_of_table(self):
        self.assertTrue(issubclass(StudiesTable, Table))

    def test_example_sql_is_constructed_properly(self):
        sample_table = ExampleTable()
        expected_sql_command = """CREATE TABLE IF NOT EXISTS Example_Table         
(
    Example_Id INTEGER NOT NULL PRIMARY KEY
    Example_Name VARCHAR(50) NOT NULL
)
"""
        actual_sql_command = sample_table._Table__get_create_table_sql_command()
        self.assertEqual(expected_sql_command, actual_sql_command)






