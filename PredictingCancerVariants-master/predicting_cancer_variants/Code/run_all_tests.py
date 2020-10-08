import unittest

from predicting_cancer_variants.Code.common_tests.configuration_unit_tests import ConfigurationUnitTests
from predicting_cancer_variants.Code.download_and_prepare_mutation_data_tests.create_database_controller_system_tests \
    import CreateDatabaseControllerSystemTests
from predicting_cancer_variants.Code.download_and_prepare_mutation_data_tests.download_and_prepare_controller_system_tests \
    import DownloadAndPrepareControllerSystemTests
from predicting_cancer_variants.Code.download_and_prepare_mutation_data_tests.table_unit_tests import TableUnitTests


def create_suite():
    # initialize the test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # add tests to the test suite
    suite.addTests(loader.loadTestsFromTestCase(ConfigurationUnitTests))
    suite.addTests(loader.loadTestsFromTestCase(TableUnitTests))
    suite.addTests(loader.loadTestsFromTestCase(CreateDatabaseControllerSystemTests))
    suite.addTests(loader.loadTestsFromTestCase(DownloadAndPrepareControllerSystemTests))

    return suite


if __name__ == '__main__':
    # get a fully loaded test suite
    test_suite = create_suite()

    # initialize a runner, pass it your suite and run it
    runner = unittest.TextTestRunner(verbosity=3)
    result = runner.run(test_suite)
