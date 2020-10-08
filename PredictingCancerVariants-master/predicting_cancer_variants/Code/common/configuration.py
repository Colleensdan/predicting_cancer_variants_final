import os
import pathlib
import yaml
import logging
import logging.config

DEFAULT_CONFIGURATION_DIRECTORY = 'Configuration Files'
DEFAULT_CONFIGURATION_FILENAME = 'predicting_cancer_variants.yaml'


# Singleton as described in pep318 https://www.python.org/dev/peps/pep-0318/#examples
# Overrides getinstance of classes decorated with @singleton so that it only ever returns the same instance
# so that we can set up a Configuration class that is initialised only once and accessible from anywhere
# This is to make configuration more efficient, so that the YAML configuration file is read and parsed only
# once, instead of every time a property is read
instances = {}


# allow the singleton to get reset so that unit tests can initialise this Configuration class with different file names
def reset_singleton():
    global instances
    instances = {}


def singleton(cls):

    def getinstance(configuration_filename=None):
        if "Configuration" not in instances:
            instances["Configuration"] = cls(configuration_filename)
        return instances["Configuration"]

    return getinstance


@singleton
class Configuration:

    def __init__(self, configuration_filename=None):
        self.__configuration_filename = configuration_filename
        self.__configuration = None
        self.__logging = None

    @property
    def application_root(self):
        return str(pathlib.Path(__file__).parent.parent.parent)

    @property
    def database_location(self):
        configuration = self.__get_configuration()
        configured_value = configuration['common']['database_location']

        # Append application root path (eg 'c:\my_code\predicting_cancer') if
        # the database location looks like it's specified as a relative path
        drive, path = os.path.splitdrive(configured_value)
        database_location = configured_value if drive else os.path.join(self.application_root, configured_value)

        return database_location

    @property
    def database_table_study(self):
        configuration = self.__get_configuration()
        return configuration['common']['table_names']['study']

    @property
    def database_table_cancer(self):
        configuration = self.__get_configuration()
        return configuration['common']['table_names']['cancer']

    @property
    def database_table_molecular(self):
        configuration = self.__get_configuration()
        return configuration['common']['table_names']['molecular']

    @property
    def database_table_mutation(self):
        configuration = self.__get_configuration()
        return configuration['common']['table_names']['mutation']

    @property
    def database_table_gene(self):
        configuration = self.__get_configuration()
        return configuration['common']['table_names']['gene']

    @property
    def database_table_staging(self):
        configuration = self.__get_configuration()
        return configuration['prepare_machine_learning_model']['table_names']['staging']

    @property
    def database_table_patient(self):
        configuration = self.__get_configuration()
        return configuration['prepare_machine_learning_model']['table_names']['patient']

    @property
    def database_table_net(self):
        configuration = self.__get_configuration()
        return configuration['prepare_machine_learning_model']['table_names']['net']

    @property
    def study_url(self):
        configuration = self.__get_configuration()
        return configuration['downloader']['urls']['study']

    @property
    def gene_url(self):
        configuration = self.__get_configuration()
        return configuration['downloader']['urls']['gene']

    @property
    def cancer_url(self):
        configuration = self.__get_configuration()
        return configuration['downloader']['urls']['cancer']

    @property
    def molecular_url(self):
        configuration = self.__get_configuration()
        return configuration['downloader']['urls']['molecular']

    @property
    def mutation_url(self):
        configuration = self.__get_configuration()
        return configuration['downloader']['urls']['mutation']

    @property
    def minimum_samples_per_study(self):
        configuration = self.__get_configuration()
        return configuration['downloader']['minimum_samples_per_study'] or 0

    @property
    def maximum_samples_per_study(self):
        configuration = self.__get_configuration()
        return configuration['downloader']['maximum_samples_per_study'] or 9999999

    def get_logger(self, name):
        # Only set up logging once
        if not self.__logging:

            # Take logging configuration from YAML file and load into a dictionary
            with open(self.__get_configuration_filename(), 'r') as f:
                config = yaml.safe_load(f.read())

                # If a logging section is declared in the configuration file then apply it here
                # The layout of this file once represented as a dictionary is described here
                # https://docs.python.org/2/library/logging.config.html#configuration-dictionary-schema
                if 'logging' in config['common']:

                    # Standard Python logging configuration is in the common/logging section of the YAML file
                    logging.config.dictConfig(config['common']['logging'])

                # To be efficient, prevent logging getting initialised a second time
                self.__logging = logging

        logger = self.__logging.getLogger(name)
        return logger

    def __get_configuration(self):
        # Check if we have a cached configuration, to save having to read and parse a text file for every property read
        if not self.__configuration:

            # Load configuration from YAML and cache for future use
            self.__configuration = self.__load_configuration()

        # Return configuration
        return self.__configuration

    def __load_configuration(self):
        # load the configuration from a file
        filename = self.__get_configuration_filename()
        with open(filename) as file:
            configuration = yaml.load(file, Loader=yaml.FullLoader)
            return configuration

    def __get_configuration_filename(self):
        # return a path to the configuration filename from the value passed in the constructor if there was one
        # otherwise use a default path
        filename = self.__get_constructor_initialised_configuration_filename() or \
                   self.__get_default_configuration_filename()

        if not os.path.exists(filename):
            raise FileNotFoundError("This configuration file does not exist: '{}'".format(filename))

        return filename

    def __get_default_configuration_filename(self):
        # return a path that combines the application root with the default subdirectory and filename
        default_filename = os.path.join(self.application_root, DEFAULT_CONFIGURATION_DIRECTORY,
                                        DEFAULT_CONFIGURATION_FILENAME)
        return default_filename

    def __get_constructor_initialised_configuration_filename(self):
        # Return None if the constructor was not initialised with a value
        if not self.__configuration_filename:
            return None

        # Return the filename initialised in the constructor if its' path is absolute (starts with a slash after
        # chopping off a potential drive letter) because it is a complete, usable path
        if os.path.isabs(self.__configuration_filename):
            return self.__configuration_filename

        # If there are path separators in the filename then it is relative so prepend with the application root
        if os.path.sep in self.__configuration_filename:
            return os.path.join(self.application_root, self.__configuration_filename)

        # As no path was specified, then return a path that combines the application root and default configuration
        # directory with the filename initialised in the constructor
        return os.path.join(self.application_root, DEFAULT_CONFIGURATION_DIRECTORY, self.__configuration_filename)
