from abc import ABC, abstractmethod

from predicting_cancer_variants.Code.common.configuration import Configuration
from predicting_cancer_variants.Code.common.database_functions import DatabaseHelper


# abstract base class
class Table(ABC):

    # this base class constructor is implicitly called when a concrete subclass is instantiated
    def __init__(self):
        self._configuration = Configuration()
        self.__logger = self._configuration.get_logger(__name__)

    # create a table as declared by a concrete subclass, and create the database if missing
    def create(self):
        self.__logger.info("Creating table {}".format(self._table_name))
        database_helper = DatabaseHelper()
        sql_command = self.__get_create_table_sql_command()
        connection = database_helper.create_database_if_missing_and_get_connection()
        cursor = connection.cursor()
        database_helper.execute_sql_command(cursor, sql_command)

#todo - proper code comments; this is an abstract protected property and is the recommended way to do it - @abstractproperty is possible but not recommended (deprecated)
    @property
    @abstractmethod
    def _table_name(self):
        pass

    @property
    @abstractmethod
    def _create_table_segment(self):
        pass

    # generate a create table statement from the abstract protected properties of a subclass
    def __get_create_table_sql_command(self):
        return "CREATE TABLE IF NOT EXISTS {0}{1}"\
            .format(self._table_name, self._create_table_segment)



