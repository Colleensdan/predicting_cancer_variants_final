from abc import ABC, abstractmethod

from predicting_cancer_variants.Code.common.configuration import Configuration
from predicting_cancer_variants.Code.common.database_functions import DatabaseHelper


# abstract base class
class PrepareForMachineLearning(ABC):

    # this base class constructor is implicitly called when a concrete subclass is instantiated
    def __init__(self):
        self._configuration = Configuration()
        self.__logger = self._configuration.get_logger(__name__)

    @property
    @abstractmethod
    def _table_name(self):
        pass

    @property
    @abstractmethod
    def _message(self):
        pass

    @property
    @abstractmethod
    def _sql_commands(self):
        pass

    def prepare(self):
        self.__logger.info(self._message)
        database_helper = DatabaseHelper()
        connection = database_helper.create_database_if_missing_and_get_connection()
        cursor = connection.cursor()

        for sql_command in self._sql_commands:
            database_helper.execute_sql_command(cursor, sql_command)
            if cursor.rowcount > 1:
                self.__logger.debug(f'{cursor.rowcount} rows were affected')

        row_count_sql_command = f'SELECT COUNT(*) FROM {self._table_name};'
        results = database_helper.execute_sql_command(cursor, row_count_sql_command)
        value = results.fetchone()
        row_count = value[0]

        connection.commit()
        message = f'{row_count:,} rows were inserted into the `{self._table_name}` table'
        self.__logger.info(message)
