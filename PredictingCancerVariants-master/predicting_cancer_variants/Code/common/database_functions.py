import sqlite3

from predicting_cancer_variants.Code.common.configuration import Configuration


class DatabaseHelper:

    def __init__(self):
        self._configuration = Configuration()
        self.__logger = self._configuration.get_logger(__name__)

    def create_database_if_missing_and_get_connection(self):
        """ get a database connection to the SQLite database as specified by configuration,
        create the database if does not already exist; log the failed location if this does not work
        :return: New SQLLite connection
        """
        database_location = self._configuration.database_location
        try:
            connection = sqlite3.connect(database_location)
        except sqlite3.Error:
            # log the location that failed and re-raise the exception
            message = f'Could not connect to or create database. The failed location is `{database_location}`'
            self.__logger.configuration.get_logger(__name__).error(message)
            raise

        return connection

    def execute_sql_command(self, cursor, sql_command, extra_information=None):
        """ run a sql command, log the failed command if this does not work
        :param cursor: SQLLite cursor object associated with an existing connection
        :param sql_command: sql command to execute
        :param extra_information: extra context-sensitive information to log if something goes wrong
        :return: results of execute statement
        """
        try:
            return cursor.execute(sql_command)
        except sqlite3.Error as e:
            # log the failed command and re-raise
            message = "failed sql command is: {}".format(sql_command)
            self.__logger.error(message)
            if extra_information:
                self.__logger.error(extra_information)

            raise

    def get_rows(self, sql_command, extra_information=None):
        """ run a sql select command and return a list of values
        :param sql_command: sql command to execute
        :param extra_information: extra context-sensitive information to log if something goes wrong
        :return: list of values returned from the select statement
        """
        connection = self.create_database_if_missing_and_get_connection()
        cursor = connection.cursor()
        self.execute_sql_command(cursor, sql_command, extra_information)
        rows = cursor.fetchall()
        return rows
