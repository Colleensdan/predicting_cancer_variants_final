import requests

from abc import ABC, abstractmethod
from requests.exceptions import HTTPError
from predicting_cancer_variants.Code.common.configuration import Configuration


# Inherits from Python's Abstract Base Class (ABC)
from predicting_cancer_variants.Code.common.database_functions import DatabaseHelper


class Downloader(ABC):

    # this base class constructor is implicitly called when a concrete subclass is instantiated
    def __init__(self):
        self._configuration = Configuration()
        self._logger = self._configuration.get_logger(__name__)

    @property
    @abstractmethod
    def _name(self):
        pass

    @property
    @abstractmethod
    def _url(self):
        pass

    @property
    @abstractmethod
    def _table(self):
        pass

    def download_and_save_all_data(self):
        self._logger.info(f'Downloading and saving the dataset for {self._name}')
        response_object = self._download(self._url)

        row_count = self._save(response_object)
        self._logger.info(f'{row_count:,} rows were inserted into the `{self._table}` table')

    def _download(self, url):
        try:
            response_object = requests.get(url)
            response_object.raise_for_status()
        except HTTPError as http_err:
            # log what URL caused the http error and re-raise
            message = "HTTP error occurred in {0} downloader requesting data from: '{1}'" \
                .format(self._name, url)
            self._logger.error(message)
            raise
        except Exception as e:
            message = "An error occurred in {0} downloader requesting data from: '{1}'" \
                .format(self._name, url)
            self._logger.error(message)
            raise

        return response_object

    def _save(self, response_object, row_count=0):
        # check if response body contains something
        if not response_object.text:
            self._logger.warning("Response body is empty")
            return 0

        # turn the response into a dictionary if it is valid JSON
        try:
            response_dictionary = response_object.json()
        except ValueError:
            self._logger.warning("Response body does not contain JSON")
            return 0

        # check for an empty dictionary
        if not response_dictionary:
            self._logger.info("No rows were returned")
            return 0

        # get the database ready
        database_helper = DatabaseHelper()
        connection = database_helper.create_database_if_missing_and_get_connection()
        cursor = connection.cursor()

        # Iterate through the list of rows returned
        for row in response_dictionary:

            # Maintain a row count
            row_count = row_count + 1

            # Get a comma separated list of column headers returned from the response row
            # These headers will be used to dynamically create a table that has these as column names
            column_names_list = list(row.keys())
            column_names_csv = ",".join(column_names_list)

            # Get a comma separated list of column values returned from the response
            # these need to be formatted in a specific way that keeps the SQL insert syntax valid
            # for types like None (null), bool, numbers and strings
            values_list = list(row.values())
            values = self.__make_sql_friendly_values(values_list)

            # Construct the insert statement
            sql_insert_statement = 'INSERT INTO {} ({}) VALUES ({});' \
                .format(self._table, column_names_csv, ", ".join(values))

            # Execute the insert statement
            database_helper.execute_sql_command(cursor, sql_insert_statement, f'Row number {row_count}')

            # Log every 100,000th row so we know that the program is still working and hasn't quietly died
            if row_count % 100000 == 0:
                self._logger.info(f'Processed row {row_count:,}')

        # After all rows have been inserted, commit the transaction and return the number of rows saved
        connection.commit()
        return row_count

    def __make_sql_friendly_values(self, query_values):
        # convert Python types of values into a form that is syntactically valid for a SQL insert statement
        return ["null" if value == "" or None or '' else
                "'{}'".format(str(value.replace("'", "''"))) if type(value) == str else
                str(value) if type(value) == int else
                str(value) if type(value) == float else
                str(1) if value else
                str(0) if not value else
                self.__raise_error("Unrecognised data type in values: {}".format(type(value)))
                for value in query_values]

    def __raise_error(self, e):
        raise Exception(e)
