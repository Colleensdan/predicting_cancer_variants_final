from predicting_cancer_variants.Code.common.database_functions import DatabaseHelper
from predicting_cancer_variants.Code.common.configuration import Configuration
import pandas as pd


class DataRetriever:
    def __init__(self):
        self.__database_helper = DatabaseHelper()
        self.__configuration = Configuration()
        self.__logger = self.__configuration.get_logger(__name__)

    def fetch(self):
        net_data = self.__database_helper.get_rows(f"select * from {self.__configuration.database_table_net}")
        message = "Data fetched from database: OK"
        self.__logger.info(message)
        message = "Starting to compile data into pandas dataframe"
        self.__logger.info(message)
        data_frame = pd.DataFrame(net_data, columns=['patient', 'Protein', 'Gene', "MtType", "cType", "studyId"])
        message = "Data parsed into dataframe: OK"
        self.__logger.info(message)
        return data_frame
