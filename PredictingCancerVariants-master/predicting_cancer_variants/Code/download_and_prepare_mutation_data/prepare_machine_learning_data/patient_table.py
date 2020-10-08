from abc import ABC

# Inherit from PrepareForMachineLearning abstract base class
from predicting_cancer_variants.Code.download_and_prepare_mutation_data.prepare_machine_learning_data.prepare_for_machine_learning import \
    PrepareForMachineLearning


class PatientTable(PrepareForMachineLearning, ABC):

    @property
    def _table_name(self):
        return self._configuration.database_table_patient

    @property
    def _message(self):
        return "Preparing the patient table by creating an auto-incrementing number for every unique patient"

    @property
    def _sql_commands(self):
        create_patient_table = """
            CREATE TABLE IF NOT EXISTS {0} 
            (
                NumericId INTEGER PRIMARY KEY AUTOINCREMENT,
                StringId VARCHAR
            ); 
         """.format(self._table_name)

        populate_patient = """
            INSERT INTO {0} (StringId)
            SELECT DISTINCT uniquePatientKey 
            FROM {1}
        """.format(
            self._table_name,
            self._configuration.database_table_staging)

        return [create_patient_table, populate_patient]
