from abc import ABC

# Inherit from PrepareForMachineLearning abstract base class
from predicting_cancer_variants.Code.download_and_prepare_mutation_data.prepare_machine_learning_data.prepare_for_machine_learning import \
    PrepareForMachineLearning


class NetDataTable(PrepareForMachineLearning, ABC):

    @property
    def _table_name(self):
        return self._configuration.database_table_net

    @property
    def _message(self):
        return "Preparing the net data table"

    @property
    def _sql_commands(self):
        create_net_data_table = """
             CREATE TABLE IF NOT EXISTS {0} AS
             SELECT 
                DISTINCT NumericId, proteinChange, hugoGeneSymbol, mutationType, cancer_name, studyId
             FROM {1}
                INNER JOIN {2} ON uniquePatientKey = stringId
             ORDER BY 
                NumericId
        """.format(
            self._table_name,
            self._configuration.database_table_staging,
            self._configuration.database_table_patient
        )

        return [create_net_data_table]