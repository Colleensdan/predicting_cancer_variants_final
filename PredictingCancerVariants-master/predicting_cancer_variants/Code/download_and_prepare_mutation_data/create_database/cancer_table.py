from abc import ABC

from predicting_cancer_variants.Code.download_and_prepare_mutation_data.create_database.table import Table


# Inherit from Table abstract base class
class CancerTable(Table, ABC):

    @property
    def _table_name(self):
        return self._configuration.database_table_cancer

    @property
    def _create_table_segment(self):
        return """         
            (
                  cancerTypeId           VARCHAR(30) NOT NULL PRIMARY KEY
                  ,name                  VARCHAR(30) NOT NULL
                  ,clinicalTrialKeywords VARCHAR(30)
                  ,dedicatedColor        VARCHAR(30)
                  ,shortName             VARCHAR(30)
                  ,parent                VARCHAR(30)
            )
        """