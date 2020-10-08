from abc import ABC

from predicting_cancer_variants.Code.download_and_prepare_mutation_data.create_database.table import Table


# Inherit from Table abstract base class
class MolecularTable(Table, ABC):

    @property
    def _table_name(self):
        return self._configuration.database_table_molecular

    @property
    def _create_table_segment(self):
        return """         
             (
                   molecularProfileId       VARCHAR(50) NOT NULL PRIMARY KEY
                  ,molecularAlterationType  VARCHAR(30) NOT NULL 
                  ,dataType                 VARCHAR(5)  NOT NULL
                  ,name                     VARCHAR(40) NOT NULL
                  ,studyId                  VARCHAR(50) NOT NULL
                  ,description              VARCHAR(200)
                  ,genericAssayType         VARCHAR(50)
                  ,showProfileInAnalysisTab VARCHAR(10)
                  ,pivotThreshold           VARCHAR(200)
                  ,sortOrder                VARCHAR(10)
                  
                  ,FOREIGN KEY(studyId) REFERENCES {0}(studyId)
            )
        """.format(self._configuration.database_table_study)