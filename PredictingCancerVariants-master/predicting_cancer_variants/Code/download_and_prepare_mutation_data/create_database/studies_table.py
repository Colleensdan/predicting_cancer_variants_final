from abc import ABC

from predicting_cancer_variants.Code.download_and_prepare_mutation_data.create_database.table import Table


# Inherit from Table abstract base class
class StudiesTable(Table, ABC):

    @property
    def _table_name(self):
        return self._configuration.database_table_study

    @property
    def _create_table_segment(self):
        return """         
                (
                     studyID                VARCHAR(100) NOT NULL PRIMARY KEY
                     ,name                  VARCHAR(200) NOT NULL
                     ,shortName             VARCHAR(200) NOT NULL
                     ,allSampleCount        INTEGER      NOT NULL
                     ,cancerTypeId          VARCHAR(5)   NOT NULL 
                     ,referenceGenome       VARCHAR(10)  NOT NULL
                     ,description           VARCHAR(200)
                     ,publicStudy           INTEGER
                     ,pmID                  VARCHAR(60) 
                     ,groups                VARCHAR(200)
                     ,citation              VARCHAR(300)
                     ,status                VARCHAR(200)
                     ,importDate            DATETIME,
                     
                     FOREIGN KEY(cancerTypeId) REFERENCES {0}(cancerTypeId)
                )
        """.format(self._configuration.database_table_cancer)
