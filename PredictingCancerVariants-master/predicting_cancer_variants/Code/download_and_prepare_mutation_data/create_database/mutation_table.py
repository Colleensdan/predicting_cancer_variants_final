from abc import ABC

from predicting_cancer_variants.Code.download_and_prepare_mutation_data.create_database.table import Table


# Inherit from Table abstract base class
class MutationTable(Table, ABC):

    @property
    def _table_name(self):
        return self._configuration.database_table_mutation

    @property
    def _create_table_segment(self):
        return """         
            (
                   uniqueSampleKey             VARCHAR(44)  
                  ,uniquePatientKey            VARCHAR(34) 
                  ,molecularProfileId          VARCHAR(25) 
                  ,sampleId                    VARCHAR(17) 
                  ,patientId                   VARCHAR(9) 
                  ,entrezGeneId                INTEGER  
                  ,studyId                     VARCHAR(15) 
                  ,center                      VARCHAR(9) 
                  ,mutationStatus              VARCHAR(2) 
                  ,validationStatus            VARCHAR(2) 
                  ,tumorAltCount               INTEGER  
                  ,tumorRefCount               INTEGER  
                  ,normalAltCount              INTEGER 
                  ,normalRefCount              INTEGER 
                  ,startPosition               INTEGER 
                  ,endPosition                 INTEGER  
                  ,referenceAllele             VARCHAR(61) 
                  ,proteinChange               VARCHAR(30) 
                  ,mutationType                VARCHAR(17) 
                  ,functionalImpactScore       VARCHAR(2)
                  ,fisValue                    NUMERIC(10,8) 
                  ,linkXvar                    VARCHAR(51)
                  ,linkPdb                     VARCHAR(61)
                  ,linkMsa                     VARCHAR(61)
                  ,ncbiBuild                   VARCHAR(6) 
                  ,variantType                 VARCHAR(3) 
                  ,keyword                     VARCHAR(45)
                  ,chr                         VARCHAR(2)
                  ,variantAllele               VARCHAR(62)
                  ,refSeqMrnaId                VARCHAR(43) 
                  ,proteinPosStart             INTEGER  
                  ,proteinPosEnd               INTEGER  
                  ,driverFilter                VARCHAR(30)
                  ,driverFilterAnnotation      VARCHAR(30)
                  ,driverTiersFilter           VARCHAR(30)
                  ,driverTiersFilterAnnotation VARCHAR(30)
                  
                  ,FOREIGN KEY(studyId) REFERENCES {0}(studyId)
                  ,FOREIGN KEY(entrezGeneId) REFERENCES {1}(entrezGeneId)
                  ,FOREIGN KEY(molecularProfileId) REFERENCES {2}(molecularProfileId)
            )
        """.format(
            self._configuration.database_table_study,
            self._configuration.database_table_gene,
            self._configuration.database_table_molecular)
