from abc import ABC

# Inherit from PrepareForMachineLearning abstract base class
from predicting_cancer_variants.Code.download_and_prepare_mutation_data.prepare_machine_learning_data.prepare_for_machine_learning import \
    PrepareForMachineLearning


class StagingTable(PrepareForMachineLearning, ABC):

    @property
    def _table_name(self):
        return self._configuration.database_table_staging

    @property
    def _message(self):
        return "Preparing the staging table by flattening and indexing gene mutation data"

    @property
    def _sql_commands(self):
        create_flattened_data_table = """
            CREATE TABLE IF NOT EXISTS {0} AS 
            SELECT 
                uniqueSampleKey, uniquePatientKey, molecularProfileId, mutation.studyId, mutation.entrezGeneId, 
                mutation.sampleId, startPosition, endPosition, referenceAllele, variantAllele, proteinChange, 
                mutationType, functionalImpactScore, fisValue, variantType, keyword, chr, refseqMrnaId, tumorAltCount,
                tumorRefCount, cancer_ref.cancerTypeId, gene.hugoGeneSymbol, parent tissue, cancer_ref.name cancer_name 
            FROM {1} cancer_ref 
                INNER JOIN {2} study ON study.cancerTypeId = cancer_ref.cancerTypeId
                INNER JOIN {3} mutation ON study.studyId = mutation.studyId 
                INNER JOIN {4} gene ON mutation.entrezGeneId = gene.entrezGeneId
        """.format(
            self._table_name,
            self._configuration.database_table_cancer,
            self._configuration.database_table_study,
            self._configuration.database_table_mutation,
            self._configuration.database_table_gene
        )

        create_index_on_flattened_table = \
            f'CREATE INDEX IF NOT EXISTS NumericId ON {self._configuration.database_table_staging} (uniquePatientKey)'

        return [create_flattened_data_table, create_index_on_flattened_table]
