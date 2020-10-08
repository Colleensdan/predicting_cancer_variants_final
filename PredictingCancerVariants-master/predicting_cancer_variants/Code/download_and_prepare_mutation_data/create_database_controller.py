from predicting_cancer_variants.Code.common.configuration import Configuration
from .create_database.gene_table import GeneTable
from .create_database.studies_table import StudiesTable
from .create_database.cancer_table import CancerTable
from .create_database.molecular_table import MolecularTable
from .create_database.mutation_table import MutationTable


def run():
    logger = Configuration().get_logger(__name__)
    try:
        logger.info("Creating tables")

        StudiesTable().create()
        CancerTable().create()
        MolecularTable().create()
        MutationTable().create()
        GeneTable().create()

        logger.info("Table creation complete")

    except Exception as e:
        message = "Failed to create tables"
        logger.exception(message, exc_info=e)
        raise SystemExit(message)


if __name__ == '__main__':
    run()