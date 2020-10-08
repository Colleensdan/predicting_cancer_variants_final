from predicting_cancer_variants.Code.common.configuration import Configuration
from predicting_cancer_variants.Code.download_and_prepare_mutation_data.prepare_machine_learning_data.net_data_table \
    import NetDataTable
from predicting_cancer_variants.Code.download_and_prepare_mutation_data.prepare_machine_learning_data.patient_table \
    import PatientTable
from predicting_cancer_variants.Code.download_and_prepare_mutation_data.prepare_machine_learning_data.staging_table \
    import StagingTable


def run():
    logger = Configuration().get_logger(__name__)
    try:
        logger.info("Preparing machine learning data")

        StagingTable().prepare()
        PatientTable().prepare()
        NetDataTable().prepare()

        logger.info("Preparation of machine learning data complete")

    except Exception as e:
        message = "Failed to prepare machine learning data"
        logger.exception(message, exc_info=e)
        raise SystemExit(message)


if __name__ == '__main__':
    run()
