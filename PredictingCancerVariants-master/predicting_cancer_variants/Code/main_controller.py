from predicting_cancer_variants.Code.common.configuration import Configuration
from predicting_cancer_variants.Code.download_and_prepare_mutation_data import create_database_controller, \
    download_cancer_genomics_controller, prepare_machine_learning_data_controller


def run():
    logger = Configuration().get_logger(__name__)
    try:
        logger.info("Starting predicting_cancer_variants")

        create_database_controller.run()
        download_cancer_genomics_controller.run()
        prepare_machine_learning_data_controller.run()

        logger.info("predicting_cancer_variants complete")

    except Exception as e:
        message = "predicting_cancer_variants has failed"
        logger.exception(message, exc_info=e)
        raise SystemExit(message)


if __name__ == '__main__':
    run()
