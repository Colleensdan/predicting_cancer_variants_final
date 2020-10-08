from predicting_cancer_variants.Code.common.configuration import Configuration
from .download_cancer_genomics.cancer_downloader import CancerDownloader
from .download_cancer_genomics.gene_downloader import GeneDownloader
from .download_cancer_genomics.molecular_downloader import MolecularDownloader
from .download_cancer_genomics.mutation_downloader import MutationDownloader
from .download_cancer_genomics.studies_downloader import StudiesDownloader


def run():
    logger = Configuration().get_logger(__name__)
    try:
        logger.info("Downloading and storing data from large-scale online Cancer Genomics database cBioPortal.org")

        StudiesDownloader().download_and_save_all_data()
        GeneDownloader().download_and_save_all_data()
        CancerDownloader().download_and_save_all_data()
        MolecularDownloader().download_and_save_data_every_large_study()
        MutationDownloader().download_and_save_data_every_large_study()

        logger.info("Downloads complete")

    except Exception as e:
        message = "Failed to download and store data"
        logger.exception(message, exc_info=e)
        raise SystemExit(message)


if __name__ == '__main__':
    run()