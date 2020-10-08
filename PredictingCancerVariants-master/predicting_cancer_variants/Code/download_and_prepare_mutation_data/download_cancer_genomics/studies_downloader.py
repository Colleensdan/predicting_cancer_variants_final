from abc import ABC

from predicting_cancer_variants.Code.download_and_prepare_mutation_data.download_cancer_genomics.downloader import Downloader


# Inherit from Downloader superclass and Python's Abstract Base Class (ABC)
class StudiesDownloader(Downloader, ABC):

    @property
    def _name(self):
        return "Studies"

    @property
    def _url(self):
        return self._configuration.study_url

    @property
    def _table(self):
        return self._configuration.database_table_study

