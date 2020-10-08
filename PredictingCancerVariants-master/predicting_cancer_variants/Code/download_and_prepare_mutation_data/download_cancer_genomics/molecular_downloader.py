from abc import ABC

# Inherit from LargeStudyDownloader, which itself inherits from Downloader superclass Abstract Base Class (ABC)
from predicting_cancer_variants.Code.download_and_prepare_mutation_data.download_cancer_genomics.large_study_downloader import \
    LargeStudyDownloader


class MolecularDownloader(LargeStudyDownloader, ABC):

    @property
    def _name(self):
        return "Molecular Profiles"

    @property
    def _url(self):
        return self._configuration.molecular_url

    @property
    def _table(self):
        return self._configuration.database_table_molecular

    @property
    def _minimum_number_of_samples_per_study(self):
        return self._configuration.minimum_samples_per_study

    @property
    def _maximum_number_of_samples_per_study(self):
        return self._configuration.maximum_samples_per_study

    def _make_url_for_study(self, study_id):
        return self._url.format(study_id)
