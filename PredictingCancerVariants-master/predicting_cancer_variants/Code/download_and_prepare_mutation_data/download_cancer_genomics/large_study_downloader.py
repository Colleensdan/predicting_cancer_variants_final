from abc import ABC, abstractmethod

from predicting_cancer_variants.Code.common import database_functions
from predicting_cancer_variants.Code.common.database_functions import DatabaseHelper
from predicting_cancer_variants.Code.download_and_prepare_mutation_data.download_cancer_genomics.downloader import \
    Downloader


# Inherit from Downloader superclass and Python's Abstract Base Class (ABC)
class LargeStudyDownloader(Downloader, ABC):

    @property
    @abstractmethod
    def _name(self):
        pass

    @property
    @abstractmethod
    def _url(self):
        pass

    @property
    @abstractmethod
    def _table(self):
        pass

    @property
    @abstractmethod
    def _minimum_number_of_samples_per_study(self):
        pass

    @property
    @abstractmethod
    def _maximum_number_of_samples_per_study(self):
        pass

    @abstractmethod
    def _make_url_for_study(self, study_id):
        pass

    def download_and_save_data_every_large_study(self):
        # get the study_id of every study with a sample size greater than the configuration value
        ids_of_large_studies = self.get_all_large_studies()

        # ensure there is at least one study
        if ids_of_large_studies.count == 0:
            self._logger.warning(f'No large studies found with a sample size greater than '
                                 f'{self._minimum_number_of_samples_per_study}')
            return 0

        self._logger.info(f'Downloading and saving {len(ids_of_large_studies)} different studies '
                          f'for the dataset {self._name}')

        row_count = 0
        study_count = 0
        for study_id in ids_of_large_studies:
            study_count = study_count + 1

            self._logger.info(
                f'Downloading and saving the dataset for {self._name}, study number {study_count}, {study_id}')

            # Form the request from a known URL combined with a study_id
            url_for_study = self._make_url_for_study(study_id)
            response_object = self._download(url_for_study)

            # instead of starting at zero for each study, this row_count will accumulate for all studies
            row_count = self._save(response_object, row_count)

        # Show how many rows were inserted in total for all studies
        self._logger.info(f'{row_count:,} rows were inserted into the `{self._table}` table')

    def get_all_large_studies(self):
        large_studies_sql_command = f'select studyID from {self._configuration.database_table_study} ' \
                                    f'where allSampleCount between {self._minimum_number_of_samples_per_study} ' \
                                    f'and {self._maximum_number_of_samples_per_study} order by allSampleCount desc '

        database_helper = DatabaseHelper()
        rows = database_helper.get_rows(large_studies_sql_command)

        large_studies = []
        for row in rows:
            large_studies.append(row[0])

        return large_studies
