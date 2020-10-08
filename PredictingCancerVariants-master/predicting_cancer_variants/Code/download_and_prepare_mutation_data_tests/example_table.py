from abc import ABC

from predicting_cancer_variants.Code.download_and_prepare_mutation_data.create_database.table import Table


class ExampleTable(Table, ABC):

    @property
    def _table_name(self):
        return "Example_Table"

    @property
    def _create_table_segment(self):
        return """         
(
    Example_Id INTEGER NOT NULL PRIMARY KEY
    Example_Name VARCHAR(50) NOT NULL
)
"""