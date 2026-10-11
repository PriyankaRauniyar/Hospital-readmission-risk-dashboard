from io import BytesIO
from unittest.mock import patch
from zipfile import ZipFile

import pandas as pd
from uci_examples import PRIVATE_COLUMNS, fetch_uci_examples, scoring_records


def test_uci_fetch_strips_identifiers_and_target():
    archive = BytesIO()
    with ZipFile(archive, 'w') as zipped:
        zipped.writestr('dataset/diabetic_data.csv', 'encounter_id,patient_nbr,readmitted,race,gender,age,discharge_disposition_id\n1,2,<30,Other,Female,[40-50),1\n')
    with patch('uci_examples.urlopen', return_value=BytesIO(archive.getvalue())):
        data = fetch_uci_examples()
    assert not set(PRIVATE_COLUMNS).intersection(data.columns)
    assert len(data) == 1 and 'discharge_disposition_id' in data
    assert scoring_records(pd.DataFrame({'Actual_outcome': ['Within 30 days'], 'race': ['Other']})).columns.tolist() == ['race']


def test_uci_fetch_failure_has_actionable_message():
    with patch('uci_examples.urlopen', side_effect=OSError('offline')):
        try:
            fetch_uci_examples()
        except ValueError as exc:
            assert 'CSV upload' in str(exc)
        else:
            raise AssertionError('Download failure accepted')
