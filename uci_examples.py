"""Fetch official UCI 296 encounters in memory; retain no identifiers or target."""
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile, BadZipFile

import pandas as pd

UCI_ARCHIVE_URL = 'https://archive.ics.uci.edu/static/public/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008.zip'
PRIVATE_COLUMNS = ['patient_nbr', 'encounter_id', 'readmitted', 'Actual_class', 'Actual_outcome',
                   'Actual class', 'Actual outcome', 'Encounter_record']


def scoring_records(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop(columns=PRIVATE_COLUMNS, errors='ignore').reset_index(drop=True)


def fetch_uci_examples() -> pd.DataFrame:
    """Download with timeout and size bounds; never extract or write the archive."""
    try:
        with urlopen(UCI_ARCHIVE_URL, timeout=30) as response:
            archive = response.read(20 * 1024 * 1024 + 1)
        if len(archive) > 20 * 1024 * 1024:
            raise ValueError('Unexpected archive size')
        with ZipFile(BytesIO(archive)) as zipped:
            member = next((item for item in zipped.infolist()
                           if item.filename.split('/')[-1] == 'diabetic_data.csv'), None)
            if member is None or member.file_size > 50 * 1024 * 1024:
                raise ValueError('Expected encounter CSV missing or too large')
            with zipped.open(member) as source:
                data = pd.read_csv(source, low_memory=False)
        required = {'race', 'gender', 'age', 'discharge_disposition_id'}
        if not required.issubset(data.columns) or data.empty:
            raise ValueError('Unexpected UCI encounter schema')
        return scoring_records(data)
    except (OSError, ValueError, BadZipFile, pd.errors.ParserError) as exc:
        raise ValueError('UCI example data could not be loaded. The source may be temporarily unavailable. Try Load UCI example data again or use a CSV upload.') from exc
