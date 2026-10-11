"""Public export safeguards, independent of private datasets and model artifacts."""
import pandas as pd

from benchmark import CLASS_NAMES
from public_predictions import PUBLIC_COLUMNS, validate_public_predictions


def example():
    return pd.DataFrame([[1, 1, CLASS_NAMES[1], .1, .7, .2, .9, .05, .05]], columns=PUBLIC_COLUMNS)


def rejects(data):
    try:
        validate_public_predictions(data)
    except ValueError:
        return
    raise AssertionError('Invalid public export accepted')


def test_public_decision_uses_uncalibrated_probabilities():
    assert validate_public_predictions(example()).iloc[0]['Predicted_class'] == 1


def test_public_export_rejects_source_columns():
    for name in ['patient_nbr', 'encounter_id', 'readmitted', 'Actual_outcome', 'race']:
        rejects(example().assign(**{name: 'private'}))


def test_public_export_rejects_invalid_probabilities_and_labels():
    for name, value in [('Uncalibrated: No readmission', .8),
                        ('Calibrated risk: No readmission', float('nan')),
                        ('Predicted_class', 0), ('Predicted_outcome', CLASS_NAMES[0]),
                        ('Display_row', 22)]:
        data = example()
        data[name] = value
        rejects(data)
