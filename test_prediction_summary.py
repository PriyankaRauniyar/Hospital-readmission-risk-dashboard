import pandas as pd
from prediction_summary import describe_predictions
from public_predictions import PUBLIC_COLUMNS


def test_summary_tracks_prediction_mix_and_calibration_disagreement():
    data = pd.DataFrame([
        [1, 1, 'Within 30 days', .1, .7, .2, .9, .05, .05],
        [2, 0, 'No readmission', .6, .3, .1, .5, .2, .3],
    ], columns=PUBLIC_COLUMNS)
    text = '\n'.join(describe_predictions(data))
    assert '1 (50.0%) within 30 days' in text
    assert 'averages 12.5%' in text
    assert 'median of 12.5%' in text
    assert 'mean calibrated early-readmission risk is 5.0%' in text
    assert '1 of 2 rows (50.0%)' in text
    second = '\n'.join(describe_predictions(data.iloc[1:]))
    assert 'No encounter in this batch' in second
    assert 'averages 20.0%' in second
    assert '0 of 1 rows (0.0%)' in second
    assert describe_predictions(data.iloc[:0]) == []
