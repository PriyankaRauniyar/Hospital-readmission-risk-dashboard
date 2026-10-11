import pandas as pd
from display_formatting import percentage_table


def test_percentage_display_preserves_source_and_hides_float_artifacts():
    source = pd.DataFrame({'Accuracy': [.5258], 'Macro F1': [.4642], 'Within 30 recall': [.3902]})
    before = source.copy(deep=True)
    display = percentage_table(source, source.columns)
    assert display.iloc[0].tolist() == ['52.58%', '46.42%', '39.02%']
    pd.testing.assert_frame_equal(source, before)
