"""Strict public-export schema and probability checks; no source fields allowed."""
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark import CLASS_NAMES

BATCH_SIZE = 2000
UNCALIBRATED = [f"Uncalibrated: {name}" for name in CLASS_NAMES]
CALIBRATED = [f"Calibrated risk: {name}" for name in CLASS_NAMES]
PUBLIC_COLUMNS = ["Display_row", "Predicted_class", "Predicted_outcome", *UNCALIBRATED, *CALIBRATED]


def validate_public_predictions(data: pd.DataFrame) -> pd.DataFrame:
    if data.columns.tolist() != PUBLIC_COLUMNS:
        raise ValueError("Public predictions must contain only the approved display and prediction columns, in order.")
    if data.empty:
        raise ValueError("Public predictions file has no encounters.")
    if not np.array_equal(data['Display_row'].to_numpy(), np.arange(1, len(data) + 1)):
        raise ValueError("Display row numbers must be consecutive from one.")
    classes = data['Predicted_class'].to_numpy()
    if not np.isin(classes, [0, 1, 2]).all():
        raise ValueError("Unexpected predicted class order.")
    expected = pd.Series(classes).map(dict(enumerate(CLASS_NAMES))).to_numpy()
    if not np.array_equal(data['Predicted_outcome'].to_numpy(), expected):
        raise ValueError("Predicted outcomes do not match class labels.")
    for columns in (UNCALIBRATED, CALIBRATED):
        values = data[columns].to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Public probabilities must be finite and between zero and one.")
        if not np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-7):
            raise ValueError("Public probability rows must sum to one.")
    if not np.array_equal(classes, data[UNCALIBRATED].to_numpy().argmax(axis=1)):
        raise ValueError("Public classes must follow uncalibrated argmax.")
    return data


def load_public_predictions(path: Path, compression="infer") -> pd.DataFrame:
    return validate_public_predictions(pd.read_csv(path, compression=compression))
