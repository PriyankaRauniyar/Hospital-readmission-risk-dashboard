"""Pure, testable transformations for dashboard encounter data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmark import CLASS_NAMES, CLASS_TO_ID


def early_readmission_counts(matrix: np.ndarray) -> dict[str, int]:
    """Review burden from actual rows and predicted columns in class order."""
    matrix = np.asarray(matrix)
    identified = int(matrix[1, 1])
    return {"identified": identified,
            "missed": int(matrix[1].sum()) - identified,
            "false_positive": int(matrix[:, 1].sum()) - identified}


ALIASES = {
    "Actual_outcome": ["Actual_outcome", "Actual outcome"],
    "Predicted_outcome": ["Predicted_outcome", "Predicted outcome"],
    "Actual_class": ["Actual_class", "Actual class"],
    "Predicted_class": ["Predicted_class", "Predicted class"],
    "Correct_prediction": ["Correct_prediction", "Correct prediction"],
    "Risk_30": ["Calibrated_risk_within_30_days", "Calibrated risk: Within 30 days"],
    "Risk_band": ["Within_30_day_risk_band", "Within_30_risk_band", "Risk_band", "Risk band"],
}


def normalize_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept the two known export naming styles without inventing missing values."""
    data = frame.copy()
    for canonical, names in ALIASES.items():
        if canonical not in data.columns:
            matching = next((name for name in names if name in data.columns), None)
            if matching:
                data[canonical] = data[matching]
    for label, numeric in (("Actual_outcome", "Actual_class"), ("Predicted_outcome", "Predicted_class")):
        if label not in data.columns and numeric in data.columns:
            data[label] = pd.to_numeric(data[numeric], errors="coerce").map(dict(enumerate(CLASS_NAMES)))
        if numeric not in data.columns and label in data.columns:
            data[numeric] = data[label].map(CLASS_TO_ID)
    required = ["Actual_class", "Predicted_class"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError("Prediction export lacks actual and predicted outcome columns: " + ", ".join(missing))
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if data[column].isna().any() or not data[column].isin([0, 1, 2]).all():
            raise ValueError(f"{column} must contain only 0, 1, 2")
        data[column] = data[column].astype(int)
    data["Actual_outcome"] = data["Actual_class"].map(dict(enumerate(CLASS_NAMES)))
    data["Predicted_outcome"] = data["Predicted_class"].map(dict(enumerate(CLASS_NAMES)))
    data["Correct_prediction"] = data["Actual_class"] == data["Predicted_class"]
    if "Risk_30" in data.columns:
        data["Risk_30"] = pd.to_numeric(data["Risk_30"], errors="coerce")
        if ((data["Risk_30"] < 0) | (data["Risk_30"] > 1)).any():
            raise ValueError("Calibrated 30-day risk must be between zero and one")
    return data


def confusion(data: pd.DataFrame) -> pd.DataFrame:
    counts = pd.crosstab(data["Actual_outcome"], data["Predicted_outcome"])
    return counts.reindex(index=CLASS_NAMES, columns=CLASS_NAMES, fill_value=0)


def basic_metrics(data: pd.DataFrame) -> dict[str, float]:
    matrix = confusion(data).to_numpy()
    tp = np.diag(matrix)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros(3, dtype=float), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(3, dtype=float), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3, dtype=float), where=(precision + recall) > 0)
    return {
        "encounters": len(data),
        "accuracy": float(tp.sum() / len(data)) if len(data) else 0.0,
        "macro_f1": float(f1.mean()),
        "within_30_recall": float(recall[1]),
        "within_30_precision": float(precision[1]),
        "within_30_f1": float(f1[1]),
        "within_30_actual": int(support[1]),
        "within_30_identified": int(tp[1]),
    }


def subgroup_table(data: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for group, subset in data.groupby(column, dropna=False, observed=True):
        m = basic_metrics(subset)
        rows.append({"Group": str(group), "Encounters": m["encounters"],
                     "Actual within 30": m["within_30_actual"],
                     "Within 30 recall": m["within_30_recall"],
                     "Accuracy": m["accuracy"], "Macro F1": m["macro_f1"]})
    return pd.DataFrame(rows).sort_values("Encounters", ascending=False)
