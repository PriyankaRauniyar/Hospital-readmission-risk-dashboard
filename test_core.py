import json

import numpy as np
import pandas as pd

from analytics import basic_metrics, confusion, early_readmission_counts, normalize_predictions
from benchmark import CLASS_NAMES, CONFUSION
from predictor import group_diagnosis, prepare_features, predict


def test_early_readmission_review_burden_uses_actual_rows():
    # Asymmetric counts catch accidental reversal of actual/predicted axes.
    assert early_readmission_counts([[8, 2, 0], [3, 4, 5], [0, 7, 9]]) == {
        "identified": 4, "missed": 8, "false_positive": 9}
    assert early_readmission_counts(CONFUSION) == {
        "identified": 879, "missed": 1337, "false_positive": 3152}


def test_frozen_confusion_totals():
    matrix = np.asarray(CONFUSION)
    assert matrix.sum() == 19802
    assert matrix.sum(axis=1).tolist() == [10496, 2216, 7090]
    assert matrix.trace() == 10382
    assert matrix[1, 1] / matrix[1].sum() > .396 and matrix[1, 1] / matrix[1].sum() < .397


def test_prediction_schema_and_metrics():
    frame = pd.DataFrame({"Actual outcome": [CLASS_NAMES[0], CLASS_NAMES[1], CLASS_NAMES[2]],
                          "Predicted outcome": [CLASS_NAMES[0], CLASS_NAMES[2], CLASS_NAMES[2]]})
    data = normalize_predictions(frame)
    assert confusion(data).to_numpy().tolist() == [[1, 0, 0], [0, 0, 1], [0, 0, 1]]
    assert basic_metrics(data)["accuracy"] == 2/3


def test_preparation_preserves_feature_order_and_exclusions():
    raw = pd.DataFrame({"diag_1": ["250.13"], "diag_2": ["V45"], "diag_3": ["401"],
                        "number_inpatient": [2], "number_outpatient": [1], "number_emergency": [0],
                        "time_in_hospital": [4], "num_lab_procedures": [12],
                        "num_medications": [8], "num_procedures": [2], "insulin": ["Up"],
                        "discharge_disposition_id": [1], "admission_type_id": [1]})
    config = {"feature_order": list(raw.columns) + ["diag_1_group", "diag_2_group", "diag_3_group",
              "total_prior_visits", "active_diabetes_medications", "labs_per_day"],
              "categorical_features": ["diag_1", "diag_2", "diag_3", "insulin",
                                       "discharge_disposition_id", "admission_type_id",
                                       "diag_1_group", "diag_2_group", "diag_3_group"]}
    data = prepare_features(raw, config)
    assert data.columns.tolist() == config["feature_order"]
    assert data.loc[0, "total_prior_visits"] == 3
    assert data.loc[0, "labs_per_day"] == 3
    assert data.loc[0, "active_diabetes_medications"] == 1
    assert data.loc[0, "diag_1_group"] == "Diabetes"
    raw.loc[0, "discharge_disposition_id"] = 11
    try:
        prepare_features(raw, config)
    except ValueError as e:
        assert "excluded" in str(e)
    else:
        raise AssertionError("Hospice/death cohort row was accepted")


def test_frozen_prediction_decision_remains_uncalibrated():
    class FakeModel:
        classes_ = np.array([0, 1, 2])
        def __init__(self, p): self.p = p
        def predict_proba(self, data): return np.asarray([self.p])
    class FakePrep:
        def transform(self, data): return data
    class FakeCal:
        classes_ = np.array([0, 1, 2])
        def predict_proba(self, data): return np.asarray([[.9, .05, .05]])
    raw = pd.DataFrame({"diag_1": ["250.13"], "diag_2": ["V45"], "diag_3": ["401"],
                        "number_inpatient": [2], "number_outpatient": [1], "number_emergency": [0],
                        "time_in_hospital": [4], "num_lab_procedures": [12], "num_medications": [8],
                        "num_procedures": [2], "discharge_disposition_id": [1]})
    cfg = {"feature_order": list(raw.columns), "categorical_features": ["diag_1", "diag_2", "diag_3", "discharge_disposition_id"]}
    scored = predict(raw, {"cat": FakeModel([.1, .7, .2]), "xgb": FakeModel([.1, .7, .2]),
                           "preprocessor": FakePrep(), "calibrator": FakeCal(), "config": cfg})
    assert scored.loc[0, "Predicted_class"] == 1
    assert scored.loc[0, "Calibrated risk: No readmission"] == .9
