"""Frozen values transcribed from the final reporting notebook (untouched test)."""

CLASS_NAMES = ["No readmission", "Within 30 days", "After 30 days"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# Rows are actual classes; columns are the frozen, uncalibrated ensemble decision.
CONFUSION = [
    [6433, 1522, 2541],
    [678, 879, 659],
    [2390, 1630, 3070],
]

TEST_METRICS = {
    "encounters": 19802,
    "accuracy": 0.5243,
    "macro_f1": 0.4615,
    "within_30_recall": 0.3967,
    "log_loss": 0.9884,
    "early_precision": 0.2181,
    "early_f1": 0.2814,
}

VALIDATION_MODELS = [
    {"Model": "CatBoost + XGBoost (60/40)", "Accuracy": 0.5258, "Macro F1": 0.4642, "Log loss": 0.9872, "Within 30 recall": 0.3902},
    {"Model": "XGBoost", "Accuracy": 0.5217, "Macro F1": 0.4596, "Log loss": 0.9818, "Within 30 recall": 0.3731},
    {"Model": "CatBoost", "Accuracy": 0.5180, "Macro F1": 0.4592, "Log loss": 0.9977, "Within 30 recall": 0.4047},
    {"Model": "Embedding neural network", "Accuracy": 0.4926, "Macro F1": 0.4406, "Log loss": 1.0071, "Within 30 recall": 0.4400},
    {"Model": "Two-stage XGBoost", "Accuracy": 0.5629, "Macro F1": 0.4147, "Log loss": 0.9186, "Within 30 recall": 0.1900},
]

CALIBRATION = [
    {"Class": "No readmission", "Observed": 0.5300, "Mean before": 0.3723, "Mean after": 0.5374, "Brier before": 0.2444, "Brier after": 0.2187, "ECE before": 0.1578, "ECE after": 0.0093},
    {"Class": "Within 30 days", "Observed": 0.1119, "Mean before": 0.2955, "Mean after": 0.1150, "Brier before": 0.1305, "Brier after": 0.0938, "ECE before": 0.1836, "ECE after": 0.0064},
    {"Class": "After 30 days", "Observed": 0.3580, "Mean before": 0.3323, "Mean after": 0.3476, "Brier before": 0.2161, "Brier after": 0.2120, "ECE before": 0.0298, "ECE after": 0.0116},
]

FEATURE_IMPORTANCE = [
    ("discharge_disposition_id", 8.77), ("diag_1", 8.19),
    ("number_inpatient", 5.25), ("diag_2", 4.99),
    ("payer_code", 4.78), ("total_prior_visits", 4.75),
    ("diag_3", 4.07), ("medical_specialty", 4.02),
    ("diag_1_group", 3.63), ("age", 3.31),
    ("labs_per_day", 3.08), ("had_prior_inpatient_visit", 2.91),
    ("diag_3_group", 2.87), ("admission_source_id", 2.86),
    ("diag_2_group", 2.55), ("medications_per_day", 2.51),
    ("num_medications", 2.29), ("insulin", 2.14),
    ("admission_type_id", 2.10), ("num_lab_procedures", 2.10),
]
