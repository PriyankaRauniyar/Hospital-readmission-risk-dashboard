"""Load and apply the frozen 60/40 model bundle from the reporting notebook."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from benchmark import CLASS_NAMES


MODEL_FILES = (
    "final_catboost_model.cbm", "final_xgboost_model.json",
    "final_xgb_preprocessor.joblib", "probability_calibrator.joblib",
    "model_configuration.json",
)
EXCLUDED_DISCHARGE_CODES = {"11", "13", "14", "19", "20", "21"}
UNKNOWN_DISCHARGE_CODES = {"18", "25", "26", "?", "nan", "None", ""}
ENGINEERED_COLUMNS = {
    "diag_1_group", "diag_2_group", "diag_3_group", "total_prior_visits",
    "had_prior_inpatient_visit", "had_prior_emergency_visit",
    "active_diabetes_medications", "medication_change_count", "labs_per_day",
    "medications_per_day", "procedures_per_day", "total_procedure_burden",
}
DIABETES_DRUGS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]


def group_diagnosis(code: object) -> str:
    if pd.isna(code) or str(code).strip() in {"?", "Unknown", "nan", "None"}:
        return "Unknown"
    code = str(code).strip()
    if code.startswith(("V", "E")):
        return "Supplementary_or_external"
    try:
        value = float(code)
    except ValueError:
        return "Other"
    if 250 <= value < 251:
        return "Diabetes"
    if 390 <= value <= 459 or int(value) == 785:
        return "Circulatory"
    if 460 <= value <= 519 or int(value) == 786:
        return "Respiratory"
    if 520 <= value <= 579 or int(value) == 787:
        return "Digestive"
    if 580 <= value <= 629 or int(value) == 788:
        return "Genitourinary"
    if 140 <= value <= 239:
        return "Neoplasm"
    if 710 <= value <= 739:
        return "Musculoskeletal"
    if 800 <= value <= 999:
        return "Injury"
    return "Other"


def _code_as_text(series: pd.Series) -> pd.Series:
    return series.map(lambda x: str(int(x)) if isinstance(x, (int, float, np.integer, np.floating)) and pd.notna(x) and float(x).is_integer() else str(x))


def prepare_features(raw: pd.DataFrame, configuration: dict) -> pd.DataFrame:
    """Recreate the notebook's engineered feature order and category handling."""
    feature_order = configuration["feature_order"]
    categoricals = configuration["categorical_features"]
    required_raw = [name for name in feature_order if name not in ENGINEERED_COLUMNS]
    missing = [name for name in required_raw if name not in raw.columns]
    if missing:
        raise ValueError("Missing source features: " + ", ".join(missing))
    if len(raw) == 0:
        raise ValueError("The uploaded CSV has no rows")
    data = raw[required_raw].copy().replace("?", "Unknown")
    for name in ("admission_type_id", "admission_source_id", "discharge_disposition_id"):
        if name in data:
            data[name] = _code_as_text(data[name]).replace({"nan": "Unknown", "None": "Unknown"})
    discharge = data["discharge_disposition_id"]
    excluded = discharge.isin(EXCLUDED_DISCHARGE_CODES)
    if excluded.any():
        raise ValueError(f"{int(excluded.sum())} encounter(s) have death/hospice discharge codes excluded from the modeling cohort. Remove those rows before scoring.")
    data["discharge_disposition_id"] = discharge.replace({x: "Unknown" for x in UNKNOWN_DISCHARGE_CODES})
    for name in ("diag_1", "diag_2", "diag_3"):
        data[f"{name}_group"] = data[name].map(group_diagnosis)
    numeric_source = ["number_inpatient", "number_outpatient", "number_emergency", "time_in_hospital", "num_lab_procedures", "num_medications", "num_procedures"]
    for name in numeric_source:
        data[name] = pd.to_numeric(data[name], errors="coerce")
    if data[numeric_source].isna().any().any():
        bad = data[numeric_source].columns[data[numeric_source].isna().any()].tolist()
        raise ValueError("Missing or invalid values in engineered numeric inputs: " + ", ".join(bad))
    data["total_prior_visits"] = data["number_inpatient"] + data["number_outpatient"] + data["number_emergency"]
    data["had_prior_inpatient_visit"] = (data["number_inpatient"] > 0).astype(int)
    data["had_prior_emergency_visit"] = (data["number_emergency"] > 0).astype(int)
    drug_data = data[[c for c in DIABETES_DRUGS if c in data.columns]].astype(str)
    data["active_diabetes_medications"] = drug_data.isin(["Steady", "Up", "Down"]).sum(axis=1)
    data["medication_change_count"] = drug_data.isin(["Up", "Down"]).sum(axis=1)
    days = data["time_in_hospital"].clip(lower=1)
    data["labs_per_day"] = data["num_lab_procedures"] / days
    data["medications_per_day"] = data["num_medications"] / days
    data["procedures_per_day"] = data["num_procedures"] / days
    data["total_procedure_burden"] = data["num_lab_procedures"] + data["num_procedures"]
    for name in categoricals:
        data[name] = data[name].astype("string").fillna("Unknown").astype(str)
    return data[feature_order]


def load_bundle(folder: Path) -> dict:
    import catboost
    import xgboost

    folder = Path(folder)
    missing = [name for name in MODEL_FILES if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing model artifacts: " + ", ".join(missing))
    cfg = json.loads((folder / "model_configuration.json").read_text())
    if abs(float(cfg["catboost_weight"]) - 0.6) > 1e-8 or abs(float(cfg["xgboost_weight"]) - 0.4) > 1e-8:
        raise ValueError("Model bundle does not match the frozen 60/40 ensemble")
    cat = catboost.CatBoostClassifier()
    cat.load_model(str(folder / "final_catboost_model.cbm"))
    xgb = xgboost.XGBClassifier()
    xgb.load_model(str(folder / "final_xgboost_model.json"))
    return {"cat": cat, "xgb": xgb, "preprocessor": joblib.load(folder / "final_xgb_preprocessor.joblib"),
            "calibrator": joblib.load(folder / "probability_calibrator.joblib"), "config": cfg}


def predict(raw: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    data = prepare_features(raw, bundle["config"])
    cat = np.asarray(bundle["cat"].predict_proba(data), dtype=float)
    xgb = np.asarray(bundle["xgb"].predict_proba(bundle["preprocessor"].transform(data)), dtype=float)
    if cat.shape != xgb.shape or cat.shape != (len(data), 3):
        raise ValueError("The saved classifiers did not return three aligned outcome probabilities")
    for name in ("cat", "xgb"):
        classes = np.asarray(bundle[name].classes_).astype(int)
        if not np.array_equal(classes, [0, 1, 2]):
            raise ValueError(f"Unexpected {name} class order: {classes.tolist()}")
    cat /= cat.sum(axis=1, keepdims=True)
    xgb /= xgb.sum(axis=1, keepdims=True)
    combined = 0.6 * cat + 0.4 * xgb
    combined /= combined.sum(axis=1, keepdims=True)
    calibrated = np.asarray(bundle["calibrator"].predict_proba(np.log(np.clip(combined, 1e-15, 1 - 1e-15))))
    if not np.array_equal(np.asarray(bundle["calibrator"].classes_).astype(int), [0, 1, 2]):
        raise ValueError("Unexpected calibration class order")
    if calibrated.shape != combined.shape or not np.isfinite(calibrated).all():
        raise ValueError("Calibrator returned invalid probabilities")
    result = pd.DataFrame(index=raw.index)
    result["Predicted_class"] = combined.argmax(axis=1)
    result["Predicted_outcome"] = result["Predicted_class"].map(dict(enumerate(CLASS_NAMES)))
    for i, name in enumerate(CLASS_NAMES):
        result[f"Uncalibrated: {name}"] = combined[:, i]
        result[f"Calibrated risk: {name}"] = calibrated[:, i]
    return result
