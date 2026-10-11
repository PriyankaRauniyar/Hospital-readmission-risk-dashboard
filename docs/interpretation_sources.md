# Interpretation records

The presentation update was checked against the project workspace's final reporting
notebook, `Hospital Readmission_Risk_and_Care_Coordination.ipynb`, and saved
`Table_2_Model_Comparison.csv` (copied here as `validation_results.csv`). The notebook
is not distributed because its outputs contain source encounter features.

- The notebook uses GroupShuffleSplit with patient_nbr as the grouping key: an outer
  development/test split and an inner training/validation split. Identifiers and
  target columns are removed from X_model.
- The calibrator is LogisticRegression fitted to log-transformed validation ensemble
  probabilities and validation outcomes. Test outcomes are not used to fit it.
- Final class decisions use uncalibrated ensemble argmax; predictor.py and the saved
  model configuration preserve 60% CatBoost / 40% XGBoost.
- The Power BI export cell creates Within_30_day_risk_band with pd.cut on the
  calibrated_probability[:, 1], bins [-inf, 0.10, 0.16, 0.25, inf], labels Low,
  Moderate, Elevated, High. Default pd.cut includes the right endpoint. Thus Low
  includes 0.10; Moderate includes 0.16; Elevated includes 0.25; High is above 0.25.
  All 19,802 saved Risk_band values match this definition. No risk bands or
  probabilities were regenerated or changed by this update.
- The notebook's ECE function sorts probabilities and partitions them into ten
  approximately equal-count groups. ECE is the count-weighted absolute difference
  between mean probability and observed frequency in each group, per class.
- Class-specific Brier scores are binary squared probability errors; the displayed
  multiclass Brier adds errors across classes. Aggregate values remain those in
  benchmark.py and the saved reporting results.

Risk bands describe this export, not validated clinical intervention thresholds.
Validation metrics document trade-offs; no additional model-selection rationale
is asserted. Test and development-inclusive prediction exports remain separate.
