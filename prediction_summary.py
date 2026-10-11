"""Descriptive summaries of scored batches, without actual-outcome metrics."""
from benchmark import CLASS_NAMES
from public_predictions import UNCALIBRATED, CALIBRATED


def describe_predictions(data):
    if data.empty:
        return []
    n = len(data)
    counts = data['Predicted_class'].value_counts()
    mix = '; '.join(f"{int(counts.get(i, 0)):,} ({counts.get(i, 0) / n:.1%}) {name.lower()}"
                    for i, name in enumerate(CLASS_NAMES))
    early = data[CALIBRATED[1]]
    means = '; '.join(f"{name.lower()}: {data[col].mean():.1%}"
                      for name, col in zip(CLASS_NAMES, CALIBRATED))
    disagreed = int((data[UNCALIBRATED].to_numpy().argmax(axis=1)
                     != data[CALIBRATED].to_numpy().argmax(axis=1)).sum())
    flagged = data.loc[data['Predicted_class'] == 1, CALIBRATED[1]]
    flag_detail = (f"Among the {len(flagged):,} encounters predicted within 30 days, the mean calibrated early-readmission risk is {flagged.mean():.1%}."
                   if len(flagged) else "No encounter in this batch has within 30 days as its final predicted outcome.")
    return [
        f"This batch contains {n:,} encounters. The final predicted outcome mix is {mix}. These are model predictions, not observed outcomes.",
        f"Across this batch, calibrated within-30-day risk averages {early.mean():.1%}, with a median of {early.median():.1%} and a range of {early.min():.1%} to {early.max():.1%}. {flag_detail}",
        f"Mean calibrated risk by outcome is {means}. Each row's three calibrated estimates sum to 100%; the three uncalibrated probabilities form a separate set that also sums to 100%.",
        f"The outcome with the highest calibrated estimate differs from the final prediction in {disagreed:,} of {n:,} rows ({disagreed/n:.1%}). The final class follows the largest uncalibrated probability from the 60% CatBoost / 40% XGBoost ensemble; calibration does not replace that decision.",
        "These estimates describe this batch's model outputs. Without observed outcomes and an independent evaluation cohort, the table does not establish prediction accuracy. The frozen test-set results remain separate."
    ]
