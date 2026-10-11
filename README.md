# Hospital Readmission Risk Dashboard

Created by **Priyanka Rauniyar**.

An interactive Streamlit dashboard for a frozen hospital readmission ensemble:
**60% CatBoost / 40% XGBoost**. The saved models are used without retraining.
Final classes follow the highest **uncalibrated** ensemble probability;
calibrated probabilities are separate exploratory risk estimates.

## Clone and run

Python 3.12 is recommended for deployment. Dependencies are pinned to the tested
local versions, including scikit-learn 1.6.1 for the serialized preprocessor and calibrator.

```bash
git clone https://github.com/PriyankaRauniyar/Hospital-readmission-risk-dashboard.git
cd Hospital-readmission-risk-dashboard
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate with `.venv\Scripts\activate`. On macOS, XGBoost requires
OpenMP (`brew install libomp`). `sh run_app.sh` is also available for macOS/Linux;
it uses a local OpenMP fallback if one exists in `.venv/lib/openmp/`, which is
not distributed with this repository. Linux deployments need `libgomp1`;
`packages.txt` supplies it on Streamlit Community Cloud.

## Dashboard pages

- **Overview:** unchanged final-test KPIs and confusion matrix: 19,802 encounters,
  52.43% accuracy, 46.15% macro F1, 39.67% within-30-day recall.
- **Explore predictions:** 99,343 precomputed historical encounter predictions,
  available without uploading data or loading models. Batches 1–49 have 2,000 rows;
  Batch 50 has 1,343. Each batch has a download and a recalculated explanation.
- **Model performance:** validation model comparisons, separate from final test results.
- **Patient cohorts:** age, gender, and race filters on the reduced final-test export.
  Filtered metrics never change the frozen overview KPIs.
- **Fairness & calibration:** aggregate calibration results and test-cohort subgroup recall.
- **Batch scoring:** either Load UCI example data or CSV upload. Both use the same
  eligibility exclusions and frozen models, with selected batches of up to 2,000 rows.
  Visitors can explore historical encounters without downloading or uploading a file.
- **Methods:** study population, development, evaluation, and limitations.

Native controls and tables follow Streamlit's light/dark theme. Charts use explicit
high-contrast surfaces. Prediction tables explain the difference between final class
decisions and calibrated risk estimates, with descriptions computed from each batch.

## Included files and data boundaries

`data/public_predictions.csv.gz` is a reviewed model-output export, about 4.2 MB.
Its nine columns are a newly generated display row number, predicted class and
outcome, three uncalibrated probabilities, and three calibrated risk estimates.
Class order is No readmission (0), Within 30 days (1), After 30 days (2).
It contains no patient/encounter identifiers, actual outcomes, or source features.
These rows include development data and **must not be used to calculate a new test
accuracy**. The 19,802-encounter untouched-test metrics remain separate.

`data/powerbi_patient_predictions.csv` is a reviewed, reduced 19,802-row final-test
export. It contains only actual/predicted classes and labels, race, gender, age,
calibrated early-readmission risk, and risk band. No patient or encounter identifiers,
original record counter, diagnosis codes, medication fields, or other source features
are included. It supports cohort filters and subgroup charts.

The five frozen artifacts in `final_model_artifacts/` are included for scoring.
The largest is approximately 19 MB, below GitHub's ordinary file-size limit, so Git LFS
is not required. CatBoost's upstream build-machine path was removed from version metadata;
leaf values and 2,000 encounter predictions were verified exactly unchanged.
Serialized artifacts must be loaded only from a trusted checkout.

Raw `data/diabetic_data.csv`, `data/IDS_mapping.csv`, `data/de-identified.csv`,
uploaded records, credentials, virtual environments, and temporary outputs stay
local and are ignored. The two reviewed prediction exports and five named model
artifacts are explicit exceptions in `.gitignore`. No raw dataset is required to
run the committed dashboard. `.gitignore` does not remove previously tracked files.

## Regenerate public predictions locally

With the original dataset available locally at `data/diabetic_data.csv`:

```bash
python generate_public_predictions.py
```

The generator reads 2,000 source rows at a time and calls the unchanged predictor.
It excludes discharge codes 11, 13, 14, 19, 20, and 21: 101,766 source encounters
minus 2,423 exclusions gives 99,343 eligible encounters. It validates the strict
column allowlist, class labels, finite probability ranges and sums, and uncalibrated
argmax before atomically replacing the compressed public export. The final display
range is 98,001–99,343. A different source may produce different counts.

## Checks

The test files contain focused functions and can be run without extra test dependencies:

```bash
python - <<'PY'
import test_core, test_public_predictions, test_prediction_summary, test_uci_examples
for module in (test_core, test_public_predictions, test_prediction_summary, test_uci_examples):
    for name in dir(module):
        if name.startswith('test_'):
            getattr(module, name)()
            print('PASS', name)
PY
```

The public export loader also validates the full predictions file whenever it loads.

## Streamlit Community Cloud

Select this repository and `app.py` as the entry point, with Python 3.12.
`requirements.txt` installs Python dependencies and `packages.txt` installs OpenMP
for Linux scoring. No secrets or raw dataset are required. After deployment, check
the four overview KPIs, Batch 50 in Explore predictions, the cohort filters, and
both the UCI example button and an upload on Batch scoring. Missing optional exports or artifacts produce availability
messages rather than preventing the fixed reporting pages from loading.

## Dataset attribution and research limitation

Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014).
[Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296).
UCI Machine Learning Repository. [DOI: 10.24432/C5230J](https://doi.org/10.24432/C5230J).
Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The published prediction exports are reduced/derived outputs from that source.

This is a retrospective research dashboard, not a validated clinical decision tool.
The data has limited longitudinal context and class imbalance; early-readmission
precision is modest. No prospective or external clinical validation was performed.


## UCI example scoring

Batch scoring offers **Load UCI example data** beside an explicit historical-research
note. `uci_examples.py` fetches the official UCI dataset 296 archive over HTTPS with
a 30-second timeout and reads the CSV in memory. It neither extracts the archive to
disk nor writes a raw/public dataset output. Identifiers and the known `readmitted`
outcome are removed before caching and scoring. The sanitized encounter frame is
cached in memory for 24 hours with one cache entry; an unavailable source produces
a retry/CSV-upload message. Uploads receive the same identifier/outcome removal.
Only model decisions and probabilities appear in result tables and downloads.

The deployment needs outbound HTTPS access to `archive.ics.uci.edu`, the existing
five saved model files, and the existing model runtime dependencies. No extra Python
package or secret is needed. Python 3.12 and `libgomp1` remain the recommended cloud
runtime. These historical examples include development records and demonstrate
scoring; no accuracy is computed from them. Held-out test KPIs stay unchanged.

## Interpretation and display formatting

`docs/validation_results.csv` preserves the saved validation comparison used to
check the dashboard wording. `docs/interpretation_sources.md` records verified
notebook definitions for patient grouping, calibration, ECE, and risk bands.
Risk bands use calibrated within-30-day probabilities: Low ≤10%, Moderate
>10%–≤16%, Elevated >16%–≤25%, High >25%. Chart bars show observed outcome shares
within each band, with filtered encounter counts available below the chart.

Visible metric tables show percentages to two decimal places and prediction
probabilities to four decimal places. Display formatting does not change source
numbers or CSV download precision. Public batch summaries lead with predicted
class counts; detailed probability statistics are in Technical probability details.
