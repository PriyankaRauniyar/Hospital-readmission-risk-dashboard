"""Hospital Readmission | Streamlit portfolio and frozen model explorer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics import basic_metrics, confusion, early_readmission_counts, normalize_predictions, subgroup_table
from benchmark import CALIBRATION, CLASS_NAMES, CONFUSION, FEATURE_IMPORTANCE, TEST_METRICS, VALIDATION_MODELS
from predictor import MODEL_FILES, load_bundle, predict
from public_predictions import BATCH_SIZE, load_public_predictions
from prediction_summary import describe_predictions
from uci_examples import fetch_uci_examples, scoring_records


ROOT = Path(__file__).resolve().parent
COLOR = {"No readmission": "#156D90", "Within 30 days": "#A9541D", "After 30 days": "#6657A7"}
st.set_page_config(page_title="Hospital Readmission Risk Dashboard", page_icon="✳", layout="wide", initial_sidebar_state="expanded")

# Native Streamlit widgets inherit the user's theme. Custom colors are scoped
# to our own elements; charts have an explicit light surface in either theme.
st.markdown("""
<style>
  .block-container {max-width: 1480px; padding-top: 2rem;}
  .atlas-hero {padding: 1.8rem 2rem; border-radius: 18px;
    background: #123D49; margin-bottom: 1.5rem; border: 1px solid #527B84;}
  .atlas-hero h1 {color: #FFFFFF; margin: .25rem 0; font-size: clamp(1.8rem, 4vw, 2.8rem);}
  .atlas-hero p {color: #E4F1F3; margin: .5rem 0 0;}
  .atlas-hero .atlas-eyebrow {color: #BCE4DF; font-size: .8rem; font-weight: 700;
    letter-spacing: .12em; text-transform: uppercase;}
  .atlas-note {background: #EDF5F5; color: #183D48; border-left: 4px solid #986014;
    padding: 1rem 1.2rem; border-radius: 0 12px 12px 0; margin: 1rem 0;}
  .atlas-note strong {color: #183D48;}
  [data-testid="stMetric"] {border: 1px solid #789297; border-radius: 12px; padding: 1rem;}
  @media (max-width: 640px) {.atlas-hero {padding: 1.2rem;} .block-container {padding: 1rem;}}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return normalize_predictions(pd.read_csv(path, low_memory=False))


def figure_base(fig: go.Figure, height: int = 400) -> go.Figure:
    fig.update_layout(template="plotly_white", height=height,
        margin=dict(l=20, r=24, t=95, b=105), font=dict(family="Arial", color="#23404C"),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        hoverlabel=dict(bgcolor="#FFFFFF", font_color="#183D48", bordercolor="#789297"),
        legend=dict(orientation="h", y=-.3, x=0, font=dict(color="#23404C")))
    fig.update_xaxes(automargin=True, gridcolor="#E1E8EA", zerolinecolor="#789297")
    fig.update_yaxes(automargin=True, gridcolor="#E1E8EA", zerolinecolor="#789297")
    return fig


def show_confusion(matrix: np.ndarray, title: str = "Final test: actual vs. predicted outcomes"):
    matrix = np.asarray(matrix)
    support = matrix.sum(axis=1, keepdims=True)
    share = np.divide(matrix, support, out=np.zeros_like(matrix, dtype=float), where=support > 0)
    fig = go.Figure(go.Heatmap(z=share, x=CLASS_NAMES, y=CLASS_NAMES,
        customdata=matrix, zmin=0, zmax=1,
        colorscale=[[0,"#EDF5F5"],[1,"#156D90"]], showscale=False,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Encounters: %{customdata:,}<br>Share of actual class: %{z:.1%}<extra></extra>"))
    for row, actual in enumerate(CLASS_NAMES):
        for col, predicted in enumerate(CLASS_NAMES):
            # Match Plotly's interpolated background and choose readable cell text.
            rgb = (np.array([237, 245, 245]) * (1 - share[row, col])
                   + np.array([21, 109, 144]) * share[row, col]) / 255
            linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
            luminance = float(linear @ np.array([.2126, .7152, .0722]))
            cell_text = "#FFFFFF" if luminance < .179 else "#000000"
            fig.add_annotation(x=predicted, y=actual,
                text=f"{matrix[row, col]:,}<br>{share[row, col]:.1%}", showarrow=False,
                font=dict(size=14, color=cell_text))
    fig.update_layout(title=title)
    fig.update_xaxes(side="top", title="Predicted outcome")
    fig.update_yaxes(autorange="reversed", title="Actual outcome")
    st.plotly_chart(figure_base(fig, 450), use_container_width=True, theme=None)


def percentages(frame: pd.DataFrame, value_columns: list[str]):
    st.dataframe(frame, hide_index=True, use_container_width=True,
        column_config={name: st.column_config.NumberColumn(name, format="%.2f%%") for name in value_columns})


def explain_prediction_table(data: pd.DataFrame):
    st.markdown("#### What this batch shows")
    for paragraph in describe_predictions(data):
        st.write(paragraph)


def prediction_table_guide():
    with st.expander("How to read predictions and risk", expanded=True):
        st.write("Each row represents one encounter. Predicted class and predicted outcome show the final decision: 0 = No readmission, 1 = Within 30 days, 2 = After 30 days. The largest of the three uncalibrated probabilities determines that decision.")
        st.write("The calibrated risk columns give a separate set of exploratory probability estimates. For example, 0.15 means an estimated 15% probability. A row can predict within 30 days while having a higher calibrated estimate for another outcome, because the class decision uses the uncalibrated probabilities.")
        st.write("The table can be scrolled horizontally to compare all outcomes. The selected-batch download contains every row shown; the description below summarizes that same batch.")


st.markdown('<div class="atlas-hero"><div class="atlas-eyebrow">RESEARCH DASHBOARD · FROZEN MODEL</div>'
    '<h1>Hospital Readmission Risk Dashboard</h1><p>Executive research dashboard · care coordination, model performance, and cohort exploration</p></div>',
    unsafe_allow_html=True)

st.sidebar.markdown("### Hospital Readmission Risk Dashboard")
st.sidebar.caption("Executive research dashboard")
SECTIONS = ["Overview", "Explore predictions", "Model performance", "Patient cohorts", "Fairness & calibration", "Batch scoring", "Methods"]
SECTION_DETAILS = {
    "Overview": "Final test performance and early-readmission detection.",
    "Explore predictions": "Precomputed predictions on historical research encounters.",
    "Model performance": "Validation comparisons and the features behind the model.",
    "Patient cohorts": "Encounter results by age, gender, and race.",
    "Fairness & calibration": "Subgroup recall and exploratory probability estimates.",
    "Batch scoring": "Score historical UCI examples or uploaded encounters.",
    "Methods": "Study population, model development, and evaluation.",
}


def navigate_section(section: str):
    st.session_state["atlas_section"] = section


page = st.sidebar.radio("Section", SECTIONS, key="atlas_section", label_visibility="collapsed")
section_index = SECTIONS.index(page)
st.sidebar.caption(SECTION_DETAILS[page])
previous, following = st.sidebar.columns(2)
previous.button("← Previous", use_container_width=True, disabled=section_index == 0,
                on_click=navigate_section, args=(SECTIONS[max(0, section_index - 1)],))
following.button("Next →", use_container_width=True, disabled=section_index == len(SECTIONS) - 1,
                 on_click=navigate_section, args=(SECTIONS[min(len(SECTIONS) - 1, section_index + 1)],))
st.sidebar.caption(f"{section_index + 1} of {len(SECTIONS)} · {page}")
st.sidebar.divider()
with st.sidebar.expander("About the ensemble"):
    st.caption("60% CatBoost / 40% XGBoost")
    st.caption("Final decisions use uncalibrated argmax. Calibrated probabilities provide exploratory risk estimates.")
    st.caption("UCI diabetes encounters · retrospective research")

csv_path = ROOT / "data" / "powerbi_patient_predictions.csv"
encounters = None
data_error = None
if csv_path.exists():
    try:
        encounters = load_csv(str(csv_path))
    except (ValueError, OSError, pd.errors.ParserError) as error:
        data_error = str(error)

with st.sidebar.expander("Data & model status"):
    if encounters is not None:
        st.caption(f"Prediction export · {len(encounters):,} encounters loaded")
    elif data_error:
        st.caption("Prediction export · unable to load")
    else:
        st.caption("Prediction export · unavailable")
    missing_assets = [name for name in MODEL_FILES if not (ROOT / "final_model_artifacts" / name).is_file()]
    st.caption("Model files · " + ("all five present" if not missing_assets else f"{len(missing_assets)} missing"))
    st.caption("Model runtime is checked when encounters are scored.")
st.sidebar.button("Back to overview", use_container_width=True, disabled=page == "Overview",
                  on_click=navigate_section, args=("Overview",))

if page == "Overview":
    st.caption("FINAL TEST RESULTS · FROZEN BENCHMARK")
    st.subheader("Readmission performance at a glance")
    columns = st.columns(4)
    for col, label, value in zip(columns,
        ["Test encounters", "Accuracy", "Macro F1", "Within 30-day recall"],
        [f"{TEST_METRICS['encounters']:,}", f"{TEST_METRICS['accuracy']:.2%}",
         f"{TEST_METRICS['macro_f1']:.2%}", f"{TEST_METRICS['within_30_recall']:.2%}"]):
        col.metric(label, value)
    st.caption("Final test cohort · overview metrics remain fixed across cohort filters.")
    a, b = st.columns([.86, 1.14], gap="large")
    with a:
        st.markdown("#### Test cohort outcome mix")
        st.write("The cohort mix sets the context for care coordination: most encounters had no readmission, while early readmissions were a smaller group.")
        counts = np.asarray(CONFUSION).sum(axis=1)
        donut = px.pie(values=counts, names=CLASS_NAMES, hole=.62,
            color=CLASS_NAMES, color_discrete_map=COLOR)
        donut.update_layout(title="Final test cohort: observed outcomes")
        donut.update_traces(textinfo="percent", textfont=dict(color="#FFFFFF"), hovertemplate="%{label}: %{value:,} encounters (%{percent})<extra></extra>")
        st.plotly_chart(figure_base(donut, 380), use_container_width=True, theme=None)
    with b:
        st.markdown("#### How class decisions compare with outcomes")
        st.write("Rows show actual outcomes and columns show predictions. The diagonal shows correct predictions; the remaining cells show misclassifications.")
        show_confusion(np.asarray(CONFUSION))
    st.markdown('<div class="atlas-note"><strong>Care coordination interpretation:</strong> '
        'Of 2,216 encounters followed by readmission within 30 days, the frozen model identified 879. '
        'Among encounters flagged for early readmission, 21.81% had that outcome.</div>',
        unsafe_allow_html=True)

    st.subheader("Early-readmission identification and review burden")
    counts = early_readmission_counts(np.asarray(CONFUSION))
    cols = st.columns(3)
    cols[0].metric("Early readmissions identified", f"{counts['identified']:,}")
    cols[1].metric("Early readmissions missed", f"{counts['missed']:,}")
    cols[2].metric("Other outcomes flagged as early", f"{counts['false_positive']:,}")
    st.write("Of the 4,031 encounters assigned to the within-30-day class, 879 actually had an early readmission (21.81% precision). The other 3,152 represent review burden. The model missed 1,337 early readmissions. These counts show the balance between early-readmission detection and false-positive flags in the final test cohort.")

elif page == "Explore predictions":
    st.subheader("Predictions on historical research encounters")
    st.write("Precomputed outputs from the frozen 60% CatBoost / 40% XGBoost ensemble. These rows include development data and cannot be used to calculate a new test accuracy. The separate overview reports the unchanged 19,802-encounter untouched-test results.")
    st.caption("Class decisions follow the highest uncalibrated probability. Calibrated probabilities are exploratory risk estimates.")
    st.markdown("Source: [UCI Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296) · Clore, Cios, DeShazo & Strack (2014) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · Model-derived predictions.")
    public_path = ROOT / "data" / "public_predictions.csv.gz"
    if not public_path.is_file():
        st.info("Public predictions are not available in this deployment.")
    else:
        try:
            @st.cache_data(show_spinner=False)
            def get_public_predictions(path: str, modified_ns: int):
                return load_public_predictions(Path(path))

            public = get_public_predictions(str(public_path), public_path.stat().st_mtime_ns)
            batch_count = (len(public) + BATCH_SIZE - 1) // BATCH_SIZE
            selected_batch = st.selectbox("Prediction batch", range(1, batch_count + 1),
                                          format_func=lambda value: f"Batch {value}")
            start = (selected_batch - 1) * BATCH_SIZE
            selected = public.iloc[start:start + BATCH_SIZE]
            st.caption(f"{len(public):,} historical encounters · Batch {selected_batch} of {batch_count} · display rows {start + 1:,}–{start + len(selected):,} · {len(selected):,} rows")
            prediction_table_guide()
            st.dataframe(selected, hide_index=True, use_container_width=True,
                         column_config={"Display_row": st.column_config.NumberColumn("Display row", format="%d")})
            explain_prediction_table(selected)
            st.download_button("Download selected batch", selected.to_csv(index=False).encode(),
                               file_name=f"historical_predictions_batch_{selected_batch}.csv", mime="text/csv")
        except (ValueError, OSError, pd.errors.ParserError) as exc:
            st.error(f"Public predictions unavailable: {exc}")

elif page == "Model performance":
    st.subheader("Model comparison · validation cohort")
    st.write("The validation cohort provided a common basis for model selection. Final test performance appears in the overview.")
    st.caption("Validation results for the five candidate models.")
    comparison = pd.DataFrame(VALIDATION_MODELS)
    metric = st.selectbox("Compare models by", ["Macro F1", "Within 30 recall", "Accuracy", "Log loss"])
    chart = px.bar(comparison.sort_values(metric), x=metric, y="Model", orientation="h",
        color="Model", color_discrete_sequence=["#156D90", "#6657A7", "#1C9A9C", "#A9541D", "#A2BDB9"])
    chart.update_layout(title=f"Validation models · {metric}", showlegend=False, yaxis_title="", xaxis_title=metric)
    chart.update_traces(hovertemplate="%{y}<br>" + metric + (": %{x:.4f}<extra></extra>" if metric == "Log loss" else ": %{x:.2%}<extra></extra>"))
    if metric != "Log loss":
        chart.update_xaxes(tickformat=".0%")
    st.caption("Lower log loss is better; higher accuracy, macro F1, and early recall are better.")
    st.plotly_chart(figure_base(chart, 390), use_container_width=True, theme=None)
    percentages(comparison.assign(**{"Accuracy":comparison["Accuracy"]*100,
        "Macro F1":comparison["Macro F1"]*100,
        "Within 30 recall":comparison["Within 30 recall"]*100}),
        ["Accuracy", "Macro F1", "Within 30 recall"])
    st.subheader("CatBoost component: top features")
    st.caption("Relative feature importance within CatBoost, not a causal explanation of readmission.")
    feat = pd.DataFrame(FEATURE_IMPORTANCE, columns=["Feature", "Importance"])
    feature_chart = px.bar(feat.iloc[::-1], x="Importance", y="Feature", orientation="h", color_discrete_sequence=["#1C7689"])
    feature_chart.update_layout(title="CatBoost: relative feature importance", xaxis_title="Relative importance", yaxis_title="")
    feature_chart.update_traces(hovertemplate="%{y}<br>Importance: %{x:.2f}<extra></extra>")
    st.plotly_chart(figure_base(feature_chart, 700), use_container_width=True, theme=None)

elif page == "Patient cohorts":
    st.subheader("Patient cohorts · filtered test results")
    st.write("Cohort results reflect the selected age, gender, and race groups. Empty filters include all values; overview metrics stay fixed.")
    if data_error:
        st.error(f"Unable to read the prediction CSV: {data_error}")
    if encounters is None:
        st.info("Encounter-level results are unavailable. The prediction export has not been added to this dashboard.")
    else:
        data = encounters
        filters = st.columns(3)
        for column, box in zip(("race", "gender", "age"), filters):
            if column in data:
                options = sorted(data[column].dropna().astype(str).unique().tolist())
                selected = box.multiselect(column.title(), options, default=[])
                if selected:
                    data = data[data[column].astype(str).isin(selected)]
        if data.empty:
            st.info("No encounters match these filters.")
        else:
            metrics = basic_metrics(data)
            cols = st.columns(4)
            for c, label, value in zip(cols,
                ["Filtered encounters", "Filtered accuracy", "Filtered macro F1", "Filtered early recall"],
                [f"{metrics['encounters']:,}", f"{metrics['accuracy']:.2%}",
                 f"{metrics['macro_f1']:.2%}", f"{metrics['within_30_recall']:.2%}"]):
                c.metric(label, value)
            st.caption("These values recalculate for the filtered test encounters; small groups may have unstable estimates.")
            show_confusion(confusion(data).to_numpy(), "Filtered encounters: actual vs. predicted outcomes")
            if "Risk_band" in data and data["Risk_band"].notna().any():
                st.subheader("Actual outcome mix within each risk band")
                cross = pd.crosstab(data["Risk_band"], data["Actual_outcome"])
                order = [x for x in ["Low", "Moderate", "Elevated", "High"] if x in cross.index]
                cross = cross.reindex(order + [x for x in cross.index if x not in order])
                share = cross.div(cross.sum(axis=1), axis=0).fillna(0)
                fig = go.Figure()
                for name in CLASS_NAMES:
                    if name in share:
                        fig.add_bar(x=share.index.astype(str), y=share[name] * 100,
                            name=name, marker_color=COLOR[name], customdata=cross[name],
                            hovertemplate="%{x}<br>"+name+": %{y:.1f}% (%{customdata:,} encounters)<extra></extra>")
                fig.update_layout(title="Filtered encounters: observed outcomes by risk band", barmode="stack", yaxis_title="Share of encounters (%)", xaxis_title="Predicted 30-day risk band")
                st.plotly_chart(figure_base(fig, 410), use_container_width=True, theme=None)
            elif "Risk_30" in data:
                st.caption("Calibrated 30-day probabilities are present; the report's risk-band labels were not included in this export.")
            st.subheader("Filtered subgroup performance")
            available = [x for x in ("race", "gender", "age") if x in data]
            if available:
                selected_group = st.selectbox("Group by", available)
                table = subgroup_table(data, selected_group)
                st.dataframe(table, hide_index=True, use_container_width=True, column_config={name: st.column_config.NumberColumn(name, format="percent") for name in ("Within 30 recall", "Accuracy", "Macro F1")})
                st.caption("Subgroup sizes provide context for differences in early-readmission recall.")

elif page == "Fairness & calibration":
    st.subheader("Exploratory probability calibration")
    st.write("Calibration brings average predicted risks closer to observed outcome rates. These probabilities are exploratory estimates; the final class decision remains the uncalibrated ensemble argmax.")
    calibration = pd.DataFrame(CALIBRATION)
    long = calibration.melt(id_vars="Class", value_vars=["Observed", "Mean before", "Mean after"],
        var_name="Measure", value_name="Share")
    fig = px.bar(long, x="Class", y="Share", color="Measure", barmode="group",
        color_discrete_map={"Observed":"#153F4B","Mean before":"#A9541D","Mean after":"#27A6A2"})
    fig.update_layout(title="Final test: observed prevalence vs. mean estimated risk", legend_title_text="", xaxis_title="Outcome")
    fig.update_traces(hovertemplate="%{x}<br>Share: %{y:.2%}<extra>%{fullData.name}</extra>")
    fig.update_yaxes(tickformat=".0%", title="Share")
    st.plotly_chart(figure_base(fig, 410), use_container_width=True, theme=None)
    st.dataframe(calibration, hide_index=True, use_container_width=True)
    a, b = st.columns(2)
    a.metric("Multiclass Brier · before", "0.5910")
    b.metric("Multiclass Brier · after", "0.5245")
    st.caption("Prevalence-only baseline Brier: 0.5783. These are fixed test-set results.")
    st.markdown('<div class="atlas-note">Calibration improves probability quality, but using calibrated argmax '
        'reduced recall for readmission within 30 days to 5.64%. The frozen class decisions still use '
        'the uncalibrated 60/40 ensemble.</div>', unsafe_allow_html=True)
    if data_error:
        st.warning(f"Encounter-level subgroup results unavailable: {data_error}")
    subgroup_variables = [x for x in ("gender", "race", "age") if encounters is not None and x in encounters]
    if subgroup_variables:
        st.subheader("Explore descriptive subgroup recall")
        group = st.selectbox("Subgroup variable", subgroup_variables)
        table = subgroup_table(encounters, group)
        fig = px.bar(table, x="Group", y="Within 30 recall", color="Encounters", color_continuous_scale="Teal")
        fig.update_layout(title=f"Encounter export: early-readmission recall by {group}", coloraxis_colorbar_title="Encounters")
        fig.update_traces(hovertemplate="Group: %{x}<br>Early recall: %{y:.2%}<extra></extra>")
        fig.update_yaxes(tickformat=".0%", range=[0,1])
        st.plotly_chart(figure_base(fig), use_container_width=True, theme=None)
        st.caption("Recall estimates are less stable in groups with few early readmissions. These comparisons describe the cohort.")

elif page == "Batch scoring":
    st.subheader("Batch encounter scoring")
    st.caption("Score historical UCI examples or an encounter CSV with the frozen ensemble. Explore predictions provides precomputed public results.")
    artifact_dir = ROOT / "final_model_artifacts"
    missing = [name for name in MODEL_FILES if not (artifact_dir / name).exists()]
    if missing:
        st.info("Batch scoring is unavailable because the saved model bundle is incomplete.")
        st.code("\n".join(missing), language="text")
    else:
        st.caption("The saved ensemble scores encounter records with the same features used during training.")
        st.info("Encounter CSVs, including diabetic_data.csv, are processed in batches of up to 2,000 rows. The modeling cohort excludes death/hospice discharge codes 11, 13, 14, 19, 20, and 21. IDS_mapping.csv contains code descriptions rather than encounter records.")
        source_choice = st.radio("Encounter source", ["UCI example data", "CSV upload"], horizontal=True)
        raw = None
        if source_choice == "UCI example data":
            load_column, note_column = st.columns([1, 3])
            load_examples = load_column.button("Load UCI example data", use_container_width=True)
            note_column.markdown("**Historical research data:** These encounters include records used during model development. Their predictions demonstrate the scoring workflow and are not a new independent accuracy test.")
            st.markdown("Source: [UCI Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296) · Clore, Cios, DeShazo & Strack (2014) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).")

            @st.cache_data(ttl=86400, max_entries=1, show_spinner=False)
            def get_uci_examples():
                return fetch_uci_examples()

            if load_examples:
                try:
                    with st.spinner("Loading historical UCI encounters..."):
                        get_uci_examples()
                    st.session_state["uci_examples_loaded"] = True
                except ValueError as exc:
                    st.session_state["uci_examples_loaded"] = False
                    st.error(str(exc))
            if st.session_state.get("uci_examples_loaded"):
                try:
                    raw = get_uci_examples()
                except ValueError as exc:
                    st.session_state["uci_examples_loaded"] = False
                    st.error(str(exc))
        else:
            uploaded = st.file_uploader("Upload an encounter CSV to see predictions and risk estimates", type=["csv"])
            if uploaded is not None:
                try:
                    if Path(uploaded.name).name.lower() == "ids_mapping.csv":
                        raise ValueError("This file contains ID code descriptions rather than encounter records. Batch scoring needs encounter data, such as diabetic_data.csv.")
                    raw = scoring_records(pd.read_csv(uploaded, low_memory=False))
                except (ValueError, OSError, pd.errors.ParserError) as exc:
                    st.error(str(exc))
        if raw is not None:
            try:
                if "description" in raw.columns and not {"race", "gender", "age"}.intersection(raw.columns):
                    raise ValueError("This file appears to be a code dictionary. Scoring requires encounter rows with patient features such as race, gender, and age.")
                st.caption(f"{source_choice} · {len(raw):,} encounter rows loaded.")
                if "discharge_disposition_id" in raw:
                    discharge = pd.to_numeric(raw["discharge_disposition_id"], errors="coerce")
                    excluded = discharge.isin([11, 13, 14, 19, 20, 21])
                    if excluded.any():
                        remove_excluded = st.checkbox(
                            f"Exclude {int(excluded.sum()):,} death/hospice encounters to match the modeling cohort",
                            value=True)
                        if not remove_excluded:
                            raise ValueError("Scoring is paused while death/hospice encounters are included. These encounters fall outside the modeling cohort.")
                        raw = raw.loc[~excluded]
                if raw.empty:
                    raise ValueError("No eligible encounter rows remain to score.")
                batches = (len(raw) + 1999) // 2000
                batch = int(st.number_input("Batch number", min_value=1, max_value=batches, value=1, step=1))
                start = (batch - 1) * 2000
                raw = raw.iloc[start:start + 2000]
                st.caption(f"Batch {batch:,} of {batches:,}: eligible rows {start + 1:,}–{start + len(raw):,}. Only this batch is scored and included in the download.")
                @st.cache_resource(show_spinner=False)
                def get_bundle():
                    return load_bundle(artifact_dir)
                with st.spinner("Scoring encounters with the frozen ensemble..."):
                    results = predict(scoring_records(raw), get_bundle())
                st.success(f"Scored {len(results):,} encounters")
                prediction_table_guide()
                st.dataframe(results, hide_index=True, use_container_width=True)
                explain_prediction_table(results)
                st.download_button("Download predictions and risks", results.to_csv(index=False).encode(),
                    file_name=f"readmission_research_predictions_batch_{batch}.csv", mime="text/csv")
                st.caption("Output contains class decisions and exploratory calibrated risks; no identifiers from the input are included.")
            except (ValueError, KeyError, FileNotFoundError) as exc:
                st.error(str(exc))

elif page == "Methods":
    st.subheader("How this model was built")
    st.markdown("""
**Population.** Diabetes encounters in the UCI dataset. Death and hospice discharge codes were excluded from the prediction cohort. Patient-grouped development, validation, and test splits prevent a patient's encounters from crossing partitions.

**Target.** Three classes: no readmission, within 30 days, and after 30 days.

**Features.** Hospital utilization, diagnoses, medications, care intensity, and discharge information available by discharge. The patient and encounter identifiers are excluded from model inputs.

**Frozen model.** 60% CatBoost + 40% XGBoost soft voting. The uncalibrated ensemble's largest class probability selects the class. A separate multinomial calibration step provides exploratory class risk estimates.

**Evaluation.** Untouched test cohort: 19,802 encounters; accuracy 52.43%; macro F1 46.15%; within-30-day recall 39.67%. Validation results in the comparison tab are separate from test results.

**Limitations.** Retrospective dataset, limited longitudinal information, class imbalance, and modest early-readmission precision. The results come from retrospective research, with no prospective or external clinical validation.
""")
    st.caption("Source of frozen results: Hospital Readmission_Risk_and_Care_Coordination final reporting notebook.")

st.divider()
st.caption("Created by Priyanka Rauniyar · Hospital Readmission Risk Dashboard · 60% CatBoost + 40% XGBoost")
