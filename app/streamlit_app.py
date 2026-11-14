"""Streamlit app: predict the severity of a road accident from its circumstances."""
import html
import json
import random
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import sklearn
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.features import CyclicalTime  # noqa: E402

# A model exported from the notebook refers to classes defined in __main__. Aliasing them here
# lets the same model.joblib load inside Streamlit, whichever way it was produced.
import __main__  # noqa: E402

if not hasattr(__main__, "CyclicalTime"):
    __main__.CyclicalTime = CyclicalTime

from src.predict import load_artifacts, predict_one  # noqa: E402

st.set_page_config(
    page_title="Accident Severity Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------- palette
NAVY, TEAL, SLATE = "#12355B", "#1F7A8C", "#5B6B7F"
SEV_COLORS = {"Slight Injury": "#2E9E6B", "Serious Injury": "#E8A33D", "Fatal injury": "#D64545"}
FALLBACK = "#6B7A8F"
UNKNOWN = "(unknown)"
TIME_KEY = "time_of_accident"

SECTIONS = {
    "Driver and vehicle": [
        "Age_band_of_driver", "Drivers_gender", "Educational_level", "Vehicle_driver_relation",
        "Driving_experience", "Type_of_vehicle", "Owner_of_vehicle", "Service_year_of_vehicle",
        "Defect_of_vehicle",
    ],
    "Road and environment": [
        "Area_accident_occured", "Lanes_or_Medians", "Road_allignment", "Types_of_Junction",
        "Road_surface_type", "Road_surface_conditions", "Light_conditions", "Weather_conditions",
    ],
    "Accident details": ["Day_of_week", "Type_of_collision", "Vehicle_movement", "Cause_of_accident"],
    "Casualty": [
        "Casualty_class", "Casualty_gender", "Age_band_of_casualty", "Work_of_casuality",
        "Fitness_of_casuality", "Pedestrian_movement",
    ],
}
SECTION_COLORS = {"Driver and vehicle": NAVY, "Road and environment": TEAL,
                  "Accident details": "#8E5BA8", "Casualty": "#C2603A"}

# ----------------------------------------------------------------------------- styling
st.markdown(
    f"""
<style>
  .block-container {{ padding-top: 1.4rem; max-width: 1400px; }}
  footer {{ visibility: hidden; }}
  .hero {{ background: linear-gradient(120deg, {NAVY} 0%, #1B5E86 55%, {TEAL} 100%);
          border-radius: 14px; padding: 26px 32px; color: #fff; margin-bottom: 1.1rem;
          box-shadow: 0 6px 18px rgba(18,53,91,.25); }}
  .hero h1 {{ color: #fff; margin: 0 0 4px 0; font-size: 2rem; letter-spacing: .2px; padding: 0; }}
  .hero p {{ color: rgba(255,255,255,.88); margin: 0 0 14px 0; font-size: 1rem; }}
  .chip {{ display: inline-block; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.3);
           padding: 3px 12px; border-radius: 999px; font-size: .8rem; margin-right: 8px; color: #fff; }}
  .sec-title {{ font-weight: 700; font-size: .95rem; text-transform: uppercase; letter-spacing: .8px;
                padding-left: 10px; margin: 2px 0 10px 0; }}
  .result {{ border-radius: 14px; padding: 20px 24px; color: #fff; box-shadow: 0 6px 16px rgba(0,0,0,.15); }}
  .result .lbl {{ font-size: .8rem; text-transform: uppercase; letter-spacing: 1px; opacity: .9; }}
  .result .val {{ font-size: 2.1rem; font-weight: 800; line-height: 1.15; margin: 2px 0 6px 0; }}
  .result .sub {{ font-size: .95rem; opacity: .95; }}
  .meter-wrap {{ margin: 14px 0 6px 0; }}
  .meter-head {{ display: flex; justify-content: space-between; font-size: .9rem; font-weight: 600; margin-bottom: 6px; }}
  .meter {{ height: 14px; border-radius: 999px; background: rgba(128,128,128,.22); overflow: hidden; }}
  .meter > div {{ height: 100%; border-radius: 999px;
                  background: linear-gradient(90deg, #2E9E6B 0%, #E8A33D 55%, #D64545 100%); }}
  .kpi {{ background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.22);
          border-top: 4px solid {NAVY}; border-radius: 10px; padding: 14px 16px; }}
  .kpi .k {{ font-size: .78rem; text-transform: uppercase; letter-spacing: .8px; opacity: .75; }}
  .kpi .v {{ font-size: 1.7rem; font-weight: 800; }}
  .stage {{ background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.22);
            border-radius: 10px; padding: 14px 16px; height: 100%; }}
  .stage h4 {{ margin: 0 0 6px 0; padding: 0; font-size: 1.02rem; }}
  .stage p {{ margin: 0; font-size: .9rem; opacity: .9; }}
  .note {{ border-left: 4px solid {TEAL}; background: rgba(31,122,140,.09); padding: 10px 14px;
           border-radius: 6px; font-size: .9rem; margin-top: 10px; }}
  div[data-testid="stTabs"] button p {{ font-size: 1rem; font-weight: 600; }}
</style>
""",
    unsafe_allow_html=True,
)


def sec_title(text: str, color: str) -> None:
    st.markdown(f'<div class="sec-title" style="border-left:4px solid {color};color:{color}">'
                f"{html.escape(text)}</div>", unsafe_allow_html=True)


def kpi(label: str, value: str, color: str = NAVY) -> str:
    return (f'<div class="kpi" style="border-top-color:{color}"><div class="k">{html.escape(label)}</div>'
            f'<div class="v">{html.escape(value)}</div></div>')


def pretty(col: str) -> str:
    return col.replace("_", " ").replace("allignment", "alignment").replace("occured", "occurred").capitalize()


# ----------------------------------------------------------------------------- artifacts
@st.cache_resource(show_spinner="Loading model...")
def get_artifacts():
    return load_artifacts()


@st.cache_data(show_spinner=False)
def get_comparison():
    if not config.METRICS_PATH.exists():
        return None
    results = json.loads(config.METRICS_PATH.read_text())["results"]
    rows = []
    for name, r in results.items():
        t = r["test"]
        rows.append({"Model": name, "CV macro-F1": r.get("cv_macro_f1"), "Test macro-F1": t["macro_f1"],
                     "Balanced accuracy": t["balanced_accuracy"], "Accuracy": t["accuracy"], "MCC": t["mcc"]})
    return pd.DataFrame(rows)


try:
    model, meta = get_artifacts()
except FileNotFoundError:
    st.error("Model not found. Run `python -m src.train` first to create `models/model.joblib`.")
    st.stop()

options = meta["categorical_options"]
labels = [meta["labels"][str(i)] for i in range(len(meta["labels"]))]
metrics = meta["test_metrics"]
sev_color = lambda c: SEV_COLORS.get(c, FALLBACK)  # noqa: E731

# ----------------------------------------------------------------------------- state
DEFAULT_TIME = pd.Timestamp("12:00").time()


def default_number(c: str) -> int:
    lo, hi = meta["numeric_ranges"][c]
    return min(max(2, lo), hi)


def reset_inputs() -> None:
    for c in options:
        st.session_state[c] = UNKNOWN
    for c in config.NUMERIC_COLS:
        st.session_state[c] = default_number(c)
    st.session_state[TIME_KEY] = DEFAULT_TIME


def random_example() -> None:
    rng = random.Random()
    for c, opts in options.items():
        st.session_state[c] = rng.choice(opts)
    for c in config.NUMERIC_COLS:
        lo, hi = meta["numeric_ranges"][c]
        st.session_state[c] = rng.randint(lo, min(hi, lo + 3))
    st.session_state[TIME_KEY] = pd.Timestamp(f"{rng.randint(0, 23)}:{rng.choice([0, 15, 30, 45])}").time()


for c in options:
    st.session_state.setdefault(c, UNKNOWN)
for c in config.NUMERIC_COLS:
    st.session_state.setdefault(c, default_number(c))
st.session_state.setdefault(TIME_KEY, DEFAULT_TIME)

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Model")
    st.markdown(
        f"**{html.escape(meta['model_name'])}**  \nTrained: {meta['trained_at'][:10]}",
    )
    st.markdown("---")
    st.markdown("### Inputs")
    st.button("Load random example", on_click=random_example, use_container_width=True)
    st.button("Reset all inputs", on_click=reset_inputs, use_container_width=True)
    st.markdown("---")
    st.caption(
        "Select 'unknown' for anything you do not know; the model treats it as missing information. "
        "Predictions update live as you change the inputs."
    )

trained_with = meta.get("versions", {}).get("scikit-learn")
if trained_with and trained_with != sklearn.__version__:
    st.warning(
        f"Model was trained with scikit-learn {trained_with} but this environment has "
        f"{sklearn.__version__}. Pin `scikit-learn=={trained_with}` in requirements.txt."
    )

# ----------------------------------------------------------------------------- hero
st.markdown(
    f"""
<div class="hero">
  <h1>Road Accident Severity Predictor</h1>
  <p>Estimate whether an accident in Addis Ababa ends in slight, serious or fatal injury from the
     circumstances in which it happened.</p>
  <span class="chip">Model: {html.escape(meta['model_name'])}</span>
  <span class="chip">Macro-F1: {metrics['macro_f1']:.2f}</span>
  <span class="chip">Classes: {len(labels)}</span>
  <span class="chip">Input features: {len(meta['input_columns'])}</span>
</div>
""",
    unsafe_allow_html=True,
)

tab_predict, tab_perf, tab_about = st.tabs(["Prediction", "Model performance", "About the project"])

# ----------------------------------------------------------------------------- prediction
with tab_predict:
    left, right = st.columns([3, 2], gap="large")
    record: dict = {}

    def render_selects(section: str) -> None:
        with st.container(border=True):
            sec_title(section, SECTION_COLORS[section])
            for c in SECTIONS[section]:
                if c in options:
                    v = st.selectbox(pretty(c), [UNKNOWN] + options[c], key=c)
                    record[c] = None if v == UNKNOWN else v

    with left:
        sec_title("Accident circumstances", NAVY)
        a, b = st.columns(2, gap="medium")
        with a:
            render_selects("Driver and vehicle")
            render_selects("Casualty")
        with b:
            render_selects("Road and environment")
            with st.container(border=True):
                sec_title("Accident details", SECTION_COLORS["Accident details"])
                for c in SECTIONS["Accident details"]:
                    if c in options:
                        v = st.selectbox(pretty(c), [UNKNOWN] + options[c], key=c)
                        record[c] = None if v == UNKNOWN else v
                t = st.time_input("Time of accident", key=TIME_KEY)
                record[config.TIME_COL] = t.strftime("%H:%M:%S")
                n1, n2 = st.columns(2)
                for col_ui, c in zip((n1, n2), config.NUMERIC_COLS):
                    lo, hi = meta["numeric_ranges"][c]
                    with col_ui:
                        record[c] = st.number_input(pretty(c), min_value=lo, max_value=max(hi, lo + 10),
                                                    step=1, key=c)

    result = predict_one(record, model, meta)
    probs = result["probabilities"]
    pred = result["prediction"]
    severe = probs.get("Serious Injury", 0.0) + probs.get("Fatal injury", 0.0)

    with right:
        sec_title("Prediction", TEAL)
        st.markdown(
            f"""
<div class="result" style="background:linear-gradient(135deg,{sev_color(pred)} 0%,{sev_color(pred)}CC 100%)">
  <div class="lbl">Predicted severity</div>
  <div class="val">{html.escape(pred)}</div>
  <div class="sub">Probability of this class: {probs[pred]:.1%}</div>
</div>
<div class="meter-wrap">
  <div class="meter-head"><span>Probability of a severe outcome (serious or fatal)</span>
       <span>{severe:.1%}</span></div>
  <div class="meter"><div style="width:{min(severe, 1) * 100:.1f}%"></div></div>
</div>
""",
            unsafe_allow_html=True,
        )

        dfp = pd.DataFrame({"Class": list(probs), "Probability": list(probs.values())})
        base = alt.Chart(dfp).encode(
            y=alt.Y("Class:N", sort=list(probs), title=None, axis=alt.Axis(labelFontSize=13, domain=False)),
            x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), title=None,
                    axis=alt.Axis(format="%", grid=True)),
        )
        bars = base.mark_bar(cornerRadiusEnd=5, size=26).encode(
            color=alt.Color("Class:N", legend=None,
                            scale=alt.Scale(domain=list(probs), range=[sev_color(c) for c in probs]))
        )
        text = base.mark_text(align="left", dx=6, fontSize=13, fontWeight="bold").encode(
            text=alt.Text("Probability:Q", format=".1%")
        )
        st.altair_chart((bars + text).properties(height=170), use_container_width=True)

        with st.expander("Input summary", expanded=False):
            summary = pd.DataFrame(
                {"Feature": [pretty(k) for k in record],
                 "Value": ["unknown" if v is None else str(v) for v in record.values()]}
            )
            st.dataframe(summary, hide_index=True, height=320)

        st.markdown(
            '<div class="note">These probabilities come from a statistical model trained on historical data '
            "with a strong class imbalance. Use them as a risk indicator, not as a certain outcome.</div>",
            unsafe_allow_html=True,
        )

# ----------------------------------------------------------------------------- performance
with tab_perf:
    sec_title("Held-out test set results", NAVY)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("Macro-F1", f"{metrics['macro_f1']:.3f}", NAVY), unsafe_allow_html=True)
    k2.markdown(kpi("Balanced accuracy", f"{metrics['balanced_accuracy']:.3f}", TEAL), unsafe_allow_html=True)
    k3.markdown(kpi("Accuracy", f"{metrics['accuracy']:.3f}", "#8E5BA8"), unsafe_allow_html=True)
    k4.markdown(kpi("MCC", f"{metrics['mcc']:.3f}", "#C2603A"), unsafe_allow_html=True)
    st.markdown(
        '<div class="note">Accuracy is inflated because most accidents are slight injuries: a model that '
        "always predicts that class already scores high. Macro-F1, balanced accuracy and the recall of the "
        "serious and fatal classes are the honest indicators.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    c_left, c_right = st.columns([2, 3], gap="large")
    with c_left:
        sec_title("Per-class performance", TEAL)
        per = pd.DataFrame(metrics["per_class"]).T
        per = per.rename(columns={"precision": "Precision", "recall": "Recall", "f1-score": "F1", "support": "Support"})
        per["Support"] = per["Support"].astype(int)
        bar = lambda name: st.column_config.ProgressColumn(name, min_value=0.0, max_value=1.0, format="%.2f")  # noqa: E731
        st.dataframe(
            per, column_config={"Precision": bar("Precision"), "Recall": bar("Recall"), "F1": bar("F1")},
            use_container_width=True,
        )
    with c_right:
        sec_title("Model comparison", "#8E5BA8")
        comp = get_comparison()
        if comp is None:
            st.info("Run `python -m src.train` to generate the comparison table.")
        else:
            st.dataframe(
                comp, hide_index=True, use_container_width=True,
                column_config={
                    "CV macro-F1": bar("CV macro-F1"), "Test macro-F1": bar("Test macro-F1"),
                    "Balanced accuracy": bar("Balanced accuracy"),
                    "Accuracy": st.column_config.NumberColumn(format="%.3f"),
                    "MCC": st.column_config.NumberColumn(format="%.3f"),
                },
            )

    figs = [("confusion_matrix.png", "Confusion matrix"), ("feature_importance.png", "Top features")]
    present = [(config.FIGURES_DIR / f, t) for f, t in figs if (config.FIGURES_DIR / f).exists()]
    if present:
        st.write("")
        cols = st.columns(len(present), gap="large")
        for col_ui, (path, title) in zip(cols, present):
            with col_ui:
                sec_title(title, NAVY)
                st.image(str(path))

# ----------------------------------------------------------------------------- about
with tab_about:
    sec_title("End-to-end pipeline", NAVY)
    stages = [
        ("1. Data", "Road traffic accident records for Addis Ababa. Duplicates removed, stratified train/test split, "
                    "exploratory analysis of severity against circumstances.", NAVY),
        ("2. Model", "Preprocessing and classifier live in one scikit-learn pipeline. Logistic regression, random "
                     "forest and XGBoost are tuned with cross-validation using class weights.", TEAL),
        ("3. Evaluation", "Models are selected by cross-validated macro-F1, never on the test set, and compared "
                          "with a majority-class baseline. Decision weights are tuned on out-of-fold predictions.",
         "#8E5BA8"),
        ("4. Deployment", "The fitted pipeline and its metadata are exported and served by this Streamlit app, "
                          "and the code is tested by continuous integration.", "#C2603A"),
    ]
    cols = st.columns(4, gap="medium")
    for col_ui, (title, body, color) in zip(cols, stages):
        col_ui.markdown(
            f'<div class="stage" style="border-top:4px solid {color}"><h4 style="color:{color}">{title}</h4>'
            f"<p>{body}</p></div>",
            unsafe_allow_html=True,
        )
    st.write("")
    sec_title("Limitations", "#D64545")
    st.markdown(
        """
- Several inputs describe the accident after it happened (for example casualty details), so this is a
  risk-assessment demonstration, not a real-time decision tool.
- The fatal class has few examples, so its estimates are noisy.
- The data comes from one city and one period; performance may not transfer elsewhere.
        """
    )
