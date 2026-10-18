# ============================================================
# app.py
# AI-Based Indoor Air Quality Health Advisor
# Streamlit dashboard: 30-minute-ahead IAQ prediction
#
# Run:  streamlit run app.py
#
# Data sources
#   1. Replay AQUAIR test period  - real readings the model never saw
#      in training; shows what actually happened 30 minutes later.
#   2. Manual entry               - type readings; history builds up.
#   3. Upload recent readings     - CSV with a timestamp + six sensors.
# ============================================================

import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from common import (  # noqa: E402
    ALIASES,
    CLASSES,
    HISTORY_STEPS,
    HORIZON_MINUTES,
    MODEL_PATH,
    SENSOR_LABELS,
    SENSORS,
    SLOPE_WINDOW,
    STEP_MINUTES,
    build_training_table,
    chronological_split,
    create_live_features,
    load_dataset,
    score_to_class,
    trend_per_30_min,
)
from explain import explain_prediction  # noqa: E402
from modeling import proba_full  # noqa: E402  (also needed to unpickle the model)
from recommendations import (  # noqa: E402
    HEALTH_RISK,
    early_warning,
    get_recommendations,
)

st.set_page_config(
    page_title="AI Indoor Air Quality Health Advisor",
    page_icon="🪟",
    layout="wide",
)

CLASS_COLOR = {"GOOD": "🟢", "MODERATE": "🟡", "POOR": "🔴"}


# ------------------------------------------------------------
# Cached resources
# ------------------------------------------------------------

@st.cache_resource
def load_bundle():
    if not Path(MODEL_PATH).exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_replay_data():
    """AQUAIR table and the indices of the held-out test partition."""
    df = load_dataset()
    table = build_training_table(df)
    _, _, test_idx = chronological_split(table)
    return df, table, test_idx


def show_class_box(label, cls):
    text = f"{label}: {CLASS_COLOR[cls]} {cls}"
    if cls == "GOOD":
        st.success(text)
    elif cls == "MODERATE":
        st.warning(text)
    else:
        st.error(text)


# ------------------------------------------------------------
# Header and model check
# ------------------------------------------------------------

st.title("AI-Based Indoor Air Quality Health Advisor")
st.markdown(
    f"Predicts the indoor air quality class **{HORIZON_MINUTES} minutes ahead** "
    "(GOOD / MODERATE / POOR) from temperature, humidity, CO₂, TVOC, PM2.5 and "
    "PM10, explains the prediction and recommends actions."
)

bundle = load_bundle()
if bundle is None:
    st.error("Model not found. Run `python src/train_models.py` first.")
    st.stop()

model = bundle["model"]
feature_columns = bundle["feature_columns"]
class_names = bundle["classes"]

# Minutes of real history the selected feature set needs.
needed_minutes = (SLOPE_WINDOW - 1) * STEP_MINUTES
if any("lag" in c or "mean" in c for c in feature_columns):
    needed_minutes = HISTORY_STEPS * STEP_MINUTES

with st.expander("About the model, its accuracy and its limits", expanded=False):
    t = bundle["test"]
    st.markdown(
        f"**Selected model:** {bundle['model_name']}  \n"
        f"**Feature set:** `{bundle['feature_kind']}` ({len(feature_columns)} features)  \n"
        f"**Held-out test accuracy:** {t['Accuracy']:.1%} | "
        f"**macro-F1:** {t['Macro_F1']:.3f} | "
        f"**balanced accuracy:** {t['Balanced_Accuracy']:.3f}"
    )
    base = bundle.get("baselines_test", {})
    if base:
        st.markdown("**Baselines on the same test set**")
        st.dataframe(pd.DataFrame(base).T.rename_axis("Baseline").reset_index().round(4),
                     hide_index=True)
    st.markdown(
        "- Trained on **AQUAIR**, recorded by an Awair HOME monitor in a trout-farm "
        "facility in Morocco (Oct–Dec 2024). It is **not** a typical occupied "
        "home or office, and the data contain two very different periods.\n"
        "- The GOOD / MODERATE / POOR bands (score ≥ 80 / 60–79 / < 60) are "
        "project assumptions applied to the monitor's IAQ score.\n"
        "- The no-change baseline that uses the monitor's own score is very "
        "strong in this data; see the report for the full comparison.\n"
        "- This is an educational decision-support tool, **not medical advice**."
    )

# ------------------------------------------------------------
# Sidebar: data source
# ------------------------------------------------------------

st.sidebar.header("Data source")
mode = st.sidebar.radio(
    "Choose how to provide readings",
    ["Replay AQUAIR test period", "Manual entry", "Upload recent readings"],
)

history = None            # DataFrame: Timestamp + six sensors
actual_now = None         # (score, class) in replay mode
actual_future = None      # (score, class) in replay mode

# ---------- Mode 1: replay -----------------------------------
if mode == "Replay AQUAIR test period":
    df_all, table, test_idx = load_replay_data()
    test_rows = table.iloc[test_idx].reset_index(drop=True)
    st.sidebar.caption(
        f"{len(test_rows):,} held-out readings from "
        f"{test_rows['Timestamp'].min():%d %b %Y} to "
        f"{test_rows['Timestamp'].max():%d %b %Y %H:%M} (UTC)."
    )
    pos = st.sidebar.slider("Position in test period", 0, len(test_rows) - 1,
                            len(test_rows) // 2)
    only_changes = st.sidebar.checkbox("Jump to moments where the class changes")
    if only_changes:
        change_pos = test_rows.index[
            test_rows["current_class"] != test_rows["future_class"]].tolist()
        if change_pos:
            pick = st.sidebar.select_slider(
                "Class-change moment", options=change_pos,
                format_func=lambda i: f"{test_rows['Timestamp'].iloc[i]:%d %b %H:%M}")
            pos = int(pick)
    row = test_rows.iloc[pos]
    end_ts = pd.Timestamp(row["Timestamp"])
    window = df_all[(df_all["Timestamp"] <= end_ts)
                    & (df_all["Timestamp"] > end_ts - timedelta(minutes=70))]
    history = window[["Timestamp"] + SENSORS].reset_index(drop=True)
    actual_now = (float(row["score"]), row["current_class"])
    actual_future = (float(row["future_score"]), row["future_class"])
    st.sidebar.info(f"Replaying: **{end_ts:%d %b %Y %H:%M} UTC**")

# ---------- Mode 2: manual -----------------------------------
elif mode == "Manual entry":
    if "manual_history" not in st.session_state:
        st.session_state.manual_history = pd.DataFrame(
            columns=["Timestamp"] + SENSORS)

    st.sidebar.caption("Enter a reading, then press **Add reading**. "
                       "Add several readings to build history (trends).")
    defaults = {"Temperature": 17.0, "Humidity": 50.0, "CO2": 600.0,
                "TVOC": 200.0, "PM25": 5.0, "PM10": 8.0}
    ranges = {"Temperature": (-10.0, 50.0, 0.5), "Humidity": (0.0, 100.0, 1.0),
              "CO2": (300.0, 5000.0, 10.0), "TVOC": (0.0, 10000.0, 10.0),
              "PM25": (0.0, 600.0, 1.0), "PM10": (0.0, 700.0, 1.0)}
    values = {}
    for s in SENSORS:
        lo, hi, step = ranges[s]
        values[s] = st.sidebar.number_input(
            SENSOR_LABELS[s], min_value=lo, max_value=hi,
            value=defaults[s], step=step, key=f"in_{s}")
    step_min = st.sidebar.number_input(
        "Minutes since previous reading", min_value=1, max_value=60,
        value=STEP_MINUTES, step=1,
        help="Lets you simulate the 5-minute sampling used in training.")

    c1, c2 = st.sidebar.columns(2)
    if c1.button("Add reading"):
        h = st.session_state.manual_history
        ts = (pd.Timestamp(datetime.now()).floor("s") if h.empty
              else h["Timestamp"].iloc[-1] + timedelta(minutes=int(step_min)))
        new = pd.DataFrame([{"Timestamp": ts, **values}])
        st.session_state.manual_history = pd.concat([h, new], ignore_index=True)
    if c2.button("Clear history"):
        st.session_state.manual_history = pd.DataFrame(
            columns=["Timestamp"] + SENSORS)
        st.rerun()
    if st.sidebar.button("Fill last hour with these values"):
        now = pd.Timestamp(datetime.now()).floor("s")
        times = [now - timedelta(minutes=STEP_MINUTES * k)
                 for k in range(HISTORY_STEPS, -1, -1)]
        st.session_state.manual_history = pd.DataFrame(
            [{"Timestamp": t, **values} for t in times])
        st.sidebar.warning("Steady-state assumption: every reading in the last "
                           "hour equals these values, so trends are zero.")

    history = st.session_state.manual_history.copy()
    if history.empty:
        st.info("Enter readings in the sidebar and press **Add reading**.")
        st.stop()

# ---------- Mode 3: upload -----------------------------------
else:
    st.sidebar.caption("CSV with a timestamp column and the six sensors "
                       "(temp, humid, co2, voc, pm25, pm10), ideally 5-minute "
                       "readings covering the last hour.")
    up = st.sidebar.file_uploader("Recent readings (CSV)", type=["csv"])
    if up is None:
        st.info("Upload a CSV file in the sidebar to continue.")
        st.stop()
    raw = pd.read_csv(up)
    rename = {}
    for col in raw.columns:
        key = str(col).strip().lower().replace(" ", "")
        if key in ALIASES:
            rename[col] = ALIASES[key]
    raw = raw.rename(columns=rename)
    missing = [c for c in ["Timestamp"] + SENSORS if c not in raw.columns]
    if missing:
        st.error(f"Missing columns: {missing}. Found: {list(raw.columns)}")
        st.stop()
    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"], errors="coerce")
    for c in SENSORS:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    history = raw.dropna(subset=["Timestamp"] + SENSORS)[["Timestamp"] + SENSORS]
    if history.empty:
        st.error("No valid rows found in the uploaded file.")
        st.stop()

history = history.sort_values("Timestamp").reset_index(drop=True)
for c in SENSORS:
    history[c] = pd.to_numeric(history[c], errors="coerce")
current = history.iloc[-1]
readings = {s: float(current[s]) for s in SENSORS}

# ------------------------------------------------------------
# Features, prediction, explanation
# ------------------------------------------------------------

feat_all, coverage = create_live_features(history)
X_live = feat_all[feature_columns].astype(float)
proba = proba_full(model, X_live)[0]
pred_idx = int(np.argmax(proba))
predicted = class_names[pred_idx]
trends = trend_per_30_min(feat_all)

if actual_now is not None:
    current_class, current_score = actual_now[1], actual_now[0]
    current_label = "Current IAQ (monitor score)"
    current_note = f"Monitor IAQ score: {current_score:.0f}/100"
else:
    est_score = float(bundle["score_estimator"].predict(
        history[SENSORS].iloc[[-1]])[0])
    est_score = float(np.clip(est_score, 0, 100))
    current_class = score_to_class(est_score)
    current_label = "Current IAQ (estimated from sensors)"
    current_note = (f"Estimated IAQ score: {est_score:.0f}/100 "
                    f"(model: {bundle['score_estimator_name']})")

# ------------------------------------------------------------
# Dashboard
# ------------------------------------------------------------

st.subheader("Current readings")
cols = st.columns(6)
for col, s in zip(cols, SENSORS):
    delta = None
    if len(history) >= 2:
        delta = f"{current[s] - history[s].iloc[-2]:+.1f}"
    col.metric(SENSOR_LABELS[s], f"{current[s]:.1f}", delta)
st.caption(f"Latest reading time: {current['Timestamp']:%d %b %Y %H:%M}")

st.subheader("Prediction")
left, right = st.columns(2)
with left:
    show_class_box(current_label, current_class)
    st.caption(current_note)
with right:
    show_class_box(f"IAQ in {HORIZON_MINUTES} minutes (predicted)", predicted)
    st.caption(f"Model confidence: {proba[pred_idx]:.1%}")

if coverage < needed_minutes:
    st.warning(
        f"Only {coverage:.0f} minutes of history are available; the model "
        f"expects about {needed_minutes} minutes. Missing history is filled "
        "with the earliest reading, so the prediction is less reliable."
    )

pcols = st.columns(len(class_names))
for col, name, p in zip(pcols, class_names, proba):
    col.metric(f"P({name})", f"{p:.1%}")

if actual_future is not None:
    st.markdown("**What actually happened 30 minutes later (replay only)**")
    ok = actual_future[1] == predicted
    msg = (f"Actual class: {CLASS_COLOR[actual_future[1]]} {actual_future[1]} "
           f"(score {actual_future[0]:.0f}). The prediction was "
           f"{'correct ' if ok else 'incorrect '}.")
    (st.success if ok else st.error)(msg)

st.subheader("⚠️ Early warning and health risk")
level, message = early_warning(current_class, predicted)
{"error": st.error, "warning": st.warning,
 "success": st.success, "info": st.info}[level](message)
risk_name, risk_text = HEALTH_RISK[predicted]
st.markdown(f"**Predicted health-risk level: {risk_name}.** {risk_text}")

st.subheader("Why did the model predict this?")
explanation = explain_prediction(model, feat_all, feature_columns, pred_idx, top_k=5)
if explanation is None:
    st.info("The selected model type does not provide per-prediction explanations.")
else:
    st.caption(
        f"Features that pushed the model towards **{predicted}** (+) or away from "
        "it (−). These are TreeSHAP values of the model, not physical causes."
    )
    show = explanation[["description", "value", "contribution"]].copy()
    show.columns = ["Feature", "Value", "Effect on prediction"]
    show["Direction"] = np.where(show["Effect on prediction"] >= 0,
                                 f"towards {predicted}", f"away from {predicted}")
    show["Value"] = show["Value"].round(3)
    show["Effect on prediction"] = show["Effect on prediction"].round(3)
    st.dataframe(show, hide_index=True)
    st.bar_chart(show.set_index("Feature")["Effect on prediction"])

st.subheader(" Recommendations")
recs = get_recommendations(current_class, predicted, readings, trends)
for r in recs:
    icon = {"urgent": "🚨", "action": "👉", "info": "ℹ️"}[r["level"]]
    st.markdown(f"{icon} {r['text']}")

st.subheader("📈 Recent history")
chart_df = history.set_index("Timestamp")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**CO₂ (ppm) and TVOC (ppb)**")
    st.line_chart(chart_df[["CO2", "TVOC"]])
    st.markdown("**Temperature (°C) and humidity (%)**")
    st.line_chart(chart_df[["Temperature", "Humidity"]])
with c2:
    st.markdown("**PM2.5 and PM10 (µg/m³)**")
    st.line_chart(chart_df[["PM25", "PM10"]])
    st.markdown("**Projected 30-minute change at the current rate**")
    st.dataframe(pd.DataFrame({
        "Sensor": [SENSOR_LABELS[s] for s in SENSORS],
        "Change in 30 min": [round(trends[s], 2) for s in SENSORS],
    }), hide_index=True)

with st.expander("Model input row (for debugging)"):
    st.dataframe(X_live.round(3), hide_index=True)

st.markdown("---")
st.caption(
    "AI-Based Indoor Air Quality Health Advisor | 30-minute-ahead prediction | "
    "Educational decision support, not medical advice."
)
