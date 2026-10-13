# ============================================================
# common.py
# AI-Based Indoor Air Quality Health Advisor
# 30-Minute-Ahead IAQ Class Prediction
#
# Shared by train_models.py, experiments.py, app.py and tests.
# One single implementation of data loading, labels and feature
# engineering is used for training AND for the live application,
# so training and deployment can never silently disagree.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 1. PROJECT SETTINGS
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "AQUAIR_1.csv"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"
FIG_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"

MODEL_PATH = MODEL_DIR / "best_iaq_30min_model.pkl"

RANDOM_STATE = 42

STEP_MINUTES = 5            # AQUAIR sampling interval
HORIZON_MINUTES = 30        # prediction horizon (30 minutes ONLY)
HORIZON_STEPS = HORIZON_MINUTES // STEP_MINUTES   # = 6 rows

# Lag steps (x5 min) and rolling window used as features, as in the
# best-performing notebook (Untitled4.ipynb).
LAGS = (1, 3, 6, 12)        # 5, 15, 30 and 60 minutes ago
ROLL_WINDOW = 12            # mean of the last hour
SLOPE_WINDOW = 6            # linear slope over the last 30 minutes
HISTORY_STEPS = max(LAGS + (ROLL_WINDOW - 1,))    # 12 previous rows needed

# ------------------------------------------------------------
# 2. SENSORS (the six variables that exist in AQUAIR_1.csv)
# ------------------------------------------------------------
# NOTE: CO and occupancy are NOT measured in AQUAIR, so they are not used.

SENSORS = ["Temperature", "Humidity", "CO2", "TVOC", "PM25", "PM10"]

SENSOR_LABELS = {
    "Temperature": "Temperature (°C)",
    "Humidity": "Relative humidity (%)",
    "CO2": "CO₂ (ppm)",
    "TVOC": "TVOC (ppb)",
    "PM25": "PM2.5 (µg/m³)",
    "PM10": "PM10 (µg/m³)",
}

SHORT_NAMES = {
    "Temperature": "temperature",
    "Humidity": "humidity",
    "CO2": "CO₂",
    "TVOC": "TVOC",
    "PM25": "PM2.5",
    "PM10": "PM10",
}

CLASSES = ["GOOD", "MODERATE", "POOR"]
CLASS_TO_INT = {name: i for i, name in enumerate(CLASSES)}

# Column-name aliases accepted when reading the CSV.
ALIASES = {
    "timestamp(utc)": "Timestamp",
    "timestamp": "Timestamp",
    "datetime": "Timestamp",
    "time": "Timestamp",
    "score": "score",
    "temp": "Temperature",
    "temperature": "Temperature",
    "humid": "Humidity",
    "humidity": "Humidity",
    "co2": "CO2",
    "voc": "TVOC",
    "tvoc": "TVOC",
    "pm25": "PM25",
    "pm2.5": "PM25",
    "pm10": "PM10",
}

# ------------------------------------------------------------
# 3. IAQ CLASS FROM THE DEVICE SCORE
# ------------------------------------------------------------
# AQUAIR was recorded with an Awair HOME monitor that reports an
# IAQ score from 0 to 100. The class bands below are PROJECT
# ASSUMPTIONS (they are not defined by the dataset authors):
#     score <  60        -> POOR
#     60 <= score < 80   -> MODERATE
#     score >= 80        -> GOOD

POOR_BELOW = 60.0
GOOD_FROM = 80.0


def score_to_class(score):
    """Convert an IAQ score (0-100) to GOOD / MODERATE / POOR."""
    if score is None or pd.isna(score):
        return np.nan
    if score < POOR_BELOW:
        return "POOR"
    if score < GOOD_FROM:
        return "MODERATE"
    return "GOOD"


def scores_to_classes(scores):
    """Vectorised version of score_to_class (returns a numpy array)."""
    s = np.asarray(scores, dtype=float)
    out = np.where(s < POOR_BELOW, "POOR",
                   np.where(s < GOOD_FROM, "MODERATE", "GOOD")).astype(object)
    out[np.isnan(s)] = None
    return out


# ------------------------------------------------------------
# 4. LOAD AND CLEAN DATASET (in memory; the CSV is never changed)
# ------------------------------------------------------------

def load_dataset(path=DATA_PATH):
    """
    Read AQUAIR_1.csv, standardise column names, parse timestamps,
    remove duplicate timestamps and mark continuous 5-minute segments.
    """
    df = pd.read_csv(path)

    rename = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "")
        if key in ALIASES:
            rename[col] = ALIASES[key]
    df = df.rename(columns=rename)

    missing = [c for c in ["Timestamp", "score"] + SENSORS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing}. Columns found: {df.columns.tolist()}"
        )

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
    df["Timestamp"] = df["Timestamp"].dt.tz_localize(None)   # keep UTC wall time

    for col in SENSORS + ["score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Timestamp"])
    df = df.sort_values("Timestamp")
    df = df.drop_duplicates(subset=["Timestamp"], keep="last")
    df = df[["Timestamp", "score"] + SENSORS].reset_index(drop=True)

    # A new segment starts whenever the gap is not exactly 5 minutes.
    gap = df["Timestamp"].diff()
    df["segment"] = (gap != pd.Timedelta(minutes=STEP_MINUTES)).cumsum()
    return df


# ------------------------------------------------------------
# 5. FEATURE ENGINEERING (shared by training and the live app)
# ------------------------------------------------------------

def feature_names(kind):
    """
    Names of the model input columns for a feature-set definition.

      raw        : the six current sensor readings
      raw_slope  : raw + 30-minute linear slope of each sensor
      lags       : raw + lags (5,15,30,60 min) + 1-hour mean   <- final
      lags_hour  : lags + hour of day (as in Untitled4.ipynb)
    """
    raw = list(SENSORS)
    slope = [f"{s}_slope6" for s in SENSORS]
    lags = [f"{s}_lag{k}" for s in SENSORS for k in LAGS]
    means = [f"{s}_mean{ROLL_WINDOW}" for s in SENSORS]

    if kind == "raw":
        return raw
    if kind == "raw_slope":
        return raw + slope
    if kind == "lags":
        return raw + lags + means
    if kind == "lags_hour":
        return raw + lags + means + ["hour"]
    raise ValueError(f"Unknown feature set: {kind}")


FEATURE_SET_DESCRIPTIONS = {
    "raw": "Current sensor values only (6 features)",
    "raw_slope": "Current values + 30-min slope per sensor (12 features)",
    "lags": "Current values + lags (5/15/30/60 min) + 1-h mean (36 features)",
    "lags_hour": "Lag set + hour of day (37 features, as in Untitled4.ipynb)",
}

_SLOPE_X = np.arange(SLOPE_WINDOW, dtype=float) * STEP_MINUTES
_SLOPE_W = (_SLOPE_X - _SLOPE_X.mean()) / np.sum((_SLOPE_X - _SLOPE_X.mean()) ** 2)


def _slope(values):
    """Least-squares slope (units per minute) over a window of regular samples."""
    return float(np.dot(_SLOPE_W, values))


def make_features(grid):
    """
    Build every candidate feature for ONE continuous 5-minute segment.

    `grid` must be sorted by Timestamp, have exactly 5 minutes between rows
    and contain the sensor columns. Only the current and PAST rows are used,
    so no future information leaks into the features. The first
    HISTORY_STEPS rows of a segment have NaN features and must be dropped.
    """
    out = pd.DataFrame(index=grid.index)
    out["Timestamp"] = grid["Timestamp"].values

    for s in SENSORS:
        series = grid[s].astype(float)
        out[s] = series
        for k in LAGS:
            out[f"{s}_lag{k}"] = series.shift(k)
        out[f"{s}_mean{ROLL_WINDOW}"] = series.rolling(ROLL_WINDOW).mean()
        out[f"{s}_slope{SLOPE_WINDOW}"] = series.rolling(SLOPE_WINDOW).apply(
            _slope, raw=True
        )

    out["hour"] = pd.to_datetime(out["Timestamp"]).dt.hour
    return out


def build_training_table(df):
    """
    Create the supervised table from the cleaned dataset:
      * features per continuous segment (no mixing across the data gap)
      * exact 30-minute-ahead target from the future IAQ score
    Rows without a full 1-hour history or without an exact +30 min
    observation are dropped.
    """
    parts = []
    for _, seg in df.groupby("segment", sort=True):
        seg = seg.reset_index(drop=True)
        feats = make_features(seg)
        feats["score"] = seg["score"].values
        feats["segment"] = seg["segment"].values
        # exact future score: row whose timestamp is exactly +30 minutes
        by_time = pd.Series(seg["score"].values, index=seg["Timestamp"])
        target_time = seg["Timestamp"] + pd.Timedelta(minutes=HORIZON_MINUTES)
        feats["target_time"] = target_time.values
        feats["future_score"] = by_time.reindex(pd.DatetimeIndex(target_time)).values
        parts.append(feats)

    table = pd.concat(parts, ignore_index=True)
    table["current_class"] = scores_to_classes(table["score"])
    table["future_class"] = scores_to_classes(table["future_score"])

    all_feats = feature_names("lags_hour") + feature_names("raw_slope")
    keep = table[all_feats].notna().all(axis=1) & table["future_class"].notna()
    table = table.loc[keep].reset_index(drop=True)
    table["y"] = table["future_class"].map(CLASS_TO_INT).astype(int)
    return table


def chronological_split(table, train_frac=0.70, val_frac=0.15):
    """
    Chronological 70/15/15 split with a purge gap.

    Rows whose 30-minute target falls inside the NEXT partition are removed
    from the earlier partition, so no target information crosses a boundary.
    Returns three integer index arrays (train, validation, test).
    """
    n = len(table)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    t = pd.to_datetime(table["Timestamp"]).reset_index(drop=True)
    tt = pd.to_datetime(table["target_time"]).reset_index(drop=True)
    idx = np.arange(n)

    train = idx[(idx < train_end) & (tt < t.iloc[train_end]).to_numpy()]
    val = idx[(idx >= train_end) & (idx < val_end) & (tt < t.iloc[val_end]).to_numpy()]
    test = idx[idx >= val_end]
    return train, val, test


# ------------------------------------------------------------
# 6. LIVE FEATURES FROM IRREGULAR READINGS (used by app.py)
# ------------------------------------------------------------

def history_to_grid(history_df, end_time=None):
    """
    Convert irregular live readings into the regular 5-minute grid used in
    training. For each grid time the most recent reading at or before that
    time is used (last observation carried forward). If the history is
    shorter than one hour, the earliest reading fills the missing past and
    `coverage_minutes` reports how much real history was available.

    Returns (grid_df with HISTORY_STEPS+1 rows, coverage_minutes).
    """
    h = history_df.copy()
    h["Timestamp"] = pd.to_datetime(h["Timestamp"])
    h = h.sort_values("Timestamp").drop_duplicates("Timestamp", keep="last")
    h = h.reset_index(drop=True)
    if h.empty:
        raise ValueError("History is empty.")

    end = pd.Timestamp(end_time) if end_time is not None else h["Timestamp"].iloc[-1]
    times = [end - pd.Timedelta(minutes=STEP_MINUTES * k)
             for k in range(HISTORY_STEPS, -1, -1)]
    grid_times = pd.DataFrame({"Timestamp": pd.DatetimeIndex(times)})

    merged = pd.merge_asof(
        grid_times, h[["Timestamp"] + SENSORS],
        on="Timestamp", direction="backward",
    )
    # fill grid points before the first reading with the earliest reading
    merged[SENSORS] = merged[SENSORS].bfill()

    coverage = (end - h["Timestamp"].iloc[0]).total_seconds() / 60.0
    coverage = float(min(max(coverage, 0.0), HISTORY_STEPS * STEP_MINUTES))
    return merged, coverage


def create_live_features(history_df, end_time=None):
    """
    Build ONE row containing every candidate feature (raw, slope, lags, means,
    hour) from the most recent readings, using exactly the same
    make_features() as training. The caller selects the model's columns:

        row, coverage = create_live_features(history)
        X = row[bundle["feature_columns"]]

    Returns (all_features_row_df, coverage_minutes).
    """
    grid, coverage = history_to_grid(history_df, end_time=end_time)
    feats = make_features(grid)
    row = feats.iloc[[-1]].reset_index(drop=True)
    return row, coverage


def trend_per_30_min(feature_row):
    """Expected change of each sensor over 30 minutes from its recent slope."""
    return {
        s: float(feature_row[f"{s}_slope{SLOPE_WINDOW}"].iloc[0]) * HORIZON_MINUTES
        for s in SENSORS
    }