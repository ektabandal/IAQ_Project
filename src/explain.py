# ============================================================
# explain.py
# Explainable AI helpers.
#
# For the XGBoost model, per-prediction explanations are TreeSHAP values
# computed natively by XGBoost (pred_contribs=True). They show how much
# each input feature pushed the model towards (+) or away from (-) the
# predicted class for ONE prediction. They describe the model's behaviour,
# not physical causes.
# ============================================================

import numpy as np
import pandas as pd

from common import CLASSES, LAGS, ROLL_WINDOW, SENSORS, SHORT_NAMES, SLOPE_WINDOW


def friendly_feature_name(name):
    """Turn a model feature name into a readable phrase."""
    if name == "hour":
        return "hour of day"
    for s in SENSORS:
        label = SHORT_NAMES[s]
        if name == s:
            return f"current {label}"
        if name == f"{s}_slope{SLOPE_WINDOW}":
            return f"{label} trend (last 30 min)"
        if name == f"{s}_mean{ROLL_WINDOW}":
            return f"{label} 1-hour average"
        for k in LAGS:
            if name == f"{s}_lag{k}":
                return f"{label} {5 * k} min ago"
    return name


def explain_prediction(model, x_row, feature_columns, class_index, top_k=5):
    """
    Top contributing features for one prediction.

    Returns a DataFrame with columns
        feature, description, value, contribution
    sorted by absolute contribution, or None if the selected model does not
    provide TreeSHAP values (only the XGBoost model does).
    """
    if not hasattr(model, "contributions"):
        return None
    contrib = model.contributions(x_row[feature_columns])[0, class_index, :-1]
    out = pd.DataFrame({
        "feature": feature_columns,
        "description": [friendly_feature_name(c) for c in feature_columns],
        "value": x_row[feature_columns].iloc[0].to_numpy(dtype=float),
        "contribution": contrib,
    })
    order = np.argsort(-np.abs(out["contribution"].to_numpy()))
    return out.iloc[order].head(top_k).reset_index(drop=True)


def global_shap_importance(model, X):
    """
    Mean absolute TreeSHAP value per feature and class.
    Returns DataFrame (index = feature, columns = class names).
    """
    contrib = model.contributions(X)[:, :, :-1]          # (n, classes, features)
    mean_abs = np.abs(contrib).mean(axis=0)               # (classes, features)
    return pd.DataFrame(mean_abs.T, index=list(X.columns), columns=CLASSES)