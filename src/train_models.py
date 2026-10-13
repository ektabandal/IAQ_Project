# ============================================================
# train_models.py
# AI-Based Indoor Air Quality Health Advisor
# 30-MINUTE-AHEAD IAQ CLASS PREDICTION (GOOD / MODERATE / POOR)
#
# Dataset : data/AQUAIR_1.csv  (never modified)
# Run     : python src/train_models.py
#
# Protocol (see report, Section 7.1)
#   1. Chronological split: 70 % train / 15 % validation / 15 % test,
#      with a 30-minute purge gap at every boundary.
#   2. Train + validation = "development set". All model selection and
#      tuning use blocked, expanding-window time-series cross-validation
#      on the development set only.
#   3. The test set (last 15 %) is used ONCE, for the final report.
#   4. Features are built only from current/past readings, per continuous
#      5-minute segment (no mixing across the data gap), as in
#      Untitled4.ipynb (lags, 1-hour rolling mean, XGBoost).
# ============================================================

import json
import platform
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import ParameterSampler
from xgboost import XGBRegressor

from common import (
    CLASSES,
    CLASS_TO_INT,
    FEATURE_SET_DESCRIPTIONS,
    GOOD_FROM,
    HORIZON_MINUTES,
    MODEL_DIR,
    MODEL_PATH,
    POOR_BELOW,
    RANDOM_STATE,
    REPORT_DIR,
    SENSORS,
    TABLE_DIR,
    build_training_table,
    chronological_split,
    feature_names,
    load_dataset,
    scores_to_classes,
)
from modeling import (
    RobustXGB,
    blocked_folds,
    build_candidates,
    cross_validate,
    metrics,
    predict_full,
)

N_TUNING_CONFIGS = 16

PARAM_GRID = {
    "n_estimators": [150, 300, 500],
    "learning_rate": [0.03, 0.05, 0.10],
    "max_depth": [3, 4, 6, 8],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 10],
    "reg_lambda": [1.0, 5.0, 10.0],
    "balanced": [False, True],
}


def banner(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78, flush=True)


# ------------------------------------------------------------
# Baselines
# ------------------------------------------------------------

def class_ints(series_of_names):
    return series_of_names.map(CLASS_TO_INT).astype(int).to_numpy()


def pick_score_estimator(table, dev, folds):
    """
    Choose the sensors -> current-score regressor by blocked-CV MAE.
    It is used (a) as the 'persistence from sensors' baseline and
    (b) by the app to display the current IAQ class from sensor values.
    """
    X = table[SENSORS]
    score = table["score"]
    candidates = {
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, min_samples_leaf=2, n_jobs=1,
            random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, n_jobs=1,
            random_state=RANDOM_STATE),
    }
    rows = []
    for name, reg in candidates.items():
        pred_all, true_all = [], []
        for tr, te in folds:
            m = clone(reg).fit(X.iloc[tr], score.iloc[tr])
            pred_all.append(m.predict(X.iloc[te]))
            true_all.append(score.iloc[te].to_numpy())
        pred_all = np.concatenate(pred_all)
        true_all = np.concatenate(true_all)
        rows.append({
            "Regressor": name,
            "CV_MAE": mean_absolute_error(true_all, pred_all),
            "CV_R2": r2_score(true_all, pred_all),
        })
    res = pd.DataFrame(rows).sort_values("CV_MAE")
    best = res.iloc[0]["Regressor"]
    return best, candidates[best], res


def persistence_estimated_oof(table, folds, estimator):
    """Out-of-fold persistence baseline using the SENSOR-estimated current class."""
    X = table[SENSORS]
    idx, pred = [], []
    for tr, te in folds:
        m = clone(estimator).fit(X.iloc[tr], table["score"].iloc[tr])
        est = scores_to_classes(m.predict(X.iloc[te]))
        pred.append(np.array([CLASS_TO_INT[c] for c in est]))
        idx.append(np.asarray(te))
    return np.concatenate(idx), np.concatenate(pred)


# ------------------------------------------------------------
# Main workflow
# ------------------------------------------------------------

def main():
    t_start = time.perf_counter()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    banner("1. LOAD DATA AND BUILD 30-MINUTE TARGET")
    df = load_dataset()
    table = build_training_table(df)
    y = table["y"]

    print(f"Raw rows after cleaning       : {len(df):,}")
    print(f"Continuous segments           : {df['segment'].nunique()}")
    print(f"Rows with full features+target: {len(table):,}")

    train, val, test = chronological_split(table)
    dev = np.concatenate([train, val])
    folds = blocked_folds(table, dev)

    split_rows = []
    for name, idx in [("Train (70%)", train), ("Validation (15%)", val),
                      ("Test (15%)", test), ("Development = Train+Val", dev)]:
        part = table.iloc[idx]
        counts = part["future_class"].value_counts()
        row = {
            "Partition": name,
            "Rows": len(idx),
            "Start": part["Timestamp"].min(),
            "End": part["Timestamp"].max(),
            "No_change_accuracy": float(
                (part["current_class"] == part["future_class"]).mean()),
        }
        for c in CLASSES:
            row[f"{c}_count"] = int(counts.get(c, 0))
            row[f"{c}_pct"] = 100.0 * counts.get(c, 0) / len(idx)
        split_rows.append(row)
    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(TABLE_DIR / "dataset_split_summary.csv", index=False)
    print(split_df[["Partition", "Rows", "Start", "End"]].to_string(index=False))
    print(split_df[["Partition"] + [f"{c}_pct" for c in CLASSES]].round(1).to_string(index=False))

    fold_info = pd.DataFrame([{
        "Fold": i + 1, "Train_rows": len(a), "Test_rows": len(b),
        "Test_start": table["Timestamp"].iloc[b[0]],
        "Test_end": table["Timestamp"].iloc[b[-1]],
        **{f"{c}_in_test": int((y.iloc[b] == CLASS_TO_INT[c]).sum()) for c in CLASSES},
    } for i, (a, b) in enumerate(folds)])
    fold_info.to_csv(TABLE_DIR / "cv_fold_definition.csv", index=False)
    print("\nBlocked CV folds:")
    print(fold_info.to_string(index=False))

    # --------------------------------------------------------
    banner("2. FEATURE-SET COMPARISON (XGBoost, notebook settings)")
    feat_rows = []
    best_feat, best_feat_f1 = None, -1.0
    for kind in ["raw", "raw_slope", "lags", "lags_hour"]:
        X = table[feature_names(kind)]
        fold_df, pooled, _ = cross_validate(RobustXGB(), X, y, folds)
        model = RobustXGB().fit(X.iloc[dev], y.iloc[dev])
        test_m = metrics(y.iloc[test], predict_full(model, X.iloc[test]))
        feat_rows.append({
            "Feature_set": kind,
            "Description": FEATURE_SET_DESCRIPTIONS[kind],
            "N_features": X.shape[1],
            "CV_pooled_Accuracy": pooled["Accuracy"],
            "CV_pooled_Macro_F1": pooled["Macro_F1"],
            "CV_fold_Accuracy_mean": fold_df["Accuracy"].mean(),
            "CV_fold_Accuracy_std": fold_df["Accuracy"].std(),
            "Test_Accuracy": test_m["Accuracy"],
            "Test_Macro_F1": test_m["Macro_F1"],
        })
        print(f"{kind:10s} CV macro-F1 {pooled['Macro_F1']:.4f}  "
              f"CV acc {pooled['Accuracy']:.4f}", flush=True)
        if pooled["Macro_F1"] > best_feat_f1:
            best_feat, best_feat_f1 = kind, pooled["Macro_F1"]
    pd.DataFrame(feat_rows).to_csv(TABLE_DIR / "feature_set_comparison.csv", index=False)

    # The hour-of-day feature depends on the facility clock (UTC in AQUAIR)
    # and ties the model to this site's schedule. It is only allowed if it is
    # clearly the best on CV; otherwise the best clock-free set is used.
    feat_df = pd.DataFrame(feat_rows).set_index("Feature_set")
    clock_free = feat_df.drop(index="lags_hour")
    best_clock_free = clock_free["CV_pooled_Macro_F1"].idxmax()
    if best_feat == "lags_hour" and (
        feat_df.loc["lags_hour", "CV_pooled_Macro_F1"]
        - feat_df.loc[best_clock_free, "CV_pooled_Macro_F1"] < 0.02
    ):
        best_feat = best_clock_free
    feature_kind = best_feat
    feature_cols = feature_names(feature_kind)
    X = table[feature_cols]
    print(f"\nSelected feature set: {feature_kind} ({len(feature_cols)} features)")

    # --------------------------------------------------------
    banner("3. SCORE ESTIMATOR (sensors -> current IAQ score) AND BASELINES")
    est_name, est_model, est_table = pick_score_estimator(table, dev, folds)
    est_table.to_csv(TABLE_DIR / "score_estimator_cv.csv", index=False)
    print(est_table.round(4).to_string(index=False))
    print(f"Selected score estimator: {est_name}")

    baseline_rows = []

    def add_baseline(name, note, oof_idx, oof_pred, test_pred):
        cvm = metrics(y.iloc[oof_idx], oof_pred)
        tm = metrics(y.iloc[test], test_pred)
        baseline_rows.append({
            "Model": name, "Type": "Baseline", "Note": note,
            **{f"CV_{k}": v for k, v in cvm.items()},
            **{f"Test_{k}": v for k, v in tm.items()},
        })

    # (a) majority class of the training rows of each fold
    oof_idx = np.concatenate([np.asarray(te) for _, te in folds])
    maj_pred = np.concatenate([
        np.full(len(te), y.iloc[tr].mode()[0]) for tr, te in folds])
    add_baseline(
        "Majority class", "Most frequent class in the training rows",
        oof_idx, maj_pred, np.full(len(test), y.iloc[dev].mode()[0]))

    # (b) persistence using the device's OWN current score (not a model input)
    cur_obs = class_ints(table["current_class"])
    add_baseline(
        "No-change (observed device score)",
        "Predicts that the monitor's current class persists; uses the "
        "device score, which sensor-only models do not receive",
        oof_idx, cur_obs[oof_idx], cur_obs[test])

    # (c) persistence using the class ESTIMATED from the sensors
    e_idx, e_pred = persistence_estimated_oof(table, folds, est_model)
    est_full = clone(est_model).fit(table[SENSORS].iloc[dev], table["score"].iloc[dev])
    est_test = np.array([CLASS_TO_INT[c] for c in
                         scores_to_classes(est_full.predict(table[SENSORS].iloc[test]))])
    add_baseline(
        "No-change (class estimated from sensors)",
        "Persistence using only sensor readings (what the app can compute)",
        e_idx, e_pred, est_test)

    est_test_score = est_full.predict(table[SENSORS].iloc[test])
    est_summary = {
        "name": est_name,
        "test_R2": float(r2_score(table["score"].iloc[test], est_test_score)),
        "test_MAE": float(mean_absolute_error(table["score"].iloc[test], est_test_score)),
        "test_class_accuracy": float(np.mean(
            est_test == class_ints(table["current_class"])[test])),
    }
    print("Score estimator on test:", {k: (round(v, 4) if not isinstance(v, str) else v)
                                       for k, v in est_summary.items()})

    # --------------------------------------------------------
    banner("4. ALGORITHM COMPARISON (same folds, same features, same metrics)")
    comparison = list(baseline_rows)
    fitted_oof = {}
    candidates = build_candidates()

    for name, estimator in candidates.items():
        t0 = time.perf_counter()
        fold_df, pooled, oof = cross_validate(estimator, X, y, folds)
        full_fit_t0 = time.perf_counter()
        model = clone(estimator).fit(X.iloc[dev], y.iloc[dev])
        full_fit_time = time.perf_counter() - full_fit_t0
        t1 = time.perf_counter()
        pred = predict_full(model, X.iloc[test])
        pred_ms = (time.perf_counter() - t1) * 1000.0
        tm = metrics(y.iloc[test], pred)
        comparison.append({
            "Model": name, "Type": "ML model",
            "Note": "Defaults documented in modeling.py",
            **{f"CV_{k}": v for k, v in pooled.items()},
            "CV_fold_Accuracy_mean": fold_df["Accuracy"].mean(),
            "CV_fold_Accuracy_std": fold_df["Accuracy"].std(),
            "CV_fold_MacroF1_mean": fold_df["Macro_F1"].mean(),
            "CV_fold_MacroF1_std": fold_df["Macro_F1"].std(),
            **{f"Test_{k}": v for k, v in tm.items()},
            "Fit_seconds_dev_set": full_fit_time,
            "Predict_ms_test_set": pred_ms,
        })
        print(f"{name:32s} CV macro-F1 {pooled['Macro_F1']:.4f} "
              f"acc {pooled['Accuracy']:.4f} | fit {full_fit_time:5.1f}s "
              f"({time.perf_counter() - t0:.0f}s total)", flush=True)

    # --------------------------------------------------------
    banner("5. XGBOOST HYPERPARAMETER TUNING (random search, blocked CV)")
    sampler = list(ParameterSampler(
        PARAM_GRID, n_iter=N_TUNING_CONFIGS, random_state=RANDOM_STATE))
    tune_rows = []
    best_cfg, best_cfg_f1 = None, -1.0
    for i, params in enumerate(sampler, start=1):
        t0 = time.perf_counter()
        fold_df, pooled, _ = cross_validate(RobustXGB(**params), X, y, folds)
        row = {"Config": i, **params,
               "CV_pooled_Macro_F1": pooled["Macro_F1"],
               "CV_pooled_Accuracy": pooled["Accuracy"],
               "CV_fold_Accuracy_std": fold_df["Accuracy"].std(),
               "Seconds": time.perf_counter() - t0}
        tune_rows.append(row)
        print(f"config {i:2d}/{N_TUNING_CONFIGS} macro-F1 {pooled['Macro_F1']:.4f} "
              f"acc {pooled['Accuracy']:.4f} {params}", flush=True)
        if pooled["Macro_F1"] > best_cfg_f1:
            best_cfg, best_cfg_f1 = params, pooled["Macro_F1"]
    tune_df = pd.DataFrame(tune_rows).sort_values("CV_pooled_Macro_F1", ascending=False)
    tune_df.to_csv(TABLE_DIR / "xgboost_tuning.csv", index=False)
    print("\nBest XGBoost configuration:", best_cfg)

    tuned = RobustXGB(**best_cfg)
    fold_df, pooled, _ = cross_validate(tuned, X, y, folds)
    t0 = time.perf_counter()
    tuned_model = clone(tuned).fit(X.iloc[dev], y.iloc[dev])
    fit_time = time.perf_counter() - t0
    t1 = time.perf_counter()
    pred = predict_full(tuned_model, X.iloc[test])
    pred_ms = (time.perf_counter() - t1) * 1000.0
    tm = metrics(y.iloc[test], pred)
    comparison.append({
        "Model": "XGBoost (tuned)", "Type": "ML model",
        "Note": f"Random search, {N_TUNING_CONFIGS} configs",
        **{f"CV_{k}": v for k, v in pooled.items()},
        "CV_fold_Accuracy_mean": fold_df["Accuracy"].mean(),
        "CV_fold_Accuracy_std": fold_df["Accuracy"].std(),
        "CV_fold_MacroF1_mean": fold_df["Macro_F1"].mean(),
        "CV_fold_MacroF1_std": fold_df["Macro_F1"].std(),
        **{f"Test_{k}": v for k, v in tm.items()},
        "Fit_seconds_dev_set": fit_time,
        "Predict_ms_test_set": pred_ms,
    })
    candidates["XGBoost (tuned)"] = tuned

    # --------------------------------------------------------
    banner("6. FINAL MODEL SELECTION (cross-validation only)")
    comp = pd.DataFrame(comparison)
    ml = comp[comp["Type"] == "ML model"].sort_values(
        ["CV_Macro_F1", "CV_Accuracy"], ascending=False)
    best_name = ml.iloc[0]["Model"]
    comp["Selected"] = comp["Model"].eq(best_name)
    comp.to_csv(TABLE_DIR / "model_comparison.csv", index=False)
    print(ml[["Model", "CV_Macro_F1", "CV_Accuracy"]].round(4).to_string(index=False))
    print(f"\nSelected model (highest CV macro-F1): {best_name}")

    final_model = clone(candidates[best_name]).fit(X.iloc[dev], y.iloc[dev])
    test_pred = predict_full(final_model, X.iloc[test])
    test_m = metrics(y.iloc[test], test_pred)

    banner("7. FINAL TEST RESULTS (held-out last 15 %, used once)")
    for k, v in test_m.items():
        print(f"{k:20s}: {v:.4f}")
    report = classification_report(
        y.iloc[test], test_pred, labels=[0, 1, 2], target_names=CLASSES,
        zero_division=0, digits=3)
    print("\n" + report)
    cm = confusion_matrix(y.iloc[test], test_pred, labels=[0, 1, 2])
    cm_df = pd.DataFrame(
        cm, index=[f"Actual_{c}" for c in CLASSES],
        columns=[f"Pred_{c}" for c in CLASSES])
    print(cm_df)

    cm_df.to_csv(TABLE_DIR / "test_confusion_matrix.csv")
    pd.DataFrame(classification_report(
        y.iloc[test], test_pred, labels=[0, 1, 2], target_names=CLASSES,
        zero_division=0, output_dict=True)).T.to_csv(
        TABLE_DIR / "test_classification_report.csv")
    (REPORT_DIR / "classification_report.txt").write_text(
        f"Selected model: {best_name}\nFeature set: {feature_kind}\n"
        f"Horizon: {HORIZON_MINUTES} minutes\n\n{report}\n{cm_df}\n")

    # --------------------------------------------------------
    banner("8. SAVE MODEL BUNDLE")
    baseline_test = {
        r["Model"]: {"Accuracy": r["Test_Accuracy"], "Macro_F1": r["Test_Macro_F1"]}
        for r in baseline_rows
    }
    best_row = comp[comp["Model"] == best_name].iloc[0]
    bundle = {
        "model": final_model,
        "model_name": best_name,
        "best_params": best_cfg if best_name == "XGBoost (tuned)" else None,
        "feature_kind": feature_kind,
        "feature_columns": feature_cols,
        "classes": CLASSES,
        "horizon_minutes": HORIZON_MINUTES,
        "score_estimator": est_full,
        "score_estimator_name": est_name,
        "score_estimator_test": est_summary,
        "class_bands": {"poor_below": POOR_BELOW, "good_from": GOOD_FROM},
        "cv_pooled": {k.replace("CV_", ""): float(best_row[k])
                      for k in best_row.index if k.startswith("CV_")
                      and k in ("CV_Accuracy", "CV_Balanced_Accuracy",
                                "CV_Macro_F1", "CV_Weighted_F1")},
        "test": {k: float(v) for k, v in test_m.items()},
        "baselines_test": baseline_test,
        "data_range": [str(table["Timestamp"].min()), str(table["Timestamp"].max())],
        "rows": {"train": len(train), "validation": len(val),
                 "test": len(test), "development": len(dev)},
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
        },
    }
    joblib.dump(bundle, MODEL_PATH)

    summary = {k: v for k, v in bundle.items()
               if k not in ("model", "score_estimator")}
    (REPORT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"Model saved   : {MODEL_PATH}")
    print(f"Tables saved  : {TABLE_DIR}")
    print(f"Total runtime : {(time.perf_counter() - t_start) / 60:.1f} min")
    print("\nReminders:")
    print("- Compare the model with the no-change baselines before trusting it.")
    print("- The test set was used once; do not re-tune against it.")
    print("- The original CSV was not modified.")


if __name__ == "__main__":
    main()