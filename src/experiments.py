# ============================================================
# experiments.py
# Additional experiments, tables and figures for the report.
# Run AFTER train_models.py:   python src/experiments.py
#
# Everything here is computed from the saved model bundle and the
# development/test split. Nothing is tuned on the test set.
# ============================================================

import io
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from xgboost import XGBClassifier, XGBRegressor

from common import (
    CLASS_TO_INT,
    CLASSES,
    FIG_DIR,
    GOOD_FROM,
    HORIZON_MINUTES,
    MODEL_PATH,
    POOR_BELOW,
    RANDOM_STATE,
    SENSOR_LABELS,
    SENSORS,
    TABLE_DIR,
    build_training_table,
    chronological_split,
    feature_names,
    load_dataset,
    scores_to_classes,
)
from explain import friendly_feature_name, global_shap_importance
from modeling import (
    RobustXGB,
    blocked_folds,
    metrics,
    predict_full,
    proba_full,
)

sns.set_theme(style="whitegrid", context="notebook")
CLASS_COLORS = {"GOOD": "#2e9e5b", "MODERATE": "#f0b429", "POOR": "#d64545"}
REGIME_BOUNDARY = pd.Timestamp("2024-11-10 12:00:00")


def save_fig(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}", flush=True)


def banner(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70, flush=True)


# ------------------------------------------------------------
# Context
# ------------------------------------------------------------

def load_context():
    bundle = joblib.load(MODEL_PATH)
    df = load_dataset()
    table = build_training_table(df)
    train, val, test = chronological_split(table)
    dev = np.concatenate([train, val])
    cols = bundle["feature_columns"]
    return dict(bundle=bundle, df=df, table=table, train=train, val=val,
                test=test, dev=dev, cols=cols, X=table[cols], y=table["y"])


# ------------------------------------------------------------
# E0. Exploratory data analysis
# ------------------------------------------------------------

def eda(ctx):
    banner("E0. Exploratory data analysis")
    df, table = ctx["df"], ctx["table"]

    # Summary table
    desc = df[SENSORS + ["score"]].describe().T[["min", "mean", "50%", "max", "std"]]
    desc.columns = ["Min", "Mean", "Median", "Max", "Std"]
    outl = {}
    for c in SENSORS:
        q1, q3 = df[c].quantile([.25, .75])
        iqr = q3 - q1
        outl[c] = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
    desc["IQR_outliers"] = pd.Series(outl)
    desc.round(2).to_csv(TABLE_DIR / "sensor_summary.csv")

    raw = pd.read_csv(Path(MODEL_PATH).parent.parent / "data" / "AQUAIR_1.csv")
    quality = pd.DataFrame([{
        "Raw_rows": len(raw),
        "Missing_values": int(raw.isna().sum().sum()),
        "Duplicate_timestamps": int(raw.duplicated(raw.columns[0]).sum()),
        "Rows_after_cleaning": len(df),
        "Continuous_segments": int(df["segment"].nunique()),
        "Largest_gap_minutes": float(df["Timestamp"].diff().max().total_seconds() / 60),
        "Negative_sensor_values": int((df[SENSORS] < 0).sum().sum()),
        "Start": df["Timestamp"].min(), "End": df["Timestamp"].max(),
    }])
    quality.to_csv(TABLE_DIR / "data_quality.csv", index=False)

    # Regime comparison table
    reg = df.copy()
    reg["Period"] = np.where(reg["Timestamp"] < REGIME_BOUNDARY,
                             "14 Oct - 10 Nov", "11 Nov - 10 Dec")
    reg_tab = reg.groupby("Period")[SENSORS + ["score"]].mean().round(1)
    reg_tab["Share_POOR_%"] = reg.groupby("Period")["score"].apply(
        lambda s: 100 * (s < POOR_BELOW).mean()).round(1)
    reg_tab["Share_GOOD_%"] = reg.groupby("Period")["score"].apply(
        lambda s: 100 * (s >= GOOD_FROM).mean()).round(1)
    reg_tab.to_csv(TABLE_DIR / "regime_comparison.csv")

    # Fig 1: time series with partitions
    fig, axes = plt.subplots(len(SENSORS) + 1, 1, figsize=(12, 13), sharex=True)
    train, val, test = ctx["train"], ctx["val"], ctx["test"]
    t_val = table["Timestamp"].iloc[val[0]]
    t_test = table["Timestamp"].iloc[test[0]]
    for ax, c in zip(axes, ["score"] + SENSORS):
        ax.plot(df["Timestamp"], df[c], lw=0.5, color="#1f4e79")
        ax.set_ylabel(SENSOR_LABELS.get(c, "IAQ score (0-100)"), fontsize=8)
        ax.axvspan(df["Timestamp"].min(), t_val, color="#d0e4f5", alpha=.35, lw=0)
        ax.axvspan(t_val, t_test, color="#fde8b0", alpha=.45, lw=0)
        ax.axvspan(t_test, df["Timestamp"].max(), color="#f6c9c9", alpha=.45, lw=0)
        ax.axvline(REGIME_BOUNDARY, color="k", ls="--", lw=.8)
    axes[0].axhline(POOR_BELOW, color="#d64545", ls=":", lw=1)
    axes[0].axhline(GOOD_FROM, color="#2e9e5b", ls=":", lw=1)
    axes[0].set_title("AQUAIR_1: monitor IAQ score and six sensors (UTC). "
                      "Blue = train, yellow = validation, red = test; "
                      "dashed line = regime change", fontsize=10)
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    save_fig(fig, "fig01_timeseries.png")

    # Fig 2: class distribution per week and per partition
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    wk = table.assign(week=pd.to_datetime(table["Timestamp"]).dt.to_period("W").dt.start_time)
    ct = wk.groupby(["week", "future_class"]).size().unstack(fill_value=0)[CLASSES]
    ct.plot(kind="bar", stacked=True, ax=axes[0],
            color=[CLASS_COLORS[c] for c in CLASSES], width=.85)
    axes[0].set_xticklabels([d.strftime("%d %b") for d in ct.index], rotation=45)
    axes[0].set_xlabel("Week starting"); axes[0].set_ylabel("Number of samples")
    axes[0].set_title("Target class (IAQ in 30 min) per week")
    sp = pd.read_csv(TABLE_DIR / "dataset_split_summary.csv").iloc[:3]
    pct = sp.set_index("Partition")[[f"{c}_pct" for c in CLASSES]]
    pct.columns = CLASSES
    pct.plot(kind="bar", stacked=True, ax=axes[1],
             color=[CLASS_COLORS[c] for c in CLASSES], width=.6)
    axes[1].set_ylabel("Share of samples (%)"); axes[1].set_xlabel("")
    axes[1].set_title("Class distribution per chronological partition")
    axes[1].tick_params(axis="x", rotation=0)
    fig.tight_layout()
    save_fig(fig, "fig02_class_distribution.png")

    # Fig 3: correlation
    fig, ax = plt.subplots(figsize=(7, 5.5))
    corr = df[["score"] + SENSORS].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation between IAQ score and sensors")
    fig.tight_layout()
    save_fig(fig, "fig03_correlation.png")

    # Fig 4: distributions by regime
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, c in zip(axes.ravel(), SENSORS):
        for name, sub in reg.groupby("Period"):
            clip = sub[c].clip(upper=sub[c].quantile(.995))
            sns.histplot(clip, bins=40, stat="density", alpha=.5, label=name,
                         ax=ax, element="step")
        ax.set_title(SENSOR_LABELS[c], fontsize=10); ax.set_xlabel("")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Sensor distributions before and after the regime change "
                 "(top 0.5 % clipped for display)", y=1.0)
    fig.tight_layout()
    save_fig(fig, "fig04_regime_distributions.png")


# ------------------------------------------------------------
# E1. Diagrams
# ------------------------------------------------------------

def _box(ax, x, y, w, h, text, color="#dbe9f6", fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=color, ec="#1f4e79", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            wrap=True)


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#1f4e79", lw=1.4))


def diagrams(ctx):
    banner("E1. Workflow and architecture diagrams")
    b = ctx["bundle"]

    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 3.6); ax.axis("off")
    steps = [
        "Problem\ndefinition\n(30-min IAQ\nclass)", "AQUAIR_1.csv\n16,533 readings\n(5-min)",
        "Cleaning\nsegments\n+30-min target", "Feature\nengineering\n(past data only)",
        "Chronological\nsplit +\nblocked CV", "Model comparison\n+ XGBoost\ntuning",
        "Single test\nevaluation +\nerror analysis", "Streamlit app\n(prediction,\nXAI, advice)",
    ]
    w, gap = 1.35, 0.27
    for i, s in enumerate(steps):
        x = 0.2 + i * (w + gap)
        _box(ax, x, 1.0, w, 1.6, s, color="#dbe9f6" if i < 7 else "#d8f0de")
        if i < len(steps) - 1:
            _arrow(ax, x + w, 1.8, x + w + gap, 1.8)
    ax.set_title("Figure: end-to-end workflow", fontsize=11)
    save_fig(fig, "fig05_workflow.png")

    # Architecture of the SELECTED model + MLP comparison model
    name = b["model_name"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.6))
    ax = axes[0]
    ax.set_xlim(0, 13); ax.set_ylim(0, 3); ax.axis("off")
    nfeat = len(b["feature_columns"])
    params = b.get("best_params") or {}
    model = b["model"]
    est = getattr(model, "named_steps", {}).get("model", model)
    if isinstance(model, RobustXGB):
        n_trees = model.n_estimators
        depth = model.max_depth
        lr = model.learning_rate
        mid = (f"Gradient-boosted trees (XGBoost)\n{n_trees} boosting rounds x 3 classes\n"
               f"max depth {depth}, learning rate {lr}")
    else:
        mid = f"{name}\n(see report Section 5)"
    _box(ax, .2, .8, 2.6, 1.6, f"Input vector\n{nfeat} features\n6 current sensors\n"
         f"+ 6 x 30-min slopes" if b["feature_kind"] == "raw_slope" else
         f"Input vector\n{nfeat} features", fs=9)
    _arrow(ax, 2.8, 1.6, 3.4, 1.6)
    _box(ax, 3.4, .8, 3.6, 1.6, mid, color="#fbeed0", fs=9)
    _arrow(ax, 7.0, 1.6, 7.6, 1.6)
    _box(ax, 7.6, .8, 2.2, 1.6, "Sum of leaf\nscores per class\n(3 logits)", fs=9)
    _arrow(ax, 9.8, 1.6, 10.4, 1.6)
    _box(ax, 10.4, .8, 2.4, 1.6, "Softmax\nP(GOOD), P(MODERATE),\nP(POOR)", color="#d8f0de", fs=9)
    ax.set_title(f"(a) Selected model architecture: {name}", fontsize=11)

    ax = axes[1]
    ax.set_xlim(0, 13); ax.set_ylim(0, 3); ax.axis("off")
    _box(ax, .2, .8, 2.4, 1.6, f"Input\n{nfeat} standardised\nfeatures", fs=9)
    _arrow(ax, 2.6, 1.6, 3.3, 1.6)
    _box(ax, 3.3, .8, 2.6, 1.6, "Dense 64\nReLU", color="#fbeed0", fs=9)
    _arrow(ax, 5.9, 1.6, 6.6, 1.6)
    _box(ax, 6.6, .8, 2.6, 1.6, "Dense 32\nReLU", color="#fbeed0", fs=9)
    _arrow(ax, 9.2, 1.6, 9.9, 1.6)
    _box(ax, 9.9, .8, 2.9, 1.6, "Dense 3 + Softmax\nP(GOOD), P(MODERATE),\nP(POOR)", color="#d8f0de", fs=9)
    ax.set_title("(b) Neural-network comparison model: MLP 64-32 "
                 "(Adam, early stopping)", fontsize=11)
    fig.tight_layout()
    save_fig(fig, "fig06_architecture.png")


# ------------------------------------------------------------
# E2. Algorithm comparison and tuning figures
# ------------------------------------------------------------

def comparison_figures(ctx):
    banner("E2. Comparison and tuning figures")
    comp = pd.read_csv(TABLE_DIR / "model_comparison.csv")
    order = comp.sort_values("CV_Macro_F1")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    colors = ["#9aa5b1" if t == "Baseline" else "#1f6fb2" for t in order["Type"]]
    for ax, col, title in [(axes[0], "CV_Macro_F1", "Blocked CV (development set)"),
                           (axes[1], "Test_Macro_F1", "Held-out test set")]:
        ax.barh(order["Model"], order[col], color=colors)
        ax.set_xlabel("Macro-F1"); ax.set_title(title)
        for y_, v in enumerate(order[col]):
            ax.text(v + .005, y_, f"{v:.3f}", va="center", fontsize=8)
        ax.set_xlim(0, 1.05)
    fig.suptitle("Macro-F1 of baselines (grey) and ML models (blue); selection used CV only")
    fig.tight_layout()
    save_fig(fig, "fig07_algorithm_comparison.png")

    tune = pd.read_csv(TABLE_DIR / "xgboost_tuning.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, p in zip(axes, ["max_depth", "learning_rate", "n_estimators"]):
        sns.stripplot(data=tune, x=p, y="CV_pooled_Macro_F1", hue="balanced",
                      ax=ax, size=8, palette={False: "#1f6fb2", True: "#d64545"})
        grp = tune.groupby(p)["CV_pooled_Macro_F1"].max()
        ax.plot(range(len(grp)), grp.values, "k--", lw=1, label="best per value")
        ax.set_title(f"CV macro-F1 vs {p}")
        ax.set_ylabel("CV pooled macro-F1")
    axes[0].legend(title="class-balanced", fontsize=8)
    for ax in axes[1:]:
        ax.get_legend().remove() if ax.get_legend() else None
    fig.tight_layout()
    save_fig(fig, "fig08_xgb_tuning.png")


# ------------------------------------------------------------
# E3. Final-model evaluation: confusion matrix, PR curves, timeline
# ------------------------------------------------------------

def final_evaluation(ctx):
    banner("E3. Final model on the held-out test set")
    b, table, test = ctx["bundle"], ctx["table"], ctx["test"]
    model = b["model"]
    Xt, yt = ctx["X"].iloc[test], ctx["y"].iloc[test]
    proba = proba_full(model, Xt)
    pred = proba.argmax(axis=1)

    cm = confusion_matrix(yt, pred, labels=[0, 1, 2])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=CLASSES, yticklabels=CLASSES, cbar=False)
    axes[0].set_title("Counts"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", ax=axes[1], vmin=0, vmax=1,
                xticklabels=CLASSES, yticklabels=CLASSES, cbar=False)
    axes[1].set_title("Row-normalised (recall)"); axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    fig.suptitle(f"Confusion matrix on the test set - {b['model_name']}")
    fig.tight_layout()
    save_fig(fig, "fig09_confusion_matrix.png")

    # Precision-recall curves (one-vs-rest)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ap_rows = []
    for i, c in enumerate(CLASSES):
        truth = (yt.to_numpy() == i).astype(int)
        if truth.sum() == 0:
            continue
        p, r, _ = precision_recall_curve(truth, proba[:, i])
        ap = average_precision_score(truth, proba[:, i])
        try:
            auc = roc_auc_score(truth, proba[:, i])
        except ValueError:
            auc = np.nan
        ap_rows.append({"Class": c, "Support": int(truth.sum()),
                        "Average_precision": ap, "ROC_AUC": auc,
                        "Prevalence": truth.mean()})
        ax.plot(r, p, color=CLASS_COLORS[c], lw=2,
                label=f"{c} (AP={ap:.2f}, n={int(truth.sum())})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves (one-vs-rest), test set")
    ax.legend(fontsize=8); ax.set_ylim(0, 1.02)
    fig.tight_layout()
    save_fig(fig, "fig10_pr_curves.png")
    pd.DataFrame(ap_rows).to_csv(TABLE_DIR / "test_pr_auc.csv", index=False)

    # Timeline of the test period
    tt = table.iloc[test]
    fig, axes = plt.subplots(2, 1, figsize=(13, 5.8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.3]})
    axes[0].plot(tt["Timestamp"], tt["score"], color="#1f4e79", lw=.8)
    axes[0].axhline(POOR_BELOW, color="#d64545", ls=":"); axes[0].axhline(GOOD_FROM, color="#2e9e5b", ls=":")
    axes[0].set_ylabel("Monitor IAQ score")
    axes[0].set_title("Test period: monitor score (top); actual vs predicted class "
                      "30 minutes ahead (bottom)")
    for lvl, arr, name in [(1, tt["future_class"].to_numpy(), "Actual"),
                           (0, np.array(CLASSES)[pred], "Predicted")]:
        for c in CLASSES:
            m = arr == c
            axes[1].scatter(tt["Timestamp"][m], np.full(m.sum(), lvl), s=7,
                            color=CLASS_COLORS[c], label=c if lvl == 1 else None)
    wrong = np.array(CLASSES)[pred] != tt["future_class"].to_numpy()
    axes[1].scatter(tt["Timestamp"][wrong], np.full(wrong.sum(), -.55), marker="|",
                    s=60, color="k", label="Error")
    axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(["Predicted", "Actual"])
    axes[1].set_ylim(-.9, 1.5); axes[1].legend(ncol=4, fontsize=8, loc="upper right")
    fig.tight_layout()
    save_fig(fig, "fig13_test_timeline.png")

    return proba, pred


# ------------------------------------------------------------
# E4. Explainability
# ------------------------------------------------------------

def explainability(ctx):
    banner("E4. Explainability: TreeSHAP and permutation importance")
    b, test = ctx["bundle"], ctx["test"]
    model = b["model"]
    Xt, yt = ctx["X"].iloc[test], ctx["y"].iloc[test]

    if hasattr(model, "contributions"):
        imp = global_shap_importance(model, Xt)
        imp["Total"] = imp.sum(axis=1)
        imp = imp.sort_values("Total", ascending=False)
        imp.insert(0, "Description", [friendly_feature_name(i) for i in imp.index])
        imp.round(4).to_csv(TABLE_DIR / "shap_global_importance.csv")
        top = imp.head(12).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        left = np.zeros(len(top))
        for c in CLASSES:
            ax.barh(top["Description"], top[c], left=left, color=CLASS_COLORS[c], label=c)
            left += top[c].to_numpy()
        ax.set_xlabel("Mean |TreeSHAP value| (log-odds units)")
        ax.set_title("Global feature importance on the test set")
        ax.legend(title="Class output")
        fig.tight_layout()
        save_fig(fig, "fig11_shap_global.png")

    scorer = lambda est, X, y: f1_score(y, predict_full(est, X), average="macro",
                                        zero_division=0)
    pi = permutation_importance(model, Xt, yt, scoring=scorer, n_repeats=8,
                                random_state=RANDOM_STATE, n_jobs=1)
    pim = pd.DataFrame({
        "Feature": ctx["cols"],
        "Description": [friendly_feature_name(c) for c in ctx["cols"]],
        "Macro_F1_drop_mean": pi.importances_mean,
        "Macro_F1_drop_std": pi.importances_std,
    }).sort_values("Macro_F1_drop_mean", ascending=False)
    pim.round(4).to_csv(TABLE_DIR / "permutation_importance.csv", index=False)
    top = pim.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(top["Description"], top["Macro_F1_drop_mean"],
            xerr=top["Macro_F1_drop_std"], color="#1f6fb2")
    ax.set_xlabel("Drop in macro-F1 when the feature is shuffled")
    ax.set_title("Permutation importance on the test set (8 repeats)")
    fig.tight_layout()
    save_fig(fig, "fig12_permutation_importance.png")


# ------------------------------------------------------------
# E5. Error analysis: stable vs transition rows
# ------------------------------------------------------------

def error_analysis(ctx, pred):
    banner("E5. Error analysis (stable rows vs class-change rows)")
    table, test, y = ctx["table"], ctx["test"], ctx["y"]
    tt = table.iloc[test].reset_index(drop=True)
    yt = y.iloc[test].to_numpy()
    cur = tt["current_class"].map(CLASS_TO_INT).to_numpy()
    est = ctx["bundle"]["score_estimator"].predict(tt[SENSORS])
    est_cls = np.array([CLASS_TO_INT[c] for c in scores_to_classes(est)])
    trans = cur != yt

    rows = []
    for name, mask in [("All test rows", np.ones(len(yt), bool)),
                       ("Stable (class does not change in 30 min)", ~trans),
                       ("Transition (class changes in 30 min)", trans)]:
        if mask.sum() == 0:
            continue
        rows.append({
            "Subset": name, "Rows": int(mask.sum()),
            "Model_accuracy": float((pred[mask] == yt[mask]).mean()),
            "No_change_observed_accuracy": float((cur[mask] == yt[mask]).mean()),
            "No_change_estimated_accuracy": float((est_cls[mask] == yt[mask]).mean()),
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "transition_analysis.csv", index=False)
    print(out.round(4).to_string(index=False))

    # Where do the errors come from? (error type counts)
    err = pd.crosstab(
        pd.Series(np.array(CLASSES)[yt], name="Actual"),
        pd.Series(np.array(CLASSES)[pred], name="Predicted"))
    err.to_csv(TABLE_DIR / "error_crosstab.csv")

    # Do errors cluster near the class thresholds?
    fs = tt["future_score"].to_numpy()
    near = (np.abs(fs - GOOD_FROM) <= 3) | (np.abs(fs - POOR_BELOW) <= 3)
    wrong = pred != yt
    thr = pd.DataFrame([{
        "Rows_within_3_points_of_a_threshold": int(near.sum()),
        "Share_of_test_rows_%": 100 * near.mean(),
        "Errors_total": int(wrong.sum()),
        "Errors_within_3_points_of_threshold": int((wrong & near).sum()),
        "Share_of_errors_near_threshold_%": 100 * (wrong & near).sum() / max(wrong.sum(), 1),
    }])
    thr.to_csv(TABLE_DIR / "errors_near_threshold.csv", index=False)
    print(thr.round(2).to_string(index=False))


# ------------------------------------------------------------
# E6. Robustness: noise and random seeds
# ------------------------------------------------------------

def _with_seed(model, seed):
    clone_m = clone(model)
    params = clone_m.get_params()
    for key in params:
        if key.endswith("random_state"):
            clone_m.set_params(**{key: seed})
    return clone_m


def robustness(ctx):
    banner("E6. Robustness: random seeds and sensor noise")
    b, table, dev, test = ctx["bundle"], ctx["table"], ctx["dev"], ctx["test"]
    X, y, cols = ctx["X"], ctx["y"], ctx["cols"]
    base = b["model"]

    # (a) Seed stability of the selected model, retrained on the development set
    rows = []
    for seed in [0, 1, 2, 3, 4]:
        m = _with_seed(base, seed).fit(X.iloc[dev], y.iloc[dev])
        mm = metrics(y.iloc[test], predict_full(m, X.iloc[test]))
        rows.append({"Seed": seed, **mm})
    seeds = pd.DataFrame(rows)
    seeds.to_csv(TABLE_DIR / "seed_stability.csv", index=False)
    print(seeds[["Seed", "Accuracy", "Macro_F1"]].round(4).to_string(index=False))

    # (b) Gaussian noise added to the raw sensor series before feature creation
    df = ctx["df"]
    stds = df[SENSORS].std()
    noise_rows = []
    for level in [0.0, 0.02, 0.05, 0.10, 0.20]:
        for rep in range(3 if level > 0 else 1):
            rng = np.random.default_rng(100 + rep)
            noisy = df.copy()
            for s in SENSORS:
                noisy[s] = noisy[s] + rng.normal(0, level * stds[s], len(noisy))
            nt = build_training_table(noisy)
            # same rows (noise does not change the keys), same order
            assert (nt["Timestamp"].values == table["Timestamp"].values).all()
            pred = predict_full(base, nt[cols].iloc[test])
            mm = metrics(nt["y"].iloc[test], pred)
            noise_rows.append({"Noise_level_pct_of_std": 100 * level, "Repeat": rep, **mm})
    noise = pd.DataFrame(noise_rows)
    noise_s = noise.groupby("Noise_level_pct_of_std")[["Accuracy", "Macro_F1"]].agg(["mean", "std"])
    noise_s.columns = ["Accuracy_mean", "Accuracy_std", "Macro_F1_mean", "Macro_F1_std"]
    noise_s = noise_s.reset_index().fillna(0.0)
    noise_s.to_csv(TABLE_DIR / "noise_robustness.csv", index=False)
    print(noise_s.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.errorbar(noise_s["Noise_level_pct_of_std"], noise_s["Accuracy_mean"],
                yerr=noise_s["Accuracy_std"], marker="o", label="Accuracy")
    ax.errorbar(noise_s["Noise_level_pct_of_std"], noise_s["Macro_F1_mean"],
                yerr=noise_s["Macro_F1_std"], marker="s", label="Macro-F1")
    ax.set_xlabel("Gaussian noise (% of each sensor's standard deviation)")
    ax.set_ylabel("Test-set score"); ax.set_ylim(0, 1.02); ax.legend()
    ax.set_title("Sensitivity to sensor noise")
    fig.tight_layout()
    save_fig(fig, "fig15_noise_robustness.png")

    # (c) Efficiency
    buf = io.BytesIO()
    joblib.dump(base, buf)
    one = X.iloc[[test[0]]]
    t0 = time.perf_counter()
    for _ in range(200):
        proba_full(base, one)
    ms = (time.perf_counter() - t0) / 200 * 1000
    pd.DataFrame([{
        "Model": b["model_name"],
        "Serialized_size_KB": buf.tell() / 1024,
        "Single_row_inference_ms": ms,
    }]).to_csv(TABLE_DIR / "efficiency.csv", index=False)
    print(f"  model size {buf.tell()/1024:.0f} KB, inference {ms:.2f} ms/row")


# ------------------------------------------------------------
# E7. Concept drift: train on the first regime, test on the second
# ------------------------------------------------------------

def drift(ctx):
    banner("E7. Concept drift (cross-regime test)")
    table, X, y = ctx["table"], ctx["X"], ctx["y"]
    ts = pd.to_datetime(table["Timestamp"])
    tt = pd.to_datetime(table["target_time"])
    first = np.flatnonzero((tt < REGIME_BOUNDARY).to_numpy())
    second = np.flatnonzero((ts >= REGIME_BOUNDARY).to_numpy())
    # within-regime reference: second regime split chronologically 70/30
    cut = int(len(second) * 0.7)
    sec_train = second[:cut]
    sec_test = second[cut:]
    sec_train = sec_train[(tt.iloc[sec_train] < ts.iloc[sec_test[0]]).to_numpy()]

    rows = []
    base = clone(ctx["bundle"]["model"])
    for name, tr, te in [
        ("Train: 14 Oct-10 Nov  ->  Test: 11 Nov-10 Dec (new regime)", first, second),
        ("Train: first 70% of 11 Nov-10 Dec  ->  Test: last 30% (same regime)",
         sec_train, sec_test),
    ]:
        m = clone(base).fit(X.iloc[tr], y.iloc[tr])
        pred = predict_full(m, X.iloc[te])
        mm = metrics(y.iloc[te], pred)
        cur = table["current_class"].map(CLASS_TO_INT).to_numpy()[te]
        rows.append({"Experiment": name, "Train_rows": len(tr), "Test_rows": len(te),
                     "Accuracy": mm["Accuracy"], "Macro_F1": mm["Macro_F1"],
                     "No_change_observed_accuracy": float((cur == y.iloc[te].to_numpy()).mean())})
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "concept_drift.csv", index=False)
    print(out.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(len(out))
    ax.bar(xs - .2, out["Accuracy"], .4, label="Model accuracy", color="#1f6fb2")
    ax.bar(xs + .2, out["No_change_observed_accuracy"], .4,
           label="No-change (monitor score)", color="#9aa5b1")
    ax.set_xticks(xs); ax.set_xticklabels(["Cross-regime", "Same regime"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy"); ax.legend()
    for x_, v in zip(xs - .2, out["Accuracy"]):
        ax.text(x_, v + .01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_title("Effect of the regime change on the model")
    fig.tight_layout()
    save_fig(fig, "fig16_concept_drift.png")


# ------------------------------------------------------------
# E8. Ablations: device score, regression alternative
# ------------------------------------------------------------

def ablations(ctx):
    banner("E8. Ablation: monitor score as input; regress-then-threshold")
    df, table, dev, test = ctx["df"], ctx["table"], ctx["dev"], ctx["test"]
    y, cols = ctx["y"], ctx["cols"]

    parts = []
    for _, seg in df.groupby("segment"):
        s = seg.reset_index(drop=True)["score"]
        f = pd.DataFrame({"score_now": s.values})
        for k in (1, 3, 6, 12):
            f[f"score_lag{k}"] = s.shift(k).values
        f["Timestamp"] = seg["Timestamp"].values
        parts.append(f)
    t2 = table.merge(pd.concat(parts), on="Timestamp", how="left")
    score_cols = ["score_now"] + [f"score_lag{k}" for k in (1, 3, 6, 12)]

    yt = y.iloc[test].to_numpy()
    cur = table["current_class"].map(CLASS_TO_INT).to_numpy()[test]
    trans = cur != yt
    rows = []

    def add(name, pred, note):
        mm = metrics(yt, pred)
        rows.append({
            "Variant": name, "Note": note, "Accuracy": mm["Accuracy"],
            "Macro_F1": mm["Macro_F1"],
            "Accuracy_on_transitions": float((pred[trans] == yt[trans]).mean()) if trans.any() else np.nan,
            "Accuracy_on_stable": float((pred[~trans] == yt[~trans]).mean()),
        })

    add("No-change (monitor score)", cur, "Baseline, uses the monitor's current class")
    sens = RobustXGB().fit(t2[cols].iloc[dev], y.iloc[dev])
    add("Sensors only (XGBoost, notebook settings)", predict_full(sens, t2[cols].iloc[test]),
        "Deployed input type: six sensors")
    for name, c, note in [
        ("Monitor score + score lags only", score_cols,
         "Needs the monitor's own score as input"),
        ("Sensors + monitor score + score lags", cols + score_cols,
         "Needs the monitor's own score as input"),
    ]:
        m = RobustXGB().fit(t2[c].iloc[dev], y.iloc[dev])
        add(name, predict_full(m, t2[c].iloc[test]), note)

    # Regress the future score, then apply the 60/80 bands
    reg = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=.05,
                       subsample=.8, colsample_bytree=.8, n_jobs=1,
                       random_state=RANDOM_STATE)
    reg.fit(t2[cols].iloc[dev], t2["future_score"].iloc[dev])
    ps = reg.predict(t2[cols].iloc[test])
    pc = np.array([CLASS_TO_INT[c] for c in scores_to_classes(ps)])
    add("Sensors -> regress future score -> apply bands", pc,
        f"Score MAE {np.mean(np.abs(ps - t2['future_score'].iloc[test])):.2f} points; "
        f"no-change MAE {np.mean(np.abs(t2['score'].iloc[test] - t2['future_score'].iloc[test])):.2f}")
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "ablation_score_and_regression.csv", index=False)
    print(out.round(4).drop(columns="Note").to_string(index=False))


# ------------------------------------------------------------
# E9. Replication of the notebook (CO2 > 1000 ppm in 30 minutes)
# ------------------------------------------------------------

def notebook_replication(ctx):
    banner("E9. Notebook replication: binary 'CO2 > 1000 ppm in 30 min'")
    df, table = ctx["df"], ctx["table"]
    by_time = df.set_index("Timestamp")["CO2"]
    fut = by_time.reindex(pd.DatetimeIndex(table["target_time"])).to_numpy()
    d = table.copy()
    d["future_co2"] = fut
    d = d.dropna(subset=["future_co2"]).reset_index(drop=True)
    d["yb"] = (d["future_co2"] > 1000).astype(int)
    feats = feature_names("lags_hour")

    def fit_eval(tr, te, label, split_name):
        m = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                          subsample=0.8, colsample_bytree=0.8, random_state=42,
                          n_jobs=1, eval_metric="logloss")
        m.fit(d[feats].iloc[tr], d["yb"].iloc[tr])
        p = m.predict_proba(d[feats].iloc[te])[:, 1]
        pred = (p > .5).astype(int)
        yt = d["yb"].iloc[te].to_numpy()
        nochange = (d["CO2"].iloc[te].to_numpy() > 1000).astype(int)
        maj = max(yt.mean(), 1 - yt.mean())
        return {
            "Split": split_name, "Train_rows": len(tr), "Test_rows": len(te),
            "XGBoost_accuracy": float((pred == yt).mean()),
            "XGBoost_ROC_AUC": float(roc_auc_score(yt, p)) if 0 < yt.mean() < 1 else np.nan,
            "XGBoost_macro_F1": float(f1_score(yt, pred, average="macro")),
            "No_change_accuracy": float((nochange == yt).mean()),
            "Test_majority_accuracy": float(maj),
            "Test_positive_share": float(yt.mean()),
        }

    rows = []
    # (a) Notebook protocol: random 25 % of calendar days as test
    d["day"] = pd.to_datetime(d["Timestamp"]).dt.date.astype(str)
    days = np.array(sorted(d["day"].unique()))
    test_days = set(np.random.RandomState(42).choice(days, size=max(2, int(len(days) * .25)),
                                                     replace=False))
    te = np.flatnonzero(d["day"].isin(test_days))
    tr = np.flatnonzero(~d["day"].isin(test_days))
    rows.append(fit_eval(tr, te, "", "Notebook protocol (random 25% of days)"))

    # (b) Chronological: same cut as the main project (last 15 % = test)
    n = len(d)
    cut = int(n * .85)
    ts = pd.to_datetime(d["Timestamp"])
    tt = pd.to_datetime(d["target_time"])
    tr = np.flatnonzero(((np.arange(n) < cut) & (tt < ts.iloc[cut])).to_numpy())
    te = np.arange(cut, n)
    rows.append(fit_eval(tr, te, "", "Chronological (last 15% = test)"))

    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "notebook_replication.csv", index=False)
    print(out.round(4).to_string(index=False))


# ------------------------------------------------------------
# E10. Training behaviour (XGBoost loss curves, MLP loss curve)
# ------------------------------------------------------------

def training_curves(ctx):
    banner("E10. Training behaviour")
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    table, X, y, train, val = ctx["table"], ctx["X"], ctx["y"], ctx["train"], ctx["val"]
    tune = pd.read_csv(TABLE_DIR / "xgboost_tuning.csv")
    top = tune.iloc[0]
    params = dict(n_estimators=int(top["n_estimators"]),
                  learning_rate=float(top["learning_rate"]),
                  max_depth=int(top["max_depth"]), subsample=float(top["subsample"]),
                  colsample_bytree=float(top["colsample_bytree"]),
                  min_child_weight=int(top["min_child_weight"]),
                  reg_lambda=float(top["reg_lambda"]))
    clf = XGBClassifier(**params, tree_method="hist", eval_metric="mlogloss",
                        random_state=RANDOM_STATE, n_jobs=1)
    clf.fit(X.iloc[train], y.iloc[train],
            eval_set=[(X.iloc[train], y.iloc[train]), (X.iloc[val], y.iloc[val])],
            verbose=False)
    ev = clf.evals_result()
    tr_loss = ev["validation_0"]["mlogloss"]
    va_loss = ev["validation_1"]["mlogloss"]

    mlp = Pipeline([("scaler", StandardScaler()),
                    ("model", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                            batch_size=256, learning_rate_init=1e-3,
                                            max_iter=150, early_stopping=True,
                                            validation_fraction=.12, n_iter_no_change=12,
                                            random_state=RANDOM_STATE))])
    mlp.fit(X.iloc[train], y.iloc[train])
    net = mlp.named_steps["model"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(tr_loss, label="Training (first 70%)")
    axes[0].plot(va_loss, label="Validation (next 15%)")
    axes[0].set_xlabel("Boosting round"); axes[0].set_ylabel("Multi-class log-loss")
    axes[0].set_title("XGBoost loss (tuned settings)"); axes[0].legend()
    axes[1].plot(net.loss_curve_, label="Training loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Cross-entropy loss")
    axes[1].set_title("MLP 64-32: training loss"); axes[1].legend()
    axes[2].plot(net.validation_scores_, color="#d64545", label="Internal validation accuracy")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Accuracy")
    axes[2].set_title("MLP 64-32: validation accuracy (random 12% of training rows)")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, "fig14_training_curves.png")

    best_round = int(np.argmin(va_loss))
    pd.DataFrame([{
        "XGB_best_validation_round": best_round + 1,
        "XGB_min_validation_logloss": float(min(va_loss)),
        "XGB_final_validation_logloss": float(va_loss[-1]),
        "XGB_final_training_logloss": float(tr_loss[-1]),
        "MLP_epochs_run": int(net.n_iter_),
        "MLP_final_training_loss": float(net.loss_curve_[-1]),
        "MLP_best_internal_val_accuracy": float(max(net.validation_scores_)),
    }]).to_csv(TABLE_DIR / "training_behaviour.csv", index=False)


# ------------------------------------------------------------

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    ctx = load_context()
    print(f"Selected model: {ctx['bundle']['model_name']} | "
          f"features: {ctx['bundle']['feature_kind']}")
    eda(ctx)
    diagrams(ctx)
    comparison_figures(ctx)
    _, pred = final_evaluation(ctx)
    explainability(ctx)
    error_analysis(ctx, pred)
    robustness(ctx)
    drift(ctx)
    ablations(ctx)
    notebook_replication(ctx)
    training_curves(ctx)
    print("\nAll experiments finished.")


if __name__ == "__main__":
    main()