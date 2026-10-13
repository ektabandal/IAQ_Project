# ============================================================
# modeling.py
# Model definitions, metrics and time-series cross-validation.
# Imported by train_models.py, experiments.py and app.py.
# ============================================================

import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from common import CLASSES, RANDOM_STATE

N_CLASSES = len(CLASSES)


# ------------------------------------------------------------
# 1. XGBoost wrapper that tolerates training folds with missing classes
# ------------------------------------------------------------

class RobustXGB(BaseEstimator, ClassifierMixin):
    """
    XGBClassifier wrapper for the 3 IAQ classes (0=GOOD, 1=MODERATE, 2=POOR).

    * Works even if a training fold does not contain every class
      (the early weeks of AQUAIR contain POOR only).
    * predict_proba always returns 3 columns in CLASSES order.
    * contributions() returns TreeSHAP values (XGBoost pred_contribs).
    """

    def __init__(
        self,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        reg_lambda=1.0,
        balanced=False,
        random_state=RANDOM_STATE,
        n_jobs=1,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.balanced = balanced
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        from xgboost import XGBClassifier

        y = np.asarray(y).astype(int)
        self.classes_ = np.arange(N_CLASSES)
        self.present_ = np.unique(y)
        self.feature_names_ = list(X.columns) if hasattr(X, "columns") else None

        if len(self.present_) == 1:
            self.model_ = None
            return self

        remap = {c: i for i, c in enumerate(self.present_)}
        y_small = np.array([remap[v] for v in y])
        weights = compute_sample_weight("balanced", y_small) if self.balanced else None

        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X, y_small, sample_weight=weights)
        return self

    def predict_proba(self, X):
        n = len(X)
        full = np.zeros((n, N_CLASSES))
        if self.model_ is None:
            full[:, self.present_[0]] = 1.0
            return full
        p = self.model_.predict_proba(X)
        full[:, self.present_] = p
        return full

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def contributions(self, X):
        """TreeSHAP values, shape (n_rows, 3 classes, n_features + 1 bias)."""
        import xgboost as xgb

        n, f = X.shape
        full = np.zeros((n, N_CLASSES, f + 1))
        if self.model_ is None:
            return full
        dm = xgb.DMatrix(X)
        c = self.model_.get_booster().predict(dm, pred_contribs=True)
        if c.ndim == 2:                    # binary case (not expected here)
            c = c[:, None, :]
        full[:, self.present_, :] = c
        return full


# ------------------------------------------------------------
# 2. Candidate algorithms
# ------------------------------------------------------------

def _scaled(estimator):
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def build_candidates():
    """The ML algorithms compared on the same data split and features."""
    return {
        "Logistic Regression": _scaled(LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=4, class_weight="balanced",
            random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=16, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=1,
            random_state=RANDOM_STATE),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=150, max_depth=16, min_samples_leaf=2,
            class_weight="balanced", n_jobs=1, random_state=RANDOM_STATE),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=RANDOM_STATE),
        "XGBoost (notebook settings)": RobustXGB(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8),
        "MLP Neural Network (64-32)": _scaled(MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            alpha=1e-3, batch_size=256, learning_rate_init=1e-3,
            max_iter=150, early_stopping=True, validation_fraction=0.12,
            n_iter_no_change=12, random_state=RANDOM_STATE)),
    }


# ------------------------------------------------------------
# 3. Probabilities in fixed CLASSES order for any estimator
# ------------------------------------------------------------

def proba_full(model, X):
    """predict_proba as an (n, 3) array in CLASSES order for any estimator."""
    p = model.predict_proba(X)
    classes = np.asarray(model.classes_).astype(int)
    full = np.zeros((len(X), N_CLASSES))
    full[:, classes] = p
    return full


def predict_full(model, X):
    return proba_full(model, X).argmax(axis=1)


# ------------------------------------------------------------
# 4. Metrics
# ------------------------------------------------------------

def metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Macro_Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro_Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro_F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted_F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


# ------------------------------------------------------------
# 5. Blocked (expanding-window) time-series cross-validation
# ------------------------------------------------------------
# AQUAIR has two regimes: until about 10 Nov 2024 the monitor reported
# almost only POOR air, afterwards mostly GOOD air. Folds that start before
# the regime change would test on conditions never seen in training, so the
# model-selection folds start at 70 % of the development data (inside the
# second regime). The cross-regime failure is reported separately as a
# concept-drift experiment (experiments.py).

CV_CUTS = (0.70, 0.78, 0.86, 0.93, 1.00)


def blocked_folds(table, pool_idx, cuts=CV_CUTS):
    """
    Expanding-window folds on the development rows (train + validation).

    Fold k trains on rows [0, cut_k) and tests on rows [cut_k, cut_k+1).
    Training rows whose 30-minute target reaches the test block are
    purged. Returns a list of (train_index, test_index).
    """
    pool_idx = np.asarray(pool_idx)
    n = len(pool_idx)
    times = pd.to_datetime(table["Timestamp"]).to_numpy()
    target_times = pd.to_datetime(table["target_time"]).to_numpy()
    folds = []
    for i in range(len(cuts) - 1):
        a = int(n * cuts[i])
        b = int(n * cuts[i + 1])
        test_idx = pool_idx[a:b]
        train_idx = pool_idx[:a]
        start = times[test_idx[0]]
        train_idx = train_idx[target_times[train_idx] < start]
        folds.append((train_idx, test_idx))
    return folds


def cross_validate(estimator, X, y, folds):
    """
    Fit a fresh clone on each fold. Returns
      fold_df : per-fold metrics and timings
      pooled  : metrics on the concatenated out-of-fold predictions
      oof     : (index, y_true, y_pred) arrays
    """
    rows, idx_all, pred_all = [], [], []
    for k, (tr, te) in enumerate(folds, start=1):
        model = clone(estimator)
        t0 = time.perf_counter()
        model.fit(X.iloc[tr], y.iloc[tr])
        fit_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = predict_full(model, X.iloc[te])
        pred_time = time.perf_counter() - t0
        m = metrics(y.iloc[te], pred)
        m.update({
            "Fold": k,
            "Fit_s": fit_time,
            "Predict_ms_per_1000_rows": pred_time * 1e6 / max(len(te), 1),
        })
        rows.append(m)
        idx_all.append(np.asarray(te))
        pred_all.append(pred)

    idx_all = np.concatenate(idx_all)
    pred_all = np.concatenate(pred_all)
    pooled = metrics(y.iloc[idx_all], pred_all)
    return pd.DataFrame(rows), pooled, (idx_all, y.iloc[idx_all].to_numpy(), pred_all)