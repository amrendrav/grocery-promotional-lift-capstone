"""Modeling utilities shared across the M24 notebooks.

Centralizes the design-matrix builder, the canonical feature lists, the
time-series CV splitter, and the metrics helpers so every notebook
(03 baseline, 04 regularized, 05 SVR, 06 comparison, 07 scenario tool)
evaluates models the same way on the same 12-week holdout.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


NUMERIC_FEATURES: list[str] = [
    "month", "week_of_year", "quarter", "is_q4",
    "promo_share", "mean_discount_depth", "sales_weighted_discount_depth",
    "promo_share_lag1", "discount_depth_lag1",
    "has_event_days", "snap_days_in_week", "n_items_priced",
    "log_unit_sales_lag1", "log_unit_sales_lag2", "log_unit_sales_lag4",
    "log_unit_sales_roll4", "log_unit_sales_roll12",
]

CATEGORICAL_FEATURES: list[str] = ["dept_id", "store_id", "state_id"]

TARGET_RAW = "unit_sales"
TARGET_LOG = "log_unit_sales"

LAG_COLS = [
    "unit_sales_lag1", "unit_sales_lag2", "unit_sales_lag4",
    "unit_sales_roll4", "unit_sales_roll12",
]


def add_log_columns(df: pd.DataFrame) -> pd.DataFrame:
    """log1p-transform the target and lagged-sales features."""
    df = df.copy()
    for c in LAG_COLS:
        df[f"log_{c}"] = np.log1p(df[c])
    df[TARGET_LOG] = np.log1p(df[TARGET_RAW])
    return df


def build_design_matrix(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardize numerics, one-hot encode categoricals, concat.

    Pass `fit_scaler=True` for the training set, then reuse the returned
    scaler for the holdout to avoid leakage.
    """
    num = df[NUMERIC_FEATURES].astype("float64").values
    if fit_scaler:
        scaler = StandardScaler().fit(num)
    if scaler is None:
        raise ValueError("Must pass a fitted scaler or set fit_scaler=True")
    num_scaled = scaler.transform(num)
    num_df = pd.DataFrame(num_scaled, columns=NUMERIC_FEATURES, index=df.index)

    cat = pd.get_dummies(df[CATEGORICAL_FEATURES], drop_first=True).astype("float64")
    X = pd.concat([num_df, cat], axis=1)
    return X, scaler


def align_columns(X_test: pd.DataFrame, X_train: pd.DataFrame) -> pd.DataFrame:
    """Make sure test has the same columns (and order) as train."""
    return X_test.reindex(columns=X_train.columns, fill_value=0.0)


def time_series_cv(n_splits: int = 5) -> TimeSeriesSplit:
    """Expanding-window CV. Use with row-sorted-by-date data."""
    return TimeSeriesSplit(n_splits=n_splits)


def compute_metrics(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    y_true_raw: np.ndarray,
) -> dict[str, float]:
    """RMSE / MAE / R² in both log space and raw-units space.

    Raw-units metrics are computed after expm1 back-transform — they're the
    business-meaningful numbers (errors in units sold).
    """
    y_pred_raw = np.expm1(y_pred_log)

    return {
        "rmse_log": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "mae_log": float(mean_absolute_error(y_true_log, y_pred_log)),
        "r2_log": float(r2_score(y_true_log, y_pred_log)),
        "rmse_raw": float(np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))),
        "mae_raw": float(mean_absolute_error(y_true_raw, y_pred_raw)),
        "r2_raw": float(r2_score(y_true_raw, y_pred_raw)),
    }


def metrics_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Stack a {model_name: metrics_dict} mapping into a DataFrame."""
    return pd.DataFrame(results).T[
        ["rmse_log", "mae_log", "r2_log", "rmse_raw", "mae_raw", "r2_raw"]
    ]


REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
RESULTS_PATH = REPORTS_DIR / "model_results.json"
PARAMS_PATH = REPORTS_DIR / "best_params.json"
PREDICTIONS_PATH = REPORTS_DIR / "holdout_predictions.parquet"


def _merge_json(path: Path, payload: dict) -> None:
    """Read, merge, and write JSON — keeps prior keys when each notebook saves."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = json.loads(path.read_text()) if path.exists() else {}
    existing.update(payload)
    path.write_text(json.dumps(existing, indent=2, default=str))


def save_results(results: dict[str, dict[str, float]]) -> None:
    """Persist holdout metrics so notebook 06 can compare across notebooks."""
    _merge_json(RESULTS_PATH, results)


def load_results() -> dict[str, dict[str, float]]:
    if not RESULTS_PATH.exists():
        return {}
    return json.loads(RESULTS_PATH.read_text())


def save_best_params(params: dict[str, dict]) -> None:
    """Persist best hyperparameters per model so notebook 07 can refit."""
    _merge_json(PARAMS_PATH, params)


def load_best_params() -> dict[str, dict]:
    if not PARAMS_PATH.exists():
        return {}
    return json.loads(PARAMS_PATH.read_text())


def save_predictions(
    test_df: pd.DataFrame,
    preds: dict[str, np.ndarray],
) -> None:
    """Persist log-space holdout predictions keyed by (week_start, dept_id, store_id).

    `preds` is {model_name: y_pred_log_array}. Merges with any existing file
    so notebooks can write predictions independently.
    """
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    key_cols = ["week_start", "dept_id", "store_id"]
    new_df = test_df[key_cols].reset_index(drop=True).copy()
    for name, arr in preds.items():
        new_df[f"pred_log_{name}"] = arr

    if PREDICTIONS_PATH.exists():
        existing = pd.read_parquet(PREDICTIONS_PATH)
        merged = existing.merge(new_df, on=key_cols, how="outer")
        # if a model is being re-saved, prefer the new version
        for col in new_df.columns:
            if col in key_cols:
                continue
            if f"{col}_x" in merged.columns:
                merged[col] = merged[f"{col}_y"].combine_first(merged[f"{col}_x"])
                merged = merged.drop(columns=[f"{col}_x", f"{col}_y"])
        new_df = merged

    new_df.to_parquet(PREDICTIONS_PATH, index=False)


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"No predictions file at {PREDICTIONS_PATH}. "
            "Run notebooks 04 and 05 first to generate it."
        )
    return pd.read_parquet(PREDICTIONS_PATH)
