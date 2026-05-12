"""Feature engineering for the M5 weekly (dept_id, store_id) panel.

Inputs: the weekly panel produced by src/data.build_weekly_panel().
Outputs: feature matrix with calendar features, lagged sales, and event flags
suitable for a multiple linear regression baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame, date_col: str = "week_start") -> pd.DataFrame:
    """Year, month, week-of-year, quarter, and Q4-holiday-window flag."""
    df = df.copy()
    d = df[date_col]
    df["year"] = d.dt.year
    df["month"] = d.dt.month.astype("int8")
    df["week_of_year"] = d.dt.isocalendar().week.astype("int16")
    df["quarter"] = d.dt.quarter.astype("int8")
    df["is_q4"] = (df["quarter"] == 4).astype("int8")
    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "unit_sales",
    group_cols: tuple[str, ...] = ("store_id", "dept_id"),
    lags: tuple[int, ...] = (1, 2, 4),
    rolling_windows: tuple[int, ...] = (4, 12),
) -> pd.DataFrame:
    """Per-(store, dept) lagged and rolling-mean sales features.

    Lags 1/2/4 weeks capture short-term momentum. The 4-week and 12-week
    rolling means capture quarterly seasonality and longer trend.
    All rolling stats use shift(1) so the current week's sales don't leak in.
    """
    df = df.sort_values(["store_id", "dept_id", "week_start"]).copy()
    grp = df.groupby(list(group_cols), observed=True)[target_col]

    for lag in lags:
        df[f"{target_col}_lag{lag}"] = grp.shift(lag).astype("float32")

    for w in rolling_windows:
        # transform applies the function per group and returns a Series aligned with df's index.
        # shift(1) prevents the current week's value from leaking into its own rolling mean.
        df[f"{target_col}_roll{w}"] = (
            df.groupby(list(group_cols), observed=True)[target_col]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=max(2, w // 2)).mean())
            .astype("float32")
        )
    return df


def add_promo_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag of promo_share / discount_depth — captures pull-forward / pantry-loading effects.

    A high promo last week may *depress* sales this week if customers stocked up.
    """
    df = df.sort_values(["store_id", "dept_id", "week_start"]).copy()
    grp = df.groupby(["store_id", "dept_id"], observed=True)
    df["promo_share_lag1"] = grp["promo_share"].shift(1).astype("float32")
    df["discount_depth_lag1"] = grp["sales_weighted_discount_depth"].shift(1).astype("float32")
    return df


def build_feature_matrix(weekly: pd.DataFrame) -> pd.DataFrame:
    """End-to-end: calendar + sales lags + promo lags. Drops rows with NaN from lag creation."""
    df = add_calendar_features(weekly)
    df = add_lag_features(df)
    df = add_promo_lag_features(df)
    return df.dropna().reset_index(drop=True)


def temporal_train_test_split(
    df: pd.DataFrame,
    holdout_weeks: int = 12,
    date_col: str = "week_start",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the last N weeks for testing — the only valid split for time-series."""
    last_date = df[date_col].max()
    cutoff = last_date - pd.Timedelta(weeks=holdout_weeks)
    train = df[df[date_col] <= cutoff].copy()
    test = df[df[date_col] > cutoff].copy()
    return train, test
