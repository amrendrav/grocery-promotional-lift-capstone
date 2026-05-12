"""Data loading and panel construction for M5 Walmart sales.

Pipeline (memory-bounded by per-store chunking):
  1. Load calendar.csv (dates, events, SNAP)
  2. Load wide sales file once; filter to "full-history" items in wide form
  3. Build per-(item, store, week) price features from sell_prices.csv
  4. For each of the 10 stores:
       a. Slice the wide sales rows for that store
       b. Melt to long (~5.9M rows per store — manageable)
       c. Aggregate to (item_id, week_start)
       d. Merge with the store's price features
       e. Aggregate to (dept_id, store_id, week_start)
  5. Concat the 10 store panels
  6. Add calendar aggregates (event days, SNAP days keyed to state)
  7. Cache as parquet

Why per-store chunking: a single global melt of 30,490 × 1,941 days produces
~59M rows. With six string columns (item_id, dept_id, cat_id, store_id, state_id, d),
each ~50-60 bytes in pandas object dtype, peak memory exceeds 20 GB. Per-store
chunking keeps peak under ~1 GB.

Known limitations are documented in the project README under "Caveats and
Mitigation Plan".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
M5_DIR = REPO_ROOT / "data" / "raw" / "m5"
INTERIM_DIR = REPO_ROOT / "data" / "interim"

DISCOUNT_THRESHOLD = 0.05            # >5% off trailing-median = "promotion"
ROLLING_PRICE_WINDOW_WEEKS = 4       # window for the "regular price" baseline
FULL_HISTORY_CUTOFF = "2013-01-01"   # items must have non-zero sales before this date


# ---------- Calendar ----------

def load_calendar() -> pd.DataFrame:
    """Load calendar.csv with parsed dates and tidy event flags."""
    cal = pd.read_csv(
        M5_DIR / "calendar.csv",
        parse_dates=["date"],
        dtype={
            "wm_yr_wk": "int32",
            "wday": "int8",
            "month": "int8",
            "year": "int16",
            "snap_CA": "int8",
            "snap_TX": "int8",
            "snap_WI": "int8",
        },
    )
    cal["week_start"] = cal["date"].dt.to_period("W-MON").dt.start_time
    cal["has_event"] = (
        cal["event_name_1"].notna() | cal["event_name_2"].notna()
    ).astype("int8")
    return cal


# ---------- Wide sales + full-history filter ----------

def _load_sales_wide_with_filter(calendar: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load the wide sales file and filter rows to full-history items.

    Filter rule: an item must have at least one non-zero sale before
    FULL_HISTORY_CUTOFF, evaluated globally (across all stores).
    """
    sales_path = M5_DIR / "sales_train_evaluation.csv"
    if not sales_path.exists():
        raise FileNotFoundError(
            f"{sales_path} not found. See data/raw/README.md for download steps."
        )

    wide = pd.read_csv(sales_path)
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    d_cols = [c for c in wide.columns if c.startswith("d_")]

    # Map d_xxx -> date via calendar; find which d_cols are before the cutoff
    cutoff = pd.Timestamp(FULL_HISTORY_CUTOFF)
    d_to_date = calendar.set_index("d")["date"].to_dict()
    early_d_cols = [c for c in d_cols if d_to_date.get(c, pd.NaT) < cutoff]

    # Item is "full history" if ANY store-row for that item had a non-zero sale
    # before cutoff. Group by item_id, max over store-rows of any-early-sale.
    early_any = (wide[early_d_cols] > 0).any(axis=1)
    item_has_early = wide.assign(early=early_any).groupby("item_id")["early"].any()
    keep_items = set(item_has_early[item_has_early].index)
    wide = wide[wide["item_id"].isin(keep_items)].reset_index(drop=True)

    # Make the id columns Categorical to save memory in downstream melts
    for c in id_cols:
        wide[c] = wide[c].astype("category")

    return wide, id_cols, d_cols


# ---------- Price features ----------

def build_item_price_features(calendar: pd.DataFrame) -> pd.DataFrame:
    """For each (item_id, store_id, week_start), compute discount depth and is_promotion.

    Pseudo-discount-depth rule (documented in README):
      regular_price  = trailing 4-week MEDIAN of sell_price per (item, store)
      discount_depth = max(0, (regular_price - sell_price) / regular_price)
      is_promotion   = discount_depth > 0.05
    """
    prices = pd.read_csv(
        M5_DIR / "sell_prices.csv",
        dtype={"wm_yr_wk": "int32", "sell_price": "float32"},
    )

    wk_map = (
        calendar[["wm_yr_wk", "week_start"]]
        .drop_duplicates()
        .sort_values("wm_yr_wk")
    )
    prices = prices.merge(wk_map, on="wm_yr_wk", how="left")
    prices = prices.sort_values(["store_id", "item_id", "week_start"]).reset_index(drop=True)

    grp = prices.groupby(["store_id", "item_id"], observed=True)["sell_price"]
    prices["regular_price"] = grp.transform(
        lambda s: s.shift(1).rolling(ROLLING_PRICE_WINDOW_WEEKS, min_periods=1).median()
    )
    prices["regular_price"] = prices["regular_price"].fillna(prices["sell_price"])

    prices["discount_depth"] = (
        (prices["regular_price"] - prices["sell_price"]) / prices["regular_price"]
    ).clip(lower=0).astype("float32")
    prices["is_promotion"] = (prices["discount_depth"] > DISCOUNT_THRESHOLD).astype("int8")

    for c in ("store_id", "item_id"):
        prices[c] = prices[c].astype("category")

    return prices[["store_id", "item_id", "week_start", "sell_price",
                   "regular_price", "discount_depth", "is_promotion"]]


# ---------- Per-store aggregation ----------

def _process_store(
    store_id: str,
    wide_rows: pd.DataFrame,
    d_cols: list[str],
    calendar: pd.DataFrame,
    item_prices_store: pd.DataFrame,
) -> pd.DataFrame:
    """Melt one store's sales, join prices, aggregate to dept-week.

    Bounded memory: ~5.9M rows after melt → ~1.4M item-weeks after first agg.
    """
    # Melt this store's rows only
    id_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    long_df = wide_rows.melt(
        id_vars=id_cols,
        value_vars=d_cols,
        var_name="d",
        value_name="sales",
    )
    long_df["sales"] = long_df["sales"].astype("float32")

    # Join calendar for week_start
    long_df = long_df.merge(
        calendar[["d", "week_start"]],
        on="d",
        how="left",
    )

    # Aggregate to (item, week)
    item_week = (
        long_df.groupby(
            ["item_id", "dept_id", "cat_id", "store_id", "state_id", "week_start"],
            observed=True,
        )["sales"].sum().reset_index()
    )
    del long_df  # free memory

    # Merge with this store's price features
    merged = item_week.merge(
        item_prices_store,
        on=["store_id", "item_id", "week_start"],
        how="left",
    )
    del item_week

    # Drop weeks where the item wasn't priced (item not in assortment that week)
    merged = merged.dropna(subset=["sell_price"]).reset_index(drop=True)
    merged["revenue"] = merged["sales"].astype("float32") * merged["sell_price"]

    # Vectorized aggregation to dept × store × week
    agg = (
        merged.groupby(
            ["dept_id", "cat_id", "store_id", "state_id", "week_start"],
            observed=True,
        )
        .agg(
            unit_sales=("sales", "sum"),
            revenue=("revenue", "sum"),
            n_items_priced=("item_id", "nunique"),
            n_items_on_promo=("is_promotion", "sum"),
            promo_share=("is_promotion", "mean"),
            mean_discount_depth=("discount_depth", "mean"),
            mean_sell_price=("sell_price", "mean"),
        )
        .reset_index()
    )

    # Sales-weighted discount depth (separate computation; weights are clipped sales)
    merged["_wgt_disc"] = merged["sales"].clip(lower=0) * merged["discount_depth"]
    swd = (
        merged.groupby(
            ["dept_id", "cat_id", "store_id", "state_id", "week_start"],
            observed=True,
        )
        .agg(
            _wgt_disc_sum=("_wgt_disc", "sum"),
            _wgt_total=("sales", lambda s: s.clip(lower=0).sum()),
        )
        .reset_index()
    )
    swd["sales_weighted_discount_depth"] = (
        swd["_wgt_disc_sum"] / swd["_wgt_total"].replace(0, np.nan)
    ).astype("float32")
    swd = swd[
        ["dept_id", "cat_id", "store_id", "state_id", "week_start",
         "sales_weighted_discount_depth"]
    ]

    return agg.merge(
        swd,
        on=["dept_id", "cat_id", "store_id", "state_id", "week_start"],
        how="left",
    )


# ---------- Calendar aggregates ----------

def _add_calendar_aggregates(weekly: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Add per-week event count and state-keyed SNAP days."""
    cal_weekly = (
        calendar.groupby("week_start", as_index=False)
        .agg(
            has_event_days=("has_event", "sum"),
            snap_ca_days=("snap_CA", "sum"),
            snap_tx_days=("snap_TX", "sum"),
            snap_wi_days=("snap_WI", "sum"),
        )
    )
    weekly = weekly.merge(cal_weekly, on="week_start", how="left")

    snap_map = {"CA": "snap_ca_days", "TX": "snap_tx_days", "WI": "snap_wi_days"}
    weekly["snap_days_in_week"] = weekly.apply(
        lambda r: r[snap_map[str(r["state_id"])]], axis=1
    ).astype("int8")

    return weekly.drop(columns=["snap_ca_days", "snap_tx_days", "snap_wi_days"])


# ---------- Public API ----------

def build_weekly_panel(
    cache_path: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """End-to-end: returns the (dept_id × store_id × week_start) panel with all base features."""
    cache_path = cache_path or (INTERIM_DIR / "m5_weekly_panel.parquet")

    cal = load_calendar()
    if verbose:
        print(f"[1/5] calendar loaded: {len(cal):,} rows, {cal['date'].min().date()} → {cal['date'].max().date()}")

    wide, _id_cols, d_cols = _load_sales_wide_with_filter(cal)
    if verbose:
        print(f"[2/5] sales wide loaded + filtered: {len(wide):,} (item, store) rows × {len(d_cols)} days")
        print(f"      kept items with non-zero sales before {FULL_HISTORY_CUTOFF}")

    item_prices = build_item_price_features(cal)
    if verbose:
        promo_pct = item_prices["is_promotion"].mean() * 100
        print(f"[3/5] price features built: {len(item_prices):,} (item, store, week) rows, "
              f"{promo_pct:.1f}% flagged is_promotion")

    parts: list[pd.DataFrame] = []
    stores = sorted(wide["store_id"].cat.categories.tolist())
    for i, store in enumerate(stores, 1):
        store_wide = wide[wide["store_id"] == store]
        store_prices = item_prices[item_prices["store_id"] == store]
        panel = _process_store(store, store_wide, d_cols, cal, store_prices)
        parts.append(panel)
        if verbose:
            print(f"[4/5]   store {i}/{len(stores)} ({store}): {len(panel):,} dept-weeks")

    weekly = pd.concat(parts, ignore_index=True)
    weekly = _add_calendar_aggregates(weekly, cal)
    if verbose:
        print(f"[5/5] final panel: {len(weekly):,} rows, "
              f"{weekly['week_start'].min().date()} → {weekly['week_start'].max().date()}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(cache_path, index=False)
    if verbose:
        print(f"      cached to {cache_path.relative_to(REPO_ROOT)}")

    return weekly


def load_weekly_panel(cache_path: Path | None = None) -> pd.DataFrame:
    """Load the cached weekly panel, or build it if missing."""
    cache_path = cache_path or (INTERIM_DIR / "m5_weekly_panel.parquet")
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    return build_weekly_panel(cache_path)
