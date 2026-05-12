# Optimizing Grocery Promotion Strategies: Quantifying Promotional Lift

**Author:** Amrendra Vimal
**Program:** UC Berkeley Professional Certificate in Machine Learning & Artificial Intelligence
**Module:** 20.1 — Initial Report and Exploratory Data Analysis

---

## Executive Summary

Grocery retailers run thousands of promotions every quarter — price discounts, weekly flyer features, end-cap displays — but rarely know *how much* of the resulting sales spike was actually caused by the promotion versus what would have sold anyway. This project quantifies that **promotional lift** using five years of daily store-level sales and weekly price data from Walmart, and builds a baseline forecasting model that lets a planner predict sales under different discount scenarios. The downstream value is sharper inventory planning: fewer empty shelves on under-forecasted promotions, and less perishable waste from over-forecasted ones.

## Rationale (Why does this question matter?)

Grocery margins are thin (1–3% net) and a large share of inventory is perishable. Two costly failure modes dominate planning:

- **Under-forecasting a promotion** → stockouts, lost sales, frustrated customers, lost loyalty.
- **Over-forecasting a promotion** → excess perishable stock → markdowns → food waste → financial loss.

A model that decomposes observed sales into "what we'd sell anyway" + "promotional lift at this discount depth" lets category managers spend their promotional budget on lifts that actually drive profit, and lets supply chain teams place orders that match the expected demand curve.

## Research Question

**How can historical sales and pricing data be used to quantify promotional lift and predict future sales volume under varying discount scenarios?**

Specifically: given a product department, a store, a calendar week, and a planned discount depth, what unit-sales volume should the retailer expect — and how confident can we be in that prediction?

## Data Sources

**Primary:** [M5 Forecasting - Accuracy (Walmart)](https://www.kaggle.com/competitions/m5-forecasting-accuracy) (Kaggle).

| Table                          | Rows                     | Key columns                                                                       |
| ------------------------------ | ------------------------ | --------------------------------------------------------------------------------- |
| `sales_train_evaluation.csv`   | 30,490 × ~1,947 (wide)   | `id, item_id, dept_id, cat_id, store_id, state_id, d_1...d_1941`                  |
| `sell_prices.csv`              | ~6.8M                    | `store_id, item_id, wm_yr_wk, sell_price` (weekly avg)                            |
| `calendar.csv`                 | ~1,969                   | `date, d, wm_yr_wk, event_name_1/2, event_type_1/2, snap_CA/TX/WI`                |

**Coverage:** 3,049 unique products × 10 Walmart stores in 3 US states (CA, TX, WI), Jan 2011 → May 2016. Categories: FOODS, HOBBIES, HOUSEHOLD; departments: FOODS_1/2/3, HOBBIES_1/2, HOUSEHOLD_1/2.

**Why this dataset:**

- **Has weekly prices** — enables the "varying discount scenarios" part of the research question (the original Favorita dataset only had a boolean promo flag, no price).
- **US retailer** — directly transferable to a US grocery context (the author works at Albertsons). Includes SNAP food-assistance flags, which are a real US demand signal.
- **5 years of daily data** — supports robust seasonality decomposition and year-over-year comparisons.
- **Bundled US calendar** — `calendar.csv` has annotated events (Christmas, SuperBowl, Easter, etc.) and three state-level SNAP flags.

**Supplemental:** [Dunnhumby Complete Journey](https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey) and [Favorita](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting) — kept on disk for a possible M24 cross-dataset validation (apply the same methodology to a second retailer to demonstrate the approach generalizes).

## Methodology

### Phase 1 — Data preparation (Notebook 01)

- Load `calendar.csv`, melt wide-format sales into long, join to dates.
- Filter to items with first non-zero sale before 2013-01-01 — keeps only items with full study-period history. Avoids contamination from launch curves.
- Compute per-(item, store, week) **discount depth** as `(regular_price - sell_price) / regular_price`, where `regular_price` is the trailing 4-week median of `sell_price`. This is an operational definition of "promotion" since M5 has no explicit flag.
- Aggregate to **(dept_id, store_id, week_start)** grain — 7 depts × 10 stores × ~280 weeks ≈ 20K rows. Plenty for regression, tractable in memory.
- Cache as parquet in `data/interim/m5_weekly_panel.parquet` for fast reload.

### Phase 2 — Exploratory Data Analysis (Notebook 02)

- Target distribution: weekly unit-sales skew, log-transform decision.
- Discount distribution: histogram of `sales_weighted_discount_depth`. How often do real promotions happen? At what depth?
- **Naive lift visualization**: mean weekly sales bucketed by discount depth (0%, 1-10%, 10-20%, 20%+) per department.
- Seasonality: weekly sales by month and week-of-year, with year-over-year overlay.
- Event-week effect: lift around SuperBowl, Thanksgiving, Christmas, Easter (from `calendar.csv` events).
- SNAP effect: weekly sales as a function of how many SNAP-eligible days fell in the week.
- Missing/zero handling: zero-sale weeks, sparse store-dept pairs, items without prices (not in assortment that week).

### Phase 3 — Feature engineering (Notebook 03)

- **Calendar**: year, month, week-of-year, quarter, Q4 holiday window flag.
- **Sales lags**: 1, 2, 4-week lags + 4-week and 12-week trailing means per (store, dept). All shifted by 1 to avoid target leakage.
- **Promo lags**: 1-week lag of `promo_share` and `sales_weighted_discount_depth` — captures pull-forward / pantry-loading effects.
- **Event flag**: `has_event_days` (count of event-flagged days in the week).
- **SNAP intensity**: `snap_days_in_week` keyed to each store's state.

### Phase 4 — Baseline model (Notebook 03)

- **Multiple Linear Regression** on the engineered feature matrix.
- Temporal train/test split: last 12 weeks held out (validity of any time-series split).
- Metrics: RMSE, MAE, R² on the holdout. Compared against a naive "last-week's-sales" benchmark.
- This baseline is the comparison anchor for M24, where Polynomial Features (for non-linear discount-response), Ridge/Lasso (for correlated lag features), and SVR (RBF/Polynomial kernels) will be evaluated.

## Caveats and Mitigation Plan

Six known limitations of the chosen dataset/approach, with concrete mitigations baked into the pipeline:

### Caveat 1 — Sparsity / intermittency at item-store-day grain

Most M5 series at the finest grain have long runs of zero-sale days, well-documented in the M5 literature.

**Mitigation:** Aggregate to **(dept_id, store_id, week)** grain. Smooths most of the zeros and gives ~20K usable training rows.

### Caveat 2 — Stockouts are not labeled

Zero sales could mean "no demand" OR "out of stock." M5 doesn't distinguish, which biases lift estimates downward when a promo causes a stockout.

**Mitigation:** Documented as a known limitation in the final report. Optionally exclude weeks with zero sales *and* a price drop as suspect (a promo with zero sales is almost certainly a stockout, not a failed promo).

### Caveat 3 — No explicit promotion flag

M5 doesn't tell us when an item was on a promo — we have to infer from prices.

**Mitigation:** Operational definition baked into the pipeline:

- `regular_price` = trailing 4-week MEDIAN of `sell_price` per (item, store).
- `discount_depth = max(0, (regular_price - sell_price) / regular_price)`.
- `is_promotion = (discount_depth > 0.05)` — anything more than 5% below the rolling regular price.

Median (vs mean) is robust to single-week price drops that would otherwise bias the regular-price baseline downward.

### Caveat 4 — Variable item launch dates

Some items have zero sales early because they hadn't launched yet, not because of low demand.

**Mitigation:** Filter to items with first non-zero sale before **2013-01-01**. Guarantees ≥3 years of full history per item.

### Caveat 5 — Weekly prices, not daily

`sell_prices.csv` averages prices over the week. Intra-week price moves are invisible.

**Mitigation:** Non-issue at our weekly grain — the price granularity matches the modeling granularity.

### Caveat 6 — Only 3 categories (Foods, Hobbies, Household)

Less product variety than a full retail dataset.

**Mitigation:** FOODS has 3 sub-departments (FOODS_1/2/3) where most grocery-promo behavior lives. Frame the analysis as "demonstrate methodology on Foods primarily, with Hobbies + Household as cross-category robustness check." This is sufficient for a capstone.

## Results

### Data and EDA findings (Notebooks 01 & 02)

- **Final panel**: 19,460 rows at `(dept_id × store_id × week_start)` grain, spanning **2011-01-25 → 2016-05-17** (~5.3 years × 7 departments × 10 stores). Pristine — zero nulls, full 278-week coverage on every (dept, store) pair, no zero-sale weeks.
- **Target is heavily right-skewed** (skew 2.31 raw → -0.60 after log1p). The baseline model uses `log1p(unit_sales)` as the target.
- **Walmart's EDLP pricing is real**: only 1.67% of (item, store, week) rows show *any* price drop from the trailing 4-week median, but when discounts do happen they're meaningful (median 8%, 95th percentile 50%).
- **At the modeling grain (dept-store-week), promo coverage is reasonable**: 57.3% of cells have at least one promoted item; 13.7% have promo_share ≥ 1%.
- **Promo prevalence varies sharply by department**: FOODS_3 has promo activity in 86% of weeks; HOBBIES_2 only in 20%.
- **Naive lift (any-promo vs no-promo, no controls)** by department: FOODS_2 +20.5%, HOUSEHOLD_1 +17.9%, FOODS_3 +14.2%, HOBBIES_2 +3.7%, FOODS_1 +2.4%, **HOBBIES_1 −4.0%** (the negative sign is exactly the kind of confounded signal a regression model needs to untangle).
- **Clear annual seasonality** with Q4 holiday peaks; week-of-year overlay shows consistent year-over-year patterns — calendar features will help the regression.
- **SNAP intensity correlates positively with FOODS sales**, especially in CA and TX.

### Baseline model performance (Notebook 03)

Holdout: last 12 weeks (2016-03-01 → 2016-05-17), 840 observations. Train: 18,200 observations across 260 weeks.

| Metric | Naive (persistence) | Linear Regression | Improvement |
| ------ | ------------------- | ----------------- | ----------- |
| RMSE (raw units) | 1,258.51 | **913.97** | **−27.4%** |
| MAE (raw units) | 632.74 | **496.12** | **−21.6%** |
| R² (raw units) | 0.970 | **0.984** | +1.4 pp |
| RMSE (log) | 0.146 | **0.121** | −17.3% |
| MAE (log) | 0.105 | **0.088** | −16.2% |
| R² (log) | 0.986 | **0.990** | +0.5 pp |

The Linear Regression baseline beats the naive last-week-sales benchmark by **~27% on RMSE**, which is a meaningful improvement on top of an already-strong persistence baseline.

### Top 5 most-predictive features (standardized coefficients)

| Rank | Feature | Coefficient |
| ---- | ------- | ----------- |
| 1 | `log_unit_sales_lag1` | **+0.624** |
| 2 | `log_unit_sales_roll4` | **+0.358** |
| 3 | `log_unit_sales_roll12` | **+0.191** |
| 4 | `log_unit_sales_lag4` | **+0.142** |
| 5 | `dept_id_HOBBIES_2` | **−0.124** |

Lagged sales dominate, as expected for a strongly persistent time-series. Department fixed effects come next, then SNAP days (+0.030) and the assortment size control (+0.024).

### Key finding: naive lift estimates are largely confounded

The most important EDA-to-model takeaway: **once the regression controls for lagged sales and calendar/seasonality, the promo-feature coefficients collapse to near-zero** (`promo_share` +0.005, `sales_weighted_discount_depth` −0.003 in standardized units). The naive +14% to +20% lift seen in EDA was almost entirely seasonal and selection bias — the items/depts/weeks that get promoted *would have sold more anyway* given their seasonal patterns and prior trajectory.

This validates the project's central methodological point: **headline lift comparisons without controls are not actionable**. The retailer-planning value comes from a model that can produce *counterfactual* sales predictions under different discount scenarios, not from raw means.

That said, the near-zero coefficient is also a signal that M20's simple linear model may be too restrictive for the discount-response surface. Polynomial features and non-linear models in M24 are well-motivated.

### Visualizations

Saved to `reports/figures/`:

1. `01_target_distribution.png` — raw vs log1p sales histograms
2. `02_sales_by_dept.png` — sales boxplots per department
3. `03_promo_prevalence.png` — promo prevalence per department + discount-depth distribution
4. `04_naive_lift.png` — mean sales by promo bucket × department
5. `05_seasonality.png` — monthly mean + year-over-year week-of-year overlay
6. `06_event_lift.png` — event-week sales lift per department
7. `07_snap_effect.png` — sales vs SNAP days, by state
8. `08_top_coefficients.png` — baseline-model top 15 standardized coefficients
9. `09_residuals.png` — predicted-vs-actual, residual distribution, residuals-vs-fitted

## Next Steps (Module 24)

- Add **Polynomial Features** on `discount_depth` to capture non-linear price elasticity (sales response is typically convex in discount).
- Add **Ridge** and **Lasso** regularization to handle correlation among lagged features.
- Add **Support Vector Regression** with RBF and Polynomial kernels.
- Compare all models on the same 12-week temporal holdout using a uniform metric (RMSE).
- Build a small **scenario tool**: given a planner's discount schedule for the next 4 weeks, output a sales forecast with a prediction interval.
- (Stretch) **Cross-dataset validation**: re-run the same pipeline on Favorita or Dunnhumby to demonstrate the methodology generalizes beyond Walmart.

## Repository Outline

```text
ucb-cap-op-promo-str/
├── README.md                          # this file
├── requirements.txt                   # pip dependencies
├── .gitignore
├── data/
│   ├── raw/
│   │   ├── m5/                        # M5 Walmart files — see data/raw/README.md
│   │   └── favorita/                  # Favorita files (supplemental, M24 use)
│   ├── interim/                       # cached weekly panel (parquet)
│   └── processed/                     # final feature matrix
├── notebooks/
│   ├── 01_data_loading.ipynb          # load + melt + filter + aggregate + cache
│   ├── 02_eda.ipynb                   # exploratory analysis
│   └── 03_baseline_model.ipynb        # feature engineering + Linear Regression baseline
├── src/
│   ├── data.py                        # M5 loader + dept-store-week aggregator
│   ├── features.py                    # calendar + lag + promo-lag features
│   └── models.py                      # (M24) model wrappers
└── reports/figures/                   # saved EDA plots
```

## How to Reproduce

```bash
# 1. Clone and enter
git clone <repo-url> && cd ucb-cap-op-promo-str

# 2. Set up environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Download M5 data (manual — see data/raw/README.md)

# 4. Run notebooks in order
jupyter lab notebooks/
```

## Contact

**Amrendra Vimal** — [amrendra.vimal@albertsons.com](mailto:amrendra.vimal@albertsons.com)
