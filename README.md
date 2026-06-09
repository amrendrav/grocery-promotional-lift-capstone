# Optimizing Grocery Promotion Strategies: Quantifying Promotional Lift

**Author:** Amrendra Vimal
**Program:** UC Berkeley Professional Certificate in Machine Learning & Artificial Intelligence
**Module:** 24.1 — Final Capstone Submission

---

## What this project does, in plain English

Grocery stores run thousands of promotions every quarter — temporary price cuts, weekly-flyer features, end-cap displays. But when sales spike during a promotion, it's hard to tell **how much** of that spike was actually caused by the promotion versus what would have sold anyway.

This project answers two questions:

1. **For past promotions:** how much extra sales did each one actually generate, after accounting for seasonality and other normal demand patterns?
2. **For future promotions:** if a planner is considering a 15% discount next week on a specific product category, what unit sales should they expect — and how confident can we be in that number?

The downstream value is **better inventory planning**:

- Fewer empty shelves (stockouts) on under-forecasted promotions.
- Less perishable waste from over-forecasted promotions.

Grocery margins are thin (1–3% net), so even modest improvements in forecast accuracy translate to meaningful P&L impact.

---

## Headline results

- Compared **six different models** on a uniform 12-week temporal holdout.
- The best models reduce forecasting error by **~41% vs the "last week's sales" benchmark** and **~17% vs a standard Linear Regression baseline**.
- A working **scenario forecasting tool** ([`notebooks/07_scenario_tool.ipynb`](notebooks/07_scenario_tool.ipynb)) takes a 4-week discount schedule and outputs a sales forecast with an 80% prediction interval — the kind of artifact a category manager could use day-to-day.
- A **key methodological finding**: simple "promo vs no-promo" lift comparisons (the kind reported in industry dashboards) overstate the real impact of promotions by 10–20 percentage points. Most of the apparent lift is **seasonality and selection bias** — the items and weeks that get promoted are the same ones that already sell more. Only after a model controls for those confounders does the true promo signal emerge.

---

## Important findings (nontechnical)

### 1. Naive lift estimates overstate promotion impact

If you compare average sales on promotion weeks vs non-promotion weeks (without any other controls), you'd report that the **Foods 2** category gets **+20.5% lift** from a promotion, **Household 1** gets **+17.9%**, and so on. But these numbers are mostly **wrong** — they include the effect of Q4 holiday seasons, year-over-year growth, and the fact that the categories that get promoted heaviest are also the ones with the highest baseline sales anyway.

Once we let a regression model control for last-week's sales, seasonality, calendar events, and SNAP food-assistance days, the naive lift numbers collapse. The takeaway for the business: **headline promotional lift reports are likely overstating the true incremental sales by a large margin**. A planner who places inventory based on those numbers will over-order.

### 2. Non-linear effects matter — but not by as much as one might expect

The biggest jump in accuracy comes from going from "last week's sales" (naive baseline) to a **multiple linear regression** with lag features and seasonality controls (~27% RMSE reduction). The further jump from linear regression to non-linear models (Polynomial Features, SVR) gives another ~17% reduction, but the bulk of the predictive signal is already captured by the right linear features. **Spending engineering effort on richer features tends to pay off more than spending it on fancier model algorithms.**

### 3. Walmart's everyday-low-price strategy is real

Only **1.67%** of (item × store × week) combinations show any price drop from the rolling regular price. When discounts do happen, they're meaningful (median 8% off, 95th percentile 50% off), but the dataset's overall promo intensity is much lower than at a typical "high-low" pricing retailer like Albertsons or Kroger. At the **(department × store × week)** grain we used for modeling, 57% of weeks have at least one item on promo — sufficient signal for the model.

### 4. The model knows when to trust itself less

When the scenario tool is asked to forecast a department-store with sparse historical promo activity (e.g., HOBBIES_2), the prediction interval widens automatically — driven by the model's larger residual variance in that segment. This is exactly the behavior we want from a planning tool.

---

## Recommendations for a category manager (non-technical reader)

1. **Stop using raw "average sales on promo weeks" as a lift estimate.** They're inflated by seasonality and the fact that high-volume products get promoted more often. Use a model-based estimate that controls for these confounders.
2. **Build promotional plans around the scenario tool's prediction interval, not just the point forecast.** If the 80% interval is wide, you have meaningful uncertainty — order conservatively. If the interval is tight, you can lean into the point forecast.
3. **Re-train the model on your own retailer's data before deploying.** The Walmart M5 data this project uses has different promotional behavior than Albertsons (EDLP vs hi-lo); the methodology transfers but the coefficients won't.
4. **Treat the operational definition of "promotion" as a business choice.** This project uses "price ≥ 5% below the rolling 4-week median" as the operational definition. A different threshold would give different lift estimates. Pick the threshold that matches how your business uses the word "promotion."

---

## Research question

**How can historical sales and pricing data be used to quantify promotional lift and predict future sales volume under varying discount scenarios?**

Specifically: given a product department, a store, a calendar week, and a planned discount depth, what unit-sales volume should the retailer expect — and how confident can we be in that prediction?

---

## Data

**Primary dataset:** [M5 Forecasting - Accuracy (Walmart)](https://www.kaggle.com/competitions/m5-forecasting-accuracy) from Kaggle.

- 3,049 unique products × 10 Walmart stores in 3 US states (CA, TX, WI), Jan 2011 → May 2016
- 3 categories (FOODS, HOBBIES, HOUSEHOLD), 7 departments
- Daily unit sales (wide-format)
- Weekly sell prices
- US calendar with annotated events (Christmas, SuperBowl, Easter, etc.)
- State-level SNAP food-assistance flags

**Why this dataset:**

- **Has weekly prices** — enables computing a discount-depth feature (an earlier candidate, Favorita, only had a boolean promo flag).
- **US retailer** — directly relevant to the author's industry context (Albertsons).
- **5+ years of daily data** — supports seasonality decomposition and year-over-year comparisons.

Download instructions live in [data/raw/README.md](data/raw/README.md). The raw files are gitignored.

---

## Methodology

The work splits cleanly into seven notebooks, executed in order.

### Phase 1 — Data preparation ([notebooks/01_data_loading.ipynb](notebooks/01_data_loading.ipynb))

- Loads `calendar.csv`, melts wide-format sales into long, and joins to dates.
- Filters to items with first non-zero sale before 2013-01-01 (full study-period history; avoids contamination from item launches).
- Computes per-(item, store, week) **discount depth** as `(regular_price - sell_price) / regular_price`, where `regular_price` is the **trailing 4-week median** of `sell_price`. This is the operational definition of "promotion" since M5 has no explicit flag.
- Aggregates to **(dept_id, store_id, week_start)** grain — 7 departments × 10 stores × ~278 weeks ≈ 19,460 rows.
- Caches the result as parquet in `data/interim/m5_weekly_panel.parquet`.

### Phase 2 — Exploratory analysis ([notebooks/02_eda.ipynb](notebooks/02_eda.ipynb))

- Target distribution and log-transform decision.
- Discount distribution and naive lift bucketing.
- Seasonality (monthly, year-over-year week-of-year overlay).
- Event-week effect (SuperBowl, Thanksgiving, Christmas, Easter).
- SNAP day effect.

Nine annotated figures saved to `reports/figures/01_*` through `09_*`.

### Phase 3 — Baseline model ([notebooks/03_baseline_model.ipynb](notebooks/03_baseline_model.ipynb))

- Feature engineering: calendar features, sales lags (1, 2, 4 weeks + 4 and 12-week rolling means), promo lags, event flags, SNAP intensity.
- **Multiple Linear Regression** on the engineered feature matrix.
- Temporal train/test split: last 12 weeks held out.
- Beats naive last-week-sales benchmark by **27.4% RMSE** in raw units.

### Phase 4 — Regularized and polynomial models ([notebooks/04_advanced_models.ipynb](notebooks/04_advanced_models.ipynb))

- **Ridge** (L2) — handles multicollinearity among lagged features.
- **Lasso** (L1) — sparse, interpretable model.
- **Polynomial Features (degree 2) + Ridge** — captures non-linear price-discount response and feature interactions.
- Hyperparameters tuned via `GridSearchCV` over `TimeSeriesSplit(5)`.

### Phase 5 — Support Vector Regression ([notebooks/05_svr.ipynb](notebooks/05_svr.ipynb))

- **SVR (RBF kernel)** — non-linear surface fitting via radial basis functions.
- **SVR (Polynomial kernel)** — implicit polynomial expansion of any degree.
- Hyperparameters tuned via `GridSearchCV` on a recent-80-weeks subsample for tractability, then refit on the full training set.

### Phase 6 — Final model comparison ([notebooks/06_model_comparison.ipynb](notebooks/06_model_comparison.ipynb))

- Unified comparison table across all six models on the same 12-week holdout.
- Per-department breakdown.
- Residual diagnostics for the top three models.
- Per-week error decomposition.
- **Final model selection** based on the combination of accuracy, interpretability, and inference speed.

### Phase 7 — Scenario forecasting tool ([notebooks/07_scenario_tool.ipynb](notebooks/07_scenario_tool.ipynb))

- Refits the chosen final model (**PolyRidge**) on all available data.
- Derives 80% prediction intervals from empirical holdout residuals.
- Implements `forecast(dept_id, store_id, discount_schedule)` returning a 4-week forecast with intervals.
- Demonstrates three example scenarios (no-promo, light-promo, sustained deep-promo) and a cross-department comparison.

---

## Evaluation metric

**Primary metric: RMSE (Root Mean Squared Error) on raw unit sales**, reported on the 12-week holdout.

**Why RMSE?** In an inventory-planning context, **large errors are disproportionately costly** — a single bad forecast can cause a stockout (lost sale + lost loyalty) or a markdown cascade (perishable waste). RMSE penalizes large errors more than MAE (which weights all errors linearly), directly matching the business cost we care about. We also report **MAE** (median forecast error in units) and **R²** (variance explained) as secondary metrics for completeness.

---

## Results

### Final model comparison (12-week holdout, March 1 – May 17, 2016)

| Model | RMSE (raw units) | MAE (raw units) | R² (raw) | vs naive | vs Linear |
| --- | ---: | ---: | ---: | ---: | ---: |
| **SVR (Polynomial)** | **746.5** | 442.6 | 0.9895 | **−40.7%** | **−18.3%** |
| SVR (RBF) | 752.3 | 440.6 | 0.9894 | −40.2% | −17.7% |
| **PolyRidge (selected final)** | **754.9** | 449.6 | 0.9893 | **−40.0%** | **−17.4%** |
| Lasso | 911.6 | 493.8 | 0.9844 | −27.6% | −0.3% |
| Linear Regression (baseline) | 914.0 | 496.1 | 0.9843 | −27.4% | — |
| Ridge | 914.6 | 495.8 | 0.9843 | −27.3% | +0.1% |
| Naive (last-week persistence) | 1,258.5 | 632.7 | 0.9701 | — | +37.7% |

**Selected final model: PolyRidge** (Polynomial Features degree 2, interaction-only, Ridge alpha=100).

The two SVR variants and PolyRidge are clustered within ~1% of each other. The choice between them comes down to secondary criteria:

| Criterion | SVR (Polynomial) | SVR (RBF) | **PolyRidge (final)** |
| --- | --- | --- | --- |
| Holdout RMSE rank | 1 | 2 | 3 |
| Interpretability | Low — kernel + support vectors | Low — kernel + support vectors | **High — named coefficients** |
| Inference speed | Slow — kernel against many SVs | Slow — kernel against many SVs | **Fast — matrix multiply** |
| Scenario-tool friendliness | Hard to explain a forecast | Same | **Coefficients explain why** |

For a planner-facing scenario tool, the ~1% RMSE gap to SVR doesn't justify losing interpretability and inference speed. **PolyRidge captures the non-linear and interaction effects** that drove the 17% improvement over the linear baseline, while remaining fully interpretable.

### Cross-validation rationale

All hyperparameters were tuned with `GridSearchCV` over `TimeSeriesSplit(n_splits=5)` — an expanding-window time-series CV that prevents future observations from leaking into past training folds. `KFold` would be invalid for time-series data.

### Top features driving the model

Lagged sales dominate, as expected for a strongly-persistent time-series:

| Rank | Feature | Standardized coefficient |
| ---: | --- | ---: |
| 1 | `log_unit_sales_lag1` (sales last week) | +0.624 |
| 2 | `log_unit_sales_roll4` (4-week trailing avg) | +0.358 |
| 3 | `log_unit_sales_roll12` (12-week trailing avg) | +0.191 |
| 4 | `log_unit_sales_lag4` (sales 4 weeks ago) | +0.142 |
| 5 | `dept_id_HOBBIES_2` (department effect) | −0.124 |

Promo-related coefficients, after controlling for the above, collapse toward zero:

| Feature | Standardized coefficient |
| --- | ---: |
| `promo_share` | +0.005 |
| `promo_share_lag1` | +0.004 |
| `sales_weighted_discount_depth` | −0.003 |

This is the methodological headline (Finding #1 in the nontechnical section).

### Selected figures

| Figure | Description |
| --- | --- |
| `04_naive_lift.png` | Mean sales by promo bucket × department — the naive lift estimates |
| `08_top_coefficients.png` | Standardized coefficients from the baseline linear regression |
| `13_nb06_six_model_comparison.png` | All six models on the same holdout, RMSE bar chart |
| `14_nb06_top3_residuals.png` | Residual diagnostics for the three best models |
| `15_nb06_per_dept_rmse.png` | Per-department RMSE for all six models |
| `17_nb07_scenario_forecasts.png` | Scenario-tool output for three promo strategies |
| `18_nb07_cross_dept_scenario.png` | Cross-department comparison for the same discount schedule |

All figures live in [reports/figures/](reports/figures/).

---

## Caveats and mitigations

Six known limitations, with concrete mitigations baked into the pipeline:

| Caveat | Mitigation |
| --- | --- |
| **Sparsity / intermittency at item-store-day grain** — long zero-sale runs at the finest grain | Aggregate to **(dept_id, store_id, week)** grain — smooths zeros and gives ~20K usable training rows |
| **Stockouts are not labeled** — zero sales could mean "no demand" or "out of stock" | Documented as a known limitation. Promo weeks with zero sales are flagged as suspect (almost certainly a stockout) |
| **No explicit promotion flag** — M5 doesn't label promotions | Operational definition: `discount_depth > 5%` from trailing 4-week median. Sensitivity to threshold can be checked in future work |
| **Variable item launch dates** — some items have early zeros because they hadn't launched | Filter to items with first non-zero sale before 2013-01-01 |
| **Weekly prices, not daily** — intra-week price moves invisible | Non-issue at our weekly grain — price granularity matches modeling granularity |
| **Only 3 categories** (Foods, Hobbies, Household) | FOODS has 3 sub-departments where most grocery-promo behavior lives. Cross-category robustness is checked across all 7 departments |

---

## Repository structure

```text
ucb-cap-op-promo-str/
├── README.md                          # this file
├── requirements.txt                   # pip dependencies
├── .gitignore
├── data/
│   ├── raw/                           # M5 download (gitignored) — see data/raw/README.md
│   ├── interim/                       # cached weekly panel (parquet, gitignored)
│   └── processed/
├── notebooks/
│   ├── 01_data_loading.ipynb          # load + melt + filter + aggregate + cache
│   ├── 02_eda.ipynb                   # exploratory analysis (9 figures)
│   ├── 03_baseline_model.ipynb        # feature engineering + Linear Regression baseline
│   ├── 04_advanced_models.ipynb       # Poly + Ridge + Lasso w/ TimeSeriesSplit CV + GridSearchCV
│   ├── 05_svr.ipynb                   # SVR (RBF + Polynomial) w/ GridSearchCV
│   ├── 06_model_comparison.ipynb      # 6-model comparison + final selection
│   └── 07_scenario_tool.ipynb         # planner-facing forecast utility
├── src/
│   ├── data.py                        # M5 loader + dept-store-week aggregator
│   ├── features.py                    # calendar + lag + promo-lag features
│   └── models.py                      # design matrix, CV splitter, metrics, save/load helpers
└── reports/
    ├── figures/                       # saved plots (referenced from notebooks and this README)
    ├── model_results.json             # holdout metrics for every model
    ├── best_params.json               # tuned hyperparameters per model
    └── holdout_predictions.parquet    # per-row predictions for all models
```

---

## How to reproduce

```bash
# 1. Clone and enter
git clone https://github.com/amrendrav/grocery-promotional-lift-capstone.git
cd grocery-promotional-lift-capstone

# 2. Set up the environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Download the M5 data (manual — see data/raw/README.md)

# 4. Run the notebooks in order
jupyter lab notebooks/
#   01 → 02 → 03 → 04 → 05 → 06 → 07
```

Notebook 05 (SVR) is the longest single notebook to execute (~10 min on a modern laptop). All others run in under 1 minute.

---

## Next steps and recommendations

### Immediate (production deployment)

1. **Re-train on Albertsons internal data** before deploying. The methodology transfers but coefficient values won't.
2. **Wire the scenario tool into the planner workflow** as a simple web form or Excel add-in, returning the 4-week forecast + interval.
3. **Implement a basic monitoring loop** — log every forecast and the realized sales; trigger re-training when residuals drift.

### Near-term modeling improvements

1. **Sensitivity analysis on the promo threshold** — compare 3% / 5% / 10% discount thresholds. The 5% number is a defensible default but isn't sacred.
2. **Add a stockout flag** if Albertsons inventory data is available — zero-sale weeks during a promo are almost certainly stockouts and should be excluded from training.
3. **Hierarchical models** — share information across stores within the same banner/region, useful for low-volume departments.

### Stretch (research)

1. **Cross-dataset validation on Favorita or Dunnhumby** — show the methodology generalizes beyond Walmart.
2. **Causal estimation** — pair the predictive model with a propensity-score-based causal estimator to isolate promo lift from confounders (instead of relying on the regression to do both jobs).
3. **Item-level rollout** — re-fit at the item-store-week grain (with appropriate sparsity handling) to give per-SKU recommendations.

---

## Contact

**Amrendra Vimal** — [amrendra.vimal@albertsons.com](mailto:amrendra.vimal@albertsons.com)
