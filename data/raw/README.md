# Raw data (manual download)

This project uses two datasets:

- **M5 Walmart** — primary dataset for the M20/M24 analysis. Has explicit weekly prices, US calendar, SNAP food-assistance flags, and 5 years of daily sales across 10 stores in 3 states.
- **Favorita** — kept as a supplemental dataset for a possible M24 cross-dataset validation (apply the same methodology to a second retailer to demonstrate generalization).

## Primary: M5 Walmart Forecasting

Live at: https://www.kaggle.com/competitions/m5-forecasting-accuracy

### Steps
1. Open the competition page, sign in, and click **"Late Submission"** to accept rules (required to unlock downloads).
2. Click **Data** → **Download All** → `m5-forecasting-accuracy.zip` (~120 MB compressed, ~450 MB uncompressed).
3. Unzip into `data/raw/m5/` so files sit at:

```
data/raw/m5/
├── calendar.csv                    # ~1,969 rows: date, d, wm_yr_wk, weekday, month, year, event_name_1/2, event_type_1/2, snap_CA/TX/WI
├── sales_train_validation.csv      # 30,490 rows × ~1,919 cols (wide format) — sales through 2016-04-24
├── sales_train_evaluation.csv      # 30,490 rows × ~1,947 cols (wide format) — sales through 2016-05-22 (USE THIS — more data)
├── sell_prices.csv                 # ~6.8M rows: store_id, item_id, wm_yr_wk, sell_price (weekly avg price)
└── sample_submission.csv           # not needed
```

### Verification
After unzipping, run from the repo root:
```bash
ls -lh data/raw/m5/
```
Expected sizes:
- `calendar.csv` ~100 KB
- `sales_train_evaluation.csv` ~125 MB
- `sales_train_validation.csv` ~120 MB
- `sell_prices.csv` ~200 MB
- `sample_submission.csv` ~5 MB (we won't use it)

If wildly off, redo the download.

### Schema notes (worth knowing before EDA)

- **Sales are in wide format** — one column per day (`d_1`, `d_2`, ... `d_1941`). The loader in `src/data.py` melts to long.
- **Prices are weekly, not daily.** `wm_yr_wk` is the Walmart fiscal week. The loader joins this to `calendar.csv` to get a `week_start` date.
- **Items have variable launch dates** — many items have NaN/zero sales before they were stocked. The loader filters to items with first non-NaN sale before 2013-01-01.
- **Zero sales ≠ no demand.** Could be stockouts. M5 does not label stockouts. Documented as a known limitation.
- **3 categories, 7 departments**: FOODS_1, FOODS_2, FOODS_3, HOBBIES_1, HOBBIES_2, HOUSEHOLD_1, HOUSEHOLD_2.

---

## Supplemental: Favorita (kept for M24)

Files are under `data/raw/favorita/` already. See the earlier README revision in git history for download steps. The Favorita download is preserved so that a future M24 chapter can demonstrate cross-dataset validation of the methodology.
