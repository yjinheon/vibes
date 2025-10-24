
# edakit (modular)

- Diagnostics: types, numeric, categorical, datetime, text, duplicates, low-variance, outliers, correlation, association, missing matrix
- Preprocessing: simple impute, scaling (standard/robust), rare bucket, one-hot
- Selection: mutual info (approx), VIF (relative)
- Drift: PSI (numeric)
- Features (time/session): calendar parts, cyclical, elapsed, diffs, rolling stats, lags/leads, sessionization

## Demo
```bash
python demo_titanic.py
```
If Seaborn datasets are unavailable, it uses `titanic_sample.csv` bundled here.

