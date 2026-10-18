# Data provenance and claim boundary

## UCI Online Retail II — primary evidence

- Source: UCI Machine Learning Repository, DOI `10.24432/C5CG6D`.
- Raw archive is stored locally as `data/raw/online_retail_ii.zip`; its SHA-256 is recorded in `outputs/run_summary.json`.
- This is the primary business case. All UCI claims use a 180-day history, a 90-day no-repeat label, and chronological train/validation/test cut-offs.

## KKBox — supplementary algorithm benchmark

- Intended source: KKBox Churn Prediction Challenge / WSDM Cup 2018.
- Current files are public pre-engineered matrices from `YLTsai0609/Kaggle-kkbox-churn-prediction-top5-percentage`, **not the official raw competition release**.
- `X_train.csv` SHA-256: `71854bc1fca6cad05a2fc0b9cf715167d411506039711fcee871751da84124bf`.
- `X_val.csv` SHA-256: `d242ae48a50cefb39ee44fdefee711eb0d1e18f38fdd7f179d502d04c3136660`.
- `Y_train.csv` SHA-256: `9d260aa84437d8afe0d633bdf9901857358ca147bed4cea5ff256677c43e0ae2`.
- `Y_val.csv` SHA-256: `988f2a302106a98542be9afc8ece4cd02cf00bee96cd7d77d4d8e00881271730`.
- Therefore KKBox metrics are labelled stratified-hold-out results only. They must not be described as temporal validation, raw-data replication, e-commerce deployment performance, or direct evidence of UCI portability.

## Reproduction boundary

Run `src/pipeline.py` for UCI, `src/kkbox_benchmark.py` for the clearly labelled supplementary benchmark, then `src/extended_visuals.py`, `src/build_report.py`, and `src/build_notebook.py`. Replacing the supplementary matrices with raw official KKBox files requires a new timestamp-aware feature pipeline and a new temporal evaluation; it must not reuse the present KKBox scores.
