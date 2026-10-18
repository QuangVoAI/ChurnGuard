"""Reproducible churn and CLV study.

Primary data: UCI Online Retail II. KKBox is benchmarked separately by
kkbox_benchmark.py because it has a different labelled-data contract.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
OUT, PROC, FIG, MODELS = [ROOT / x for x in ("outputs", "data/processed", "reports/figures", "models")]
LOOKBACK, HORIZON = 180, 90
FEATURES = ["recency_days", "frequency", "monetary_log", "aov_log", "product_diversity", "interpurchase_mean", "tenure_days"]
RFM_FEATURES = ["recency_days", "frequency", "monetary_log"]
TARGET_FRACTIONS = (0.10, 0.20, 0.30, 0.40, 0.50)


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_uci():
    xlsx = next(RAW.glob("*.xlsx"))
    # Both UCI sheets are required so every 180-day observation window is complete.
    # The first sheet supplies pre-2011 history; the model outcomes remain in 2011.
    sheets = pd.read_excel(xlsx, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)
    df.columns = [str(c).strip().replace(" ", "_").lower() for c in df.columns]
    df["invoice_date"] = pd.to_datetime(df["invoicedate"])
    df = df[df["customer_id"].notna() & df["quantity"].gt(0) & df["price"].gt(0)].copy()
    df = df[~df["invoice"].astype(str).str.startswith("C")]
    df["customer_id"] = df["customer_id"].astype("Int64").astype(str)
    df["revenue"] = df["quantity"] * df["price"]
    orders = (df.groupby(["customer_id", "invoice"], as_index=False)
                .agg(event_date=("invoice_date", "min"), monetary=("revenue", "sum"),
                     products=("stockcode", "nunique"), item_lines=("stockcode", "size"), country=("country", "first")))
    orders = orders.rename(columns={"customer_id": "customer", "invoice": "order_id"})
    orders["source"] = "UCI Online Retail II"
    return orders, df


def build_snapshot(orders, cutoff, label=True, lookback=LOOKBACK, horizon=HORIZON):
    cutoff = pd.Timestamp(cutoff)
    hist = orders[(orders.event_date >= cutoff - pd.Timedelta(days=lookback)) & (orders.event_date < cutoff)].copy()
    hist = hist.sort_values(["customer", "event_date"])
    g = hist.groupby("customer", as_index=False)
    s = g.agg(frequency=("order_id", "nunique"), monetary=("monetary", "sum"), aov=("monetary", "mean"),
              product_diversity=("products", "sum"), last_event=("event_date", "max"), first_event=("event_date", "min"))
    gaps = hist.groupby("customer").event_date.diff().dt.total_seconds().div(86400)
    s["interpurchase_mean"] = gaps.groupby(hist.customer).mean().reindex(s.customer).to_numpy()
    s["recency_days"] = (cutoff - s.last_event).dt.total_seconds() / 86400
    s["tenure_days"] = (cutoff - s.first_event).dt.total_seconds() / 86400
    s["monetary_log"] = np.log1p(s.monetary)
    s["aov_log"] = np.log1p(s.aov)
    s["snapshot_date"] = cutoff
    if label:
        future = orders[(orders.event_date >= cutoff) & (orders.event_date < cutoff + pd.Timedelta(days=horizon))]
        s["churn"] = (~s.customer.isin(future.customer)).astype(int)
    return s


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    n = max(1, int(np.ceil(len(y) * .10)))
    top = np.argsort(-p)[:n]
    prevalence = float(y.mean())
    brier = float(brier_score_loss(y, p))
    brier_baseline = prevalence * (1 - prevalence)
    return {"n": int(len(y)), "churn_prevalence": prevalence, "roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)), "brier": brier,
            "brier_baseline": float(brier_baseline), "brier_skill_score": float(1 - brier / brier_baseline),
            "precision_at_10": float(y[top].mean()), "recall_at_10": float(y[top].sum() / max(1, y.sum())),
            "lift_at_10": float(y[top].mean() / prevalence)}


def gains_table(y, p, fractions=TARGET_FRACTIONS):
    """Cumulative precision, recall and lift at decision-budget cutoffs."""
    y, p = np.asarray(y), np.asarray(p)
    rows = []
    for f in fractions:
        top = np.argsort(-p)[:max(1, int(np.ceil(len(y) * f)))]
        precision = float(y[top].mean())
        rows.append({"target_fraction": f, "customers_targeted": len(top),
                     "precision": precision, "recall": float(y[top].sum() / y.sum()),
                     "lift": float(precision / y.mean())})
    return pd.DataFrame(rows)


def bootstrap_ci(y, p, n_boot=1000, seed=42):
    """Percentile 95% CIs; resampling is test-set only and never selects a model."""
    y, p = np.asarray(y), np.asarray(p)
    rng, rows = np.random.default_rng(seed), []
    while len(rows) < n_boot:
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size < 2:
            continue
        m = metrics(y[idx], p[idx])
        rows.append({k: m[k] for k in ("roc_auc", "pr_auc", "precision_at_10")})
    b = pd.DataFrame(rows)
    result = []
    for metric in b.columns:
        result.append({"metric": metric, "estimate": float(metrics(y, p)[metric]),
                       "ci_95_low": float(b[metric].quantile(.025)),
                       "ci_95_high": float(b[metric].quantile(.975)), "n_bootstrap": n_boot})
    return pd.DataFrame(result)


def fit_models(train, validation, test, dataset_name, metrics_filename, save_model=False, coefficient_filename=None):
    candidates = {
        "RFM Logistic Regression": (RFM_FEATURES, Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler()), ("model", LogisticRegression(C=.4, class_weight="balanced", max_iter=3000, random_state=42))])),
        "Logistic Regression": (FEATURES, Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler()), ("model", LogisticRegression(C=.4, class_weight="balanced", max_iter=3000, random_state=42))])),
        "HistGradientBoosting": (FEATURES, Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("model", HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=15, min_samples_leaf=35, l2_regularization=4, random_state=42))])),
    }
    rows, fitted, candidate_features = [], {}, {}
    for name, (features, model) in candidates.items():
        model.fit(train[features], train.churn)
        p = model.predict_proba(validation[features])[:, 1]
        fitted[name], candidate_features[name] = model, features
        rows.append({"dataset": f"{dataset_name} validation", "model": name, "feature_set": "+".join(features), **metrics(validation.churn, p)})
    validation_metrics = pd.DataFrame(rows).sort_values(["pr_auc", "roc_auc"], ascending=False)
    selected_name = validation_metrics.iloc[0].model
    selected = fitted[selected_name]
    selected_features = candidate_features[selected_name]
    # Platt calibration learns only from the validation period; final test remains untouched.
    cal = CalibratedClassifierCV(selected, method="sigmoid", cv="prefit")
    cal.fit(validation[selected_features], validation.churn)
    result_rows = rows.copy()
    for label, model in [("Selected uncalibrated", selected), ("Selected calibrated", cal)]:
        result_rows.append({"dataset": f"{dataset_name} test", "model": label, "feature_set": "+".join(selected_features), **metrics(test.churn, model.predict_proba(test[selected_features])[:, 1])})
    pd.DataFrame(result_rows).to_csv(OUT / metrics_filename, index=False)
    # Model interpretation is fitted only on training data (not test labels).
    if coefficient_filename and selected_name.endswith("Logistic Regression"):
        coef = selected.named_steps["model"].coef_.ravel()[:len(selected_features)]
        pd.DataFrame({"feature": selected_features, "standardized_coefficient": coef,
                      "odds_ratio_per_1sd": np.exp(coef)}).sort_values("standardized_coefficient", key=np.abs, ascending=False).to_csv(OUT / coefficient_filename, index=False)
    if save_model:
        joblib.dump({"model": cal, "features": selected_features, "selected_base_model": selected_name, "lookback_days": LOOKBACK, "horizon_days": HORIZON}, MODELS / "portable_churn_model.joblib")
    return cal, selected_name, selected_features, pd.DataFrame(result_rows)


def fit_clv(uci_txn, cutoff):
    txn = uci_txn[uci_txn.invoice_date < cutoff].copy()
    summary = summary_data_from_transaction_data(txn, customer_id_col="customer_id", datetime_col="invoice_date", monetary_value_col="revenue", observation_period_end=cutoff, freq="D")
    # A modest penalty stabilizes the maximum-likelihood fit on this wholesale-heavy data.
    bgnbd = BetaGeoFitter(penalizer_coef=0.1)
    bgnbd.fit(summary["frequency"], summary["recency"], summary["T"])
    summary["expected_purchases_90d"] = bgnbd.conditional_expected_number_of_purchases_up_to_time(90, summary["frequency"], summary["recency"], summary["T"])
    purchasers = summary[summary.frequency > 0].copy()
    ggf = GammaGammaFitter(penalizer_coef=0.1)
    ggf.fit(purchasers["frequency"], purchasers["monetary_value"])
    purchasers["expected_average_profit"] = ggf.conditional_expected_average_profit(purchasers["frequency"], purchasers["monetary_value"])
    purchasers["clv_90d"] = purchasers.expected_purchases_90d * purchasers.expected_average_profit
    summary = summary.join(purchasers[["expected_average_profit", "clv_90d"]], how="left")
    summary.reset_index().to_csv(OUT / "bgnbd_gammagamma_clv.csv", index=False)
    return summary


def chart_all(uci_orders, uci_test, model, clv, selected_features, gains):
    sns.set_theme(style="whitegrid", palette="colorblind")
    monthly = uci_orders.assign(month=uci_orders.event_date.dt.to_period("M").astype(str)).groupby("month").agg(revenue=("monetary", "sum"), orders=("order_id", "nunique")).reset_index()
    fig, ax = plt.subplots(figsize=(10, 4)); ax.plot(monthly.month, monthly.revenue, marker="o", color="#2166ac", label="Revenue (GBP)"); ax.set(title="UCI monthly transaction value", xlabel="Month", ylabel="Revenue (GBP)"); ax.tick_params(axis="x", rotation=45); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "01_monthly_revenue.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4)); sns.histplot(uci_test, x="recency_days", hue="churn", bins=30, stat="density", common_norm=False, ax=ax); ax.set(title="Recency distribution by 90-day churn label", xlabel="Days since last purchase", ylabel="Density"); fig.tight_layout(); fig.savefig(FIG / "02_recency_churn.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4)); sns.boxplot(data=uci_test.assign(churn=uci_test.churn.astype(str)), x="churn", y="monetary_log", ax=ax, palette={"0":"#4daf4a","1":"#e41a1c"}); ax.set(title="Historical spend distribution by 90-day churn label", xlabel="Churn label (1 = no repeat)", ylabel="log(1 + historical GBP spend)"); fig.tight_layout(); fig.savefig(FIG / "06_spend_boxplot.png", dpi=180); plt.close(fig)
    pt, pp = calibration_curve(uci_test.churn, model.predict_proba(uci_test[selected_features])[:,1], n_bins=10, strategy="quantile"); fig, ax=plt.subplots(figsize=(5,5)); ax.plot([0,1],[0,1],"--",color="grey",label="Perfect calibration"); ax.plot(pp,pt,"o-",label="UCI calibrated model"); ax.set(title="Reliability curve on untouched UCI test",xlabel="Mean predicted churn probability",ylabel="Observed churn frequency"); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "04_calibration.png",dpi=180);plt.close(fig)
    d=clv.clv_90d.dropna(); fig,ax=plt.subplots(figsize=(8,4)); sns.histplot(np.log1p(d), bins=35, ax=ax, color="#4daf4a");ax.set(title="BG NBD and Gamma Gamma estimated 90-day CLV",xlabel="log(1 + estimated CLV in GBP)",ylabel="Customers");fig.tight_layout();fig.savefig(FIG / "05_clv_distribution.png",dpi=180);plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(gains.target_fraction * 100, gains.lift, marker="o", label="Lift"); ax.plot(gains.target_fraction * 100, gains.recall, marker="s", label="Cumulative recall"); ax.axhline(1, color="grey", linestyle="--", label="Random lift"); ax.set(title="UCI test: targeting trade-off by campaign size", xlabel="Customers targeted (%)", ylabel="Lift / recall"); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "07_gains_lift.png", dpi=180); plt.close(fig)


def main():
    for d in (OUT, PROC, FIG, MODELS): d.mkdir(parents=True, exist_ok=True)
    uci_orders, uci_txn = load_uci()
    split_cutoffs = {"train": "2011-01-30", "validation": "2011-04-30", "test": "2011-07-30"}
    assert uci_orders.event_date.min() <= pd.Timestamp(split_cutoffs["train"]) - pd.Timedelta(days=LOOKBACK)
    train = build_snapshot(uci_orders, split_cutoffs["train"])
    validation = build_snapshot(uci_orders, split_cutoffs["validation"])
    test = build_snapshot(uci_orders, split_cutoffs["test"])
    assert train.snapshot_date.max() + pd.Timedelta(days=HORIZON) <= validation.snapshot_date.min()
    model, selected_name, selected_features, table = fit_models(
        train, validation, test, "UCI", "model_metrics.csv", save_model=True,
        coefficient_filename="uci_logistic_coefficients.csv"
    )
    test_p = model.predict_proba(test[selected_features])[:, 1]
    gains = gains_table(test.churn, test_p)
    gains.to_csv(OUT / "uci_test_gains_table.csv", index=False)
    bootstrap_ci(test.churn, test_p).to_csv(OUT / "uci_test_bootstrap_ci.csv", index=False)
    sensitivity_rows = []
    for horizon in (60, 90, 120):
        sens = build_snapshot(uci_orders, split_cutoffs["test"], horizon=horizon)
        p = model.predict_proba(sens[selected_features])[:, 1]
        sensitivity_rows.append({"horizon_days": horizon, **metrics(sens.churn, p)})
    pd.DataFrame(sensitivity_rows).to_csv(OUT / "uci_horizon_sensitivity.csv", index=False)
    clv = fit_clv(uci_txn, pd.Timestamp(split_cutoffs["test"]))
    chart_all(uci_orders, test, model, clv, selected_features, gains)
    pd.concat([train.assign(split="train"), validation.assign(split="validation"), test.assign(split="test")]).to_parquet(PROC / "uci_snapshots.parquet", index=False)
    uci_orders.to_parquet(PROC / "uci_orders.parquet", index=False)
    source_zip = next(RAW.glob("*.zip"))
    summary = {"primary_dataset": "UCI Online Retail II", "benchmark_dataset": "KKBox Churn Prediction", "lookback_days": LOOKBACK, "horizon_days": HORIZON, "split_cutoffs": split_cutoffs, "selected_model": selected_name, "selected_features": selected_features, "uci_test": table[table.dataset.eq("UCI test")].to_dict("records"), "uci_test_bootstrap_ci": pd.read_csv(OUT / "uci_test_bootstrap_ci.csv").to_dict("records"), "uci_orders": int(len(uci_orders)), "uci_customers": int(uci_orders.customer.nunique()), "data_hash": hashlib.sha256(source_zip.read_bytes()).hexdigest()}
    save_json("run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
