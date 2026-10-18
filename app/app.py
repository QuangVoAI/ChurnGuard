"""Interactive dashboard for the UCI case study and KKBox churn benchmark."""
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs", ROOT / "reports/figures"
st.set_page_config(page_title="ChurnGuard Research", page_icon="📈", layout="wide")

@st.cache_data
def load():
    return (json.loads((OUT / "run_summary.json").read_text()), pd.read_csv(OUT / "model_metrics.csv"),
            pd.read_csv(OUT / "kkbox_benchmark_metrics.csv"),
            pd.read_csv(OUT / "bgnbd_gammagamma_clv.csv"), pd.read_csv(OUT / "uci_test_gains_table.csv"),
            pd.read_csv(OUT / "uci_test_bootstrap_ci.csv"), pd.read_csv(OUT / "uci_horizon_sensitivity.csv"),
            pd.read_csv(OUT / "uci_logistic_coefficients.csv"))

summary, metrics, kkbox, clv, gains, ci, sensitivity, coefficients = load()
st.title("ChurnGuard Research")
st.caption("UCI Online Retail II là case study nghiệp vụ theo temporal split. KKBox là benchmark churn có nhãn; benchmark dùng stratified user hold-out do feature matrix công khai không còn timestamp gốc.")

tab1, tab2, tab3, tab4 = st.tabs(["Kết quả mô hình", "Benchmark ngoài mẫu", "CLV học thuật", "Phương pháp"])
with tab1:
    test = metrics[metrics.dataset.eq("UCI test")].copy()
    a, b, c, d = st.columns(4)
    chosen = test[test.model.eq("Selected calibrated")].iloc[0]
    a.metric("UCI test ROC-AUC", f"{chosen.roc_auc:.3f}")
    b.metric("UCI test PR-AUC", f"{chosen.pr_auc:.3f}")
    c.metric("Lift@10%", f"{chosen.lift_at_10:.2f}x")
    d.metric("Brier Skill Score", f"{chosen.brier_skill_score:.1%}")
    st.caption(f"Brier baseline (luôn dự báo prevalence) = {chosen.brier_baseline:.3f}; calibrated Brier = {chosen.brier:.3f}.")
    st.dataframe(metrics.style.format({"roc_auc":"{:.3f}","pr_auc":"{:.3f}","brier":"{:.3f}","brier_baseline":"{:.3f}","brier_skill_score":"{:.1%}","lift_at_10":"{:.2f}x"}), use_container_width=True)
    st.image(str(FIG / "04_calibration.png"), caption="Calibration trên test UCI không được dùng để chọn mô hình")
    st.plotly_chart(px.line(gains, x="target_fraction", y=["lift", "recall"], markers=True, labels={"target_fraction":"Tỷ lệ khách được target", "value":"Lift / recall", "variable":"Metric"}, title="Trade-off theo quy mô campaign"), use_container_width=True)
    st.dataframe(ci.style.format({"estimate":"{:.3f}","ci_95_low":"{:.3f}","ci_95_high":"{:.3f}"}), use_container_width=True)
    st.subheader("Diễn giải RFM Logistic Regression")
    st.dataframe(coefficients.style.format({"standardized_coefficient":"{:.3f}","odds_ratio_per_1sd":"{:.2f}"}), use_container_width=True)
    st.caption("Dấu dương làm tăng risk dự báo; đây không phải là kết luận nhân quả.")
with tab2:
    advanced = kkbox[kkbox.model.eq("HistGradientBoosting")].iloc[0]
    a,b,c=st.columns(3)
    a.metric("KKBox ROC-AUC", f"{advanced.roc_auc:.3f}")
    b.metric("KKBox PR-AUC", f"{advanced.pr_auc:.3f}")
    c.metric("KKBox Lift@10%", f"{advanced.lift_at_10:.2f}x")
    st.info("KKBox được dùng để benchmark thuật toán trên nhãn churn có sẵn. Đây là stratified held-out test, không phải temporal benchmark; không so sánh trực tiếp AUC với UCI.")
    st.plotly_chart(px.bar(kkbox, x="model", y=["roc_auc", "pr_auc", "lift_at_10"], barmode="group", labels={"model":"Mô hình", "value":"Metric", "variable":"Metric"}, title="So sánh baseline và mô hình nâng cao trên KKBox"), use_container_width=True)
    st.dataframe(kkbox.style.format({"roc_auc":"{:.3f}","pr_auc":"{:.3f}","brier_skill_score":"{:.1%}","precision_at_10":"{:.1%}","recall_at_10":"{:.1%}","lift_at_10":"{:.2f}x"}), use_container_width=True)
with tab3:
    eligible = clv.dropna(subset=["clv_90d"]).copy()
    a,b,c=st.columns(3)
    a.metric("Khách đủ điều kiện Gamma-Gamma", f"{len(eligible):,}")
    b.metric("Median CLV 90 ngày", f"£{eligible.clv_90d.median():,.0f}")
    c.metric("P95 CLV 90 ngày", f"£{eligible.clv_90d.quantile(.95):,.0f}")
    st.plotly_chart(px.histogram(eligible, x="clv_90d", nbins=50, title="Phân phối CLV 90 ngày từ BG/NBD + Gamma-Gamma", labels={"clv_90d":"Estimated CLV in GBP"}), use_container_width=True)
    st.dataframe(eligible.reset_index().sort_values("clv_90d", ascending=False).head(100), use_container_width=True)
with tab4:
    st.markdown("""
    **Đơn vị phân tích:** khách hàng tại một snapshot. Features chỉ dùng lịch sử 180 ngày trước snapshot.

    **Nhãn:** churn = không có giao dịch mới trong 90 ngày sau snapshot. Đây là proxy cho không mua lại, không khẳng định rời bỏ vĩnh viễn.

    **Tách thời gian:** UCI train {train}, validation {validation}, test {test}. Mỗi snapshot dùng đủ 180 ngày lịch sử. Model chọn bằng PR-AUC trên validation. Platt calibration fit trên validation, test giữ nguyên.

    **Benchmark KKBox:** feature matrix public đã tiền xử lý được tách stratified thành calibration và final test. Vì không có timestamp gốc trong matrix, kết quả này không được diễn giải là temporal validation hoặc external deployment performance.

    **Tầng học thuật:** BG/NBD ước lượng số lần mua kỳ vọng và Gamma-Gamma ước lượng giá trị giao dịch, tạo CLV 90 ngày. Churn score là dự báo, không phải bằng chứng campaign tạo tác động nhân quả.

    **Độ nhạy churn:** Dashboard lưu kết quả với horizon 60/90/120 ngày. Nhãn “không mua lại” là proxy và không đồng nghĩa khách đã rời bỏ vĩnh viễn.
    """.format(**summary["split_cutoffs"]))
    st.dataframe(sensitivity.style.format({"churn_prevalence":"{:.1%}","roc_auc":"{:.3f}","pr_auc":"{:.3f}","brier_skill_score":"{:.1%}","lift_at_10":"{:.2f}x"}), use_container_width=True)
