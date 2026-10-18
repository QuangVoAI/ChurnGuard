# ChurnGuard Research

Project dự đoán khách hàng **không mua lại trong 90 ngày** cho thương mại điện tử/retail. Thiết kế có hai tầng, nhằm vừa trả lời bài toán kinh doanh vừa tránh đánh giá mô hình chỉ trên một nguồn dữ liệu.

1. **Nghiên cứu chính — UCI Online Retail II:** dùng lịch sử 180 ngày để dự đoán 90 ngày tiếp theo. Hai sheet 2009–2011 được gộp để mọi snapshot có đủ lịch sử; train, validation và test tách theo thời gian.
2. **Benchmark thuật toán — KKBox Churn Prediction:** benchmark churn có nhãn. Feature matrix công khai đã tiền xử lý được dùng minh bạch như labelled hold-out benchmark, không thay thế raw KKBox release.

Tầng học thuật dùng **BG/NBD + Gamma-Gamma** để ước lượng số lần mua và customer lifetime value (CLV) 90 ngày. Classification dự báo rủi ro; nó không đo tác động nhân quả của voucher/campaign.

## Chạy lại

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
curl -L -o data/raw/online_retail_ii.zip https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
unzip data/raw/online_retail_ii.zip -d data/raw
.venv/bin/python src/pipeline.py
.venv/bin/python src/kkbox_benchmark.py
.venv/bin/python src/extended_visuals.py
.venv/bin/python src/build_report.py
.venv/bin/python src/build_notebook.py
.venv/bin/jupyter execute ChurnGuard.ipynb --inplace --timeout=180
.venv/bin/streamlit run app/app.py
```

## Đọc kết quả đúng

- UCI test là đánh giá nội bộ theo thời gian: train 2011-01-30, validation 2011-04-30, test 2011-07-30. Mô hình chỉ được chọn trên validation.
- So sánh gồm RFM Logistic Regression, Logistic Regression đầy đủ và HistGradientBoosting. Báo cáo ROC-AUC, PR-AUC, Brier Skill Score, Lift@10%, bootstrap 95% CI và gains theo 10–50% khách target.
- Nhãn churn được kiểm tra độ nhạy với horizon 60/90/120 ngày; “không mua lại” là proxy, không đồng nghĩa rời bỏ vĩnh viễn.
- KKBox dùng stratified held-out test vì feature matrix công khai không có timestamp gốc. Không so sánh trực tiếp raw PR-AUC giữa hai nguồn vì prevalence, ngành hàng và hành vi khác nhau.
- Accuracy không được dùng làm metric chính khi lớp mất cân bằng.
- CLV từ BG/NBD + Gamma-Gamma là ước lượng hành vi giao dịch, không phải lợi nhuận thực vì dataset không có margin.
- Điểm churn là dự báo; chỉ A/B test mới trả lời được uplift và ROI campaign.

## Nguồn

- Chen, D. (2012). *Online Retail II*. UCI Machine Learning Repository. DOI: 10.24432/C5CG6D.
- KKBox Churn Prediction Challenge, WSDM Cup 2018; feature-matrix provenance được ghi trong `outputs/kkbox_benchmark_protocol.json`.
- Fader, Hardie & Lee (2005), Counting Your Customers the Easy Way.
- Fader, Hardie & Lee (2005), RFM and CLV using Gamma-Gamma.
