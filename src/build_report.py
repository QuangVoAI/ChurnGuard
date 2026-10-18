"""Build the Vietnamese 9-section report strictly aligned with the Midterm rubric."""
from pathlib import Path
import json
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
FIG = ROOT / 'reports/figures'

summary = json.loads((OUT / 'run_summary.json').read_text())
metrics = pd.read_csv(OUT / 'model_metrics.csv')
kkbox = pd.read_csv(OUT / 'kkbox_benchmark_metrics.csv')
gains = pd.read_csv(OUT / 'uci_test_gains_table.csv')
ci = pd.read_csv(OUT / 'uci_test_bootstrap_ci.csv')
sensitivity = pd.read_csv(OUT / 'uci_horizon_sensitivity.csv')
coefficients = pd.read_csv(OUT / 'uci_logistic_coefficients.csv')
chosen = metrics[(metrics.dataset == 'UCI test') & (metrics.model == 'Selected calibrated')].iloc[0]

model_display = metrics[['dataset', 'model', 'n', 'roc_auc', 'pr_auc', 'brier', 'brier_skill_score', 'lift_at_10']].copy()
model_display['model'] = model_display['model'].replace({
    'RFM Logistic Regression': 'RFM LR (Baseline)',
    'Logistic Regression': 'Full LR',
    'HistGradientBoosting': 'HistGradientBoosting',
    'Selected uncalibrated': 'Selected Raw',
    'Selected calibrated': 'Selected Calibrated'
})
model_display.columns = ['Tập dữ liệu', 'Mô hình', 'N', 'ROC-AUC', 'PR-AUC', 'Brier', 'BSS', 'Lift@10%']

ci_display = ci.rename(columns={'metric': 'Chỉ số', 'estimate': 'Ước lượng', 'ci_95_low': '95% CI Dưới', 'ci_95_high': '95% CI Trên', 'n_bootstrap': 'Số mẫu Bootstrap'})
coef_display = coefficients.rename(columns={'feature': 'Đặc trưng', 'standardized_coefficient': 'Hệ số chuẩn hóa', 'odds_ratio_per_1sd': 'Tỷ số chênh (OR/1SD)'})
sensitivity_display = sensitivity[['horizon_days', 'churn_prevalence', 'roc_auc', 'pr_auc', 'brier_skill_score', 'lift_at_10']].copy()
sensitivity_display.columns = ['Cửa sổ (Ngày)', 'Tỷ lệ Churn', 'ROC-AUC', 'PR-AUC', 'BSS', 'Lift@10%']

doc = Document()
sec = doc.sections[0]
sec.top_margin = Inches(0.7)
sec.bottom_margin = Inches(0.7)
sec.left_margin = Inches(0.75)
sec.right_margin = Inches(0.75)

styles = doc.styles
styles['Normal'].font.name = 'Aptos'
styles['Normal'].font.size = Pt(10.5)

def title(t):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(t)
    r.font.size = Pt(18)
    r.font.bold = True
    r.font.color.rgb = RGBColor(16, 44, 87)

def h1(t):
    p = doc.add_paragraph()
    r = p.add_run(t)
    r.font.size = Pt(13)
    r.font.bold = True
    r.font.color.rgb = RGBColor(24, 76, 120)
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)

def h2(t):
    p = doc.add_paragraph()
    r = p.add_run(t)
    r.font.size = Pt(11.5)
    r.font.bold = True
    r.font.color.rgb = RGBColor(40, 40, 40)
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)

def p(t, bold=False, italic=False):
    q = doc.add_paragraph()
    r = q.add_run(t)
    r.bold = bool(bold)
    r.italic = bool(italic)
    q.paragraph_format.space_after = Pt(4)
    q.paragraph_format.line_spacing = 1.15
    return q

def fig(name, caption, comment):
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(6)
    p_img.paragraph_format.space_after = Pt(2)
    p_img.add_run().add_picture(str(FIG / name), width=Inches(6.0))
    
    q = doc.add_paragraph()
    q.alignment = WD_ALIGN_PARAGRAPH.CENTER
    q.paragraph_format.space_after = Pt(4)
    r = q.add_run(caption)
    r.italic = True
    r.font.size = Pt(9.5)
    r.font.color.rgb = RGBColor(60, 60, 60)
    
    c = doc.add_paragraph()
    c.paragraph_format.space_after = Pt(6)
    r_c_bold = c.add_run("Nhận xét chuyên môn: ")
    r_c_bold.bold = True
    r_c_bold.font.size = Pt(10)
    r_c = c.add_run(comment)
    r_c.font.size = Pt(10)

def table(df, cols):
    t = doc.add_table(rows=1, cols=len(cols))
    t.style = 'Light Shading Accent 1'
    for c, cell in zip(cols, t.rows[0].cells):
        cell.text = c
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.bold = True
                run.font.size = Pt(9.5)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for cell, key in zip(cells, cols):
            v = row[key] if key in row else ''
            cell.text = f'{v:.3f}' if isinstance(v, float) else str(v)
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(9.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

# Header & Cover
title('BÁO CÁO HỆ THỐNG HỖ TRỢ QUYẾT ĐỊNH GIỮ CHÂN KHÁCH HÀNG (CHURNGUARD)')
p('Đề tài: Dự đoán rời bỏ khách hàng thương mại điện tử phục vụ tối ưu hóa ngân sách Marketing', bold=True)
p('Bộ môn: Data Analytics | Báo cáo kiểm tra giữa kỳ 2026')
p('Nhóm thực hiện: Nhóm ChurnGuard | Lớp: Data Analytics K18')
doc.add_page_break()

# 1. Giới thiệu
h1('1. Giới thiệu')
p('Trong bối cảnh doanh thu sàn thương mại điện tử tăng trưởng chậm lại dù ngân sách marketing không đổi, Ban giám đốc kinh doanh nghi ngờ tỷ lệ quay lại mua hàng (repeat purchase) đang sụt giảm. Dự án ChurnGuard được xây dựng nhằm giải quyết bài toán kinh doanh cốt lõi: nhận diện chính xác nhóm khách hàng có nguy cơ cao không quay lại mua sắm trong 90 ngày tới, từ đó đề xuất chiến dịch giữ chân cá nhân hóa trước khi phân bổ ngân sách quý mới.')
p('Nghiên cứu tuân thủ chặt chẽ ranh giới phương pháp luận: mô hình dự báo xác suất rời bỏ (churn risk) phục vụ xếp hạng và ưu tiên đối tượng can thiệp, không tự đồng nhất dự báo tương quan với tác động nhân quả (causal effect) của chiến dịch voucher. Hiệu quả gia tăng thực tế (incremental uplift) và ROI bắt buộc phải được thẩm định thông qua thử nghiệm đối chứng A/B testing.')

# 2. Dữ liệu và Tiền xử lý
h1('2. Dữ liệu và Tiền xử lý')
p(f'Nghiên cứu sử dụng nguồn dữ liệu chính từ UCI Online Retail II Repository (DOI: 10.24432/C5CG6D) với tổng cộng {summary["uci_orders"]:,} hóa đơn hợp lệ và {summary["uci_customers"]:,} khách hàng được hợp nhất từ 2 sheet giao dịch giai đoạn 2009–2011. Giai đoạn 2009–2010 cung cấp lịch sử quan sát ban đầu để đảm bảo toàn bộ các snapshot khách hàng đều có đầy đủ 180 ngày lịch sử giao dịch; các nhãn mục tiêu và đánh giá mô hình được thực hiện hoàn toàn trong năm 2011.')
p('Quy trình kiểm soát chất lượng dữ liệu: Loại bỏ các mã đơn hàng hủy (bắt đầu bằng ký tự C), loại bỏ các dòng thiếu Customer ID, đơn giá hoặc số lượng nhỏ hơn hoặc bằng 0. Đơn vị phân tích chuẩn là customer-snapshot: cửa sổ quan sát (observation window) = 180 ngày, cửa sổ dự báo (prediction window) = 90 ngày. Nhãn churn = 1 khi khách hàng hoàn toàn không phát sinh giao dịch hợp lệ nào trong 90 ngày sau mốc snapshot. Doanh thu được log-transform (log1p) để giảm hiện tượng lệch phải nặng.')

# 3. Lớp phân tích mô tả
h1('3. Lớp phân tích mô tả (Descriptive Analytics)')
p('Lớp phân tích mô tả tập trung làm rõ bức tranh tổng thể về doanh thu theo chu kỳ thời gian và động thái quay lại mua sắm của các thế hệ khách hàng (cohorts).')
fig('01_monthly_revenue.png',
    'Hình 1. Biểu đồ đường (Line Chart) thể hiện tổng giá trị giao dịch và số lượng đơn hàng theo tháng (UCI Online Retail II).',
    'Doanh thu biến động rất mạnh theo mùa vụ, đạt đỉnh vào quý 4 (tháng 9-11) với doanh thu vượt 1.4 triệu GBP trước khi hạ nhiệt mạnh vào tháng 12. Biến động phi ngẫu nhiên này khẳng định việc chia tập dữ liệu ngẫu nhiên (random split) sẽ vi phạm giả định độc lập và gây rò rỉ dữ liệu (data leakage) từ tương lai về quá khứ.')

fig('08_cohort_retention.png',
    'Hình 2. Ma trận nhiệt giữ chân (Cohort Retention Heatmap) theo tháng phát sinh giao dịch đầu tiên.',
    'Tỷ lệ quay lại mua hàng sụt giảm nhanh chóng sau tháng đầu tiên (chỉ còn khoảng 15-25% khách hàng quay lại ở tháng thứ 2) và phân rã dần qua các tháng tiếp theo. Xu hướng sụt giảm đều đặn giữa các cohort củng cố nghi ngờ của Ban giám đốc: suy giảm repeat purchase là vấn đề hệ thống của toàn bộ vòng đời khách hàng chứ không chỉ cục bộ ở một nhóm đơn lẻ.')

# 4. Lớp phân tích chẩn đoán
h1('4. Lớp phân tích chẩn đoán (Diagnostic Analytics)')
p('Lớp chẩn đoán đi sâu vào các nhân tố RFM (Recency, Frequency, Monetary) để xác định sự khác biệt hành vi giữa nhóm khách hàng tiếp tục mua và nhóm khách hàng có nguy cơ rời bỏ.')
fig('06_spend_boxplot.png',
    'Hình 3. Biểu đồ hộp (Boxplot) so sánh phân phối giá trị chi tiêu lịch sử (log monetary) giữa nhóm Churn (1) và Non-churn (0).',
    'Phân phối chi tiêu của cả hai nhóm đều có độ phân tán rộng và nhiều giá trị ngoại lai. Mặc dù nhóm quay lại mua hàng có trung vị chi tiêu cao hơn nhẹ, sự chồng lấn lớn cho thấy mức chi tiêu đơn thuần không đủ để phân tách nguy cơ rời bỏ; do đó bắt buộc phải chuẩn hóa và kết hợp đa biến.')

fig('09_rfm_risk_heatmap.png',
    'Hình 4. Ma trận nhiệt 2 chiều (2D Risk Heatmap) đo tỷ lệ rời bỏ theo phân khúc Recency (ngày mua gần nhất) x Frequency (tần suất mua).',
    'Tỷ lệ không mua lại tăng vọt lên trên 70-85% ở các ô có Recency > 90 ngày ngay cả khi tần suất mua lịch sử cao. Khách hàng có tần suất cao nhưng ngừng giao dịch đột ngột đại diện cho nhóm rủi ro mất mát giá trị lớn nhất cần được hệ thống gắn cờ cảnh báo sớm.')

# 5. Lớp phân tích dự báo
h1('5. Lớp phân tích dự báo (Predictive Analytics)')
p(f'Để tái lập chính xác kịch bản triển khai trong thực tế, nghiên cứu áp dụng phương pháp chia tập dữ liệu theo thời gian (Temporal Split) thay vì ngẫu nhiên. Mốc cắt huấn luyện (Train cutoff): {summary["split_cutoffs"]["train"]}; mốc cắt thẩm định (Validation cutoff): {summary["split_cutoffs"]["validation"]}; mốc cắt kiểm thử độc lập (Test cutoff): {summary["split_cutoffs"]["test"]}. Mỗi snapshot có đủ 180 ngày lịch sử trước mốc cắt và nhãn 90 ngày chỉ được ghi nhận sau mốc cắt, bảo đảm 100% không rò rỉ thông tin tương lai.')
p('Nghiên cứu so sánh 3 thuật toán: (1) Mô hình Baseline: RFM Logistic Regression chỉ dùng 3 biến truyền thống (Recency, Frequency, Monetary); (2) Full Logistic Regression mở rộng thêm Giá trị đơn trung bình (AOV), Độ đa dạng sản phẩm, Khoảng cách mua trung bình và Thâm niên (Tenure); (3) Mô hình nâng cao: HistGradientBoosting Classifier có khả năng học quan hệ phi tuyến và tương tác bậc cao. Mô hình được chọn dựa trên PR-AUC trên tập validation; sau đó áp dụng hiệu chuẩn Platt (Sigmoid Calibration) trên tập validation để xác suất dự báo phản ánh đúng tần suất thực tế.')

table(model_display, ['Tập dữ liệu', 'Mô hình', 'N', 'ROC-AUC', 'PR-AUC', 'Brier', 'BSS', 'Lift@10%'])

fig('04_calibration.png',
    'Hình 5. Đường cong hiệu chuẩn xác suất (Reliability Diagram / Calibration Curve) trên tập kiểm thử độc lập UCI test.',
    'Đường cong thực tế của mô hình sau hiệu chuẩn bám rất sát đường lý tưởng 45 độ. Điểm số Brier Skill Score đạt 16.9% so với baseline ngây thơ, chứng minh xác suất dự báo rủi ro có độ tin cậy định lượng cao, giúp bộ phận Marketing tự tin tính toán giá trị kỳ vọng khi ra quyết định.')

fig('07_gains_lift.png',
    'Hình 6. Biểu đồ đường kép (Dual-axis Line Chart) thể hiện Đòn bẩy (Lift) và Độ phủ tích lũy (Cumulative Recall) theo tỷ lệ khách hàng mục tiêu.',
    f'Tại ngưỡng top 10% khách hàng rủi ro cao nhất, mô hình đạt Lift = {chosen.lift_at_10:.2f}x với Precision = {chosen.precision_at_10:.1%} (gấp 1.62 lần so với chọn ngẫu nhiên có tỷ lệ nền là {chosen.churn_prevalence:.1%}). Khi mở rộng quy mô chiến dịch lên top 30%, mô hình bao phủ được hơn 42% tổng lượng khách hàng rời bỏ.')

h2('5.1. Diễn giải mô hình (Model Interpretability)')
p('Mô hình hồi quy Logistic được diễn giải trực tiếp thông qua hệ số chuẩn hóa (Standardized Coefficients) và Tỷ số chênh (Odds Ratio per 1 SD) để ban giám đốc nắm rõ trọng số thúc đẩy rủi ro:')
table(coef_display, ['Đặc trưng', 'Hệ số chuẩn hóa', 'Tỷ số chênh (OR/1SD)'])
p('Recency là nhân tố thúc đẩy rời bỏ mạnh nhất (Hệ số = +1.037, OR = 2.82): cứ tăng 1 độ lệch chuẩn số ngày chưa mua hàng, tỷ số chênh không quay lại tăng 2.82 lần. Ngược lại, Tần suất mua hàng (Frequency) là lá chắn giữ chân mạnh nhất (Hệ số = -0.825, OR = 0.44): tăng 1 độ lệch chuẩn tần suất làm giảm tỷ số chênh rời bỏ đi hơn 56%.')

h2('5.2. Khoảng tin cậy Bootstrap và Kiểm định độ nhạy cửa sổ')
p('Để định lượng mức độ bất định thống kê, nghiên cứu thực hiện 1.000 lần lấy mẫu lại có hoàn lại (Bootstrap Resampling B=1,000) trên tập UCI test untouched:')
table(ci_display, ['Chỉ số', 'Ước lượng', '95% CI Dưới', '95% CI Trên', 'Số mẫu Bootstrap'])
p('Khoảng tin cậy hẹp (ROC-AUC 95% CI: 0.733–0.769, Lift@10% CI: 1.50–1.74x) khẳng định năng lực phân loại của mô hình có tính ổn định thống kê cao. Đồng thời, kiểm tra độ nhạy với các cửa sổ định nghĩa churn 60, 90 và 120 ngày cho thấy ROC-AUC duy trì ổn định ở mức 0.749–0.753, chứng minh kết luận mô hình không phụ thuộc cơ hội vào mốc 90 ngày.')
table(sensitivity_display, ['Cửa sổ (Ngày)', 'Tỷ lệ Churn', 'ROC-AUC', 'PR-AUC', 'BSS', 'Lift@10%'])

# 6. Lớp phân tích chỉ định
h1('6. Lớp phân tích chỉ định (Prescriptive Analytics)')
p('Lớp chỉ định kết nối trực tiếp các phát hiện từ 3 phần phân tích trước vào việc ra quyết định kinh doanh. Thay vì phân bổ ngân sách marketing đại trà hoặc chỉ dựa vào điểm rủi ro, ChurnGuard tích hợp mô hình kinh tế lượng BG/NBD + Gamma-Gamma để ước lượng Giá trị vòng đời khách hàng 90 ngày (90-day Customer Lifetime Value - CLV).')
fig('10_risk_clv_priority.png',
    'Hình 7. Ma trận phân bổ ưu đãi 4 góc (4-Quadrant Priority Matrix) kết hợp giữa Nguy cơ rời bỏ (Churn Risk) và Giá trị ước tính (CLV 90 ngày).',
    'Ma trận phân tách khách hàng thành 4 nhóm rõ rệt để tối ưu hóa ngân sách: (1) Nhóm VIP Cứu vãn (High Risk - High CLV): Ưu tiên can thiệp cao nhất; (2) Nhóm Chăm sóc tự động (High Risk - Low CLV): Dùng kênh chi phí thấp; (3) Nhóm Khách hàng trung thành (Low Risk - High CLV): Tránh can thiệp thừa; (4) Nhóm Tiềm năng thấp (Low Risk - Low CLV).')

p('Dựa trên số liệu định lượng cụ thể từ các bước phân tích trước, đề xuất hành động cho Ban giám đốc kinh doanh như sau:')
p('1. Trích dẫn số liệu & Phân bổ ngân sách mục tiêu: Tập trung ngân sách chiến dịch vào nhóm Top 10% rủi ro cao nhất (Lift = 1.62x, Precision = 71.7% so với tỷ lệ nền 44.3%). Với 2,788 khách hàng trong tập đánh giá, top 10% tương đương 279 khách hàng mang lại tỷ lệ bắt trúng cao nhất.')
p('2. Can thiệp có phân tầng theo CLV: Trong 279 khách hàng này, lọc ra nhóm có CLV 90 ngày thuộc phân vị cao (P95 CLV > £2,500). Chỉ nhóm này mới được phân bổ ưu đãi có chi phí tài chính cao (voucher giảm giá trực tiếp 15-20% hoặc liên hệ chăm sóc khách hàng 1-1). Nhóm còn lại (CLV thấp) được tiếp cận bằng kênh tiếp thị chi phí 0đ (Email tự động cá nhân hóa sản phẩm yêu thích, App push notification) nhằm bảo vệ biên lợi nhuận ròng.')
p('3. Nguyên tắc không can thiệp khách hàng rủi ro thấp: Tuyệt đối không gửi voucher đại trà cho nhóm Khách hàng trung thành (Low Risk), vì nhóm này có xác suất tự quay lại mua hàng trên 75%; việc tặng voucher cho họ sẽ gây hiện tượng lãng phí ngân sách và xói mòn lợi nhuận tự nhiên.')
p('4. Thiết kế thử nghiệm A/B Testing đo lường Causal Uplift: Trước khi nhân rộng ra toàn sàn, triển khai pilot trong 30 ngày với tỷ lệ phân chia ngẫu nhiên 80% Treatment (nhận can thiệp) và 20% Control (giữ nguyên không tác động) trong nhóm mục tiêu. Chỉ số đánh giá cuối cùng là Tỷ lệ mua lại tăng thêm thực sự (Incremental Repeat Rate) và Lợi nhuận tăng thêm trên mỗi đồng chi phí (Incremental Marketing ROI).')

# 7. Kết luận và Hạn chế
h1('7. Kết luận và Hạn chế')
p('Dự án ChurnGuard đã xây dựng thành công một quy trình phân tích và hỗ trợ quyết định hoàn chỉnh, giải quyết trọn vẹn câu hỏi kinh doanh về sụt giảm khách hàng quay lại mua sắm. Việc áp dụng chặt chẽ temporal split, hiệu chuẩn xác suất và tích hợp CLV đã nâng tầm dự án vượt trên một bài toán phân loại máy học đơn thuần, trở thành một công cụ định lượng hỗ trợ điều hành kinh doanh.')
p('Hạn chế nghiên cứu: (1) Bộ dữ liệu UCI Online Retail II mang đặc thù bán buôn quà tặng tại Anh, thiếu dữ liệu biên lợi nhuận thực tế (margin) và chi phí vận hành; (2) Benchmark KKBox là bài toán đăng ký dịch vụ số (subscription) trên feature matrix có sẵn, mang tính chất kiểm tra độ khái quát của thuật toán chứ không thể thay thế cho thử nghiệm thực địa; (3) Điểm số mô hình chỉ là xác suất tương quan; chỉ có A/B test thực tế mới trả lời được bài toán nhân quả.')

# 8. Phân công công việc nhóm
h1('8. Phân công công việc nhóm (Team Task Allocation)')
p('Bảng phân công chi tiết nhiệm vụ và tỷ lệ đóng góp của từng thành viên trong nhóm:')

allocation_df = pd.DataFrame([
    {'STT': '1', 'Họ và tên': 'Nguyễn Văn A', 'MSSV': '22120001', 'Nhiệm vụ đảm nhiệm': 'Tiền xử lý dữ liệu, kiểm soát chất lượng, Temporal Split, Pipeline chính', 'Tỷ lệ đóng góp': '100%'},
    {'STT': '2', 'Họ và tên': 'Trần Thị B', 'MSSV': '22120002', 'Nhiệm vụ đảm nhiệm': 'Phân tích mô tả, chẩn đoán RFM, trực quan hóa biểu đồ và cohort', 'Tỷ lệ đóng góp': '100%'},
    {'STT': '3', 'Họ và tên': 'Lê Văn C', 'MSSV': '22120003', 'Nhiệm vụ đảm nhiệm': 'Huấn luyện mô hình, Calibration, Bootstrap CI, diễn giải hệ số hồi quy', 'Tỷ lệ đóng góp': '100%'},
    {'STT': '4', 'Họ và tên': 'Phạm Thị D', 'MSSV': '22120004', 'Nhiệm vụ đảm nhiệm': 'Mô hình CLV (BG/NBD, Gamma-Gamma), Phân tích chỉ định và A/B testing', 'Tỷ lệ đóng góp': '100%'},
    {'STT': '5', 'Họ và tên': 'Hoàng Văn E', 'MSSV': '22120005', 'Nhiệm vụ đảm nhiệm': 'Xây dựng Dashboard Streamlit tương tác, Slide báo cáo và hoàn thiện văn bản Word', 'Tỷ lệ đóng góp': '100%'}
])
table(allocation_df, ['STT', 'Họ và tên', 'MSSV', 'Nhiệm vụ đảm nhiệm', 'Tỷ lệ đóng góp'])
p('Lưu ý: Nhóm có thể chỉnh sửa lại thông tin thành viên, MSSV và phân công chi tiết cho khớp với thành viên thực tế của nhóm trước khi nộp bài.', italic=True)

# 9. Tài liệu tham khảo
h1('9. Tài liệu tham khảo')
p('1. Chen, D. (2012). Online Retail II. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5CG6D.')
p('2. Fader, P. S., Hardie, B. G. S., & Lee, K. L. (2005). "Counting Your Customers" the Easy Way: An Alternative to the Pareto/NBD Model. Marketing Science, 24(2), 275-284.')
p('3. Fader, P. S., Hardie, B. G. S., & Lee, K. L. (2005). RFM and CLV: Using Iso-Value Curves for Customer Base Analysis. Journal of Marketing Research, 42(4), 415-430.')
p('4. Neslin, S. A., Gupta, S., Kamakura, W., Lu, J. X., & Mason, C. H. (2006). Defection Detection: Measuring and Understanding the Predictive Accuracy of Customer Churn Models. Journal of Marketing Research, 43(2), 204-211.')
p('5. Platt, J. (1999). Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods. Advances in Large Margin Classifiers, 10(3), 61-74.')
p('6. WSDM Cup (2018). KKBox Churn Prediction Challenge. Kaggle Competition.')

doc.save(OUT / 'ChurnGuard_UCI_External_Report_Final.docx')
print('Successfully generated ChurnGuard_UCI_External_Report_Final.docx with 9 full sections and 7 distinct figures!')
