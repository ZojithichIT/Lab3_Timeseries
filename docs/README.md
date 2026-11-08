<div align="center">

# Lab 3: Dự Đoán Chuỗi Thời Gian Sử Dụng Machine Learning Models
### Chuyển đổi Supervised Learning, Lag Features, Rolling Statistics, Huấn luyện XGBoost / Random Forest, TimeSeriesSplit & Multi-step Forecasting

<p>
  <img src="https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter" alt="Jupyter Notebook" />
  <img src="https://img.shields.io/badge/Topic-Time--Series%20ML-brightgreen?style=for-the-badge" alt="Time-Series ML" />
  <img src="https://img.shields.io/badge/Model-XGBoost%20%7C%20RandomForest-red?style=for-the-badge" alt="Models" />
  <img src="https://img.shields.io/badge/Library-XGBoost%20%7C%20Scikit--Learn%20%7C%20Pandas-purple?style=for-the-badge" alt="Libraries" />
</p>

| Thuộc tính | Chi tiết |
|---|---|
| **Notebook chính** | [`notebooks/main.ipynb`](../notebooks/main.ipynb) (59 cells — Đầy đủ 7 bước cơ bản, 4 bài nâng cao & 3 câu thảo luận) |
| **Dữ liệu chính (Dataset)** | [Kaggle: Hourly Energy Consumption (AEP_hourly.csv)](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption?select=AEP_hourly.csv) (Lưu tại [`data/raw/AEP_hourly.csv`](../data/raw/AEP_hourly.csv) — $121.296$ giờ liên tục, $13,8$ năm) |
| **Mô hình cốt lõi** | `XGBoost Regressor` (Baseline & Tuned), `Random Forest Regressor`, `Linear Regression` |
| **Kỹ thuật thẩm định** | Time-based Train/Val/Test Split ($75\% / 12,5\% / 12,5\%$), `TimeSeriesSplit` ($5$ Folds Expanding Window) |
| **Độ đo đánh giá** | MAE ($\text{MW}$), RMSE ($\text{MW}$), MAPE ($\%$), $R^2$ Score |

</div>

---

## 1. Giới Thiệu Dự Án

Dự án này tập trung vào phương pháp **Machine Learning ứng dụng cho bài toán dự báo chuỗi thời gian (Time Series Forecasting)**. Thay vì dựa vào các giả định phân phối chuẩn và tính dừng tuyến tính như các mô hình thống kê cổ điển (ARIMA/SARIMA), ta tiếp cận bài toán bằng cách biến đổi chuỗi thời gian đơn biến thành bài toán học có giám sát (**Supervised Learning**).

Mục tiêu chính bao gồm:
* **Feature Engineering cho Time Series:** Tạo các đặc trưng trễ đa chu kỳ (`29 Selective Lags`: `lag_1..24`, `lag_48`, `lag_72`, `lag_167..169`) và các đặc trưng thống kê trượt (`Rolling Mean`, `Rolling Std` với `window = 6, 24, 168`).
* **Phân chia dữ liệu theo thời gian (Time-based Splitting):** Đảm bảo tính toàn vẹn nhân quả thời gian, tuyệt đối không xáo trộn (shuffle) tránh rò rỉ dữ liệu từ tương lai (Data Leakage).
* **Mô hình hóa với Gradient Boosting (XGBoost):** Huấn luyện mô hình `XGBRegressor`, tinh chỉnh siêu tham số bằng `TimeSeriesSplit` (5 Folds) và đạt kỷ lục dự báo trên tập Test (**$\text{MAE} = 123,28\text{ MW}$, $\text{RMSE} = 163,25\text{ MW}$, $\text{MAPE} = 0,84\%$, $R^2 = 0,9953$**).
* **Đánh giá & Diễn giải mô hình:** So sánh sai số dự báo và phân tích tầm quan trọng của đặc trưng (**Feature Importance** qua Gain, Weight, Permutation Importance) để giải mã vai trò của từng bước trễ thời gian.
* **Mở rộng nâng cao & Thực nghiệm lý thuyết:** Thực hiện kỹ thuật dự báo đa bước (**Multi-step Forecasting** $H = 24$ giờ: Direct vs Recursive), đọ sức giữa `XGBoost`, `Random Forest`, `Linear Regression`, khảo sát độ nhạy bậc trễ $p \in [1..336]$ và đối chiếu chuyên sâu với `ARIMA/SARIMA`.

---

## 2. Cấu Trúc Thư Mục

```text
Lab3_TS/
├── data/
│   ├── raw/
│   │   └── AEP_hourly.csv                          # Dữ liệu tiêu thụ điện hàng giờ AEP (Dataset chính)
│   └── processed/
│       ├── aep_train.csv, aep_val.csv, aep_test.csv# Dữ liệu đã phân chia Train (75%) - Val (12.5%) - Test (12.5%)
│       └── X_train.csv, y_train.csv ...            # Ma trận đặc trưng và nhãn mục tiêu tách rời
├── notebooks/
│   └── main.ipynb                                  # Jupyter Notebook thực hành toàn diện (59 cells hoàn chỉnh)
├── scripts/                                        # Module Python đóng gói chuẩn mực (Class, Models, Configs)
│   ├── __init__.py                                 # Package init
│   ├── config.py                                   # Quản lý siêu tham số, tỷ lệ chia, đường dẫn
│   ├── data_loader.py                              # Xử lý nạp dữ liệu thô, DST, interpolate
│   ├── feature_engineering.py                      # Tạo lag features, rolling statistics, time-based split
│   ├── models.py                                   # Các Class mô hình (XGBoostForecaster, RandomForest...)
│   ├── evaluation.py                               # Độ đo đánh giá (MAE, RMSE, MAPE, Residual Diagnostics)
│   ├── visualization.py                            # Các hàm trực quan hóa và tự động lưu biểu đồ
│   └── train.py                                    # Script chạy toàn bộ pipeline tự động từ dòng lệnh (CLI)
└── docs/
    ├── README.md                                   # Trang chủ tài liệu & chỉ mục
    ├── 01_data_preparation_and_lag_features.md     # Tài liệu: Chuẩn bị dữ liệu & Kỹ thuật tạo Lag Features (Bước 1-2)
    ├── 02_feature_matrix_and_time_split.md         # Tài liệu: Phân tách X-y & Chia Train - Val - Test (Bước 3-4)
    ├── 03_xgboost_modeling_and_evaluation.md       # Tài liệu: Huấn luyện XGBoost, Đánh giá & Feature Importance (Bước 5-7)
    ├── 03_advanced_tuning_and_multistep.md         # Tài liệu: 4 Bài tập Nâng cao (Rolling, Multi-step, TimeSeriesSplit, RF)
    ├── 04_model_comparisons_and_insights.md        # Tài liệu: 3 Câu hỏi Thảo luận (LR, Độ nhạy bậc trễ p, ARIMA vs XGBoost)
    ├── report.md                                   # Báo cáo tổng hợp kết quả thực nghiệm hoàn chỉnh
    └── images/                                     # 15 biểu đồ trực quan hóa độ phân giải cao (01_.. -> 14_..)
```

---

## 3. Danh Mục Tài Liệu Kỹ Thuật (Đã Hoàn Thành 100%)

* 📘 **[01. Chuẩn Bị Dữ Liệu & Biến Đổi Supervised Learning (`01_data_preparation_and_lag_features.md`)](01_data_preparation_and_lag_features.md)**  
  Xử lý bất thường lưới thời gian DST của `AEP_hourly.csv` ($121.296$ giờ), phân tích tự tương quan ACF/PACF và thiết kế $29$ đặc trưng trễ đa chu kỳ (`lag_1..24`, `48`, `72`, `167..169`).

* 🗂️ **[02. Phân Tách Ma Trận $X - y$ & Chia Tập Dữ Liệu Theo Thời Gian (`02_feature_matrix_and_time_split.md`)](02_feature_matrix_and_time_split.md)**  
  Tách ma trận $\mathbf{X}$ ($121.127 \times 29$) và $\mathbf{y}$, phân chia tuần tự theo thời gian tỷ lệ $75\% / 12,5\% / 12,5\%$ chống rò rỉ dữ liệu và lưu trữ tại `data/processed/`.

* 📊 **[03A. Huấn Luyện, Đánh Giá Mô Hình XGBoost & Giải Mã Feature Importance (`03_xgboost_modeling_and_evaluation.md`)](03_xgboost_modeling_and_evaluation.md)**  
  Phân tích quá trình hội tụ của `XGBRegressor`, đánh giá sai số trên tập Test ($\text{MAE} = 125,32\text{ MW}, \text{RMSE} = 165,39\text{ MW}, \text{MAPE} = 0,86\%$), chẩn đoán phần dư và giải mã Feature Importance qua 3 thước đo (`Gain`, `Weight`, `Permutation Importance`).

* ⚙️ **[03B. Bốn Bài Tập Nâng Cao: Rolling Features, Multi-step, TimeSeriesSplit & Random Forest (`03_advanced_tuning_and_multistep.md`)](03_advanced_tuning_and_multistep.md)**  
  Chi tiết 4 kỹ thuật nâng cao: bổ sung 6 biến cửa sổ trượt (`window = 6, 24, 168`), dự báo đa bước $H = 24$ giờ (Direct giảm $13,86\%$ RMSE so với Recursive), tinh chỉnh siêu tham số bằng `TimeSeriesSplit` 5 Folds (đạt $\text{RMSE} = 163,25\text{ MW}$, $\text{MAPE} = 0,84\%$), và đối chiếu Bagging (`RandomForestRegressor`) vs Boosting (`XGBRegressor`).

* 📈 **[04. So Sánh Mô Hình & Thực Nghiệm Lý Thuyết (`04_model_comparisons_and_insights.md`)](04_model_comparisons_and_insights.md)**  
  Giải đáp chuyên sâu kèm thực nghiệm chứng minh cho 3 câu hỏi thảo luận:
  1. **XGBoost vs Linear Regression:** Phân tích hệ số $\beta_k$ và ưu thế học chuyển pha phi tuyến của XGBoost.
  2. **Tác động khi tăng số lag $p$ ($p \in [1..336]$):** Đường cong đánh đổi Độ chệch – Phương sai – Chi phí tính toán và ưu thế của `29 Selective Lags`.
  3. **Khi nào XGBoost phù hợp hơn ARIMA:** Đối chiếu 8 tiêu chí kỹ thuật và kiến trúc lai `Hybrid ARIMA + XGBoost`.

* 📑 **[Báo Cáo Tổng Hợp Toàn Dự Án (`report.md`)](report.md)**  
  Báo cáo học thuật hoàn chỉnh tổng kết toàn bộ quy trình 7 bước cơ bản, 4 bài tập nâng cao, 3 câu hỏi nghiên cứu và bảng xếp hạng 10 mô hình/phương pháp.

---

## 4. Hướng Dẫn Cài Đặt & Thực Thi

### 4.1. Thiết lập môi trường
```bash
pip install numpy pandas matplotlib seaborn statsmodels scikit-learn xgboost
```

### 4.2. Khởi chạy thực nghiệm
* **Cách 1 — Chạy trực tiếp trên Jupyter Notebook:**
  ```bash
  jupyter notebook notebooks/main.ipynb
  ```
* **Cách 2 — Chạy pipeline tự động qua CLI:**
  ```bash
  python scripts/train.py
  ```
