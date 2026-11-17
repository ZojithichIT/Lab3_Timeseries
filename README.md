<div align="center">

# Lab 3: Dự Đoán Chuỗi Thời Gian Sử Dụng Machine Learning Models
### Chuyển đổi Supervised Learning (24 Lags), Huấn luyện XGBoost / Random Forest / Linear Regression, TimeSeriesSplit Tuning & Multi-step Forecasting

<p>
  <img src="https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter" alt="Jupyter Notebook" />
  <img src="https://img.shields.io/badge/Topic-Time--Series%20ML-brightgreen?style=for-the-badge" alt="Time-Series ML" />
  <img src="https://img.shields.io/badge/Model-XGBoost%20%7C%20RandomForest%20%7C%20LinearRegression-red?style=for-the-badge" alt="Models" />
  <img src="https://img.shields.io/badge/Library-XGBoost%20%7C%20Scikit--Learn%20%7C%20Pandas-purple?style=for-the-badge" alt="Libraries" />
</p>

| Thuộc tính | Chi tiết |
|---|---|
| **Notebook chính** | [`notebooks/main.ipynb`](notebooks/main.ipynb) (59 cells — Đầy đủ 7 bước cơ bản, 4 bài nâng cao & 3 câu hỏi thảo luận) |
| **Dữ liệu chính (Dataset)** | [Kaggle: Hourly Energy Consumption (AEP_hourly.csv)](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption?select=AEP_hourly.csv) (`data/raw/AEP_hourly.csv` — $121.296$ giờ liên tục, gần $14$ năm) |
| **Mô hình cốt lõi** | `XGBoost Regressor` (Baseline & Tuned), `Random Forest Regressor`, `Linear Regression` |
| **Tập đặc trưng chuẩn** | **24 Lags chu kỳ tự nhiên** (`lag_1` đến `lag_24`), mở rộng 30 đặc trưng (+ 6 Rolling Statistics) |
| **Kỹ thuật thẩm định** | Time-based Train/Val/Test Split ($75\% / 12,5\% / 12,5\%$), `TimeSeriesSplit` ($5$ Folds Expanding Window) |
| **Độ đo đánh giá** | MAE ($\text{MW}$), RMSE ($\text{MW}$), MAPE ($\%$), $R^2$ Score |

</div>

---

## 1. Giới Thiệu Dự Án

Dự án này tập trung vào phương pháp **Machine Learning ứng dụng cho bài toán dự báo chuỗi thời gian (Time Series Forecasting)**. Thay vì dựa vào các giả định phân phối chuẩn và tính dừng tuyến tính như các mô hình thống kê cổ điển (ARIMA/SARIMA), ta tiếp cận bài toán bằng cách biến đổi chuỗi thời gian đơn biến thành bài toán học có giám sát (**Supervised Learning**).

Mục tiêu chính bao gồm:
* **Feature Engineering chuẩn 24 Lags:** Tạo bộ 24 đặc trưng trễ chu kỳ tự nhiên (`lag_1..24`) bao trọn quán tính ngắn hạn và chu kỳ ngày–đêm ($24\text{h}$), mở rộng thêm các đặc trưng thống kê trượt (`Rolling Mean`, `Rolling Std` với `window = 6, 12, 24`).
* **Phân chia dữ liệu theo thời gian (Time-based Splitting):** Phân chia theo tỷ lệ $75\%$ Train ($90.954$ giờ) - $12,5\%$ Val ($15.159$ giờ) - $12,5\%$ Test ($15.159$ giờ), đảm bảo tính toàn vẹn nhân quả thời gian, chống rò rỉ dữ liệu (Data Leakage).
* **Mô hình hóa với Gradient Boosting (XGBoost):** Huấn luyện mô hình `XGBRegressor`, tinh chỉnh siêu tham số bằng `TimeSeriesSplit` (5 Folds) kết hợp Regularization (L1/L2) và Refit trên toàn bộ tập Development, đạt kết quả xuất sắc trên tập Test (**$\text{MAE} = 127,07\text{ MW}$, $\text{RMSE} = 171,74\text{ MW}$, $\text{MAPE} = 0,87\%$, $R^2 = 0,9948$**).
* **Đánh giá & Diễn giải mô hình:** Phân tích tầm quan trọng của đặc trưng (**Feature Importance** qua Gain metric) chứng minh vai trò áp đảo của quán tính tức thời (`lag_1`, `lag_2`) và chu kỳ ngày hôm trước (`lag_24`).
* **Mở rộng nâng cao & Đối chứng:** Thực hiện dự báo đa bước (**Multi-step Forecasting** $H = 24$ giờ: Direct vs Recursive), đọ sức giữa `XGBoost`, `Random Forest` và `Linear Regression`, khảo sát độ nhạy bậc trễ $p \in [1..336]$ và đối chiếu chuyên sâu 8 tiêu chí kỹ thuật với `ARIMA/SARIMA`.

---

## 2. Bảng Xếp Hạng Kết Quả Thực Nghiệm Trên Tập Test

| Hạng | Mô hình / Phương pháp | Thuật toán | Số lượng đặc trưng | Test MAE (MW) | Test RMSE (MW) | Test MAPE (%) | Hệ số $R^2$ |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|
| **🥇 1** | **`XGBoost Tuned (TS-Split + Refit)`** | XGBRegressor (Tuned Regularization) | 24 Lags | **$127,07$** | **$171,74$** | **$0,87\%$** | **$0,9948$** |
| **🥈 2** | **`XGBoost Baseline`** | XGBRegressor | 24 Lags | $132,37$ | $179,22$ | $0,91\%$ | $0,9943$ |
| **🥉 3** | **`XGBoost Enhanced`** | XGBRegressor (+ 6 Rolling Features) | 30 Features | $135,75$ | $182,86$ | $0,93\%$ | $0,9941$ |
| **4** | `Random Forest Regressor` | RandomForestRegressor (200 cây) | 24 Lags | $138,41$ | $188,67$ | $0,94\%$ | $0,9937$ |
| **5** | `Linear Regression (AR Tuyến tính)` | LinearRegression (OLS) | 24 Lags | $145,39$ | $195,04$ | $1,00\%$ | $0,9932$ |

---

## 3. Cấu Trúc Thư Mục

```text
Lab3_Timeseries/
├── README.md                                       # Trang chủ dự án hiển thị trên GitHub
├── requirements.txt                                # Thư viện phụ thuộc
├── .gitignore                                      # Loại bỏ file tạm và cache
├── data/
│   ├── raw/
│   │   └── AEP_hourly.csv                          # Dữ liệu phụ tải điện hàng giờ AEP (Dataset gốc)
│   └── processed/
│       ├── aep_supervised_24lags.csv               # Bảng dữ liệu học có giám sát chuẩn 24 Lags (121.272 dòng x 25 cột)
│       ├── aep_train.csv, aep_val.csv, aep_test.csv# Tập dữ liệu đầy đủ sau phân chia thời gian 75/12.5/12.5
│       ├── X_train.csv, X_val.csv, X_test.csv      # Ma trận 24 đặc trưng trễ độc lập
│       └── y_train.csv, y_val.csv, y_test.csv      # Vector nhãn mục tiêu phụ tải
├── notebooks/
│   └── main.ipynb                                  # Jupyter Notebook thực hành toàn diện (59 cells bám sát 100% đề bài)
├── scripts/                                        # Pipeline mã nguồn Python mô-đun hóa
│   ├── config.py                                   # Cấu hình tập trung tham số, đường dẫn, tỷ lệ chia
│   ├── data_loader.py                              # Xử lý nạp dữ liệu thô, khử trùng lặp DST, nội suy chuỗi đều
│   ├── feature_engineering.py                      # Tạo 24 lag features, rolling statistics, time-based split
│   ├── models.py                                   # Lớp Forecasters (XGBoost, RandomForest, LinearRegression)
│   ├── evaluation.py                               # Bộ tính toán độ đo MAE, RMSE, MAPE, R2, Residual Diagnostics
│   ├── visualization.py                            # Module vẽ biểu đồ chuẩn hóa xuất file tự động
│   ├── train.py                                    # Pipeline 4 Phase huấn luyện toàn bộ mô hình và xuất log
│   └── predict.py                                  # Pipeline suy luận độc lập & dự báo đa bước tương lai 24h
├── models/                                         # File lưu trữ siêu tham số, trọng số & kết quả từng mô hình (.json)
│   ├── xgboost_baseline_24lags.json                # Kết quả & booster JSON của XGBoost Baseline
│   ├── xgboost_enhanced_rolling.json               # Kết quả & booster JSON của XGBoost Enhanced
│   ├── xgboost_tuned_24lags.json                   # Kết quả & booster JSON của XGBoost Tuned (Best Model)
│   ├── random_forest_24lags.json                   # Kết quả, tham số & feature importances của Random Forest
│   └── linear_regression_24lags.json               # Kết quả, intercept & toàn bộ 24 hệ số beta của Linear Regression
├── logs/                                           # Nhật ký thực nghiệm & bảng xếp hạng kết quả
│   ├── pipeline_training.log                       # Toàn văn log quá trình huấn luyện 4 Phase chi tiết
│   ├── metrics_leaderboard.csv                     # Bảng xếp hạng 5 mô hình dạng CSV
│   ├── model_comparison_summary.txt                # Báo cáo so sánh tổng kết các mô hình
│   ├── inference.log                               # Log quá trình kiểm thử suy luận & dự báo
│   └── latest_predictions.csv                      # Bảng giá trị dự báo tương lai 24 giờ tiếp theo
└── docs/
    ├── report.md                                   # Báo cáo học thuật chi tiết toàn bộ dự án
    └── images/                                     # 15 biểu đồ độ phân giải cao (01 đến 15)
        ├── 01_aep_full_timeseries.png              # Toàn bộ chuỗi thời gian AEP (2004-2018)
        ├── 02_aep_zoomin_2weeks.png                # Phóng đại 2 tuần phụ tải chu kỳ ngày/đêm
        ├── 03_train_val_test_split.png             # Phân chia dữ liệu theo thời gian 75/12.5/12.5
        ├── 04_acf_pacf_analysis.png                # Đồ thị ACF & PACF xác lập 24 bậc trễ
        ├── 05_xgboost_learning_curves.png          # Đường cong học tập Train/Val RMSE của XGBoost
        ├── 06_predictions_vs_actual.png            # So sánh thực tế vs dự báo và phân tích phần dư
        ├── 07_feature_importance.png               # Tầm quan trọng 24 lags theo điểm số Gain
        ├── 08_rolling_features_analysis.png        # Phân tích đặc trưng Rolling Statistics
        ├── 09_multistep_forecasting.png            # Đối chiếu Recursive vs Direct Multi-Step 24h
        ├── 10_timeseriessplit_tuning.png           # Sơ đồ và kết quả kiểm định chéo TimeSeriesSplit
        ├── 11_xgboost_vs_random_forest.png         # Đối chiếu toàn diện Boosting vs Bagging (24 lags)
        ├── 12_xgboost_vs_linear_regression.png     # Đối chiếu phi tuyến vs tuyến tính (24 lags)
        ├── 13_lag_p_sensitivity_analysis.png       # Khảo sát độ nhạy khi tăng bậc trễ p (1..336)
        ├── 14_lab3_comprehensive_summary.png       # Bảng xếp hạng 5 mô hình & đối chiếu ARIMA vs XGBoost
        └── 15_model_comparison_metrics.png        # Biểu đồ cột so sánh sai số Test 5 mô hình (train.py)
```

---

## 4. Hướng Dẫn Cài Đặt & Thực Thi

### 4.1. Thiết lập môi trường
```bash
pip install -r requirements.txt
```

### 4.2. Khởi chạy thực nghiệm
* **Cách 1 — Chạy trực tiếp trên Jupyter Notebook:**
  ```bash
  jupyter notebook notebooks/main.ipynb
  ```
* **Cách 2 — Chạy toàn bộ pipeline huấn luyện 4 Phase tự động qua CLI:**
  ```bash
  python scripts/train.py
  ```
* **Cách 3 — Chạy suy luận độc lập & dự báo 24 giờ tương lai từ mô hình đã lưu:**
  ```bash
  python scripts/predict.py --model xgboost_tuned_24lags.json --horizon 24
  ```
