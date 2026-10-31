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
| **Notebook chính** | [`scripts/main.ipynb`](scripts/main.ipynb) |
| **Dữ liệu chính (Dataset)** | [Kaggle: Hourly Energy Consumption (AEP_hourly.csv)](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption?select=AEP_hourly.csv) (Lưu tại [`data/AEP_hourly.csv`](data/AEP_hourly.csv) - ~121k dòng, 10+ năm) |
| **Mô hình cốt lõi** | XGBoost Regressor, Random Forest Regressor, Linear Regression |
| **Kỹ thuật thẩm định** | Time-based Train/Test Split (không shuffle), TimeSeriesSplit (Cross Validation) |
| **Độ đo đánh giá** | MAE, RMSE, MAPE |

</div>

---

## 1. Giới Thiệu Dự Án

Dự án này tập trung vào phương pháp **Machine Learning ứng dụng cho bài toán dự báo chuỗi thời gian (Time Series Forecasting)**. Thay vì dựa vào các giả định phân phối và tính dừng tuyến tính như các mô hình thống kê cổ điển (ARIMA/SARIMA), ta tiếp cận bài toán bằng cách biến đổi chuỗi thời gian đơn biến thành bài toán học có giám sát (**Supervised Learning**).

Mục tiêu chính bao gồm:
* **Feature Engineering cho Time Series:** Tạo các đặc trưng trễ (Lag features $y_{t-1}, y_{t-2}, \dots, y_{t-p}$), các đặc trưng thống kê trượt (Rolling Mean, Rolling Std).
* **Phân chia dữ liệu theo thời gian (Time-based Splitting):** Đảm bảo tính toàn vẹn thời gian, tuyệt đối không xáo trộn (shuffle) tránh rò rỉ dữ liệu từ tương lai (Data Leakage).
* **Mô hình hóa với Gradient Boosting (XGBoost):** Huấn luyện mô hình `XGBRegressor`, tinh chỉnh siêu tham số bằng `TimeSeriesSplit`.
* **Đánh giá & Diễn giải mô hình:** So sánh sai số dự báo (MAE, RMSE, MAPE) và phân tích tầm quan trọng của đặc trưng (**Feature Importance**) để làm rõ vai trò của từng bước trễ thời gian.
* **Mở rộng nâng cao:** Thực hiện kỹ thuật dự báo đa bước (**Multi-step Forecasting**), so sánh hiệu năng giữa XGBoost, Random Forest, Linear Regression và phân tích so sánh chuyên sâu với ARIMA.

---

## 2. Cấu Trúc Thư Mục

```text
Lab3_TS/
├── data/
│   ├── AEP_hourly.csv                              # Dữ liệu tiêu thụ điện hàng giờ AEP (Dataset chính)
│   ├── COMED_hourly.csv                            # Dữ liệu đối chứng vùng Commonwealth Edison
│   ├── DAYTON_hourly.csv                           # Dữ liệu đối chứng Dayton
│   └── ...                                         # Các dataset phụ tải PJM khác
├── scripts/
│   └── main.ipynb                                  # Jupyter Notebook thực hành toàn diện (Các bước 1 -> 7 & Nâng cao)
└── docs/
    ├── README.md                                   # Trang chủ tài liệu & chỉ mục
    ├── 01_data_preparation_and_lag_features.md     # Tài liệu: Chuẩn bị dữ liệu & Kỹ thuật tạo Lag / Rolling Features
    ├── 02_feature_matrix_and_time_split.md         # Tài liệu: Phân tách X-y & Chia Train (75%) - Val (12.5%) - Test (12.5%)
    ├── 03_xgboost_modeling_and_evaluation.md       # Tài liệu: Huấn luyện XGBoost, Đánh giá sai số & Feature Importance
    ├── 03_advanced_tuning_and_multistep.md         # Tài liệu: TimeSeriesSplit, Tuning Hyperparameters & Multi-step
    ├── 04_model_comparisons_and_insights.md        # Tài liệu: So sánh XGBoost vs Random Forest vs Linear Regression vs ARIMA
    ├── report.md                                   # Báo cáo tổng hợp kết quả thực nghiệm hoàn chỉnh
    └── images/                                     # Biểu đồ và hình ảnh trực quan hóa
```

---

## 3. Danh Mục Nhiệm Vụ & Tài Liệu Dự Kiến

* 📘 **01. Chuẩn Bị Dữ Liệu & Biến Đổi Supervised Learning (Data Prep & Lag Features)**  
  Load dữ liệu [AEP_hourly.csv](file:///e:/TimeSeries/Lab3_TS/data/AEP_hourly.csv), sắp xếp và xử lý timeline. Chuyển bài toán $y_t$ thành học có giám sát bằng kỹ thuật trượt cửa sổ thời gian (Lag Features $p=24$ hoặc $p=48$). Tạo các đặc trưng bổ trợ: Rolling Mean, Rolling Standard Deviation.

* 📊 **02. Huấn Luyện & Đánh Giá Mô Hình XGBoost (XGBoost Modeling & Feature Importance)**  
  Tách tập $X - y$, phân chia Train/Test theo mốc thời gian (ví dụ 80% quá khứ làm Train, 20% gần nhất làm Test). Fit `XGBRegressor`, dự báo trên tập Test, đo lường chất lượng bằng MAE, RMSE, MAPE. Trích xuất và trực quan hóa Feature Importance của các lag để xác định các mốc trễ có sức ảnh hưởng mạnh nhất (chu kỳ 24h, 168h...).

* ⚙️ **03. Tối Ưu Siêu Tham Số & Dự Báo Đa Bước (Hyperparameter Tuning & Multi-step Forecasting)**  
  Sử dụng `TimeSeriesSplit` (K-fold theo thứ tự thời gian) để cross-validate và tune các siêu tham số chính: `n_estimators`, `max_depth`, `learning_rate`, `subsample`. Triển khai chiến lược dự báo nhiều bước tương lai (**Multi-step Forecasting**) thông qua Recursive Strategy hoặc Direct Strategy.

* 📈 **04. So Sánh Mô Hình & Đúc Kết Lý Thuyết (Model Comparisons & Theoretical Insights)**  
  Huấn luyện đối chứng `RandomForestRegressor` và `LinearRegression`. Phân tích so sánh toàn diện:
  1. **XGBoost vs Linear Regression:** Năng lực nắm bắt mối quan hệ phi tuyến tính và tương tác phức tạp.
  2. **Tác động của số bậc lag $p$:** Hiện tượng Overfitting, hiện tượng đa cộng tuyến (Multicollinearity), và sự suy giảm kích thước mẫu do `shift()`.
  3. **XGBoost vs ARIMA:** Khi nào nên dùng XGBoost (dữ liệu lớn, phi tuyến tính, nhiều features ngoại sinh) và khi nào ARIMA vượt trội (dữ liệu nhỏ, đơn biến, quan hệ tuyến tính chặt chẽ).

* 📑 **Báo Cáo Tổng Hợp (report.md)**  
  Báo cáo học thuật chuẩn mực tổng kết toàn bộ quy trình, bảng so sánh định lượng các mô hình, biểu đồ dự báo thực tế và lời giải chi tiết cho các câu hỏi nghiên cứu.

---

## 4. Hướng Dẫn Cài Đặt & Thực Thi

### 4.1. Thiết lập môi trường

Cài đặt các thư viện cần thiết phục vụ Machine Learning cho chuỗi thời gian:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

### 4.2. Khởi chạy thực nghiệm
Khởi chạy Jupyter Notebook để chạy từng bước thực nghiệm:
```bash
jupyter notebook scripts/main.ipynb
```
