# BÁO CÁO THỰC HÀNH BÀI 3: DỰ ĐOÁN CHUỖI THỜI GIAN SỬ DỤNG MACHINE LEARNING MODELS

> **Dataset:** Hourly Energy Consumption (`AEP_hourly.csv` — PJM Interconnection)  
> **Topic:** Chuyển đổi Supervised Learning (24 Lags), Huấn luyện XGBoost / Random Forest / Linear Regression, TimeSeriesSplit Tuning & Multi-step Forecasting  
> **Mã nguồn chính:** [`notebooks/main.ipynb`](notebooks/main.ipynb) (59 cells) | [`scripts/`](scripts/) (Pipeline hoàn chỉnh)

---

## BẢNG XẾP HẠNG HIỆU NĂNG TỔNG KẾT TRÊN TẬP TEST

| Hạng | Tên mô hình | Thuật toán | Số đặc trưng | Train Time (s) | Test MAE (MW) | Test RMSE (MW) | Test MAPE (%) | Hệ số $R^2$ |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 **1** | **`XGBoost Tuned (TS-Split + Refit)`** | XGBRegressor (Tuned Regularization) | 24 Lags | 26.59s | **127.07** | **171.74** | **0.87%** | **0.9948** |
| 🥈 **2** | **`XGBoost Baseline`** | XGBRegressor | 24 Lags | 22.52s | 132.37 | 179.22 | 0.91% | 0.9943 |
| 🥉 **3** | **`XGBoost Enhanced`** | XGBRegressor (+ 6 Rolling Statistics) | 30 Features | 29.73s | 135.75 | 182.86 | 0.93% | 0.9941 |
| 4 | **`Random Forest Regressor`** | RandomForestRegressor (200 cây) | 24 Lags | 135.88s | 138.41 | 188.67 | 0.94% | 0.9937 |
| 5 | **`Linear Regression (AR Tuyến tính)`** | LinearRegression (OLS) | 24 Lags | 0.12s | 145.39 | 195.04 | 1.00% | 0.9932 |

---

## I. MỤC TIÊU ĐỀ BÀI
1. **Chuyển bài toán chuỗi thời gian $\rightarrow$ Học có giám sát (Supervised Learning):** Biến đổi chuỗi đơn biến $y_t$ thành bài toán hồi quy hàm mục tiêu $y_t = f(y_{t-1}, y_{t-2}, \dots, y_{t-p})$.
2. **Huấn luyện mô hình XGBoost Regressor:** Tận dụng thuật toán Gradient Boosted Decision Trees với các kỹ thuật Regularization và Early Stopping để tối ưu dự báo.
3. **Đánh giá chất lượng dự báo:** Sử dụng hệ thống độ đo toàn diện (MAE, RMSE, MAPE, $R^2$) kết hợp phân tích chẩn đoán phần dư (Residual Diagnostics).

---

## II. BÀI TẬP THỰC HÀNH (7 BƯỚC CƠ BẢN)

### Bước 1. Chuẩn bị dữ liệu
* **Mô tả & Xử lý:** Bộ dữ liệu gốc có $121.273$ dòng. Xử lý triệt để $4$ mốc giờ bị trùng lặp do đổi giờ mùa hè (DST Autumn) bằng cách lấy trung bình (`groupby.mean()`), tái lập lưới thời gian đều đặn hàng giờ (`asfreq('h')`) và bù đắp $27$ mốc khuyết bằng nội suy tuyến tính theo thời gian (`interpolate(method='time')`).
* **Kết quả:** Chuỗi dữ liệu liên tục gồm **$121.296$ giờ** (từ `2004-10-01 01:00` đến `2018-08-03 00:00`), không còn giá trị khuyết thiếu.
* **Vị trí tài nguyên:**
  * Code: `notebooks/main.ipynb` (Cells 1 – 6) | `scripts/data_loader.py`
  * Dữ liệu: `data/raw/AEP_hourly.csv`
  * Hình ảnh: `docs/images/01_aep_full_timeseries.png`, `docs/images/02_aep_zoomin_2weeks.png`

### Bước 2. Tạo lag features (Supervised Learning)
* **Cơ sở khoa học:** Đồ thị tự tương quan (ACF/PACF) xác nhận độ tương quan vượt trội tại chu kỳ **24 giờ** (chu kỳ ngày-đêm). Do đó, thiết lập chuẩn **$p = 24$ bậc trễ liên tục** (`lag_1` đến `lag_24`).
* **Kỹ thuật:** Dịch trễ chuỗi bằng hàm `shift(i)` với $i \in [1, 24]$.
* **Vị trí tài nguyên:**
  * Code: `notebooks/main.ipynb` (Cells 7 – 12) | `scripts/feature_engineering.py`
  * Hình ảnh: `docs/images/04_acf_pacf_analysis.png`

### Bước 3. Chia $X - y$
* **Mô tả:** Tách cột mục tiêu $y$ (`AEP_MW` tại thời điểm $t$) và ma trận đặc trưng $X$ gồm 24 cột trễ (`lag_1` đến `lag_24`). Loại bỏ 24 dòng đầu chứa `NaN` do dịch trễ.
* **Kích thước ma trận:** $X \in \mathbb{R}^{121.272 \times 24}$ và $y \in \mathbb{R}^{121.272}$.
* **Vị trí tài nguyên:**
  * Code: `notebooks/main.ipynb` (Cells 13 – 15) | `scripts/feature_engineering.py`
  * Dữ liệu: `data/processed/aep_supervised_24lags.csv`, `X_train.csv`, `y_train.csv`

### Bước 4. Chia train / test theo thời gian
* **Nguyên tắc cốt lõi:** Tuyệt đối không xáo trộn dữ liệu (No Shuffling) để bảo toàn tính nhân quả thời gian.
* **Tỷ lệ phân chia:** **75% Train** ($90.845$ giờ) — **12.5% Val** ($15.140$ giờ) — **12.5% Test** ($15.142$ giờ).
* **Vị trí tài nguyên:**
  * Code: `notebooks/main.ipynb` (Cells 16 – 19) | `scripts/feature_engineering.py`
  * Hình ảnh: `docs/images/03_train_val_test_split.png`

### Bước 5. Huấn luyện mô hình XGBoost
* **Cấu hình siêu tham số Baseline:**
  `n_estimators=1000`, `learning_rate=0.05`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `early_stopping_rounds=50`.
* **Cơ chế:** Dùng tập Validation để kích hoạt Early Stopping, ngăn chặn quá khớp (mô hình tự động dừng ở vòng lặp ~280 - 320 thay vì chạy hết 1000 cây). Thời gian train: **22.52 giây**.
* **Vị trí tài nguyên:**
  * Code: `notebooks/main.ipynb` (Cells 20 – 23) | `scripts/models.py`
  * Mô hình lưu trữ: `models/xgboost_baseline_24lags.json`
  * Hình ảnh: `docs/images/05_xgboost_learning_curves.png`

### Bước 6. Dự đoán và đánh giá
* **Kết quả định lượng mô hình Baseline:**
  * **Test MAE:** $132.37\text{ MW}$
  * **Test RMSE:** $179.22\text{ MW}$
  * **Test MAPE:** $0.91\%$
  * **Hệ số $R^2$:** $0.9943$
* **Chẩn đoán phần dư:** Phân phối sai số chuẩn đối xứng quanh $0$ (Mean bias $\approx -5.61\text{ MW}$), $92.3\%$ số điểm dự báo có sai số nhỏ hơn $2\%$.
* **Vị trí tài nguyên:**
  * Code: `notebooks/main.ipynb` (Cells 24 – 27) | `scripts/evaluation.py`
  * Hình ảnh: `docs/images/06_predictions_vs_actual.png`

### Bước 7. Phân tích Feature Importance
* **Kết quả đánh giá theo Gain (% đóng góp giảm sai số):**
  1. `lag_1` chiếm **$67.52\%$** tầm quan trọng: Điểm neo mức công suất tức thời gần nhất.
  2. `lag_2` chiếm **$14.64\%$**: Kết hợp với `lag_1` hình thành vector quán tính và gia tốc phụ tải.
  3. `lag_24` (và `lag_23`, `lag_22`) chiếm **$9.08\%$**: Chu kỳ cùng giờ ngày hôm trước, tín hiệu quan trọng đón đầu đỉnh sáng và đỉnh tối.
* **Vị trí tài nguyên:**
  * Code: `notebooks/main.ipynb` (Cells 28 – 31)
  * Hình ảnh: `docs/images/07_feature_importance.png`

---

## III. BÀI TẬP NÂNG CAO

### 1. Thêm rolling mean / rolling std
* **Cách thực hiện:** Tạo 6 đặc trưng bổ trợ qua 3 kích thước cửa sổ đại diện: $6\text{h}$ (buổi/ca làm việc), $12\text{h}$ (nửa ngày), $24\text{h}$ (chu kỳ ngày).
* **Quy tắc chống rò rỉ dữ liệu:** Áp dụng `.shift(1)` trước khi tính rolling:
  `df['rolling_mean_24'] = series.shift(1).rolling(24).mean()`
* **Ý nghĩa:** `rolling_mean` đo mức phụ tải nền (Baseline Trend), `rolling_std` đo mức độ biến động (phân biệt ngày thường dao động mạnh với cuối tuần dao động phẳng).
* **Vị trí tài nguyên:**
  * Code: `notebooks/main.ipynb` (Cells 32 – 35) | `scripts/feature_engineering.py`
  * Mô hình lưu trữ: `models/xgboost_enhanced_rolling.json`
  * Hình ảnh: `docs/images/08_rolling_features_analysis.png`

### 2. Dự báo multi-step
* **Bài toán:** Dự báo liên tục cho $H = 24$ giờ tương lai (một ngày tiếp theo).
* **So sánh hai chiến lược:**
  * **Recursive Multi-Step (Đệ quy):** Dùng 1 mô hình duy nhất, lấy giá trị dự báo làm lag giả lập cho bước tiếp theo. Nhược điểm: Tích lũy sai số (*Error Accumulation*), RMSE trung bình 24h là $1034.83\text{ MW}$.
  * **Direct Multi-Step (Trực tiếp):** Huấn luyện 24 mô hình độc lập tương ứng với từng bước $t+1, \dots, t+24$. Kết quả: RMSE trung bình giảm xuống **$891.40\text{ MW}$** (vượt trội hơn **$13.86\%$** so với Recursive).
* **Vị trí tài nguyên:**
  * Code: `notebooks/main.ipynb` (Cells 36 – 39) | `scripts/predict.py`
  * Hình ảnh: `docs/images/09_multistep_forecasting.png`

### 3. Tune hyperparameter bằng TimeSeriesSplit
* **Kỹ thuật:** Áp dụng `TimeSeriesSplit(n_splits=5)` với cơ chế Expanding Window (cửa sổ mở rộng) trên tập Development ($105.985$ giờ), đảm bảo không vi phạm thứ tự thời gian.
* **Cấu hình tối ưu được chọn:**
  `max_depth=6`, `learning_rate=0.05`, `subsample=0.85`, `colsample_bytree=0.85`, `min_child_weight=3`, `reg_alpha=0.1`, `reg_lambda=1.5`.
* **Hiệu năng sau khi Refit (Top 1 Bảng xếp hạng):**
  * **Test MAE:** **$127.07\text{ MW}$** (giảm $5.30\text{ MW}$ so với Baseline)
  * **Test RMSE:** **$171.74\text{ MW}$** (giảm $7.48\text{ MW}$ so với Baseline)
  * **Test MAPE:** **$0.87\%$** | **$R^2$:** **$0.9948$**
* **Vị trí tài nguyên:**
  * Code: `notebooks/main.ipynb` (Cells 40 – 43)
  * Mô hình lưu trữ: `models/xgboost_tuned_24lags.json`
  * Hình ảnh: `docs/images/10_timeseriessplit_tuning.png`

### 4. So sánh với Random Forest
* **Đối chiếu thuật toán:**
  * `XGBoost` (Boosting - xây cây tuần tự học từ phần dư) vs `Random Forest` (Bagging - xây 200 cây song song lấy trung bình).
* **Kết quả thực nghiệm:**
  * **Tốc độ:** XGBoost ($22.52\text{s}$) huấn luyện **nhanh gấp $6.0$ lần** so với Random Forest ($135.88\text{s}$).
  * **Độ chính xác:** XGBoost đạt RMSE $179.22\text{ MW}$, vượt trội hơn Random Forest ($188.67\text{ MW}$) nhờ cơ chế tối ưu hàm mất mát trực tiếp và độ sâu cây linh hoạt.
* **Vị trí tài nguyên:**
  * Code: `notebooks/main.ipynb` (Cells 44 – 47) | `scripts/models.py`
  * Mô hình lưu trữ: `models/random_forest_24lags.json`
  * Hình ảnh: `docs/images/11_xgboost_vs_random_forest.png`

---

## IV. CÂU HỎI CẦN TRẢ LỜI

### 1. So sánh XGBoost với Linear Regression
* **Thực nghiệm:** Huấn luyện mô hình hồi quy tuyến tính OLS trên cùng 24 đặc trưng trễ.
* **Kết quả:**
  * `Linear Regression`: Huấn luyện siêu tốc ($0.12\text{s}$), đạt Test RMSE = $195.04\text{ MW}$ nhờ phương trình $y_t \approx 1.62 y_{t-1} - 0.76 y_{t-2}$. Tuy nhiên, mô hình tuyến tính bị cố định hệ số toàn cục, phát sinh sai số lớn tại các thời điểm chuyển giao chế độ tải cao điểm/thấp điểm.
  * `XGBoost`: Đạt Test RMSE = $179.22\text{ MW}$ (Tuned: $171.74\text{ MW}$), vượt trội nhờ khả năng phân nhánh phi tuyến tính và bắt được các mối quan hệ phi đối xứng theo thời gian.
* **Vị trí tài nguyên:**
  * Code: `notebooks/main.ipynb` (Cells 48 – 51)
  * Mô hình lưu trữ: `models/linear_regression_24lags.json`
  * Hình ảnh: `docs/images/12_xgboost_vs_linear_regression.png`

### 2. Điều gì xảy ra khi tăng số lag $p$?
* **Thực nghiệm khảo sát 9 mốc:** $p \in \{1, 3, 6, 12, 24, 48, 72, 168, 336\}$.
* **Hiện tượng rút ra:**
  1. **Giai đoạn $p = 1 \rightarrow 24$:** Sai số giảm mạnh (RMSE hạ từ $511.21\text{ MW}$ xuống $179.22\text{ MW}$) do mô hình thu nạp đủ thông tin của một chu kỳ ngày hoàn chỉnh.
  2. **Giai đoạn $p = 48 \rightarrow 336$:** Lợi ích cận biên giảm rõ rệt. Xảy ra hiện tượng **đa cộng tuyến**, khoảng cách quá khớp (*Overfitting Gap*) tăng lên $+36.92\text{ MW}$, mất nhiều dòng dữ liệu ban đầu do `dropna()`, và thời gian huấn luyện tăng gần 40 lần ($0.96\text{s} \rightarrow 37.55\text{s}$).
  3. **Kết luận:** Lựa chọn **$p = 24$** là điểm cân bằng tối ưu giữa độ chính xác, tốc độ tính toán và khả năng kiểm soát quá khớp.
* **Vị trí tài nguyên:**
  * Code: `notebooks/main.ipynb` (Cells 52 – 55)
  * Hình ảnh: `docs/images/13_lag_p_sensitivity_analysis.png`

### 3. Khi nào XGBoost phù hợp hơn ARIMA?
* **XGBoost phù hợp hơn ARIMA/SARIMA khi:**
  * Dữ liệu có kích thước lớn ($N > 10.000$ mẫu).
  * Chuỗi thời gian chứa **đa chu kỳ lồng nhau** (Chu kỳ ngày 24h + Chu kỳ tuần 168h + Chu kỳ mùa hàng năm).
  * Mối quan hệ giữa các biến có tính chất **phi tuyến tính phức tạp**.
  * Cần bổ sung các **biến ngoại sinh phong phú** (Nhiệt độ, độ ẩm, ngày lễ, chỉ số kinh tế).
* **ARIMA phù hợp hơn XGBoost khi:**
  * Dữ liệu chuỗi đơn biến ngắn ($N < 500$ mẫu).
  * Chuỗi có **xu hướng tăng/giảm mạnh vượt đỉnh lịch sử (*Trend Extrapolation*)**, do cây quyết định của XGBoost bị chặn cận trên/dưới trong khoảng giá trị đã thấy ở tập Train.
* **Vị trí tài nguyên:**
  * Code: `notebooks/main.ipynb` (Cells 56 – 58)
  * Hình ảnh: `docs/images/14_lab3_comprehensive_summary.png`

---

## V. DATASET & HƯỚNG DẪN ĐỊNH VỊ THƯ MỤC / FOLDER

### 1. Thông tin Dataset
* **Tên tệp:** `AEP_hourly.csv` (American Electric Power Service Corp).
* **Nguồn dữ liệu:** [Kaggle - Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption?select=AEP_hourly.csv).
* **Đặc tính:** Chuỗi thời gian đo đạc công suất tải điện (Megawatt - MW) từ 2004 đến 2018 (hơn 10 năm liên tục).

### 2. Sơ đồ cấu trúc thư mục & định vị tài nguyên
```text
Lab3_Timeseries/
│
├── README.md                           <- Báo cáo tổng hợp toàn bài theo format chuẩn đề bài
├── requirements.txt                    <- Danh sách các thư viện Python (xgboost, scikit-learn,...)
│
├── data/
│   ├── raw/
│   │   └── AEP_hourly.csv              <- Tệp dữ liệu gốc tải từ Kaggle
│   └── processed/
│       ├── aep_supervised_24lags.csv   <- Bảng dữ liệu học có giám sát 24 Lags (121.272 x 25)
│       ├── X_train.csv, X_val.csv, X_test.csv  <- Ma trận 24 đặc trưng trễ sau phân chia
│       └── y_train.csv, y_val.csv, y_test.csv  <- Nhãn công suất mục tiêu tương ứng
│
├── notebooks/
│   └── main.ipynb                      <- Notebook hoàn chỉnh 59 cells (7 Bước + 4 Nâng cao + 3 Thảo luận)
│
├── scripts/
│   ├── config.py                       <- Quản lý tập trung siêu tham số, tỷ lệ split, đường dẫn
│   ├── data_loader.py                  <- Xử lý DST, nội suy chuỗi đều đặn 121.296 giờ
│   ├── feature_engineering.py          <- Hàm tạo 24 Lags, Rolling Features, Time Split
│   ├── models.py                       <- Đóng gói các lớp Forecaster (XGB, RF, LR)
│   ├── evaluation.py                   <- Module tính MAE, RMSE, MAPE, R2, Residual Bias
│   ├── visualization.py                <- Module vẽ biểu đồ chuẩn hóa xuất ảnh tự động
│   ├── train.py                        <- Pipeline tự động hóa huấn luyện 4 Phase
│   └── predict.py                      <- Pipeline suy luận & dự báo đa bước H=24
│
├── models/                             <- Tệp JSON lưu trữ trọng số và siêu tham số mô hình
│   ├── xgboost_baseline_24lags.json    <- Mô hình XGBoost cơ bản (Bước 5)
│   ├── xgboost_enhanced_rolling.json   <- Mô hình bổ sung Rolling Features (Nâng cao 1)
│   ├── xgboost_tuned_24lags.json       <- Mô hình tinh chỉnh TimeSeriesSplit tối ưu (Nâng cao 3)
│   ├── random_forest_24lags.json       <- Mô hình Random Forest đối chứng (Nâng cao 4)
│   └── linear_regression_24lags.json   <- Mô hình hồi quy tuyến tính đối chứng (Thảo luận 1)
│
├── logs/
│   ├── metrics_leaderboard.csv         <- Bảng xếp hạng định lượng xuất tự động từ code
│   ├── pipeline_training.log           <- Nhật ký huấn luyện chi tiết 4 Phase
│   ├── latest_predictions.csv          <- Giá trị dự báo 24 giờ tương lai của mô hình tốt nhất
│   └── model_comparison_summary.txt    <- Bản tóm tắt so sánh chỉ số giữa các mô hình
│
└── docs/
    ├── report.md                       <- Báo cáo học thuật mở rộng
    └── images/                         <- 15 ảnh kết quả trực quan hóa phân giải cao (01 -> 15)
```

---

## VI. HƯỚNG DẪN THỰC THI NHANH (QUICK START)

1. **Cài đặt thư viện:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Chạy thực nghiệm trên Jupyter Notebook:**
   ```bash
   jupyter notebook notebooks/main.ipynb
   ```
3. **Chạy tự động toàn bộ quy trình huấn luyện & xuất báo cáo (CLI):**
   ```bash
   python scripts/train.py
   ```
4. **Chạy suy luận dự báo 24 giờ tiếp theo bằng mô hình tốt nhất:**
   ```bash
   python scripts/predict.py --model xgboost_tuned_24lags.json --horizon 24
   ```
