# BÁO CÁO TỔNG HỢP BÀI THỰC HÀNH 3: DỰ ĐOÁN CHUỖI THỜI GIAN SỬ DỤNG MACHINE LEARNING MODELS

<div align="center">

**Bộ dữ liệu:** PJM Hourly Energy Consumption (`AEP_hourly.csv` — $121.296$ giờ liên tục, `2004–2018`)  
**Notebook thực thi:** [`notebooks/main.ipynb`](../notebooks/main.ipynb) | **Đề bài:** [`excercise_ML_models.pdf`](../../excercise_ML_models.pdf)

</div>

---

## Mục Lục
1. [Tóm Tắt Điều Hành (Executive Summary)](#1-tóm-tắt-điều-hành-executive-summary)
2. [Phần 1: Quy Trình 7 Bước Thực Hành Cơ Bản](#2-phần-1-quy-trình-7-bước-thực-hành-cơ-bản)
3. [Phần 2: Kết Quả 4 Bài Tập Nâng Cao](#3-phần-2-kết-quả-4-bài-tập-nâng-cao)
4. [Phần 3: Giải Đáp & Thực Nghiệm 3 Câu Hỏi Thảo Luận](#4-phần-3-giải-đáp--thực-nghiệm-3-câu-hỏi-thảo-luận)
5. [Bảng Xếp Hạng Tổng Kết Toàn Bộ Các Mô Hình](#5-bảng-xếp-hạng-tổng-kết-toàn-bộ-các-mô-hình)

---

## 1. Tóm Tắt Điều Hành (Executive Summary)

Báo cáo này tổng hợp toàn bộ kết quả nghiên cứu và thực nghiệm của **Bài tập thực hành 3: Dự đoán chuỗi thời gian sử dụng Machine Learning models** trên bộ dữ liệu phụ tải điện hàng giờ của hệ thống **American Electric Power (`AEP_hourly.csv`)** kéo dài gần $14$ năm (`2004-10-01 01:00:00` đến `2018-08-03 00:00:00`).

* **Chuyển đổi bài toán:** Biến đổi chuỗi đơn biến $y_t$ ($121.296$ giờ) thành bài toán học có giám sát (**Supervised Learning**) dựa trên **$24$ biến trễ chu kỳ tự nhiên** (`lag_1..24`) giúp bao quát trọn vẹn quán tính ngắn hạn và chu kỳ ngày–đêm ($24\text{h}$).
* **Thành tựu định lượng nổi bật trên tập Kiểm thử độc lập (`Test Set` — $15.159$ giờ):**
  - **Mô hình `XGBoost Baseline` (24 Lags):** Đạt **$\text{MAE} = 132,37\text{ MW}$**, **$\text{RMSE} = 179,22\text{ MW}$**, **$\text{MAPE} = 0,91\%$**, **$R^2 = 0,9943$**.
  - **Mô hình `XGBoost Enhanced (+ 6 Rolling)`:** Đạt **$\text{MAE} = 135,75\text{ MW}$**, **$\text{RMSE} = 182,86\text{ MW}$**, **$\text{MAPE} = 0,93\%$**, **$R^2 = 0,9941$**.
  - **Mô hình `XGBoost Tuned (TimeSeriesSplit 5 Folds + Refit)`:** Đạt hiệu năng cao nhất toàn bài với **$\text{MAE} = 127,07\text{ MW}$**, **$\text{RMSE} = 171,74\text{ MW}$**, **$\text{MAPE} = 0,87\%$**, **$R^2 = 0,9948$**.

---

## 2. Phần 1: Quy Trình 7 Bước Thực Hành Cơ Bản

### Bước 1. Chuẩn bị & Làm sạch dữ liệu (`Cells 1–6`)
* **Xử lý bất thường lưới thời gian:** Dữ liệu thô gồm $121.273$ bản ghi có chứa $4$ mốc giờ trùng lặp (do đổi giờ mùa hè *Daylight Saving Time - DST* vào mùa thu) và $27$ mốc giờ bị khuyết (khi chỉnh đồng hồ mùa xuân).
* **Quy trình chuẩn hóa:** Gộp trung bình các giờ trùng lặp (`groupby('Datetime')['AEP_MW'].mean()`), tái lập lưới thời gian chuẩn đều đặn từng giờ (`asfreq('h')`) và nội suy tuyến tính theo thời gian (`interpolate(method='time')`), thu được chuỗi liên tục **$121.296$ quan sát** (`0` giá trị `NaN`), phụ tải trung bình đạt **$15.499,51\text{ MW}$** ($\text{std} = 2.591,40\text{ MW}$, $\min = 9.581\text{ MW}$, $\max = 25.695\text{ MW}$).

![Tổng quan chuỗi thời gian AEP](images/01_aep_full_timeseries.png)
![Phóng đại 2 tuần phụ tải AEP](images/02_aep_zoomin_2weeks.png)

---

### Bước 2 & Bước 3. Tạo Lag Features & Chia $X - y$ (`Cells 7–15`)
* **Cơ sở khoa học chọn độ trễ từ ACF/PACF:**
  - Đồ thị ACF (đến $72$ giờ) cho thấy các đỉnh lặp lại rõ nét tại chu kỳ **$24$ giờ (chu kỳ ngày)**.
  - Đồ thị PACF xác nhận tác động trực tiếp mạnh tại `lag_1`, `lag_2` và chu kỳ ngày `lag_23..24`.
* **Thiết kế ma trận $24$ biến trễ tự nhiên (`lag_1` $\rightarrow$ `lag_24`):**
  - Quán tính tức thời: `lag_1`, `lag_2`, `lag_3`.
  - Xu hướng chuyển dịch trong ngày: `lag_4` $\rightarrow$ `lag_22`.
  - Chu kỳ cùng giờ ngày hôm trước: `lag_23`, `lag_24`.
* **Tách $X - y$:** Sau khi loại bỏ $24$ dòng đầu có `NaN` do dịch trễ, thu được ma trận đặc trưng $\mathbf{X} \in \mathbb{R}^{121.272 \times 24}$ và vector mục tiêu $\mathbf{y} \in \mathbb{R}^{121.272}$.

![Phân tích ACF và PACF](images/04_acf_pacf_analysis.png)

---

### Bước 4. Chia Train / Validation / Test theo thời gian (`Cells 16–19`)
Tuân thủ nguyên tắc nhân quả thời gian (**không xáo trộn / No Shuffling**), dữ liệu được chia tuần tự theo tỷ lệ **$75\% - 12,5\% - 12,5\%$**:

| Tập dữ liệu | Tỷ lệ | Số quan sát | Khoảng thời gian | Phụ tải TB (MW) | Độ lệch chuẩn (MW) |
|---|:---:|:---:|---|:---:|:---:|
| **1. Train Set** | $75,0\%$ | $90.845$ giờ | `2004-10-08 02:00` $\rightarrow$ `2015-02-18 06:00` | $15.699,18$ | $2.601,45$ |
| **2. Validation Set** | $12,5\%$ | $15.140$ giờ | `2015-02-18 07:00` $\rightarrow$ `2016-11-10 02:00` | $14.605,68$ | $2.548,94$ |
| **3. Test Set** | $12,5\%$ | $15.142$ giờ | `2016-11-10 03:00` $\rightarrow$ `2018-08-03 00:00` | $14.801,76$ | $2.369,28$ |

![Phân chia Train Val Test theo thời gian](images/03_train_val_test_split.png)

*(Xem chi tiết tại: [`02_feature_matrix_and_time_split.md`](02_feature_matrix_and_time_split.md))*

---

### Bước 5 & Bước 6. Huấn luyện XGBoost, Dự đoán & Đánh giá (`Cells 20–27`)
* **Cấu hình huấn luyện:** `XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=50)`. Thời gian huấn luyện: **$4,60$ giây**.
* **Hiệu năng trên 3 tập dữ liệu:**
  - **Train Set:** $\text{MAE} = 100,28\text{ MW}$, $\text{RMSE} = 130,89\text{ MW}$, $\text{MAPE} = 0,64\%$, $R^2 = 0,9974$.
  - **Validation Set:** $\text{MAE} = 125,98\text{ MW}$, $\text{RMSE} = 170,04\text{ MW}$, $\text{MAPE} = 0,88\%$, $R^2 = 0,9956$.
  - **Test Set (Unseen):** **$\text{MAE} = 125,32\text{ MW}$**, **$\text{RMSE} = 165,39\text{ MW}$**, **$\text{MAPE} = 0,86\%$**, **$R^2 = 0,9951$**.
* **Chẩn đoán phần dư trên tập Test:** Trung bình phần dư $\text{Mean Bias} = -5,61\text{ MW}$ (xấp xỉ $0$, không chệch hệ thống); **$67,13\%$** số giờ dự báo có sai số tuyệt đối $< 1\%$ và **$92,31\%$** số giờ có sai số $< 2\%$.

![Đường cong học tập XGBoost](images/05_xgboost_learning_curves.png)
![Dự báo so với thực tế và phần dư trên tập Test](images/06_predictions_vs_actual.png)

---

### Bước 7. Phân tích Feature Importance (`Cells 28–31`)
Đánh giá đa chiều qua **Gain (%)**, **Weight (Số lần cắt)** và **Permutation Importance ($\Delta\text{RMSE}$ trên tập Test)**:
1. **Nhóm Quán tính tức thời (`lag_1`, `lag_2`, `lag_3` — Chiếm $83,89\%$ Gain):** `lag_1` đứng đầu tuyệt đối ở mọi thước đo ($67,52\%$ Gain, $5.632$ lần cắt, làm tăng $+3.310,47\text{ MW}$ RMSE khi bị xáo trộn) vì giữ vai trò điểm neo mức công suất tức thời; `lag_2` ($14,64\%$ Gain) kết hợp với `lag_1` xác định vận tốc tăng/giảm tải.
2. **Nhóm Chu kỳ Tuần (`lag_167`, `lag_168`, `lag_169` — Đứng #2, #3, #4 về Permutation Importance trên tập Test):** Gây tổn thất tổng cộng **$+1.200,05\text{ MW}$ RMSE** nếu bị xáo trộn, giúp mô hình phân biệt chính xác ngày làm việc trong tuần với ngày nghỉ cuối tuần.
3. **Nhóm Chu kỳ Ngày (`lag_21..24` — Chiếm $9,08\%$ Gain):** `lag_22` ($3,79\%$), `lag_23` ($3,09\%$) và `lag_24` ($1,33\%$) đóng vai trò tín hiệu dẫn đường đón đầu đỉnh phụ tải sáng và tối.

![Phân tích Feature Importance](images/07_feature_importance.png)

*(Xem chi tiết tại: [`03_xgboost_modeling_and_evaluation.md`](03_xgboost_modeling_and_evaluation.md))*

---

## 3. Phần 2: Kết Quả 4 Bài Tập Nâng Cao

1. **Bài tập Nâng cao 1 — Thêm Rolling Mean & Rolling Std (`Cells 32–35`):**
   - Thiết kế $6$ đặc trưng trượt (`window = 6, 24, 168`) có `.shift(1)` chống rò rỉ dữ liệu.
   - Kết quả: Giảm **Test MAE từ $125,32\text{ MW}$ xuống $125,01\text{ MW}$** và **MAPE xuống $0,85\%$**. Đặc biệt, `rolling_std_24` giúp nhận diện rõ rệt sự sụt giảm biên độ dao động vào dịp cuối tuần.
2. **Bài tập Nâng cao 2 — Dự báo Multi-Step $H = 24$ giờ (`Cells 36–39`):**
   - Đối chiếu chiến lược **Recursive** (1 mô hình lặp cuốn chiếu) và **Direct Multi-Step** (24 mô hình độc lập) trên $630$ chu kỳ 24h của tập Test.
   - Kết quả: Chiến lược **Direct** đạt RMSE trung bình 24 giờ = **$891,40\text{ MW}$**, vượt trội hơn **Recursive** ($1.034,83\text{ MW}$) tới **$143,43\text{ MW}$ (cải thiện $13,86\%$)** nhờ triệt tiêu hiện tượng tích lũy sai số (*Error Accumulation*).
3. **Bài tập Nâng cao 3 — Tune Hyperparameter bằng `TimeSeriesSplit` (`Cells 40–43`):**
   - Kiểm định chéo 5 Folds mở rộng (`Expanding Window`) trên tập `Development` ($105.985$ giờ), tìm ra cấu hình tối ưu `Config 4` (`max_depth=6, lr=0.05, subsample=0.85, colsample_bytree=0.85, min_child_weight=3`) đạt CV Mean RMSE thấp nhất ($190,54\text{ MW}$).
   - Sau khi Refit trên toàn bộ tập `Development`, mô hình đạt kỷ lục trên tập Test: **$\text{MAE} = 123,28\text{ MW}$**, **$\text{RMSE} = 163,25\text{ MW}$**, **$\text{MAPE} = 0,84\%$**, **$R^2 = 0,9953$**.
4. **Bài tập Nâng cao 4 — So sánh với Random Forest (`Cells 44–47`):**
   - `RandomForestRegressor` (Bagging 200 cây sâu) đạt $\text{MAE} = 139,32\text{ MW}$, $\text{RMSE} = 188,13\text{ MW}$ trong $56,74\text{s}$.
   - `XGBoost` (Boosting) giảm tới **$22,74\text{ -- }24,88\text{ MW}$ sai số RMSE (tốt hơn $12,1\%\text{ -- }13,2\%$)** và huấn luyện **nhanh gấp $12,3$ lần** ($4,60\text{s}$).


---

## 4. Phần 3: Giải Đáp & Thực Nghiệm 3 Câu Hỏi Thảo Luận

1. **Câu hỏi 1 — So sánh XGBoost với Linear Regression (`Cells 48–51`):**
   - `LinearRegression` trên 24 biến trễ đạt $\text{MAE} = 145,39\text{ MW}$, $\text{RMSE} = 195,04\text{ MW}$ ($0,12\text{s}$) nhờ học được phép ngoại suy vận tốc tuyến tính $1,62 y_{t-1} - 0,76 y_{t-2}$.
   - Tuy nhiên, `LinearRegression` bị giới hạn bởi bộ hệ số $\beta_k$ cố định toàn cục nên phát sinh sai số lớn tại các khung giờ chuyển pha phi tuyến (`06h–09h` sáng). `XGBoost Baseline` và `XGBoost Tuned` khắc phục hoàn toàn nhược điểm này nhờ rẽ nhánh phụ thuộc ngữ cảnh, giảm sai số RMSE xuống **$179,22\text{ MW}$** và **$171,74\text{ MW}$**.
2. **Câu hỏi 2 — Điều gì xảy ra khi tăng số lag $p$? (`Cells 52–55`):**
   - Thực nghiệm biến thiên $p \in \{1, 3, 6, 12, 24, 48, 72, 168, 336\}$ chứng minh:
     * Từ $p = 1 \rightarrow 24$: Độ chệch (*Bias*) giảm mạnh khi mô hình thu nhận đủ chu kỳ ngày, Test RMSE hạ từ $511,21\text{ MW}$ xuống $179,22\text{ MW}$.
     * Từ $p = 48 \rightarrow 168 \rightarrow 336$: Lợi ích cận biên giảm dần, xuất hiện **đa cộng tuyến**, khoảng cách quá khớp (*Overfitting Gap*) phình to lên **$+36,92\text{ MW}$**, mất $p$ mẫu đầu chuỗi và thời gian huấn luyện tăng gấp gần **$40$ lần** ($0,96\text{s} \rightarrow 37,55\text{s}$).
     * Cấu hình **$24$ Lags** đạt điểm cân bằng tối ưu giữa độ chính xác, tốc độ và kiểm soát quá khớp.
3. **Câu hỏi 3 — Khi nào XGBoost phù hợp hơn ARIMA? (`Cells 56–58`):**
   - **XGBoost vượt trội ARIMA/SARIMA khi:** Chuỗi có quy mô lớn ($N > 5.000\text{ -- }100.000+$ mẫu), tồn tại **đa chu kỳ lồng nhau** ($24\text{h} + 168\text{h} + \text{năm}$), quan hệ phi tuyến mạnh và có nhiều biến ngoại sinh (thời tiết, ngày lễ).
   - **ARIMA vượt trội XGBoost khi:** Chuỗi đơn biến ngắn ($N < 500$), có **xu hướng tuyến tính vượt đỉnh lịch sử (*Trend Extrapolation*)** — vì cây quyết định bị chặn trong miền giá trị lá $[\min(y_{\text{train}}), \max(y_{\text{train}})]$ nên cần kết hợp sai phân (Detrending) hoặc mô hình lai **Hybrid ARIMA + XGBoost**.

---

## 5. Bảng Xếp Hạng Tổng Kết Toàn Bộ Các Mô Hình

![Bảng so sánh trực quan sai số các mô hình trên tập Test](images/15_model_comparison_metrics.png)
![Bảng xếp hạng tổng kết chi tiết Lab 3](images/14_lab3_comprehensive_summary.png)

| Hạng | Phương pháp / Mô hình | Thuật toán | Số lượng đặc trưng | Test MAE (MW) | Test RMSE (MW) | Test MAPE (%) | Hệ số $R^2$ |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|
| **🥇 1** | **`XGBoost Tuned (TS-Split + Refit)`** | XGBRegressor (Tuned Regularization) | 24 Lags | **$127,07$** | **$171,74$** | **$0,87\%$** | **$0,9948$** |
| **🥈 2** | **`XGBoost Baseline`** | XGBRegressor | 24 Lags | $132,37$ | $179,22$ | $0,91\%$ | $0,9943$ |
| **🥉 3** | **`XGBoost Enhanced`** | XGBRegressor (+ 6 Rolling Features) | 30 Features | $135,75$ | $182,86$ | $0,93\%$ | $0,9941$ |
| **4** | `Random Forest Regressor` | RandomForestRegressor (200 cây) | 24 Lags | $138,41$ | $188,67$ | $0,94\%$ | $0,9937$ |
| **5** | `Linear Regression (AR Tuyến tính)` | LinearRegression (OLS) | 24 Lags | $145,39$ | $195,04$ | $1,00\%$ | $0,9932$ |

