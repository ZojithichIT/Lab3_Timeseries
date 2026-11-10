# Phân Tách Đặc Trưng & Phân Chia Train / Validation / Test (Feature Matrix & Time-Based Splitting)

Tài liệu này chi tiết hóa toàn bộ quy trình thực hiện **Bước 3: Chia $X - y$** và **Bước 4: Chia Train / Validation / Test theo thời gian** với tỷ lệ **$75\% - 12.5\% - 12.5\%$** trong notebook [`notebooks/main.ipynb`](../notebooks/main.ipynb).

---

## 1. Cơ Sở Lý Thuyết & Biểu Diễn Toán Học

### 1.1. Ma trận đặc trưng $X$ và Vector biến mục tiêu $y$ (Bước 3)
Từ bảng dữ liệu học có giám sát `df_supervised` thu được sau Bước 2 ($121.127$ dòng $\times 30$ cột), ta phân tách thành:
1. **Vector mục tiêu ($y \in \mathbb{R}^N$):** Cột phụ tải $y_t$ tại thời điểm hiện tại cần dự đoán ($121.127$ quan sát).
2. **Ma trận đặc trưng ($X \in \mathbb{R}^{N \times 29}$):** Gồm $29$ đặc trưng trễ đa chu kỳ trong quá khứ:
   $$\mathbf{x}_t = \big[ y_{t-1}, \dots, y_{t-24}, y_{t-48}, y_{t-72}, y_{t-167}, y_{t-168}, y_{t-169} \big]^T \in \mathbb{R}^{29}$$
3. **Mục tiêu học máy:** Tìm hàm ánh xạ phi tuyến xấp xỉ:
   $$\hat{y}_t = f(\mathbf{x}_t) = f(y_{t-1}, \dots, y_{t-24}, y_{t-48}, y_{t-72}, y_{t-167}, y_{t-168}, y_{t-169})$$

### 1.2. Nguyên tắc phân chia theo trình tự thời gian (Time-Based Splitting - Bước 4)
Khác với bài toán học máy trên dữ liệu bảng độc lập (tabular i.i.d.), chuỗi thời gian có đặc tính **phụ thuộc thời gian** và **tính nhân quả (Causality)**:
* **Quy tắc vàng (Strict Rule):** **Tuyệt đối KHÔNG shuffle (xáo trộn ngẫu nhiên)**.
* **Nguy cơ rò rỉ dữ liệu (Data Leakage / Look-ahead Bias):** Nếu sử dụng `train_test_split(..., shuffle=True)`, thông tin từ tương lai ($t+k$) sẽ lọt vào tập huấn luyện ($t$), làm mô hình đạt điểm số cao ảo nhưng hoàn toàn tê liệt khi dự báo thực tế.
* **Giải pháp chuẩn:** Cắt lát tuần tự (Slicing) theo dòng thời gian tăng dần.

---

## 2. Chiến Lược Phân Bổ 3 Tập Dữ Liệu (75% - 12.5% - 12.5%)

Việc phân chia thành 3 phần thay vì 2 phần mang lại lợi thế vượt trội cho các mô hình Gradient Boosting (XGBoost):

```text
[======================= TRAIN (75.0%) =======================][== VALIDATION (12.5%) ==][===== TEST (12.5%) =====]
2004-10-08 02:00                                            2015-02-18 06:00          2016-11-10 02:00       2018-08-03 00:00
```

1. **Tập Huấn Luyện (Train - 75%):**
   - Đóng vai trò cung cấp dữ liệu lịch sử để mô hình học các mẫu chu kỳ (ngày, tuần, mùa), xu hướng và cấu trúc phân nhánh của cây quyết định.
2. **Tập Thẩm Định (Validation - 12.5%):**
   - Đóng vai trò là tập giám sát độc lập trong quá trình huấn luyện XGBoost.
   - Cho phép áp dụng cơ chế **Early Stopping** (`early_stopping_rounds`): dừng quá trình thêm cây mới khi hàm mất mát trên tập Validation không còn giảm, triệt tiêu nguy cơ Overfitting.
   - Hỗ trợ tinh chỉnh siêu tham số (Hyperparameter Tuning) trước khi đánh giá cuối cùng.
3. **Tập Kiểm Thử (Test - 12.5%):**
   - Dữ liệu tương lai hoàn toàn chưa từng được nhìn thấy ("unseen hold-out data").
   - Đảm bảo đánh giá khách quan, trung thực năng lực dự báo thực tế của mô hình ở Bước 6.

---

## 3. Kết Quả Định Lượng Thực Nghiệm

Toàn bộ $121.127$ mẫu thời gian được phân chia chính xác tuyệt đối:

| Thuộc tính | Toàn bộ chuỗi ($N$) | Tập Train ($75\%$) | Tập Validation ($12.5\%$) | Tập Test ($12.5\%$) |
|---|---|---|---|---|
| **Số quan sát (giờ)** | **$121.127$** | **$90.845$** | **$15.140$** | **$15.142$** |
| **Tỷ lệ phần trăm** | $100,0\%$ | $75,00\%$ | $12,50\%$ | $12,50\%$ |
| **Kích thước ma trận $X$** | $(121127, 29)$ | $(90845, 29)$ | $(15140, 29)$ | $(15142, 29)$ |
| **Kích thước vector $y$** | $(121127,)$ | $(90845,)$ | $(15140,)$ | $(15142,)$ |
| **Mốc bắt đầu** | `2004-10-08 02:00:00` | `2004-10-08 02:00:00` | `2015-02-18 07:00:00` | `2016-11-10 03:00:00` |
| **Mốc kết thúc** | `2018-08-03 00:00:00` | `2015-02-18 06:00:00` | `2016-11-10 02:00:00` | `2018-08-03 00:00:00` |
| **Thời lượng tương đương** | $\approx 13,82\text{ năm}$ | $\approx 10,36\text{ năm}$ | $\approx 1,73\text{ năm}$ | $\approx 1,73\text{ năm}$ |
| **Phụ tải trung bình (mean)** | $15.501,43\text{ MW}$ | $15.767,34\text{ MW}$ | $14.605,68\text{ MW}$ | $14.801,76\text{ MW}$ |
| **Độ lệch chuẩn (std)** | $2.591,78\text{ MW}$ | $2.575,97\text{ MW}$ | $2.563,56\text{ MW}$ | $2.374,38\text{ MW}$ |

### Kiểm định tính liên tục & bảo toàn dữ liệu:
* **Liên tục giữa Train và Validation:**  
  $$\Delta t = t_{\text{val, start}} - t_{\text{train, end}} = \text{2015-02-18 07:00:00} - \text{2015-02-18 06:00:00} = 1\text{ giờ} \quad (\text{Đạt})$$
* **Liên tục giữa Validation và Test:**  
  $$\Delta t = t_{\text{test, start}} - t_{\text{val, end}} = \text{2016-11-10 03:00:00} - \text{2016-11-10 02:00:00} = 1\text{ giờ} \quad (\text{Đạt})$$
* **Bảo toàn số lượng mẫu:**  
  $$N_{\text{train}} + N_{\text{val}} + N_{\text{test}} = 90.845 + 15.140 + 15.142 = 121.127 = N \quad (\text{Đạt } 100\%)$$

---

## 4. Trực Quan Hóa Phân Bổ Chuỗi Thời Gian

Biểu đồ trực quan hóa 3 tập dữ liệu được tự động tạo và lưu trữ tại [`docs/images/03_train_val_test_split.png`](images/03_train_val_test_split.png):

![Phân chia Train Val Test](../docs/images/03_train_val_test_split.png)

* **Vùng xanh dương (Train - 75%):** Bao phủ hơn 10 năm phụ tải từ 2004 đến đầu 2015, chứa trọn vẹn các đỉnh phụ tải mùa đông và mùa hè.
* **Vùng cam (Validation - 12.5%):** Kéo dài từ tháng 02/2015 đến tháng 11/2016, đại diện cho giai đoạn chuyển tiếp.
* **Vùng đỏ (Test - 12.5%):** Kéo dài từ tháng 11/2016 đến tháng 08/2018, dùng làm chuẩn thẩm định cuối cùng.

---

## 5. Danh Mục Dữ Liệu Đã Lưu Vào data/processed/

Toàn bộ các tập dữ liệu phân chia đã được xuất và lưu trữ trong thư mục [`data/processed/`](../data/processed):

| Tên File | Loại Dữ Liệu | Số Dòng | Số Cột | Kích Thước | Mô Tả Chi Tiết |
|---|---|:---:|:---:|:---:|---|
| [`aep_train.csv`](../data/processed/aep_train.csv) | Bảng hoàn chỉnh (Train) | $90.845$ | $30$ | $\approx 23,5\text{ MB}$ | Chứa `DatetimeIndex`, Target $y$ và $29$ cột biến trễ đa chu kỳ ($75\%$). |
| [`aep_val.csv`](../data/processed/aep_val.csv) | Bảng hoàn chỉnh (Val) | $15.140$ | $30$ | $\approx 3,9\text{ MB}$ | Chứa `DatetimeIndex`, Target $y$ và $29$ cột biến trễ đa chu kỳ ($12,5\%$). |
| [`aep_test.csv`](../data/processed/aep_test.csv) | Bảng hoàn chỉnh (Test) | $15.142$ | $30$ | $\approx 3,9\text{ MB}$ | Chứa `DatetimeIndex`, Target $y$ và $29$ cột biến trễ đa chu kỳ ($12,5\%$). |
| [`X_train.csv`](../data/processed/X_train.csv) | Ma trận đặc trưng (Train) | $90.845$ | $29$ | $\approx 22,8\text{ MB}$ | Chỉ chứa $29$ cột biến trễ của tập Train. |
| [`y_train.csv`](../data/processed/y_train.csv) | Vector mục tiêu (Train) | $90.845$ | $1$ | $\approx 2,6\text{ MB}$ | Chỉ chứa nhãn mục tiêu phụ tải $y$ của tập Train. |
| [`X_val.csv`](../data/processed/X_val.csv) | Ma trận đặc trưng (Val) | $15.140$ | $29$ | $\approx 3,8\text{ MB}$ | Chỉ chứa $29$ cột biến trễ của tập Validation. |
| [`y_val.csv`](../data/processed/y_val.csv) | Vector mục tiêu (Val) | $15.140$ | $1$ | $\approx 0,4\text{ MB}$ | Chỉ chứa nhãn mục tiêu phụ tải $y$ của tập Validation. |
| [`X_test.csv`](../data/processed/X_test.csv) | Ma trận đặc trưng (Test) | $15.142$ | $29$ | $\approx 3,8\text{ MB}$ | Chỉ chứa $29$ cột biến trễ của tập Test. |
| [`y_test.csv`](../data/processed/y_test.csv) | Vector mục tiêu (Test) | $15.142$ | $1$ | $\approx 0,4\text{ MB}$ | Chỉ chứa nhãn mục tiêu phụ tải $y$ của tập Test. |

---

## 6. Đúc Kết & Sẵn Sàng Cho Bước 5 (Huấn Luyện XGBoost)

Việc có sẵn tập Validation độc lập tạo điều kiện lý tưởng cho **Bước 5: Huấn luyện mô hình XGBoost Regressor**:
```python
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False
)
```
Mô hình sẽ theo dõi hàm mất mát trên cả `eval_set` để tự động xác định điểm dừng tối ưu, tránh tình trạng Overfitting trên tập Train.
