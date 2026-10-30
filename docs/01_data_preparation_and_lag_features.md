# Tiền Xử Lý & Khám Phá Dữ Liệu Chuỗi Thời Gian (Data Preparation & EDA)

Tài liệu này chi tiết hóa toàn bộ quy trình kiểm tra chất lượng dữ liệu, xử lý tính liên tục thời gian (Daylight Saving Time, mốc trùng lặp / thiếu hụt), thống kê mô tả (`describe`) và phân tích trực quan hóa chuỗi dữ liệu tiêu thụ điện **AEP Hourly Energy Consumption** được triển khai trong notebook [`scripts/main.ipynb`](../scripts/main.ipynb).

---

## 1. Giới Thiệu Tập Dữ Liệu AEP Hourly

* **Nguồn dữ liệu:** PJM Interconnection LLC (Dataset: [Hourly Energy Consumption - AEP](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption?select=AEP_hourly.csv)).
* **Đơn vị quản trị:** American Electric Power (AEP) - hệ thống điện lưới phụ tải lớn tại miền Đông Bắc Hoa Kỳ.
* **Thời gian quan sát:** Từ **01/10/2004 01:00:00** đến **03/08/2018 00:00:00** (~14 năm liên tục).
* **Đơn vị đo lường:** Megawatt ($MW$).
* **Đặc tính chuỗi:** Chuỗi thời gian đơn biến $y_t$ có tần suất hàng giờ (`Hourly`), chu kỳ ngày (24 giờ), chu kỳ tuần (168 giờ) và chu kỳ mùa trong năm.

---

## 2. Kiểm Tra Chất Lượng Dữ Liệu & Tiền Xử Lý Dòng Thời Gian

### 2.1. Đánh giá hiện trạng dữ liệu thô
* Kích thước file ban đầu: **121.273 dòng** và **2 cột** (`Datetime`, `AEP_MW`).
* Giá trị khuyết thiếu ban đầu: **0 giá trị NaN**.
* **Vấn đề trật tự thời gian (Non-monotonic):** Dữ liệu thô bị đảo lộn thứ tự ngày tháng (ví dụ: ngày `2004-12-31` nằm trước ngày `2004-12-30`).  
  👉 **Giải pháp:** Bắt buộc sắp xếp lại dòng thời gian tăng dần tuyệt đối:
  ```python
  df['Datetime'] = pd.to_datetime(df['Datetime'])
  df = df.sort_values('Datetime').reset_index(drop=True)
  ```

### 2.2. Xử lý hiện tượng Quy ước giờ mùa hè (Daylight Saving Time - DST)
Chuỗi thời gian thực tế tại Bắc Mỹ chịu ảnh hưởng của DST với 2 hiện tượng vật lý:

1. **Trùng lặp mốc thời gian (Fall Back - Lùi 1 giờ vào mùa thu):**  
   Chủ nhật đầu tiên của tháng 11 hàng năm, đồng hồ lùi từ 02:00 sáng về 01:00 sáng, dẫn đến giờ `02:00:00` xuất hiện 2 lần:
   * `2014-11-02 02:00:00`: $12.994\text{ MW}$ và $13.190\text{ MW}$
   * `2015-11-01 02:00:00`: $10.785\text{ MW}$ và $10.542\text{ MW}$
   * `2016-11-06 02:00:00`: $10.964\text{ MW}$ và $11.008\text{ MW}$
   * `2017-11-05 02:00:00`: $10.596\text{ MW}$ và $10.446\text{ MW}$  
   👉 **Giải pháp:** Nhóm theo `Datetime` và tính trung bình phụ tải của 2 lần đo: `df.groupby('Datetime')['AEP_MW'].mean()`.

2. **Khuyết mốc thời gian (Spring Forward - Nhảy cóc 1 giờ vào mùa xuân):**  
   Vào tháng 3 hàng năm, đồng hồ nhảy từ 02:00 sáng sang 03:00 sáng, làm khuyết mốc `02:00:00`. Tổng số giờ dự kiến liên tục trong 14 năm là **121.296 giờ**, trong khi dữ liệu thực tế chỉ có 121.269 mốc duy nhất (thiếu **27 giờ** $\approx 0.022\%$).  
   👉 **Giải pháp:** Thiết lập lưới thời gian liên tục với tần suất 1 giờ (`asfreq('h')`) và nội suy tuyến tính theo thời gian:
   ```python
   series = df.groupby('Datetime')['AEP_MW'].mean().asfreq('h')
   series = series.interpolate(method='time')
   ```

* **Kết quả:** Thu được chuỗi thời gian sạch hoàn hảo với **121.296 quan sát liên tục**, không có giá trị khuyết và khoảng cách giữa các bước luôn đúng $1\text{ giờ}$.

---

## 3. Thống Kê Mô Tả Chuỗi Phụ Tải (`describe`)

Kết quả thống kê mô tả từ hàm `series.describe()` trên toàn bộ chuỗi $y_t$:

| Chỉ số thống kê | Giá trị ($MW$) | Diễn giải & Ý nghĩa nghiệp vụ |
|---|---|---|
| **Số quan sát (Count)** | **121.296 giờ** | Chuỗi đầy đủ liên tục 100% trong suốt ~14 năm (2004–2018). |
| **Giá trị trung bình (Mean)** | **15.499,15 MW** | Mức tiêu thụ điện năng bình quân của lưới điện AEP. |
| **Độ lệch chuẩn (Std)** | **2.591,38 MW** | Biên độ dao động lớn phản ánh sự chênh lệch phụ tải ngày/đêm và mùa hè/mùa đông. |
| **Thấp nhất (Min)** | **9.581,00 MW** | Đáy cực tiểu xuất hiện vào lúc `02/10/2016 05:00:00` (đêm thời tiết ôn hòa, thấp điểm). |
| **Tứ phân vị 25% (Q1)** | **13.629,00 MW** | 25% thời gian trong 14 năm phụ tải dưới mức này (chủ yếu là đêm và cuối tuần). |
| **Trung vị 50% (Median)** | **15.309,00 MW** | Rất gần với Mean ($15.499\text{ MW}$), phân phối phụ tải mang tính cân xứng tốt. |
| **Tứ phân vị 75% (Q3)** | **17.200,00 MW** | 75% thời gian phụ tải không vượt quá $17.200\text{ MW}$. |
| **Cao nhất (Max)** | **25.695,00 MW** | Đỉnh cực đại kỷ lục đạt vào lúc `20/10/2008 14:00:00` (đợt rét đậm bất thường). |

---

## 4. Trực Quan Hóa Dữ Liệu & Chú Thích Biểu Đồ

### 4.1. Toàn cảnh chuỗi thời gian & Các đường ngưỡng thống kê

![Toàn cảnh chuỗi thời gian AEP 2004 - 2018](images/01_aep_full_timeseries.png)

* **Đường Mean (đỏ nét đứt):** Mức trung bình $15.499,2\text{ MW}$ duy trì vững chắc qua 14 năm, cho thấy nhu cầu phụ tải nền (Base Load) rất ổn định.
* **Đường Max (tím nét chấm):** Ngưỡng $25.695\text{ MW}$ và các đợt đỉnh mùa đông lạnh cực đoan (Polar Vortex) chứng minh các giá trị ngoại lai này mang bản chất vật lý thật của thời tiết, cần được giữ lại trong tập huấn luyện.
* **Đường Min (xanh lá nét chấm):** Ngưỡng $9.581\text{ MW}$ thể hiện mức sàn tiêu thụ điện tối thiểu của vùng.
* **Tính mùa vụ lặp lại:** Chu kỳ 1 năm lặp đi lặp lại rất rõ rệt với 2 đỉnh phụ tải (mùa đông sưởi ấm và mùa hè làm mát).

---

### 4.2. Thu phóng 2 tuần điển hình: Nhịp điệu ngày/đêm & Sự sụt giảm cuối tuần

![Thu phóng 2 tuần điển hình phụ tải AEP](images/02_aep_zoomin_2weeks.png)

Quan sát lát cắt 2 tuần liên tục (từ `09/01/2017` đến `22/01/2017`) làm sáng tỏ hai đặc trưng cấu trúc then chốt:

1. **Chu kỳ hàng ngày (Daily 24-hour cycle):**
   - Đỉnh sáng (~08:00–10:00) và đỉnh tối (~18:00–20:00) xuất hiện đều đặn mỗi ngày do lịch trình sinh hoạt và làm việc.
   - Đáy thấp điểm (~03:00–05:00 sáng) tụt sâu khi hoạt động sinh hoạt giảm.
2. **Chu kỳ hàng tuần (Weekly 168-hour cycle):**
   - Các vùng tô xám (Thứ Bảy và Chủ Nhật) chứng kiến phụ tải sụt giảm rõ rệt so với các ngày trong tuần (Thứ Hai đến Thứ Sáu) do các cơ sở công nghiệp, công sở ngừng làm việc.

---

## 5. Cơ Sở Lý Thuyết Biến Đổi Chuỗi Thời Gian Sang Học Có Giám Sát

### 5.1. Bản chất toán học của bài toán
Trong bài toán chuỗi thời gian đơn biến cổ điển, ta có một dãy các quan sát theo thứ tự thời gian:
$$y_1, y_2, y_3, \dots, y_T$$

Mô hình Machine Learning (như XGBoost, Random Forest, Linear Regression) yêu cầu dữ liệu đầu vào dạng bảng có cấu trúc ma trận gồm tập đặc trưng đầu vào $X \in \mathbb{R}^{N \times p}$ và vector nhãn mục tiêu $y \in \mathbb{R}^N$.

Kỹ thuật **Trượt cửa sổ thời gian (Sliding Window / Lagging)** chuyển đổi bài toán dự báo bước tiếp theo thành bài toán hồi quy có giám sát:
$$\hat{y}_t = f(y_{t-1}, y_{t-2}, \dots, y_{t-p})$$

Trong đó:
* $y_t$ (**Biến mục tiêu / Target**): Giá trị phụ tải tại thời điểm $t$ cần dự báo.
* $y_{t-1}, y_{t-2}, \dots, y_{t-p}$ (**Biến trễ / Lag Features**): Các giá trị thực tế đã quan sát được trong quá khứ tại các thời điểm $t-1, t-2, \dots, t-p$.
* $p$ (**Bậc trễ / Lag Order**): Số bước thời gian quá khứ được sử dụng làm đặc trưng dự đoán.

---

### 5.2. Minh họa cơ chế trượt cửa sổ thời gian (Sliding Window Table)

Bảng dưới đây minh họa quá trình biến đổi từ chuỗi 1 chiều sang bảng học có giám sát với $p = 3$:

| Thời gian $t$ | Chuỗi gốc $y_t$ | $\text{Target } (y)$ | $\text{Feature } lag_1$ | $\text{Feature } lag_2$ | $\text{Feature } lag_3$ | Trạng thái sau `shift()` |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $t_1$ | $12.379$ | $12.379$ | `NaN` | `NaN` | `NaN` | Bị loại bỏ |
| $t_2$ | $11.935$ | $11.935$ | $12.379$ | `NaN` | `NaN` | Bị loại bỏ |
| $t_3$ | $11.692$ | $11.692$ | $11.935$ | $12.379$ | `NaN` | Bị loại bỏ |
| $t_4$ | $11.597$ | $11.597$ | $11.692$ | $11.935$ | $12.379$ | **Mẫu hợp lệ đầu tiên** |
| $t_5$ | $11.681$ | $11.681$ | $11.597$ | $11.692$ | $11.935$ | Hợp lệ |
| $t_6$ | $12.260$ | $12.260$ | $11.681$ | $11.597$ | $11.692$ | Hợp lệ |

---

## 6. Thiết Kế Bậc Trễ $p = 24$ (Chu Kỳ 1 Ngày)

### 6.1. Căn cứ lựa chọn $p = 24$
1. **Chu kỳ vật lý của lưới điện:** Kết quả phân tích EDA ở Bước 1 chứng minh phụ tải điện có tính chu kỳ ngày lặp lại rất mạnh sau mỗi 24 giờ.
2. **Độ bao phủ thông tin:** Với $p = 24$, mô hình được cung cấp thông tin của:
   - Các giờ liền kề gần nhất ($lag_1, lag_2$): nắm bắt xu hướng tăng/giảm tức thời và quán tính ngắn hạn.
   - Các giờ nửa ngày trước ($lag_{12}$): nắm bắt pha đối lập ngày/đêm.
   - Cùng thời điểm ngày hôm trước ($lag_{24}$): nắm bắt mức nền phụ tải tại đúng thời khắc đó của chu kỳ trước.
3. **Cân đối chi phí tính toán & rủi ro đa cộng tuyến:** $p = 24$ là bậc trễ tiêu chuẩn, vừa đủ để bắt quy luật mà không làm bùng nổ số chiều đặc trưng dẫn đến overfitting.

---

### 6.2. Triển khai mã nguồn trong Notebook

Hàm `create_lag_features` được triển khai chuẩn hóa trong [`scripts/main.ipynb`](../scripts/main.ipynb):

```python
def create_lag_features(data_series, p=24):
    """
    Chuyển đổi chuỗi thời gian đơn biến thành bài toán học có giám sát với p bậc trễ.
    """
    df_feat = pd.DataFrame({'y': data_series})
    for i in range(1, p + 1):
        df_feat[f'lag_{i}'] = data_series.shift(i)
        
    # Loại bỏ p dòng đầu tiên chứa NaN
    df_clean = df_feat.dropna()
    return df_clean

# Thực thi tạo ma trận đặc trưng với p = 24
p_lags = 24
df_supervised = create_lag_features(series, p=p_lags)
```

---

## 7. Xử Lý Giá Trị Khuyết Thiếu & Tính Toàn Vẹn Dữ Liệu

### 7.1. Hiện tượng suy giảm mẫu do phép `shift()`
* Khi thực hiện trượt $p$ bước, $p$ quan sát đầu tiên của chuỗi không có đủ giá trị quá khứ để điền vào các cột $lag_1 \dots lag_p$, sinh ra các giá trị `NaN`.
* **Quy tắc xử lý:** Loại bỏ hoàn toàn các dòng `NaN` này bằng `.dropna()`.
* **Đánh giá mức độ tổn thất:**
  $$\text{Tỷ lệ mất mát} = \frac{p}{N} = \frac{24}{121.296} \approx 0,0198\%$$
  Mức suy giảm dưới $0,02\%$ là không đáng kể, không làm ảnh hưởng đến tính đại diện của chuỗi dữ liệu 14 năm.

### 7.2. Nguyên tắc chống rò rỉ dữ liệu (No Data Leakage)
* Tuyệt đối không sử dụng bất kỳ thông tin nào tại thời điểm $t$ (hoặc tương lai $t+1$) để làm đặc trưng cho việc dự đoán $y_t$.
* Mọi biến đặc trưng $lag_i$ đều thỏa mãn:
  $$\text{Feature}_i(t) = y_{t-i} \quad \text{với } i \ge 1$$
  Đảm bảo tính nhân quả thời gian (Causality) được tôn trọng 100%.

---

## 8. Kết Quả Thực Nghiệm Ma Trận Học Có Giám Sát

Bảng tổng hợp đặc tính ma trận dữ liệu thu được sau Bước 2:

| Thuộc tính | Giá trị định lượng | Diễn giải |
|---|---|---|
| **Kích thước ma trận (Shape)** | **$(121.272, 25)$** | $121.272$ dòng quan sát và $25$ cột dữ liệu. |
| **Cột mục tiêu (Target)** | `y` ($1$ cột) | Phụ tải điện tại thời điểm $t$ ($MW$). |
| **Cột đặc trưng (Features)** | `lag_1` $\rightarrow$ `lag_24` ($24$ cột) | Các giá trị trễ từ $1$ đến $24$ giờ trước đó ($MW$). |
| **Số giá trị khuyết (NaN)** | **$0$** | Dữ liệu sạch hoàn hảo sau khi `dropna()`. |
| **Dải thời gian hữu dụng** | `2004-10-02 01:00:00` $\rightarrow$ `2018-08-03 00:00:00` | Bắt đầu từ 01:00 ngày thứ hai của chuỗi quan sát. |

### Dữ liệu mẫu kiểm tra:
* **Mẫu đầu tiên (`2004-10-02 01:00:00`):**
  - $y_t = 12.260\text{ MW}$
  - $lag_1 (t-1 = \text{00:00}) = 13.147\text{ MW}$
  - $lag_2 (t-2 = \text{23:00 ngày 01/10}) = 14.067\text{ MW}$
  - $lag_{24} (t-24 = \text{01:00 ngày 01/10}) = 12.379\text{ MW}$ (khớp chính xác giá trị giờ đầu tiên của dataset).

---

## 9. Đúc Kết & Sẵn Sàng Cho Bước 3 (Chia $X - y$)

* Quá trình biến đổi bài toán chuỗi thời gian $y_t$ thành bài toán học có giám sát đã hoàn thành mỹ mãn.
* Ma trận dữ liệu `df_supervised` đã sẵn sàng để chuyển giao sang **Bước 3: Tách ma trận đầu vào $X$ và vector mục tiêu $y$**, làm tiền đề cho việc phân chia tập Train/Test theo trình tự thời gian ở Bước 4.
