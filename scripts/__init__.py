"""
Lab 3 Time Series Machine Learning Package
Package cung cấp các module chuẩn mực cho bài toán dự báo chuỗi thời gian:
- config: Quản lý siêu tham số và đường dẫn
- data_loader: Tiền xử lý dữ liệu và làm sạch dòng thời gian
- feature_engineering: Tạo đặc trưng trễ (Lag) và phân chia Train/Val/Test
- models: Đóng gói các mô hình Machine Learning (XGBoost, Random Forest...)
- evaluation: Tính toán các độ đo định lượng MAE, RMSE, MAPE
- visualization: Trực quan hóa chuỗi thời gian và biểu đồ học tập
"""

from .config import Config

__all__ = ['Config']

