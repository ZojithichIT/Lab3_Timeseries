"""
Module chuyên trách trực quan hóa dữ liệu và xuất biểu đồ:
- Biểu đồ chuỗi thời gian cùng các ngưỡng thống kê min/max/mean
- Biểu đồ tự tương quan (ACF & PACF)
- Biểu đồ phân chia Train / Validation / Test
- Biểu đồ đường cong học tập (Learning Curves)
- Biểu đồ so sánh giá trị Thực tế vs Dự báo
- Biểu đồ tầm quan trọng của đặc trưng (Feature Importance)
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from .config import Config

# Thiết lập theme mặc định
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def save_fig_helper(fig: plt.Figure, save_path: Optional[Union[str, Path]]):
    """Hàm phụ trợ tự động tạo thư mục và lưu biểu đồ."""
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        print(f"Đã lưu biểu đồ thành công tại: {path}")


def plot_raw_time_series(
    series: pd.Series, 
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """Vẽ toàn bộ chuỗi thời gian cùng các đường min, max, mean."""
    fig, ax = plt.subplots(figsize=(15, 6), dpi=150)
    ax.plot(series.index, series.values, label="Phụ tải AEP_MW", color="#1f77b4", linewidth=0.6, alpha=0.85)

    mean_val = series.mean()
    min_val, max_val = series.min(), series.max()
    min_date, max_date = series.idxmin(), series.idxmax()

    ax.axhline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:,.1f} MW")
    ax.axhline(min_val, color="green", linestyle=":", linewidth=1.5, label=f"Min: {min_val:,.1f} MW")
    ax.axhline(max_val, color="purple", linestyle=":", linewidth=1.5, label=f"Max: {max_val:,.1f} MW")

    ax.set_title("Chuỗi Tiêu Thụ Điện Hàng Giờ AEP (AEP_MW) Toàn Bộ 2004 - 2018", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Thời gian (Datetime)", fontsize=11)
    ax.set_ylabel("Phụ tải (Megawatt - MW)", fontsize=11)
    ax.legend(loc="upper right", frameon=True, facecolor="white")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    save_fig_helper(fig, save_path)
    return fig


def plot_acf_pacf_analysis(
    series: pd.Series, 
    lags: int = 72, 
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """Vẽ biểu đồ ACF và PACF trên 2 subplots để nhận diện chu kỳ và quán tính."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), dpi=150)

    # 1. ACF
    plot_acf(series, lags=lags, ax=ax1, alpha=0.05, 
             title=f"(A) Biểu Đồ Tự Tương Quan (ACF) - Nhận Diện Chu Kỳ ({lags} Giờ)")
    ax1.axvline(x=24, color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="Chu kỳ 24h (1 ngày)")
    ax1.axvline(x=48, color="orange", linestyle="--", linewidth=1.5, alpha=0.8, label="Chu kỳ 48h (2 ngày)")
    ax1.axvline(x=72, color="green", linestyle="--", linewidth=1.5, alpha=0.8, label="Chu kỳ 72h (3 ngày)")
    ax1.set_xlabel("Số bậc trễ (Lags - giờ)", fontsize=11)
    ax1.set_ylabel("Hệ số ACF", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True, facecolor="white")

    # 2. PACF
    plot_pacf(series, lags=lags, ax=ax2, alpha=0.05, method="ywm",
              title=f"(B) Biểu Đồ Tự Tương Quan Từng Phần (PACF) - Tương Quan Trực Tiếp ({lags} Giờ)")
    ax2.axvline(x=24, color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="Chu kỳ 24h (1 ngày)")
    ax2.axvline(x=48, color="orange", linestyle="--", linewidth=1.5, alpha=0.8, label="Chu kỳ 48h (2 ngày)")
    ax2.axvline(x=72, color="green", linestyle="--", linewidth=1.5, alpha=0.8, label="Chu kỳ 72h (3 ngày)")
    ax2.set_xlabel("Số bậc trễ (Lags - giờ)", fontsize=11)
    ax2.set_ylabel("Hệ số PACF", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True, facecolor="white")

    plt.tight_layout()
    save_fig_helper(fig, save_path)
    return fig


def plot_train_val_test_split(
    y_train: pd.Series, 
    y_val: pd.Series, 
    y_test: pd.Series, 
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """Vẽ phân bổ chuỗi thời gian của 3 tập Train - Val - Test."""
    fig, ax = plt.subplots(figsize=(15, 6), dpi=150)
    ax.plot(y_train.index, y_train.values, label=f"Train ({len(y_train):,} giờ - 75%)", color="#1f77b4", linewidth=0.8, alpha=0.85)
    ax.plot(y_val.index, y_val.values, label=f"Validation ({len(y_val):,} giờ - 12.5%)", color="#ff7f0e", linewidth=0.8, alpha=0.9)
    ax.plot(y_test.index, y_test.values, label=f"Test ({len(y_test):,} giờ - 12.5%)", color="#d62728", linewidth=0.8, alpha=0.9)

    ax.axvline(x=y_val.index.min(), color="black", linestyle="--", linewidth=1.5, alpha=0.8,
               label=f"Ranh giới Train/Val: {y_val.index.min().strftime('%Y-%m-%d')}")
    ax.axvline(x=y_test.index.min(), color="purple", linestyle="--", linewidth=1.5, alpha=0.8,
               label=f"Ranh giới Val/Test: {y_test.index.min().strftime('%Y-%m-%d')}")

    ax.set_title("Phân Chia Dữ Liệu AEP_MW: Train (75%) - Validation (12.5%) - Test (12.5%)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Thời gian (Datetime)", fontsize=11)
    ax.set_ylabel("Phụ tải (Megawatt - MW)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, facecolor="white")
    plt.tight_layout()

    save_fig_helper(fig, save_path)
    return fig


def plot_learning_curves(
    evals_result: Dict[str, Any], 
    best_iteration: Optional[int] = None, 
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """Vẽ đường cong học tập (Learning Curves) của mô hình XGBoost."""
    train_rmse = evals_result["validation_0"]["rmse"]
    val_rmse = evals_result["validation_1"]["rmse"]
    iterations = range(1, len(train_rmse) + 1)
    best_idx = (best_iteration + 1) if best_iteration is not None else len(train_rmse)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), dpi=150)

    # 1. Toàn cảnh
    ax1.plot(iterations, train_rmse, label="Train RMSE", color="#1f77b4", linewidth=1.5)
    ax1.plot(iterations, val_rmse, label="Validation RMSE", color="#ff7f0e", linewidth=1.5)
    ax1.axvline(x=best_idx, color="red", linestyle="--", linewidth=1.5, alpha=0.8, label=f"Best: {best_idx} cây")
    ax1.set_title("(A) Toàn Cảnh Quá Trình Học", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Số lượng cây (Boosting Iterations)", fontsize=11)
    ax1.set_ylabel("Sai số RMSE (MW)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True, facecolor="white")

    # 2. Phóng to giai đoạn hội tụ
    zoom_start = min(50, len(train_rmse) // 4)
    ax2.plot(iterations[zoom_start:], train_rmse[zoom_start:], label="Train RMSE", color="#1f77b4", linewidth=1.5)
    ax2.plot(iterations[zoom_start:], val_rmse[zoom_start:], label="Validation RMSE", color="#ff7f0e", linewidth=1.5)
    ax2.axvline(x=best_idx, color="red", linestyle="--", linewidth=1.5, alpha=0.8, label=f"Best RMSE: {val_rmse[best_idx-1]:.2f} MW")
    ax2.set_title(f"(B) Giai Đoạn Hội Tụ ({zoom_start} - {len(train_rmse)} Cây)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Số lượng cây (Boosting Iterations)", fontsize=11)
    ax2.set_ylabel("Sai số RMSE (MW)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True, facecolor="white")

    plt.suptitle("Đường Cong Học Tập (Learning Curves) Của Mô Hình XGBoost Regressor", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    save_fig_helper(fig, save_path)
    return fig


def plot_predictions_vs_actual(
    y_true: pd.Series, 
    y_pred: Union[pd.Series, np.ndarray], 
    zoom_days: int = 14, 
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """Vẽ so sánh giá trị Thực tế vs Dự báo (toàn tập Test và phóng to zoom_days ngày)."""
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred, index=y_true.index)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), dpi=150)

    # 1. Toàn bộ tập Test
    ax1.plot(y_true.index, y_true.values, label="Thực tế (Actual)", color="#1f77b4", linewidth=0.8, alpha=0.85)
    ax1.plot(y_pred.index, y_pred.values, label="Dự báo (XGBoost)", color="#d62728", linewidth=0.8, alpha=0.85, linestyle="--")
    ax1.set_title("So Sánh Giá Trị Thực Tế và Dự Báo Trên Toàn Bộ Tập Kiểm Thử (Test Set)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Thời gian (Datetime)", fontsize=11)
    ax1.set_ylabel("Phụ tải (Megawatt - MW)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True, facecolor="white")

    # 2. Phóng to 2 tuần đại diện
    zoom_hours = zoom_days * 24
    zoom_y_true = y_true.iloc[:zoom_hours]
    zoom_y_pred = y_pred.iloc[:zoom_hours]

    ax2.plot(zoom_y_true.index, zoom_y_true.values, label="Thực tế (Actual)", color="#1f77b4", marker="o", markersize=3, linewidth=1.2)
    ax2.plot(zoom_y_pred.index, zoom_y_pred.values, label="Dự báo (XGBoost)", color="#d62728", marker="x", markersize=3, linewidth=1.2, linestyle="--")
    ax2.set_title(f"Góc Nhìn Phóng To {zoom_days} Ngày Đầu Tiên Của Tập Test", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Thời gian (Datetime)", fontsize=11)
    ax2.set_ylabel("Phụ tải (Megawatt - MW)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True, facecolor="white")

    plt.tight_layout()
    save_fig_helper(fig, save_path)
    return fig


def plot_feature_importance(
    importance_series: pd.Series, 
    top_n: int = 15, 
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """Vẽ biểu đồ cột ngang thể hiện tầm quan trọng của các đặc trưng."""
    top_imp = importance_series.head(top_n).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    bars = ax.barh(top_imp.index, top_imp.values, color="#2ca02c", alpha=0.85, edgecolor="black", linewidth=0.5)

    # Hiển thị giá trị trên từng cột
    for bar in bars:
        width = bar.get_width()
        ax.text(width + (width * 0.01), bar.get_y() + bar.get_height() / 2, f"{width:,.2f}",
                va="center", ha="left", fontsize=9)

    ax.set_title(f"Top {top_n} Đặc Trưng Quan Trọng Nhất (Feature Importance - Gain)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Tầm quan trọng (Gain Score)", fontsize=11)
    ax.set_ylabel("Tên đặc trưng trễ (Lags)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    save_fig_helper(fig, save_path)
    return fig

