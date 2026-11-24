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


def plot_tree_extrapolation_benchmark(
    extrap_results: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot 4-panel diagnostic figure (300 DPI) demonstrating the Tree Extrapolation Limitation
    and evaluating Differenced XGBoost, Detrended XGBoost, and Hybrid Linear + XGBoost.
    """
    exp1 = extrap_results["exp1_trend_stress"]
    exp2 = extrap_results["exp2_peak_stress"]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=300)
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Panel A: Trend Extrapolation Stress-Test around global Test Peak
    y1 = exp1["y_test"]
    peak_pos1 = int(np.argmax(y1.values))
    s1 = max(0, peak_pos1 - 120)
    e1 = min(len(y1), peak_pos1 + 120)
    sl1 = slice(s1, e1)

    ax1.plot(y1.index[sl1], y1.values[sl1], label="Thực tế (Actual Trend Peak)", color="#0f172a", linewidth=2.0, zorder=5)
    ax1.axhline(exp1["train_max"], color="#b42318", linestyle="--", linewidth=1.6, label=f"Trần Train max(y_train) = {exp1['train_max']:,.0f} MW")
    style_map = [
        ("Random Forest (Standard)", "#f97316", ":", 1.6),
        ("XGBoost (Standard)", "#dc2626", "-.", 1.8),
        ("Detrended XGBoost", "#7c3aed", "--", 1.7),
        ("Differenced XGBoost (Delta y)", "#0f766e", "--", 1.9),
        ("Hybrid Linear + XGBoost", "#1f4e79", "-", 1.9),
    ]
    for m_name, col, ls, lw in style_map:
        if m_name in exp1["predictions"]:
            pred_s = exp1["predictions"][m_name].iloc[sl1]
            ax1.plot(pred_s.index, pred_s.values, label=m_name, color=col, linestyle=ls, linewidth=lw, alpha=0.92)
    ax1.fill_between(y1.index[sl1], exp1["train_max"], y1.values[sl1], where=(y1.values[sl1] > exp1["train_max"]),
                     color="#fee2e2", alpha=0.45, label="Vùng vượt đỉnh lịch sử")
    ax1.set_title("(A) Stress-Test 1: Xu Hướng Tăng (+0,25 MW/h) Vượt Trần +6.140 MW", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Phụ tải (MW)", fontsize=10.5)
    ax1.legend(loc="lower left", fontsize=8.3, ncol=2, frameon=True, facecolor="white", framealpha=0.92)
    ax1.tick_params(axis="x", rotation=20)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel B: Real AEP Unseen Extreme Peak Regime (Train <= 19,500 MW)
    y2 = exp2["y_test"]
    peak_pos2 = int(np.argmax(y2.values))
    s2 = max(0, peak_pos2 - 96)
    e2 = min(len(y2), peak_pos2 + 96)
    sl2 = slice(s2, e2)

    ax2.plot(y2.index[sl2], y2.values[sl2], label="Thực tế AEP (Peak 22.759 MW)", color="#0f172a", linewidth=2.0, zorder=5)
    ax2.axhline(exp2["train_max"], color="#b42318", linestyle="--", linewidth=1.6, label=f"Trần Train = {exp2['train_max']:,.0f} MW")
    for m_name, col, ls, lw in style_map[1:]:
        if m_name in exp2["predictions"]:
            pred_s = exp2["predictions"][m_name].iloc[sl2]
            ax2.plot(pred_s.index, pred_s.values, label=m_name, color=col, linestyle=ls, linewidth=lw, alpha=0.92)
    ax2.fill_between(y2.index[sl2], exp2["train_max"], y2.values[sl2], where=(y2.values[sl2] > exp2["train_max"]),
                     color="#ffedd5", alpha=0.55, label="Vùng đỉnh cực đoan chưa thấy")
    ax2.set_title("(B) Stress-Test 2: Dữ Liệu Thật AEP (Train ≤ 19.500 MW, Test Tới 22.759 MW)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Phụ tải (MW)", fontsize=10.5)
    ax2.legend(loc="lower left", fontsize=8.3, ncol=2, frameon=True, facecolor="white", framealpha=0.92)
    ax2.tick_params(axis="x", rotation=20)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Panel C: Out-of-Bounds Parity Scatter in Stress-Test 1
    oob_mask1 = exp1["oob_mask"]
    y_oob = exp1["y_test"].values[oob_mask1]
    ax3.scatter(y_oob, exp1["predictions"]["XGBoost (Standard)"].values[oob_mask1],
                s=10, alpha=0.45, color="#dc2626", label="XGBoost Standard (Bị cắt phẳng ngọn)")
    ax3.scatter(y_oob, exp1["predictions"]["Differenced XGBoost (Delta y)"].values[oob_mask1],
                s=10, alpha=0.45, color="#0f766e", label="Differenced XGBoost (Δy)")
    ax3.scatter(y_oob, exp1["predictions"]["Hybrid Linear + XGBoost"].values[oob_mask1],
                s=10, alpha=0.45, color="#1f4e79", label="Hybrid Linear + XGBoost")
    lo, hi = float(y_oob.min()), float(y_oob.max())
    ax3.plot([lo, hi], [lo, hi], "k--", linewidth=1.4, label="Đường lý tưởng y_pred = y_true")
    ax3.axhline(exp1["train_max"], color="#b42318", linestyle=":", linewidth=1.3, label=f"Trần max(y_train) = {exp1['train_max']:,.0f}")
    ax3.set_title(f"(C) Phân Tán Ngoại Suy Trên {exp1['oob_count']:,} Giờ Vượt Trần (Exp 1)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Phụ tải thực tế y_true (MW)", fontsize=10.5)
    ax3.set_ylabel("Phụ tải dự báo y_pred (MW)", fontsize=10.5)
    ax3.legend(loc="upper left", fontsize=8.5, frameon=True, facecolor="white")
    ax3.grid(True, linestyle=":", alpha=0.6)

    # Panel D: Out-of-Bounds RMSE Bar Chart across Exp 1 & Exp 2
    df_m1 = pd.DataFrame(exp1["metrics"]).set_index("model")
    df_m2 = pd.DataFrame(exp2["metrics"]).set_index("model")
    models_order = [
        "Random Forest (Standard)",
        "XGBoost (Standard)",
        "Detrended XGBoost",
        "Differenced XGBoost (Delta y)",
        "Hybrid Linear + XGBoost",
    ]
    short_labels = ["Random\nForest", "XGBoost\nStandard", "Detrended\nXGBoost", "Differenced\nXGBoost (Δy)", "Hybrid Linear\n+ XGBoost"]
    x = np.arange(len(models_order))
    w = 0.36
    vals1 = [df_m1.loc[m, "oob_rmse"] for m in models_order]
    vals2 = [df_m2.loc[m, "oob_rmse"] for m in models_order]

    b1 = ax4.bar(x - w / 2, vals1, width=w, color="#1f4e79", label="Exp 1: Trend Growth Vượt Trần", edgecolor="black", linewidth=0.5)
    b2 = ax4.bar(x + w / 2, vals2, width=w, color="#c2610c", label="Exp 2: Đỉnh Cực Đoan Thật AEP", edgecolor="black", linewidth=0.5)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, h + max(max(vals1), max(vals2)) * 0.015,
                 f"{h:,.0f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(short_labels, fontsize=9.5)
    ax4.set_ylim(0, max(max(vals1), max(vals2)) * 1.18)
    ax4.set_title("(D) Sai Số RMSE Trên Vùng Vượt Đỉnh Lịch Sử (Thấp Hơn = Tốt Hơn)", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Out-of-Bounds RMSE (MW)", fontsize=10.5)
    ax4.legend(loc="upper right", fontsize=9, frameon=True, facecolor="white")
    ax4.grid(True, linestyle=":", alpha=0.6, axis="y")

    plt.suptitle("Phân Tích Giới Hạn Ngoại Suy Của Mô Hình Cây & 3 Phương Pháp Khắc Phục Vượt Đỉnh Lịch Sử", fontsize=14.5, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig_helper(fig, save_path)
    return fig


def plot_calendar_and_hybrid_ablation(
    extrap_results: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot 4-panel figure (300 DPI) showing Full AEP Test improvements from Hybrid Linear + XGBoost
    and Calendar Features (hourly ramp, residual decomposition, and feature importance).
    """
    exp3 = extrap_results["exp3_full_aep"]
    df_rows = pd.DataFrame(exp3["rows"]).sort_values("test_rmse", ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=300)
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Panel A: Full Test Set RMSE Horizontal Bar Chart
    colors_a = []
    for m in df_rows["model"]:
        if "Hybrid" in m:
            colors_a.append("#1f4e79")
        elif "Calendar" in m and "XGBoost" in m:
            colors_a.append("#2f7d4f")
        elif "Differenced" in m:
            colors_a.append("#0f766e")
        elif "Linear" in m:
            colors_a.append("#64748b")
        else:
            colors_a.append("#c2610c")

    bars_a = ax1.barh(df_rows["model"], df_rows["test_rmse"], color=colors_a, alpha=0.9, edgecolor="black", linewidth=0.5)
    for bar, (_, row) in zip(bars_a, df_rows.iterrows()):
        w = bar.get_width()
        ax1.text(w + 1.2, bar.get_y() + bar.get_height() / 2,
                 f"{w:.2f} MW (MAE {row['test_mae']:.1f})", va="center", ha="left", fontsize=8.5, fontweight="bold")
    ax1.set_xlim(0, df_rows["test_rmse"].max() * 1.28)
    ax1.set_title("(A) Bảng Xếp Hạng Test RMSE Trên Toàn Tập AEP (15.142 Giờ)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Test RMSE (MW)", fontsize=10.5)
    ax1.grid(True, linestyle=":", alpha=0.6, axis="x")

    # Panel B: Hourly RMSE Profile 00h..23h
    h_map = exp3["hourly_rmse"]
    hours = list(range(24))
    line_specs = [
        ("Linear Regression (29 Lags)", "#0f766e", "--", 1.8),
        ("XGBoost Baseline (29 Lags)", "#c2610c", "-", 2.0),
        ("XGBoost + Calendar (35 Features)", "#2f7d4f", "-", 2.1),
        ("Hybrid Linear + XGBoost + Calendar (Refit)", "#1f4e79", "-", 2.4),
    ]
    for m_name, col, ls, lw in line_specs:
        if m_name in h_map:
            ax2.plot(hours, h_map[m_name], label=m_name, color=col, linestyle=ls, linewidth=lw, marker="o", markersize=3.5)
    ax2.axvspan(6, 9, color="#ffedd5", alpha=0.5, label="Khung tăng tải sáng (06h–09h)")
    ax2.axvspan(18, 21, color="#e3edf7", alpha=0.5, label="Khung đỉnh/giảm tối (18h–21h)")
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_xticklabels([f"{h:02d}h" for h in range(0, 24, 2)])
    ax2.set_title("(B) Sai Số RMSE Theo Khung Giờ: Calendar + Hybrid Khắc Phục Chuyển Pha", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Giờ trong ngày", fontsize=10.5)
    ax2.set_ylabel("Hourly RMSE (MW)", fontsize=10.5)
    ax2.legend(loc="upper left", fontsize=8.3, frameon=True, facecolor="white")
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Panel C: Hybrid Decomposition over 72h Test Window
    comp = exp3["hybrid_components"]
    sl_72 = slice(0, 72)
    idx72 = comp["y_true"].index[sl_72]
    y_t72 = comp["y_true"].values[sl_72]
    y_l72 = comp["y_linear"].values[sl_72]
    y_h72 = comp["y_total"].values[sl_72]
    e_x72 = comp["e_xgb"].values[sl_72]

    ax3.plot(idx72, y_t72, label="Thực tế AEP_MW", color="#0f172a", linewidth=2.0)
    ax3.plot(idx72, y_l72, label="Tầng 1: Linear AR Backbone (ŷ_linear)", color="#0f766e", linestyle="--", linewidth=1.6)
    ax3.plot(idx72, y_h72, label="Tầng 1+2: Hybrid Total (ŷ_linear + ê_xgb)", color="#1f4e79", linestyle="-.", linewidth=1.8)
    ax3_r = ax3.twinx()
    ax3_r.bar(idx72, e_x72, width=0.028, color=np.where(e_x72 >= 0, "#2f7d4f", "#dc2626"), alpha=0.35, label="Tầng 2: Hiệu chỉnh phi tuyến ê_xgb (trục phải)")
    ax3_r.set_ylabel("Phần hiệu chỉnh XGBoost ê_xgb (MW)", fontsize=9.5, color="#475569")
    ax3_r.set_ylim(-900, 900)
    ax3.set_title("(C) Giải Phẫu Mô Hình Lai Hybrid (Linear Backbone + XGBoost Residual, 72h)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Phụ tải (MW)", fontsize=10.5)
    ax3.tick_params(axis="x", rotation=20)
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_r.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8.0, frameon=True, facecolor="white")
    ax3.grid(True, linestyle=":", alpha=0.6)

    # Panel D: Top 12 Feature Importance (Gain Share %) of XGBoost + Calendar
    imp_cal = exp3["cal_xgb_importance"]
    imp_pct = (imp_cal / imp_cal.sum() * 100.0).head(12).sort_values(ascending=True)
    bar_cols = ["#2f7d4f" if f in Config.CALENDAR_COLS else "#1f4e79" for f in imp_pct.index]
    bars_d = ax4.barh(imp_pct.index, imp_pct.values, color=bar_cols, alpha=0.9, edgecolor="black", linewidth=0.5)
    for bar in bars_d:
        w = bar.get_width()
        ax4.text(w + imp_pct.max() * 0.015, bar.get_y() + bar.get_height() / 2,
                 f"{w:.2f}%", va="center", ha="left", fontsize=8.8, fontweight="bold")
    ax4.set_xlim(0, imp_pct.max() * 1.18)
    ax4.set_title("(D) Top 12 Đặc Trưng Quan Trọng Nhất (XGBoost + 6 Calendar, Gain %)", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Tỷ trọng Gain (%)", fontsize=10.5)
    from matplotlib.patches import Patch
    ax4.legend(handles=[
        Patch(facecolor="#2f7d4f", edgecolor="black", label="Đặc trưng Lịch (Calendar)"),
        Patch(facecolor="#1f4e79", edgecolor="black", label="Đặc trưng Trễ (Lags)"),
    ], loc="lower right", fontsize=9, frameon=True, facecolor="white")
    ax4.grid(True, linestyle=":", alpha=0.6, axis="x")

    plt.suptitle("Cải Thiện Dự Báo Với Hybrid Linear + XGBoost & Đặc Trưng Lịch (Calendar Features)", fontsize=14.5, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig_helper(fig, save_path)
    return fig


