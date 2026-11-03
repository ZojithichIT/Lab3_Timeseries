"""
Module chuyên trách đo lường, tính toán các độ đo định lượng và chẩn đoán phần dư:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R2 Score
- Kiểm định tự tương quan phần dư (Ljung-Box Test)
"""

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Tính Mean Absolute Percentage Error (đơn vị %)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Tránh chia cho 0
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_forecast(
    y_true: Union[pd.Series, np.ndarray], 
    y_pred: Union[pd.Series, np.ndarray]
) -> Dict[str, float]:
    """
    Tính toán toàn diện các độ đo định lượng cho bài toán dự báo phụ tải điện.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    mape = calculate_mape(y_true_arr, y_pred_arr)
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }


def residual_diagnostics(
    y_true: Union[pd.Series, np.ndarray], 
    y_pred: Union[pd.Series, np.ndarray],
    lags: int = 24
) -> Dict[str, Any]:
    r"""
    Chẩn đoán phần dư e_t = y_t - \hat{y}_t và kiểm định Ljung-Box (Nhiễu trắng).
    """
    residuals = np.array(y_true) - np.array(y_pred)
    lb_test = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    
    p_value = float(lb_test["lb_pvalue"].iloc[0])
    is_white_noise = p_value > 0.05

    return {
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "ljung_box_stat": float(lb_test["lb_stat"].iloc[0]),
        "ljung_box_pvalue": p_value,
        "is_white_noise": is_white_noise
    }


def print_metrics_summary(metrics: Dict[str, float], model_name: str = "XGBoost") -> None:
    """In định dạng bảng tổng kết các độ đo."""
    print(f"\n=== ĐÁNH GIÁ ĐỘ CHÍNH XÁC MÔ HÌNH: {model_name.upper()} ===")
    print(f"- MAE  (Sai số tuyệt đối TB):            {metrics['MAE']:,.2f} MW")
    print(f"- RMSE (Căn bậc hai sai số bình phương):  {metrics['RMSE']:,.2f} MW")
    print(f"- MAPE (Sai số phần trăm tuyệt đối TB):   {metrics['MAPE']:.2f}%")
    print(f"- R^2  (Hệ số xác định):                 {metrics['R2']:.4f}")
