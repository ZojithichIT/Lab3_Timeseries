"""
Script thực thi huấn luyện mô hình dự báo chuỗi thời gian tự động từ Command Line (CLI).
Sử dụng:
    python scripts/train.py
    python scripts/train.py --lags 24 --lr 0.05
"""

import sys
from pathlib import Path
import argparse
import time

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Đảm bảo đường dẫn import hoạt động độc lập
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.config import Config
from scripts.data_loader import DataLoader
from scripts.feature_engineering import TimeSeriesFeatureBuilder
from scripts.models import XGBoostForecaster
from scripts.evaluation import evaluate_forecast, print_metrics_summary, residual_diagnostics
from scripts.visualization import (
    plot_acf_pacf_analysis, 
    plot_train_val_test_split, 
    plot_learning_curves, 
    plot_predictions_vs_actual, 
    plot_feature_importance
)


def run_pipeline(lags: int = Config.DEFAULT_LAGS, lr: float = 0.05):
    print("=" * 75)
    print(f"KHỞI CHẠY PIPELINE DỰ BÁO CHUỖI THỜI GIAN VỚI XGBOOST (Lags = {lags}, LR = {lr})")
    print("=" * 75)

    # 1. Nạp và làm sạch dữ liệu
    print("\n[1/5] Đang nạp và làm sạch dữ liệu...")
    loader = DataLoader(Config.RAW_DATA_PATH)
    series = loader.load_clean_series()
    print(f"-> Chuỗi thời gian sạch: {len(series):,} quan sát từ {series.index.min()} đến {series.index.max()}")

    # 2. Tạo lag features và chia dữ liệu
    print(f"\n[2/5] Đang tạo {lags} đặc trưng trễ và phân chia Train / Val / Test...")
    df_supervised = TimeSeriesFeatureBuilder.create_lag_features(series, p=lags)
    splits = TimeSeriesFeatureBuilder.time_based_split(
        df_supervised, 
        ratios=Config.SPLIT_RATIOS, 
        export_dir=Config.PROCESSED_DATA_DIR
    )
    print(f"-> Train: {len(splits.X_train):,} dòng | Val: {len(splits.X_val):,} dòng | Test: {len(splits.X_test):,} dòng")

    # 3. Huấn luyện mô hình XGBoost
    print("\n[3/5] Đang huấn luyện mô hình XGBoost Regressor...")
    params = Config.XGB_BASELINE_PARAMS.copy()
    params["learning_rate"] = lr
    forecaster = XGBoostForecaster(params)

    start_train = time.time()
    forecaster.fit(splits.X_train, splits.y_train, splits.X_val, splits.y_val, verbose=200)
    train_duration = time.time() - start_train
    print(f"-> Huấn luyện xong trong {train_duration:.2f}s! Best iteration: {forecaster.best_iteration + 1}, Val RMSE: {forecaster.best_score:.2f} MW")

    # 4. Dự báo và đánh giá trên tập Test
    print("\n[4/5] Đang dự báo và đánh giá trên tập Kiểm thử (Test Set)...")
    y_pred = forecaster.predict(splits.X_test)
    metrics = evaluate_forecast(splits.y_test, y_pred)
    print_metrics_summary(metrics, model_name="XGBoost Baseline")

    res_diag = residual_diagnostics(splits.y_test, y_pred)
    print(f"- Ljung-Box p-value phần dư:              {res_diag['ljung_box_pvalue']:.4e} (White noise: {res_diag['is_white_noise']})")

    # 5. Xuất biểu đồ phân tích
    print("\n[5/5] Đang xuất toàn bộ biểu đồ vào thư mục docs/images/...")
    plot_train_val_test_split(splits.y_train, splits.y_val, splits.y_test, save_path=Config.IMAGE_DIR / "03_train_val_test_split.png")
    plot_learning_curves(forecaster.evals_result, best_iteration=forecaster.best_iteration, save_path=Config.IMAGE_DIR / "05_xgboost_learning_curves.png")
    plot_predictions_vs_actual(splits.y_test, y_pred, save_path=Config.IMAGE_DIR / "06_predictions_vs_actual.png")
    plot_feature_importance(forecaster.get_feature_importance(), save_path=Config.IMAGE_DIR / "07_feature_importance.png")

    print("\n" + "=" * 75)
    print("PIPELINE HOÀN TẤT THÀNH CÔNG VÀ ĐÃ LƯU KẾT QUẢ ĐẦY ĐỦ!")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy Pipeline Dự Báo Chuỗi Thời Gian Lab 3")
    parser.add_argument("--lags", type=int, default=Config.DEFAULT_LAGS, help="Số bậc trễ (default: 24)")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate (default: 0.05)")
    args = parser.parse_args()

    run_pipeline(lags=args.lags, lr=args.lr)
