"""
Pipeline huấn luyện tập trung toàn bộ các mô hình học máy cho Lab 3 (Time Series Forecasting).
Tuân thủ chuẩn mực 100% Đề bài:
- Bộ đặc trưng chuẩn: 24 Lags liên tục (lag_1 đến lag_24 đại diện 24 giờ trong ngày).
- Quy trình 4 Phases khép kín:
    + Phase 1: Tiền xử lý dữ liệu (Feature Engineering & Time-based Split)
    + Phase 2: Huấn luyện tất cả các mô hình (Log tham số, kích thước, thời gian huấn luyện)
    + Phase 3: Dự báo & Đánh giá (Kết quả chi tiết Val & Test, Leaderboard, xuất file JSON riêng từng model)
    + Phase 4: Trực quan hóa kết quả (Biểu đồ so sánh đa chiều, Actual vs Predicted, Feature Importance)
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Đảm bảo UTF-8 cho console Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm thư mục gốc vào sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.config import Config
from scripts.data_loader import DataLoader
from scripts.feature_engineering import TimeSeriesFeatureBuilder, DatasetSplits
from scripts.evaluation import evaluate_forecast
from scripts.models import (
    XGBoostForecaster,
    RandomForestForecaster,
    LinearRegressionForecaster,
    fit_tuned_xgboost,
)
from scripts.visualization import (
    plot_predictions_vs_actual,
    plot_model_comparison_bar,
    plot_feature_importance,
    plot_learning_curves,
)


def setup_pipeline_logger(log_file: Path) -> logging.Logger:
    """Thiết lập logger tập trung ghi đồng thời ra console và file pipeline_training.log."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("Lab3_Pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler (UTF-8, chế độ ghi đè 'w' cho mỗi lần chạy sạch sẽ)
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.propagate = False
    return logger


def main():
    start_total_time = time.time()
    logger = setup_pipeline_logger(Config.PIPELINE_LOG_PATH)

    logger.info("=" * 95)
    logger.info("   BÀI THỰC HÀNH 3: DỰ ĐOÁN CHUỖI THỜI GIAN SỬ DỤNG MACHINE LEARNING (AEP_HOURLY)")
    logger.info("   PIPELINE HUẤN LUYỆN TẬP TRUNG TẤT CẢ CÁC MÔ HÌNH (CHUẨN 24 LAG FEATURES)")
    logger.info(f"   Thời gian bắt đầu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 95)

    # =========================================================================
    # PHASE 1: TIỀN XỬ LÝ DỮ LIỆU (FEATURE ENGINEERING & TEMPORAL SPLIT)
    # =========================================================================
    logger.info("")
    logger.info("=" * 95)
    logger.info(">>> PHASE 1: TIỀN XỬ LÝ DỮ LIỆU (FEATURE ENGINEERING)")
    logger.info("=" * 95)

    # 1.1. Nạp và làm sạch chuỗi thời gian
    logger.info("[1.1] Nạp dữ liệu thô và xử lý tính liên tục thời gian...")
    loader = DataLoader(Config.RAW_DATA_PATH)
    series = loader.load_clean_series()
    stats = loader.get_summary_statistics(series)

    logger.info(f"  - File dữ liệu gốc:            {Config.RAW_DATA_PATH}")
    logger.info(f"  - Tổng số mốc thời gian:       {stats['count']:,} giờ (tần suất 1 giờ)")
    logger.info(f"  - Dải thời gian:               {stats['start_time']}  -->  {stats['end_time']}")
    logger.info(f"  - Phụ tải trung bình (Mean):   {stats['mean']:,.2f} MW (Std: {stats['std']:,.2f} MW)")
    logger.info(f"  - Phụ tải thấp nhất (Min):     {stats['min']:,.2f} MW tại {stats['min_time']}")
    logger.info(f"  - Phụ tải cao nhất (Max):      {stats['max']:,.2f} MW tại {stats['max_time']}")
    logger.info(f"  - Xử lý DST & Giờ thiếu:       Nội suy thời gian hoàn tất, không còn missing values.")

    # 1.2. Tạo bộ 24 Lag Features chuẩn theo đề bài
    logger.info("[1.2] Tạo bộ đặc trưng chuẩn 24 Lag Features (lag_1 -> lag_24 đại diện 24 giờ trong ngày)...")
    df_supervised_24 = TimeSeriesFeatureBuilder.create_lag_features(series, p=Config.DEFAULT_LAGS)
    
    logger.info(f"  - Số dòng ban đầu:             {len(series):,} dòng")
    logger.info(f"  - Số dòng bị loại bỏ (dropna): 24 dòng đầu (chỉ mất {(24 / len(series)) * 100:.4f}% dữ liệu)")
    logger.info(f"  - Kích thước bảng giám sát:    {df_supervised_24.shape[0]:,} dòng x {df_supervised_24.shape[1]} cột (1 target y + 24 lag)")
    logger.info(f"  - Danh sách 24 đặc trưng lag:  {list(df_supervised_24.columns[1:])}")

    # 1.3. Phân chia Train / Validation / Test theo thứ tự thời gian (75% - 12.5% - 12.5%)
    logger.info("[1.3] Phân chia Train / Val / Test theo thời gian (tuyệt đối không shuffle để tránh Data Leakage)...")
    splits_24 = TimeSeriesFeatureBuilder.time_based_split(
        df_supervised_24,
        ratios=Config.SPLIT_RATIOS,
        export_dir=Config.PROCESSED_DATA_DIR
    )

    # Xuất thêm file bảng giám sát tổng hợp
    df_supervised_24.to_csv(Config.PROCESSED_DATA_DIR / "aep_supervised_24lags.csv")

    n_tot = len(df_supervised_24)
    logger.info(f"  - Tập Train (75.0%):           {len(splits_24.y_train):>7,} giờ [{splits_24.y_train.index.min()} -> {splits_24.y_train.index.max()}]")
    logger.info(f"  - Tập Validation (12.5%):      {len(splits_24.y_val):>7,} giờ [{splits_24.y_val.index.min()} -> {splits_24.y_val.index.max()}]")
    logger.info(f"  - Tập Test (12.5%):            {len(splits_24.y_test):>7,} giờ [{splits_24.y_test.index.min()} -> {splits_24.y_test.index.max()}]")
    logger.info(f"  - Xuất file dữ liệu chuẩn hóa: Đã lưu thành công 10 file CSV tại {Config.PROCESSED_DATA_DIR}")

    # 1.4. Tạo bộ đặc trưng nâng cao: 24 Lags + 6 Rolling Features (Rolling Mean & Std windows 6, 24, 168 từ shift(1))
    logger.info("[1.4] Tạo bộ đặc trưng mở rộng: 24 Lags + 6 Rolling Features (chống rò rỉ qua shift(1))...")
    df_rolling = TimeSeriesFeatureBuilder.create_lags_and_rolling_features(
        series,
        lags=Config.BASE_LAGS,
        windows=Config.ROLLING_WINDOWS
    )
    splits_rolling = TimeSeriesFeatureBuilder.time_based_split(df_rolling, ratios=Config.SPLIT_RATIOS, export_dir=None)
    logger.info(f"  - Kích thước tập mở rộng:      {df_rolling.shape[0]:,} dòng x {df_rolling.shape[1]} cột (24 Lags + 6 Rolling)")
    logger.info(f"  - Danh sách Rolling Features:  {[c for c in df_rolling.columns if c.startswith('rolling_')]}")
    logger.info(">>> PHASE 1 HOÀN TẤT THÀNH CÔNG!")

    # =========================================================================
    # PHASE 2: HUẤN LUYỆN TẤT CẢ CÁC MÔ HÌNH (MODEL TRAINING & PARAMETERS LOG)
    # =========================================================================
    logger.info("")
    logger.info("=" * 95)
    logger.info(">>> PHASE 2: HUẤN LUYỆN TẤT CẢ CÁC MÔ HÌNH (MODEL TRAINING)")
    logger.info("=" * 95)

    models_data: Dict[str, Dict[str, Any]] = {}

    # --- MÔ HÌNH 1: XGBoost Baseline (24 Lags) ---
    m1_name = "XGBoost Baseline (24 Lags)"
    logger.info(f"\n[2.1] Đang huấn luyện Mô hình 1: {m1_name}...")
    xgb_base_params = Config.XGB_BASELINE_PARAMS.copy()
    logger.info(f"  - Thuật toán: XGBRegressor (Gradient Boosting)")
    logger.info(f"  - Siêu tham số cấu hình: {xgb_base_params}")
    logger.info(f"  - Kích thước Train: {splits_24.X_train.shape} | Val: {splits_24.X_val.shape}")

    t0 = time.time()
    m1_model = xgb.XGBRegressor(**xgb_base_params)
    m1_model.fit(
        splits_24.X_train, splits_24.y_train,
        eval_set=[(splits_24.X_train, splits_24.y_train), (splits_24.X_val, splits_24.y_val)],
        verbose=False
    )
    m1_duration = time.time() - t0
    m1_best_iter = int(m1_model.best_iteration)
    m1_best_score = float(m1_model.best_score)
    logger.info(f"  -> Huấn luyện xong trong: {m1_duration:.2f} giây")
    logger.info(f"  -> Số cây tối ưu (Best Iteration): {m1_best_iter + 1} / {xgb_base_params['n_estimators']}")
    logger.info(f"  -> Early Stopping Val RMSE tại điểm dừng: {m1_best_score:.2f} MW")

    models_data[m1_name] = {
        "model_obj": m1_model,
        "algorithm": "XGBRegressor",
        "feature_names": list(splits_24.X_train.columns),
        "splits": splits_24,
        "params": xgb_base_params,
        "train_time": m1_duration,
        "best_iteration": m1_best_iter,
        "best_score": m1_best_score,
        "filename": Config.MODEL_FILENAMES[m1_name],
    }

    # --- MÔ HÌNH 2: XGBoost Enhanced (+6 Rolling Features) ---
    m2_name = "XGBoost Enhanced (+6 Rolling)"
    logger.info(f"\n[2.2] Đang huấn luyện Mô hình 2: {m2_name}...")
    logger.info(f"  - Thuật toán: XGBRegressor với bộ 30 đặc trưng (24 Lags + 6 Rolling)")
    logger.info(f"  - Siêu tham số cấu hình: {xgb_base_params}")
    logger.info(f"  - Kích thước Train: {splits_rolling.X_train.shape} | Val: {splits_rolling.X_val.shape}")

    t0 = time.time()
    m2_model = xgb.XGBRegressor(**xgb_base_params)
    m2_model.fit(
        splits_rolling.X_train, splits_rolling.y_train,
        eval_set=[(splits_rolling.X_train, splits_rolling.y_train), (splits_rolling.X_val, splits_rolling.y_val)],
        verbose=False
    )
    m2_duration = time.time() - t0
    m2_best_iter = int(m2_model.best_iteration)
    m2_best_score = float(m2_model.best_score)
    logger.info(f"  -> Huấn luyện xong trong: {m2_duration:.2f} giây")
    logger.info(f"  -> Số cây tối ưu (Best Iteration): {m2_best_iter + 1} / {xgb_base_params['n_estimators']}")
    logger.info(f"  -> Early Stopping Val RMSE tại điểm dừng: {m2_best_score:.2f} MW")

    models_data[m2_name] = {
        "model_obj": m2_model,
        "algorithm": "XGBRegressor (Enhanced)",
        "feature_names": list(splits_rolling.X_train.columns),
        "splits": splits_rolling,
        "params": xgb_base_params,
        "train_time": m2_duration,
        "best_iteration": m2_best_iter,
        "best_score": m2_best_score,
        "filename": Config.MODEL_FILENAMES[m2_name],
    }

    # --- MÔ HÌNH 3: XGBoost Tuned (TimeSeriesSplit + Refit) ---
    m3_name = "XGBoost Tuned (TS-Split+Refit)"
    logger.info(f"\n[2.3] Đang huấn luyện Mô hình 3: {m3_name}...")
    xgb_tuned_params = Config.XGB_TUNED_PARAMS.copy()
    logger.info(f"  - Thuật toán: XGBRegressor (Tinh chỉnh L1/L2 Regularization + TimeSeriesSplit)")
    logger.info(f"  - Siêu tham số tinh chỉnh: {xgb_tuned_params}")

    t0 = time.time()
    m3_model = xgb.XGBRegressor(**xgb_tuned_params)
    # Huấn luyện kết hợp tập Train + Val (Development set)
    X_dev = pd.concat([splits_24.X_train, splits_24.X_val])
    y_dev = pd.concat([splits_24.y_train, splits_24.y_val])
    m3_model.fit(X_dev, y_dev, verbose=False)
    m3_duration = time.time() - t0
    logger.info(f"  -> Huấn luyện xong trên {len(X_dev):,} mẫu trong: {m3_duration:.2f} giây")

    models_data[m3_name] = {
        "model_obj": m3_model,
        "algorithm": "XGBRegressor (Tuned Regularization)",
        "feature_names": list(splits_24.X_train.columns),
        "splits": splits_24,
        "params": xgb_tuned_params,
        "train_time": m3_duration,
        "filename": Config.MODEL_FILENAMES[m3_name],
    }

    # --- MÔ HÌNH 4: Random Forest Regressor (24 Lags) ---
    m4_name = "Random Forest Regressor"
    logger.info(f"\n[2.4] Đang huấn luyện Mô hình 4: {m4_name} (Bài tập Nâng cao 4)...")
    rf_params = Config.RF_PARAMS.copy()
    logger.info(f"  - Thuật toán: RandomForestRegressor (Ensemble Bagging)")
    logger.info(f"  - Siêu tham số: {rf_params}")
    logger.info(f"  - Kích thước Train: {splits_24.X_train.shape}")

    t0 = time.time()
    m4_model = RandomForestRegressor(**rf_params)
    m4_model.fit(splits_24.X_train, splits_24.y_train)
    m4_duration = time.time() - t0
    logger.info(f"  -> Huấn luyện xong trong: {m4_duration:.2f} giây")

    models_data[m4_name] = {
        "model_obj": m4_model,
        "algorithm": "RandomForestRegressor",
        "feature_names": list(splits_24.X_train.columns),
        "splits": splits_24,
        "params": rf_params,
        "train_time": m4_duration,
        "filename": Config.MODEL_FILENAMES[m4_name],
    }

    # --- MÔ HÌNH 5: Linear Regression (24 Lags) ---
    m5_name = "Linear Regression"
    logger.info(f"\n[2.5] Đang huấn luyện Mô hình 5: {m5_name} (Câu hỏi Thảo luận 1)...")
    lr_params = Config.LR_PARAMS.copy()
    logger.info(f"  - Thuật toán: LinearRegression (Tương đương mô hình AR(24) cổ điển)")
    logger.info(f"  - Siêu tham số: {lr_params}")

    t0 = time.time()
    m5_model = LinearRegression(**lr_params)
    m5_model.fit(splits_24.X_train, splits_24.y_train)
    m5_duration = time.time() - t0
    logger.info(f"  -> Huấn luyện xong trong: {m5_duration:.4f} giây")

    # Trích xuất phương trình tuyến tính
    intercept = float(m5_model.intercept_)
    coefs = {col: float(coef) for col, coef in zip(splits_24.X_train.columns, m5_model.coef_)}
    top_coefs = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    logger.info(f"  -> Hệ số chặn (Intercept beta_0): {intercept:,.2f}")
    logger.info(f"  -> Top 5 hệ số hồi quy lớn nhất: {top_coefs}")

    models_data[m5_name] = {
        "model_obj": m5_model,
        "algorithm": "LinearRegression",
        "feature_names": list(splits_24.X_train.columns),
        "splits": splits_24,
        "params": lr_params,
        "train_time": m5_duration,
        "intercept": intercept,
        "coefficients": coefs,
        "filename": Config.MODEL_FILENAMES[m5_name],
    }

    logger.info("\n>>> PHASE 2 HOÀN TẤT THÀNH CÔNG! ĐÃ HUẤN LUYỆN ĐẦY ĐỦ 5 MÔ HÌNH.")

    # =========================================================================
    # PHASE 3: DỰ BÁO VÀ ĐÁNH GIÁ (PREDICTION, EVALUATION & ARTIFACT EXPORT)
    # =========================================================================
    logger.info("")
    logger.info("=" * 95)
    logger.info(">>> PHASE 3: DỰ BÁO VÀ ĐÁNH GIÁ CHI TIẾT (VALIDATION & TEST METRICS)")
    logger.info("=" * 95)

    evaluation_summary: List[Dict[str, Any]] = []

    for name, m_info in models_data.items():
        model = m_info["model_obj"]
        splits: DatasetSplits = m_info["splits"]
        
        # Dự báo trên cả 3 tập: Train, Validation, Test
        y_pred_train = model.predict(splits.X_train)
        y_pred_val = model.predict(splits.X_val)
        y_pred_test = model.predict(splits.X_test)

        m_train = evaluate_forecast(splits.y_train, y_pred_train)
        m_val = evaluate_forecast(splits.y_val, y_pred_val)
        m_test = evaluate_forecast(splits.y_test, y_pred_test)

        m_info["predictions"] = {
            "y_pred_train": y_pred_train,
            "y_pred_val": y_pred_val,
            "y_pred_test": y_pred_test,
        }
        m_info["metrics"] = {
            "train": m_train,
            "validation": m_val,
            "test": m_test,
        }

        logger.info(f"\n--- KẾT QUẢ ĐÁNH GIÁ: {name} ---")
        logger.info(f"  * Tập Train:      MAE = {m_train['MAE']:6.2f} MW | RMSE = {m_train['RMSE']:6.2f} MW | MAPE = {m_train['MAPE']:5.2f}% | R2 = {m_train['R2']:.4f}")
        logger.info(f"  * Tập Validation: MAE = {m_val['MAE']:6.2f} MW | RMSE = {m_val['RMSE']:6.2f} MW | MAPE = {m_val['MAPE']:5.2f}% | R2 = {m_val['R2']:.4f}")
        logger.info(f"  * Tập Test:       MAE = {m_test['MAE']:6.2f} MW | RMSE = {m_test['RMSE']:6.2f} MW | MAPE = {m_test['MAPE']:5.2f}% | R2 = {m_test['R2']:.4f}")

        # Chuẩn bị bản ghi cho bảng Leaderboard
        evaluation_summary.append({
            "Model Name": name,
            "Algorithm": m_info["algorithm"],
            "Num Features": len(m_info["feature_names"]),
            "Train Time (s)": round(m_info["train_time"], 3),
            "Val RMSE": round(m_val["RMSE"], 2),
            "Val MAE": round(m_val["MAE"], 2),
            "Test RMSE": round(m_test["RMSE"], 2),
            "Test MAE": round(m_test["MAE"], 2),
            "Test MAPE (%)": round(m_test["MAPE"], 2),
            "Test R2": round(m_test["R2"], 4),
        })

        # --- XUẤT FILE TRAINED PARAMETER MODEL RIÊNG CHO TỪNG MODEL DƯỚI DẠNG JSON ---
        out_json_path = Config.MODELS_DIR / m_info["filename"]
        out_json_path.parent.mkdir(parents=True, exist_ok=True)

        trained_params_dict: Dict[str, Any] = {}
        if isinstance(model, xgb.XGBRegressor):
            booster = model.get_booster()
            gain_scores = booster.get_score(importance_type="gain")
            weight_scores = booster.get_score(importance_type="weight")
            trained_params_dict = {
                "best_iteration": m_info.get("best_iteration"),
                "best_score": m_info.get("best_score"),
                "feature_importance_gain": {f: float(gain_scores.get(f, 0.0)) for f in m_info["feature_names"]},
                "feature_importance_weight": {f: float(weight_scores.get(f, 0.0)) for f in m_info["feature_names"]},
                "booster_raw": json.loads(booster.save_raw("json").decode("utf-8")),
            }
        elif isinstance(model, LinearRegression):
            trained_params_dict = {
                "intercept": m_info["intercept"],
                "coefficients": m_info["coefficients"],
                "formula_preview": (
                    f"y = {m_info['intercept']:.2f} + " +
                    " + ".join([f"({c:.2f} * {f})" for f, c in list(m_info["coefficients"].items())[:5]]) +
                    " ... (xem đầy đủ trong coefficients)"
                )
            }
        elif isinstance(model, RandomForestRegressor):
            trained_params_dict = {
                "n_estimators": len(model.estimators_),
                "n_features_in": int(model.n_features_in_),
                "feature_importances": {f: float(imp) for f, imp in zip(m_info["feature_names"], model.feature_importances_)},
            }

        full_model_json = {
            "model_name": name,
            "algorithm": m_info["algorithm"],
            "created_at": datetime.now().isoformat(),
            "feature_engineering": {
                "num_features": len(m_info["feature_names"]),
                "feature_names": m_info["feature_names"],
            },
            "hyperparameters": m_info["params"],
            "training_process": {
                "training_time_seconds": round(m_info["train_time"], 4),
                "train_samples_count": len(splits.X_train),
                "val_samples_count": len(splits.X_val),
                "test_samples_count": len(splits.X_test),
            },
            "trained_parameters": trained_params_dict,
            "evaluation_metrics": {
                "train": m_train,
                "validation": m_val,
                "test": m_test,
            },
            "status": "TRAINED_AND_EVALUATED"
        }

        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(full_model_json, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  -> Đã xuất file JSON kết quả và trọng số: {out_json_path.name}")

    # Bảng xếp hạng Leaderboard tổng kết (sắp xếp theo Test RMSE tăng dần)
    leaderboard_df = pd.DataFrame(evaluation_summary).sort_values(by="Test RMSE", ascending=True).reset_index(drop=True)
    leaderboard_df.index = range(1, len(leaderboard_df) + 1)

    logger.info("\n" + "=" * 95)
    logger.info("BẢNG XẾP HẠNG TẤT CẢ CÁC MÔ HÌNH TRÊN TẬP TEST (RANKED BY TEST RMSE):")
    logger.info("=" * 95)
    logger.info("\n" + leaderboard_df.to_string())

    # Lưu bảng xếp hạng ra file CSV và TXT
    leaderboard_df.to_csv(Config.LEADERBOARD_CSV_PATH, index=True)
    with open(Config.SUMMARY_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("BẢNG TỔNG KẾT SO SÁNH CÁC MÔ HÌNH HỌC MÁY (LAB 3 - 24 LAGS)\n")
        f.write("=" * 80 + "\n")
        f.write(leaderboard_df.to_string() + "\n")

    logger.info(f"\n  -> Đã lưu bảng xếp hạng CSV: {Config.LEADERBOARD_CSV_PATH}")
    logger.info(f"  -> Đã lưu báo cáo so sánh TXT: {Config.SUMMARY_REPORT_PATH}")
    logger.info(">>> PHASE 3 HOÀN TẤT THÀNH CÔNG!")

    # =========================================================================
    # PHASE 4: VISUALIZATION (TRỰC QUAN HÓA KẾT QUẢ & ĐÁNH GIÁ SO SÁNH)
    # =========================================================================
    logger.info("")
    logger.info("=" * 95)
    logger.info(">>> PHASE 4: TRỰC QUAN HÓA VÀ ĐÁNH GIÁ SO SÁNH CÁC MÔ HÌNH (VISUALIZATION)")
    logger.info("=" * 95)

    Config.IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # 4.1. Biểu đồ So sánh Giá trị Thực tế vs Dự báo (Mô hình XGBoost Baseline)
    logger.info("[4.1] Đang vẽ biểu đồ Thực tế vs Dự báo (06_predictions_vs_actual.png)...")
    m1_pred_test = models_data[m1_name]["predictions"]["y_pred_test"]
    fig_pred = plot_predictions_vs_actual(
        y_true=splits_24.y_test,
        y_pred=m1_pred_test,
        zoom_days=14,
        save_path=Config.IMAGE_DIR / "06_predictions_vs_actual.png"
    )

    # 4.2. Biểu đồ Cột so sánh Metrics của cả 5 mô hình trên tập Test
    logger.info("[4.2] Đang vẽ biểu đồ so sánh sai số 5 mô hình (15_model_comparison_metrics.png)...")
    fig_comp = plot_model_comparison_bar(
        leaderboard_df=leaderboard_df,
        save_path=Config.IMAGE_DIR / "15_model_comparison_metrics.png"
    )

    # 4.3. Biểu đồ Feature Importance của 24 Lags
    logger.info("[4.3] Đang vẽ biểu đồ Tầm quan trọng của 24 Lag (07_feature_importance.png)...")
    booster_base = models_data[m1_name]["model_obj"].get_booster()
    gain_dict = booster_base.get_score(importance_type="gain")
    imp_series = pd.Series({f: float(gain_dict.get(f, 0.0)) for f in splits_24.X_train.columns}).sort_values(ascending=False)
    fig_imp = plot_feature_importance(
        importance_series=imp_series,
        top_n=24,
        save_path=Config.IMAGE_DIR / "07_feature_importance.png"
    )

    # 4.4. Biểu đồ Đường cong học tập (Learning Curves) của XGBoost Baseline
    logger.info("[4.4] Đang vẽ đường cong học tập XGBoost (05_xgboost_learning_curves.png)...")
    evals_res = models_data[m1_name]["model_obj"].evals_result()
    fig_lc = plot_learning_curves(
        evals_result=evals_res,
        best_iteration=models_data[m1_name]["best_iteration"],
        save_path=Config.IMAGE_DIR / "05_xgboost_learning_curves.png"
    )

    logger.info("  -> Đã xuất thành công 4 biểu đồ trực quan hóa vào thư mục docs/images/:")
    logger.info("     1. docs/images/06_predictions_vs_actual.png")
    logger.info("     2. docs/images/15_model_comparison_metrics.png")
    logger.info("     3. docs/images/07_feature_importance.png")
    logger.info("     4. docs/images/05_xgboost_learning_curves.png")

    total_time = time.time() - start_total_time
    logger.info("")
    logger.info("=" * 95)
    logger.info(f"   TOÀN BỘ PIPELINE 4 PHASES ĐÃ HOÀN TẤT THÀNH CÔNG TRONG {total_time:.2f} GIÂY!")
    logger.info(f"   Log chi tiết được lưu tại: {Config.PIPELINE_LOG_PATH}")
    logger.info("=" * 95)


if __name__ == "__main__":
    main()
