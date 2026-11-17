"""
Module quản lý tập trung toàn bộ cấu hình, siêu tham số và đường dẫn thư mục cho Lab 3.
Thiết kế bám sát 100% Đề bài:
- Nhóm 1: Mô hình cốt lõi 7 bước cơ bản (XGBoost 24 lags)
- Nhóm 2: Các mô hình mở rộng 4 bài nâng cao (Rolling features, Multi-step H=24, TimeSeriesSplit Tuning, Random Forest)
- Nhóm 3: Các mô hình đối chứng 3 câu hỏi thảo luận (Linear Regression, Khảo sát độ nhạy p lags, So sánh ARIMA)
"""

from pathlib import Path


class Config:
    # =========================================================================
    # 1. ĐƯỜNG DẪN THƯ MỤC VÀ TỆP TIN DỰ ÁN
    # =========================================================================
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_PATH = DATA_DIR / "raw" / "AEP_hourly.csv"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    DOCS_DIR = BASE_DIR / "docs"
    IMAGE_DIR = DOCS_DIR / "images"
    NOTEBOOKS_DIR = BASE_DIR / "notebooks"
    SCRIPTS_DIR = BASE_DIR / "scripts"
    LOGS_DIR = BASE_DIR / "logs"
    MODELS_DIR = BASE_DIR / "models"
    
    # File báo cáo & bảng xếp hạng kết quả
    PIPELINE_LOG_PATH = LOGS_DIR / "pipeline_training.log"
    INFERENCE_LOG_PATH = LOGS_DIR / "inference.log"
    PREDICTIONS_CSV_PATH = LOGS_DIR / "latest_predictions.csv"
    SUMMARY_REPORT_PATH = LOGS_DIR / "model_comparison_summary.txt"
    LEADERBOARD_CSV_PATH = LOGS_DIR / "metrics_leaderboard.csv"
    EXPERIMENTS_JSON_PATH = LOGS_DIR / "experiments_history.json"
    MODEL_REGISTRY_PATH = MODELS_DIR / "model_registry.json"

    # =========================================================================
    # 2. THUỘC TÍNH DỮ LIỆU CHUỖI THỜI GIAN
    # =========================================================================
    TIME_COL = "Datetime"
    TARGET_COL = "AEP_MW"
    FREQUENCY = "h"  # Tần suất hàng giờ (Hourly)
    
    # Tỷ lệ phân chia tập dữ liệu theo thời gian (Bước 4)
    # 75% Train (học trọng số) - 12.5% Val (early stopping chống overfit) - 12.5% Test (kiểm thử mù)
    SPLIT_RATIOS = {
        "train": 0.75,
        "val": 0.125,
        "test": 0.125
    }
    RANDOM_STATE = 42  # Đảm bảo tính tái lập 100%

    # =========================================================================
    # 3. NHÓM 1: MÔ HÌNH CỐT LÕI (7 BƯỚC THỰC HÀNH CƠ BẢN)
    # =========================================================================
    # Bước 2: Tạo p = 24 lag liên tục (đại diện cho chu kỳ một ngày đêm 24h theo chuẩn đề bài)
    BASE_LAGS = list(range(1, 25))
    DEFAULT_LAGS = 24
    
    # Bước 5: Siêu tham số mô hình XGBoost cơ bản
    XGB_BASELINE_PARAMS = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "early_stopping_rounds": 50,
        "random_state": 42,
        "n_jobs": -1
    }

    # =========================================================================
    # 4. NHÓM 2: CÁC MÔ HÌNH 4 BÀI TẬP NÂNG CAO
    # =========================================================================
    # Bài nâng cao 1: Cửa sổ trượt thống kê (Rolling Mean & Rolling Std với shift(1))
    ROLLING_WINDOWS = [6, 24, 168]  # 6h ngắn hạn, 24h ngày, 168h tuần
    
    # Bài nâng cao 2: Dự báo đa bước (Multi-step Horizon H = 24 giờ tới)
    MULTISTEP_HORIZON = 24
    
    # Bài nâng cao 3: Tinh chỉnh siêu tham số bằng TimeSeriesSplit (5 Folds Expanding Window)
    TSCV_N_SPLITS = 5
    TSCV_PARAM_CANDIDATES = [
        {"name": "Config 1 (Shallow, lr=0.10)", "max_depth": 4, "learning_rate": 0.10, "subsample": 0.80, "colsample_bytree": 0.80, "min_child_weight": 1, "n_estimators": 600},
        {"name": "Config 2 (Shallow, lr=0.05)", "max_depth": 4, "learning_rate": 0.05, "subsample": 0.80, "colsample_bytree": 0.80, "min_child_weight": 3, "n_estimators": 800},
        {"name": "Config 3 (Baseline, d=6)", "max_depth": 6, "learning_rate": 0.05, "subsample": 0.80, "colsample_bytree": 0.80, "min_child_weight": 1, "n_estimators": 800},
        {"name": "Config 4 (Balanced, d=6, reg)", "max_depth": 6, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 3, "n_estimators": 1000},
        {"name": "Config 5 (Deep, d=7, lr=0.04)", "max_depth": 7, "learning_rate": 0.04, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 3, "n_estimators": 1000},
        {"name": "Config 6 (Deep, d=8, lr=0.03)", "max_depth": 8, "learning_rate": 0.03, "subsample": 0.80, "colsample_bytree": 0.80, "min_child_weight": 5, "n_estimators": 1000},
    ]
    # Tham số tốt nhất sau khi tune TimeSeriesSplit
    XGB_TUNED_PARAMS = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.5,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": 42,
        "n_jobs": -1
    }
    
    # Bài nâng cao 4: Mô hình đối chứng Random Forest (Bagging)
    RF_PARAMS = {
        "n_estimators": 200,
        "max_depth": 16,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": 0.7,
        "random_state": 42,
        "n_jobs": -1
    }

    # =========================================================================
    # 5. NHÓM 3: CÁC MÔ HÌNH ĐỐI CHỨNG (3 CÂU HỎI THẢO LUẬN)
    # =========================================================================
    # Câu hỏi 1: Linear Regression (Hồi quy tuyến tính tương đương mô hình AR)
    LR_PARAMS = {
        "fit_intercept": True
    }
    
    # Câu hỏi 2: Khảo sát ảnh hưởng khi thay đổi số lag p (quá ít vs quá nhiều lag)
    LAG_SENSITIVITY_LIST = [1, 6, 12, 24, 48, 72, 168]

    # Ánh xạ tên mô hình sang tên file JSON lưu trữ (đồng bộ chuẩn 24 Lags theo đề bài)
    MODEL_FILENAMES = {
        "XGBoost Baseline (24 Lags)": "xgboost_baseline_24lags.json",
        "XGBoost Enhanced (+6 Rolling)": "xgboost_enhanced_rolling.json",
        "XGBoost Tuned (TS-Split+Refit)": "xgboost_tuned_24lags.json",
        "Random Forest Regressor": "random_forest_24lags.json",
        "Linear Regression": "linear_regression_24lags.json",
    }
