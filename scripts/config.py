"""
Module quản lý tập trung toàn bộ cấu hình, siêu tham số và đường dẫn thư mục cho Lab 3.
"""

from pathlib import Path


class Config:
    # 1. Đường dẫn thư mục gốc và các thư mục con
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
    PIPELINE_LOG_PATH = LOGS_DIR / "pipeline_training.log"
    SUMMARY_REPORT_PATH = LOGS_DIR / "model_comparison_summary.txt"
    LEADERBOARD_CSV_PATH = LOGS_DIR / "metrics_leaderboard.csv"
    EXPERIMENTS_JSON_PATH = LOGS_DIR / "experiments_history.json"
    MODEL_REGISTRY_PATH = MODELS_DIR / "model_registry.json"
    INFERENCE_LOG_PATH = LOGS_DIR / "inference.log"
    PREDICTIONS_CSV_PATH = LOGS_DIR / "latest_predictions.csv"

    MODEL_FILENAMES = {
        "Linear Regression (29 Lags)": "linear_regression_29lags.joblib",
        "Random Forest (29 Lags)": "random_forest_29lags.joblib",
        "XGBoost Dense (24 Lags)": "xgboost_dense_24lags.joblib",
        "XGBoost Baseline (29 Lags)": "xgboost_baseline_29lags.joblib",
        "XGBoost Enhanced (+6 Rolling)": "xgboost_enhanced_35features.joblib",
        "XGBoost Tuned (TS-Split+Refit)": "xgboost_tuned_29lags.joblib",
    }

    # 2. Thuộc tính chuỗi thời gian
    TIME_COL = "Datetime"
    TARGET_COL = "AEP_MW"
    FREQUENCY = "h"  # Tần suất hàng giờ (1 hour)
    DEFAULT_LAGS = 24  # Số bậc trễ cơ sở cho chu kỳ ngày (24 giờ)
    SELECTIVE_LAGS = list(range(1, 25)) + [48, 72, 167, 168, 169]
    ROLLING_WINDOWS = [6, 24, 168]
    MULTISTEP_HORIZON = 24
    LAG_SENSITIVITY_LIST = [1, 3, 6, 12, 24, 48, 72, 168, 336]

    # 3. Tỷ lệ phân chia tập dữ liệu (Time-based Splitting)
    SPLIT_RATIOS = {
        "train": 0.75,
        "val": 0.125,
        "test": 0.125
    }

    # 4. Siêu tham số mô hình cơ sở XGBoost (Bước 5)
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

    # 5. Siêu tham số mô hình đối chứng Random Forest (Bài tập nâng cao 4)
    RF_PARAMS = {
        "n_estimators": 200,
        "max_depth": 16,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": 0.7,
        "random_state": 42,
        "n_jobs": -1
    }

    # 6. Tuned XGBoost hyperparameters and TimeSeriesSplit candidates (Advanced Exercise 3)
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

    TSCV_PARAM_CANDIDATES = [
        {"name": "Config 1 (Shallow, lr=0.10)", "max_depth": 4, "learning_rate": 0.10, "subsample": 0.80, "colsample_bytree": 0.80, "min_child_weight": 1, "n_estimators": 600},
        {"name": "Config 2 (Shallow, lr=0.05)", "max_depth": 4, "learning_rate": 0.05, "subsample": 0.80, "colsample_bytree": 0.80, "min_child_weight": 3, "n_estimators": 800},
        {"name": "Config 3 (Baseline, d=6)", "max_depth": 6, "learning_rate": 0.05, "subsample": 0.80, "colsample_bytree": 0.80, "min_child_weight": 1, "n_estimators": 800},
        {"name": "Config 4 (Balanced, d=6, reg)", "max_depth": 6, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 3, "n_estimators": 1000},
        {"name": "Config 5 (Deep, d=7, lr=0.04)", "max_depth": 7, "learning_rate": 0.04, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 3, "n_estimators": 1000},
        {"name": "Config 6 (Deep, d=8, lr=0.03)", "max_depth": 8, "learning_rate": 0.03, "subsample": 0.80, "colsample_bytree": 0.80, "min_child_weight": 5, "n_estimators": 1000},
    ]

    # 7. Ngẫu nhiên và tái lập
    RANDOM_STATE = 42

