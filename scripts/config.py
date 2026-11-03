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

    # 2. Thuộc tính chuỗi thời gian
    TIME_COL = "Datetime"
    TARGET_COL = "AEP_MW"
    FREQUENCY = "h"  # Tần suất hàng giờ (1 hour)
    DEFAULT_LAGS = 24  # Số bậc trễ cơ sở cho chu kỳ ngày (24 giờ)

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
        "n_estimators": 300,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1
    }

    # 6. Ngẫu nhiên và tái lập
    RANDOM_STATE = 42

