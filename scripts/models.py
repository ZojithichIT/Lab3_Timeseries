"""
Module định nghĩa các Class đóng gói mô hình học máy cho bài toán dự báo chuỗi thời gian:
- XGBoostForecaster (Mô hình chính của Lab 3)
- RandomForestForecaster (Mô hình đối chứng nâng cao)
- LinearRegressionForecaster (Mô hình cơ sở so sánh)
"""

from typing import Dict, Any, Optional, Tuple
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from .config import Config


class XGBoostForecaster:
    """
    Class đóng gói mô hình XGBoost Regressor cho dự báo chuỗi thời gian.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or Config.XGB_BASELINE_PARAMS.copy()
        self.model = xgb.XGBRegressor(**self.params)
        self.is_fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        verbose: int = 100
    ) -> "XGBoostForecaster":
        """
        Huấn luyện mô hình XGBoost với cơ chế Early Stopping trên tập Validation.
        """
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=verbose
        )
        self.is_fitted = True
        self.feature_names_ = list(X_train.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Dự báo giá trị phụ tải trên ma trận đặc trưng X."""
        if not self.is_fitted:
            raise ValueError("Mô hình chưa được huấn luyện! Hãy gọi fit() trước.")
        return self.model.predict(X)

    @property
    def best_iteration(self) -> int:
        return self.model.best_iteration

    @property
    def best_score(self) -> float:
        return self.model.best_score

    @property
    def evals_result(self) -> Dict[str, Any]:
        return self.model.evals_result()

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """
        Trích xuất tầm quan trọng của các đặc trưng (Feature Importance).
        Mặc định sử dụng độ đo 'gain' (mức độ giảm sai số trung bình khi phân nhánh).
        """
        if not self.is_fitted:
            raise ValueError("Mô hình chưa được huấn luyện!")
        
        # Lấy điểm số importance từ booter
        booster = self.model.get_booster()
        score_dict = booster.get_score(importance_type=importance_type)
        
        # Đảm bảo đủ các cột đặc trưng (nếu cột nào không được phân nhánh thì gán 0)
        importances = {col: score_dict.get(col, 0.0) for col in self.feature_names_}
        ser = pd.Series(importances).sort_values(ascending=False)
        return ser

    def save(self, filepath: Path):
        """Lưu trọng số mô hình ra file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: Path) -> "XGBoostForecaster":
        """Nạp lại mô hình đã lưu."""
        return joblib.load(filepath)


class RandomForestForecaster:
    """Class đóng gói Random Forest cho bài tập đối chứng nâng cao."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or Config.RF_PARAMS.copy()
        self.model = RandomForestRegressor(**self.params)
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "RandomForestForecaster":
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        self.feature_names_ = list(X_train.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importance(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_, 
            index=self.feature_names_
        ).sort_values(ascending=False)


class LinearRegressionForecaster:
    """Class đóng gói Linear Regression làm baseline tuyến tính cổ điển."""

    def __init__(self):
        self.model = LinearRegression()
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LinearRegressionForecaster":
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        self.feature_names_ = list(X_train.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

