"""
Module định nghĩa các Class đóng gói mô hình học máy cho bài toán dự báo chuỗi thời gian:
- XGBoostForecaster (Mô hình chính của Lab 3)
- RandomForestForecaster (Mô hình đối chứng nâng cao)
- LinearRegressionForecaster (Mô hình cơ sở so sánh)
"""

from typing import Dict, Any, List, Optional, Tuple
import time
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from .config import Config
from .evaluation import evaluate_forecast
from .feature_engineering import TimeSeriesFeatureBuilder


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
    def best_iteration(self) -> Optional[int]:
        return getattr(self.model, "best_iteration", None)

    @property
    def best_score(self) -> Optional[float]:
        return getattr(self.model, "best_score", None)

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

    def save(self, filepath: Path) -> None:
        """Save fitted Random Forest model to disk using joblib."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: Path) -> "RandomForestForecaster":
        """Load saved Random Forest model from disk."""
        return joblib.load(filepath)


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

    def get_coefficients(self) -> pd.Series:
        """Return signed linear regression coefficients sorted by absolute magnitude descending."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet! Call fit() first.")
        coefs = pd.Series(self.model.coef_, index=self.feature_names_)
        return coefs.reindex(coefs.abs().sort_values(ascending=False).index)

    def get_feature_importance(self) -> pd.Series:
        """Return absolute coefficient magnitudes sorted descending for uniform model comparison."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet! Call fit() first.")
        return pd.Series(
            np.abs(self.model.coef_),
            index=self.feature_names_
        ).sort_values(ascending=False)

    def save(self, filepath: Path) -> None:
        """Save fitted Linear Regression model to disk using joblib."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: Path) -> "LinearRegressionForecaster":
        """Load saved Linear Regression model from disk."""
        return joblib.load(filepath)


def fit_tuned_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    tuned_params: Optional[Dict[str, Any]] = None,
) -> Tuple[XGBoostForecaster, float, int]:
    """Fit tuned XGBoost model on the combined Development set (Train + Validation)."""
    params = (tuned_params or Config.XGB_TUNED_PARAMS).copy()
    X_dev = pd.concat([X_train, X_val])
    y_dev = pd.concat([y_train, y_val])

    t0 = time.time()
    tuned_forecaster = XGBoostForecaster(params)
    tuned_forecaster.model.fit(X_dev, y_dev, verbose=False)
    tuned_forecaster.is_fitted = True
    tuned_forecaster.feature_names_ = list(X_dev.columns)
    refit_time = time.time() - t0
    return tuned_forecaster, refit_time, len(X_dev)


def run_timeseries_split_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    candidates: Optional[List[Dict[str, Any]]] = None,
    tuned_params: Optional[Dict[str, Any]] = None,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """
    Run Expanding Window TimeSeriesSplit cross-validation across hyperparameter candidates
    and refit the tuned XGBoost model on the combined Development set (Train + Validation).
    """
    candidates = candidates or Config.TSCV_PARAM_CANDIDATES
    tuned_params = (tuned_params or Config.XGB_TUNED_PARAMS).copy()

    X_dev = pd.concat([X_train, X_val])
    y_dev = pd.concat([y_train, y_val])
    tscv = TimeSeriesSplit(n_splits=n_splits)

    folds_meta: List[Dict[str, Any]] = []
    splits_indices = list(tscv.split(X_dev))
    for fold_idx, (tr_idx, va_idx) in enumerate(splits_indices, start=1):
        folds_meta.append({
            "fold": fold_idx,
            "train_size": len(tr_idx),
            "val_size": len(va_idx),
            "train_start": str(X_dev.index[tr_idx[0]]),
            "train_end": str(X_dev.index[tr_idx[-1]]),
            "val_start": str(X_dev.index[va_idx[0]]),
            "val_end": str(X_dev.index[va_idx[-1]]),
        })

    cv_results: List[Dict[str, Any]] = []
    for cfg in candidates:
        fold_rmses: List[float] = []
        t0 = time.time()
        for tr_idx, va_idx in splits_indices:
            X_tr_f, y_tr_f = X_dev.iloc[tr_idx], y_dev.iloc[tr_idx]
            X_va_f, y_va_f = X_dev.iloc[va_idx], y_dev.iloc[va_idx]
            m = xgb.XGBRegressor(
                n_estimators=cfg["n_estimators"],
                learning_rate=cfg["learning_rate"],
                max_depth=cfg["max_depth"],
                subsample=cfg["subsample"],
                colsample_bytree=cfg["colsample_bytree"],
                min_child_weight=cfg["min_child_weight"],
                objective="reg:squarederror",
                eval_metric="rmse",
                early_stopping_rounds=40,
                random_state=Config.RANDOM_STATE,
                n_jobs=-1,
            )
            m.fit(X_tr_f, y_tr_f, eval_set=[(X_va_f, y_va_f)], verbose=False)
            pred_f = m.predict(X_va_f)
            fold_rmses.append(float(np.sqrt(mean_squared_error(y_va_f, pred_f))))

        cv_results.append({
            "name": cfg["name"],
            "fold_rmses": fold_rmses,
            "mean_rmse": float(np.mean(fold_rmses)),
            "std_rmse": float(np.std(fold_rmses)),
            "time_s": time.time() - t0,
        })

    best_cv = min(cv_results, key=lambda x: x["mean_rmse"])

    # Refit optimal model on full Development set
    tuned_forecaster, refit_time, dev_rows = fit_tuned_xgboost(
        X_train, y_train, X_val, y_val, tuned_params=tuned_params
    )

    return {
        "forecaster": tuned_forecaster,
        "folds_meta": folds_meta,
        "cv_results": cv_results,
        "best_config_name": best_cv["name"],
        "best_cv_mean_rmse": best_cv["mean_rmse"],
        "refit_time": refit_time,
        "dev_rows": dev_rows,
    }


def run_multistep_experiment(
    series: pd.Series,
    lags: Optional[List[int]] = None,
    horizon: int = Config.MULTISTEP_HORIZON,
) -> Dict[str, Any]:
    """
    Evaluate Direct Multi-Step (H independent XGBRegressor models) vs Recursive Multi-Step
    forecasting across independent 24-hour test cycles.
    """
    lags = lags or Config.SELECTIVE_LAGS
    t_start = time.time()

    df_multi = pd.DataFrame({"y_step_1": series})
    for h in range(2, horizon + 1):
        df_multi[f"y_step_{h}"] = series.shift(-(h - 1))
    for lag in lags:
        df_multi[f"lag_{lag}"] = series.shift(lag)
    df_multi = df_multi.dropna()

    n_m = len(df_multi)
    n_train_m = int(n_m * Config.SPLIT_RATIOS["train"])
    n_val_m = int(n_m * Config.SPLIT_RATIOS["val"])

    train_m = df_multi.iloc[:n_train_m]
    val_m = df_multi.iloc[n_train_m:n_train_m + n_val_m]
    test_m = df_multi.iloc[n_train_m + n_val_m:]

    feat_cols = [f"lag_{l}" for l in lags]
    target_cols = [f"y_step_{h}" for h in range(1, horizon + 1)]

    X_train_m, Y_train_m = train_m[feat_cols], train_m[target_cols]
    X_val_m, Y_val_m = val_m[feat_cols], val_m[target_cols]
    X_test_m, Y_test_m = test_m[feat_cols], test_m[target_cols]

    # 1. Train H independent Direct models
    direct_models: List[xgb.XGBRegressor] = []
    preds_direct = np.zeros((len(X_test_m), horizon))
    for h in range(1, horizon + 1):
        m = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            eval_metric="rmse",
            early_stopping_rounds=30,
            random_state=Config.RANDOM_STATE + h,
            n_jobs=-1,
        )
        m.fit(
            X_train_m,
            Y_train_m[f"y_step_{h}"],
            eval_set=[(X_val_m, Y_val_m[f"y_step_{h}"])],
            verbose=False,
        )
        direct_models.append(m)
        preds_direct[:, h - 1] = m.predict(X_test_m)

    # 2. Vectorized Recursive Multi-Step evaluation on independent 24h cycles
    rec_model = direct_models[0]
    origins = np.arange(0, len(X_test_m) - horizon, horizon)
    max_lag = max(lags)
    series_values = series.values

    hist_rows = []
    for origin in origins:
        origin_dt = X_test_m.index[origin]
        pos = series.index.get_loc(origin_dt)
        hist_rows.append(series_values[pos - max_lag:pos])
    hist_matrix = np.array(hist_rows, dtype=float)

    preds_rec = np.zeros((len(origins), horizon))
    for step in range(horizon):
        X_step = np.column_stack([hist_matrix[:, -lag] for lag in lags])
        y_next = rec_model.predict(pd.DataFrame(X_step, columns=feat_cols))
        preds_rec[:, step] = y_next
        hist_matrix = np.column_stack([hist_matrix, y_next])

    actual_sub = Y_test_m.iloc[origins].values
    direct_sub = preds_direct[origins, :]

    dir_rmse = [float(np.sqrt(mean_squared_error(actual_sub[:, h], direct_sub[:, h]))) for h in range(horizon)]
    rec_rmse = [float(np.sqrt(mean_squared_error(actual_sub[:, h], preds_rec[:, h]))) for h in range(horizon)]

    return {
        "horizon": horizon,
        "n_cycles": int(len(origins)),
        "direct_rmse_by_step": dir_rmse,
        "recursive_rmse_by_step": rec_rmse,
        "direct_mean_rmse": float(np.mean(dir_rmse)),
        "recursive_mean_rmse": float(np.mean(rec_rmse)),
        "elapsed_s": time.time() - t_start,
    }


def run_lag_sensitivity_experiment(
    series: pd.Series,
    p_list: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluate XGBRegressor across increasing continuous lag orders p to analyze
    the Bias-Variance-Compute trade-off (Discussion Question 2).
    """
    p_list = p_list or Config.LAG_SENSITIVITY_LIST
    results: List[Dict[str, Any]] = []

    for p in p_list:
        df_p = pd.DataFrame({"y": series})
        for lag in range(1, p + 1):
            df_p[f"lag_{lag}"] = series.shift(lag)
        df_p = df_p.dropna()

        n_p = len(df_p)
        nt_p = int(n_p * Config.SPLIT_RATIOS["train"])
        nv_p = int(n_p * Config.SPLIT_RATIOS["val"])

        fcols = [f"lag_{lag}" for lag in range(1, p + 1)]
        X_tr_p, y_tr_p = df_p[fcols].iloc[:nt_p], df_p["y"].iloc[:nt_p]
        X_va_p, y_va_p = df_p[fcols].iloc[nt_p:nt_p + nv_p], df_p["y"].iloc[nt_p:nt_p + nv_p]
        X_te_p, y_te_p = df_p[fcols].iloc[nt_p + nv_p:], df_p["y"].iloc[nt_p + nv_p:]

        m_p = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            eval_metric="rmse",
            early_stopping_rounds=40,
            random_state=Config.RANDOM_STATE,
            n_jobs=-1,
        )
        t0 = time.time()
        m_p.fit(X_tr_p, y_tr_p, eval_set=[(X_va_p, y_va_p)], verbose=False)
        el_p = time.time() - t0

        pred_tr = m_p.predict(X_tr_p)
        pred_te = m_p.predict(X_te_p)

        train_rmse = float(np.sqrt(mean_squared_error(y_tr_p, pred_tr)))
        test_rmse = float(np.sqrt(mean_squared_error(y_te_p, pred_te)))
        test_mae = float(mean_absolute_error(y_te_p, pred_te))
        mask = y_te_p.values != 0
        test_mape = float(np.mean(np.abs((y_te_p.values[mask] - pred_te[mask]) / y_te_p.values[mask])) * 100.0)

        results.append({
            "p": p,
            "lost_samples": int(len(series) - n_p),
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "test_mape": test_mape,
            "gap_rmse": test_rmse - train_rmse,
            "train_time": el_p,
        })

    return results


def _build_shift_invariant_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw lag features into relative differences w.r.t. lag_1:
    diff_lag_k = lag_k - lag_1 for k > 1, and keep calendar/volatility features.
    This guarantees input and target invariance under upward trends or peak shocks.
    """
    if "lag_1" not in X.columns:
        return X.copy()

    anchor = X["lag_1"]
    transformed: Dict[str, pd.Series] = {}
    for col in X.columns:
        if col == "lag_1":
            continue
        elif col.startswith("lag_"):
            k = col.split("_")[1]
            transformed[f"diff_lag_{k}"] = X[col] - anchor
        elif col.startswith("rolling_mean_"):
            w = col.split("_")[-1]
            transformed[f"diff_rolling_mean_{w}"] = X[col] - anchor
        else:
            transformed[col] = X[col]
    return pd.DataFrame(transformed, index=X.index)


def _strip_early_stopping(params: Dict[str, Any]) -> Dict[str, Any]:
    """Remove early_stopping_rounds when fitting without a validation set."""
    clean = params.copy()
    clean.pop("early_stopping_rounds", None)
    return clean


class DifferencedXGBoostForecaster:
    """
    Remediation #1 for Tree Extrapolation Limitation:
    Trains XGBoost to predict hourly differences (Delta y_t = y_t - y_{t-1}) using
    shift-invariant lag differences, then reconstructs y_hat_t = y_{t-1} + Delta y_hat_t.
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
        verbose: int = 0,
    ) -> "DifferencedXGBoostForecaster":
        if "lag_1" not in X_train.columns:
            raise ValueError("DifferencedXGBoostForecaster requires 'lag_1' in X_train.")

        X_tr_diff = _build_shift_invariant_features(X_train)
        dy_train = y_train - X_train["lag_1"]

        if X_val is not None and y_val is not None:
            X_va_diff = _build_shift_invariant_features(X_val)
            dy_val = y_val - X_val["lag_1"]
            self.model.fit(
                X_tr_diff,
                dy_train,
                eval_set=[(X_tr_diff, dy_train), (X_va_diff, dy_val)],
                verbose=verbose,
            )
        else:
            self.model = xgb.XGBRegressor(**_strip_early_stopping(self.params))
            self.model.fit(X_tr_diff, dy_train, verbose=verbose)

        self.is_fitted = True
        self.feature_names_ = list(X_train.columns)
        self.tree_feature_names_ = list(X_tr_diff.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet! Call fit() first.")
        X_diff = _build_shift_invariant_features(X[self.feature_names_])
        delta_hat = self.model.predict(X_diff)
        return X["lag_1"].values + delta_hat

    @property
    def best_iteration(self) -> Optional[int]:
        return getattr(self.model, "best_iteration", None)

    @property
    def best_score(self) -> Optional[float]:
        return getattr(self.model, "best_score", None)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet!")
        score_dict = self.model.get_booster().get_score(importance_type=importance_type)
        importances = {col: score_dict.get(col, 0.0) for col in self.tree_feature_names_}
        return pd.Series(importances).sort_values(ascending=False)

    def save(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: Path) -> "DifferencedXGBoostForecaster":
        return joblib.load(filepath)


class DetrendedXGBoostForecaster:
    """
    Remediation #2 for Tree Extrapolation Limitation:
    Fits a global linear trend T(t) on Train, removes T(t) from both target and lag
    columns, trains XGBoost on stationary residuals r_t, then restores T(t) + r_hat_t.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or Config.XGB_BASELINE_PARAMS.copy()
        self.trend_model = LinearRegression()
        self.model = xgb.XGBRegressor(**self.params)
        self.origin_ts_: Optional[pd.Timestamp] = None
        self.is_fitted = False

    def _hours_from_origin(self, index: pd.Index) -> np.ndarray:
        dt_idx = pd.DatetimeIndex(index)
        return ((dt_idx - self.origin_ts_).total_seconds() / 3600.0).to_numpy(dtype=float)

    def _detrend_X(self, X: pd.DataFrame) -> pd.DataFrame:
        t_hours = self._hours_from_origin(X.index)
        X_det = X.copy()
        for col in X.columns:
            if col.startswith("lag_"):
                k = int(col.split("_")[1])
                lag_trend = self.trend_model.predict((t_hours - k).reshape(-1, 1))
                X_det[col] = X[col].values - lag_trend
            elif col.startswith("rolling_mean_"):
                w = int(col.split("_")[-1])
                roll_trend = self.trend_model.predict((t_hours - (w + 1) / 2.0).reshape(-1, 1))
                X_det[col] = X[col].values - roll_trend
        return X_det

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        verbose: int = 0,
    ) -> "DetrendedXGBoostForecaster":
        if not isinstance(X_train.index, pd.DatetimeIndex):
            raise TypeError("DetrendedXGBoostForecaster requires a DatetimeIndex.")

        self.origin_ts_ = X_train.index[0]
        t_tr = self._hours_from_origin(X_train.index)
        self.trend_model.fit(t_tr.reshape(-1, 1), y_train.values)

        r_train = y_train.values - self.trend_model.predict(t_tr.reshape(-1, 1))
        X_tr_det = self._detrend_X(X_train)

        if X_val is not None and y_val is not None:
            t_va = self._hours_from_origin(X_val.index)
            r_val = y_val.values - self.trend_model.predict(t_va.reshape(-1, 1))
            X_va_det = self._detrend_X(X_val)
            self.model.fit(
                X_tr_det,
                r_train,
                eval_set=[(X_tr_det, r_train), (X_va_det, r_val)],
                verbose=verbose,
            )
        else:
            self.model = xgb.XGBRegressor(**_strip_early_stopping(self.params))
            self.model.fit(X_tr_det, r_train, verbose=verbose)

        self.is_fitted = True
        self.feature_names_ = list(X_train.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet! Call fit() first.")
        t_hours = self._hours_from_origin(X.index)
        trend_pred = self.trend_model.predict(t_hours.reshape(-1, 1))
        res_pred = self.model.predict(self._detrend_X(X[self.feature_names_]))
        return trend_pred + res_pred

    @property
    def trend_slope_mw_per_hour(self) -> float:
        return float(self.trend_model.coef_[0])

    @property
    def best_iteration(self) -> Optional[int]:
        return getattr(self.model, "best_iteration", None)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet!")
        score_dict = self.model.get_booster().get_score(importance_type=importance_type)
        importances = {col: score_dict.get(col, 0.0) for col in self.feature_names_}
        return pd.Series(importances).sort_values(ascending=False)

    def save(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: Path) -> "DetrendedXGBoostForecaster":
        return joblib.load(filepath)


class HybridLinearXGBoostForecaster:
    """
    Remediation #3 (Best Practice) for Tree Extrapolation Limitation:
    Two-stage Residual Boosting architecture:
      Stage 1 (Linear AR Backbone): LinearRegression on lag features captures unbounded
      linear extrapolation and local level/velocity without leaf clipping.
      Stage 2 (Nonlinear Residual Tree): XGBoost learns high-order nonlinear interactions
      (morning/evening transitions, calendar phases) on residuals e_t = y_t - y_hat_linear
      using shift-invariant difference features so extrapolation remains accurate at any scale.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        scale_invariant: bool = True,
    ):
        self.params = params or Config.XGB_BASELINE_PARAMS.copy()
        self.scale_invariant = scale_invariant
        self.linear_model = LinearRegression()
        self.model = xgb.XGBRegressor(**self.params)
        self.is_fitted = False

    def _prepare_tree_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.scale_invariant:
            return _build_shift_invariant_features(X)
        return X.copy()

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        verbose: int = 0,
    ) -> "HybridLinearXGBoostForecaster":
        self.feature_names_ = list(X_train.columns)
        lag_cols = [c for c in self.feature_names_ if c.startswith("lag_")]
        self.linear_cols_ = lag_cols if lag_cols else self.feature_names_

        # Stage 1: Linear autoregressive backbone
        self.linear_model.fit(X_train[self.linear_cols_], y_train)
        lin_train = self.linear_model.predict(X_train[self.linear_cols_])
        res_train = y_train.values - lin_train

        # Stage 2: XGBoost on residuals
        X_tr_tree = self._prepare_tree_features(X_train)
        if X_val is not None and y_val is not None:
            lin_val = self.linear_model.predict(X_val[self.linear_cols_])
            res_val = y_val.values - lin_val
            X_va_tree = self._prepare_tree_features(X_val)
            self.model.fit(
                X_tr_tree,
                res_train,
                eval_set=[(X_tr_tree, res_train), (X_va_tree, res_val)],
                verbose=verbose,
            )
        else:
            self.model = xgb.XGBRegressor(**_strip_early_stopping(self.params))
            self.model.fit(X_tr_tree, res_train, verbose=verbose)

        self.tree_feature_names_ = list(X_tr_tree.columns)
        self.is_fitted = True
        return self

    def predict_components(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (y_total, y_linear_backbone, e_xgb_residual)."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet! Call fit() first.")
        X_aligned = X[self.feature_names_]
        y_lin = self.linear_model.predict(X_aligned[self.linear_cols_])
        X_tree = self._prepare_tree_features(X_aligned)
        e_xgb = self.model.predict(X_tree)
        return y_lin + e_xgb, y_lin, e_xgb

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y_total, _, _ = self.predict_components(X)
        return y_total

    @property
    def best_iteration(self) -> Optional[int]:
        return getattr(self.model, "best_iteration", None)

    @property
    def best_score(self) -> Optional[float]:
        return getattr(self.model, "best_score", None)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet!")
        score_dict = self.model.get_booster().get_score(importance_type=importance_type)
        importances = {col: score_dict.get(col, 0.0) for col in self.tree_feature_names_}
        return pd.Series(importances).sort_values(ascending=False)

    def save(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: Path) -> "HybridLinearXGBoostForecaster":
        return joblib.load(filepath)


def _evaluate_extrapolation_suite(
    splits: Any,
    include_detrended: bool = True,
) -> Dict[str, Any]:
    """
    Fit and compare standard tree models against extrapolation-capable tree remedies
    on a dataset whose Test set contains values exceeding max(y_train).
    """
    train_max = float(splits.y_train.max())
    oob_mask = (splits.y_test > train_max).values

    fast_xgb = Config.XGB_BASELINE_PARAMS.copy()
    fast_xgb["n_estimators"] = 800
    fast_rf = {
        "n_estimators": 100,
        "max_depth": 16,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": 0.7,
        "random_state": Config.RANDOM_STATE,
        "n_jobs": -1,
    }

    candidates: List[Tuple[str, Any, bool]] = [
        ("Random Forest (Standard)", RandomForestForecaster(fast_rf), False),
        ("XGBoost (Standard)", XGBoostForecaster(fast_xgb.copy()), True),
        ("Differenced XGBoost (Delta y)", DifferencedXGBoostForecaster(fast_xgb.copy()), True),
    ]
    if include_detrended:
        candidates.append(("Detrended XGBoost", DetrendedXGBoostForecaster(fast_xgb.copy()), True))
    candidates.append(("Hybrid Linear + XGBoost", HybridLinearXGBoostForecaster(fast_xgb.copy(), scale_invariant=True), True))

    predictions: Dict[str, pd.Series] = {}
    metrics_rows: List[Dict[str, Any]] = []

    for name, forecaster, uses_val in candidates:
        t0 = time.time()
        if uses_val:
            forecaster.fit(splits.X_train, splits.y_train, splits.X_val, splits.y_val, verbose=False)
        else:
            forecaster.fit(splits.X_train, splits.y_train)
        elapsed = time.time() - t0

        y_pred = forecaster.predict(splits.X_test)
        predictions[name] = pd.Series(y_pred, index=splits.y_test.index)

        full_m = evaluate_forecast(splits.y_test, y_pred)
        oob_true = splits.y_test.values[oob_mask]
        oob_pred = y_pred[oob_mask]
        oob_m = evaluate_forecast(oob_true, oob_pred) if oob_mask.sum() > 0 else {"RMSE": 0.0, "MAE": 0.0}
        peak_bias = float(np.mean(oob_pred - oob_true)) if oob_mask.sum() > 0 else 0.0

        metrics_rows.append({
            "model": name,
            "full_rmse": full_m["RMSE"],
            "full_mae": full_m["MAE"],
            "oob_rmse": oob_m["RMSE"],
            "oob_mae": oob_m["MAE"],
            "peak_bias_mw": peak_bias,
            "pred_max_mw": float(np.max(y_pred)),
            "train_time_s": elapsed,
        })

    return {
        "train_max": train_max,
        "test_max": float(splits.y_test.max()),
        "oob_count": int(oob_mask.sum()),
        "test_count": int(len(splits.y_test)),
        "oob_mask": oob_mask,
        "y_test": splits.y_test,
        "predictions": predictions,
        "metrics": metrics_rows,
    }


def run_extrapolation_and_supplementary_experiments(
    series: pd.Series,
    save_models: bool = True,
) -> Dict[str, Any]:
    """
    Run the 3 Lab 3 Supplementary Experiments inspired by presentation.html:
      1. Trend Extrapolation Stress-Test (+0.25 MW/hour deterministic trend -> +6,140 MW above max(y_train))
      2. Real AEP Unseen Extreme Peak Regime Stress-Test (Train capped <= 19,500 MW; Test reaches 22,759 MW)
      3. Full Original AEP Test Benchmark comparing Standard XGBoost, Differenced XGBoost,
         Hybrid Linear + XGBoost, and Calendar Features (hour, dayofweek, month, is_weekend, sin/cos).
    """
    # --- Experiment 1: Linear Growth Trend Stress-Test ---
    slope_mw_h = 0.25
    trend_series = series + slope_mw_h * np.arange(len(series), dtype=float)
    df_trend = TimeSeriesFeatureBuilder.create_selective_lags(trend_series, lags=Config.SELECTIVE_LAGS)
    splits_trend = TimeSeriesFeatureBuilder.time_based_split(df_trend, ratios=Config.SPLIT_RATIOS, export_dir=None)
    exp1_res = _evaluate_extrapolation_suite(splits_trend, include_detrended=True)
    exp1_res["slope_mw_h"] = slope_mw_h

    # --- Experiment 2: Unseen Extreme Peak Regime (Real AEP Train capped <= 19,500 MW) ---
    cap_mw = 19500.0
    df_sel = TimeSeriesFeatureBuilder.create_selective_lags(series, lags=Config.SELECTIVE_LAGS)
    splits_raw = TimeSeriesFeatureBuilder.time_based_split(df_sel, ratios=Config.SPLIT_RATIOS, export_dir=None)
    tr_mask = splits_raw.y_train <= cap_mw
    va_mask = splits_raw.y_val <= cap_mw

    capped_splits = splits_raw._replace(
        X_train=splits_raw.X_train.loc[tr_mask],
        y_train=splits_raw.y_train.loc[tr_mask],
        X_val=splits_raw.X_val.loc[va_mask],
        y_val=splits_raw.y_val.loc[va_mask],
    )
    exp2_res = _evaluate_extrapolation_suite(capped_splits, include_detrended=True)
    exp2_res["cap_mw"] = cap_mw

    # --- Experiment 3: Full Original AEP Benchmark (Extrapolation Remedies + Calendar Features) ---
    df_cal = TimeSeriesFeatureBuilder.create_lags_and_calendar_features(series, lags=Config.SELECTIVE_LAGS)
    splits_cal = TimeSeriesFeatureBuilder.time_based_split(df_cal, ratios=Config.SPLIT_RATIOS, export_dir=None)

    xgb_params = Config.XGB_BASELINE_PARAMS.copy()
    xgb_tuned_params = Config.XGB_TUNED_PARAMS.copy()

    exp3_models: List[Tuple[str, Any, Any, bool, Optional[str]]] = [
        ("Linear Regression (29 Lags)", LinearRegressionForecaster(), splits_raw, False, None),
        ("XGBoost Baseline (29 Lags)", XGBoostForecaster(xgb_params.copy()), splits_raw, True, None),
        ("Differenced XGBoost (29 Lags)", DifferencedXGBoostForecaster(xgb_params.copy()), splits_raw, True, Config.MODEL_FILENAMES["Differenced XGBoost (29 Lags)"]),
        ("Hybrid Linear + XGBoost (29 Lags)", HybridLinearXGBoostForecaster(xgb_params.copy(), scale_invariant=True), splits_raw, True, Config.MODEL_FILENAMES["Hybrid Linear + XGBoost (29 Lags)"]),
        ("Linear Regression (+6 Calendar)", LinearRegressionForecaster(), splits_cal, False, None),
        ("XGBoost + Calendar (35 Features)", XGBoostForecaster(xgb_params.copy()), splits_cal, True, Config.MODEL_FILENAMES["XGBoost + Calendar (35 Features)"]),
        ("Hybrid Linear + XGBoost + Calendar (35 Features)", HybridLinearXGBoostForecaster(xgb_params.copy(), scale_invariant=True), splits_cal, True, None),
    ]

    exp3_rows: List[Dict[str, Any]] = []
    exp3_preds: Dict[str, np.ndarray] = {}
    exp3_hourly_rmse: Dict[str, List[float]] = {}
    cal_xgb_importance: Optional[pd.Series] = None
    hybrid_cal_components: Optional[Dict[str, pd.Series]] = None

    ramp_mask = splits_raw.y_test.index.hour.isin([6, 7, 8, 9])

    for name, model, sp, uses_val, save_fname in exp3_models:
        t0 = time.time()
        if uses_val:
            model.fit(sp.X_train, sp.y_train, sp.X_val, sp.y_val, verbose=False)
        else:
            model.fit(sp.X_train, sp.y_train)
        elapsed = time.time() - t0

        y_pred = model.predict(sp.X_test)
        m = evaluate_forecast(sp.y_test, y_pred)
        ramp_rmse = evaluate_forecast(sp.y_test[ramp_mask], y_pred[ramp_mask])["RMSE"]

        # Hourly RMSE profile 00h..23h
        err_sq = pd.Series((sp.y_test.values - y_pred) ** 2, index=sp.y_test.index)
        h_rmse = [float(np.sqrt(err_sq[err_sq.index.hour == h].mean())) for h in range(24)]

        exp3_preds[name] = y_pred
        exp3_hourly_rmse[name] = h_rmse
        exp3_rows.append({
            "model": name,
            "features": int(sp.X_train.shape[1]),
            "test_mae": m["MAE"],
            "test_rmse": m["RMSE"],
            "test_mape": m["MAPE"],
            "test_r2": m["R2"],
            "ramp_06_09_rmse": ramp_rmse,
            "train_time_s": elapsed,
        })

        if name == "XGBoost + Calendar (35 Features)":
            cal_xgb_importance = model.get_feature_importance()
        if save_models and save_fname:
            model.save(Config.MODELS_DIR / save_fname)

    # Also fit Tuned Refit Hybrid Linear + XGBoost + Calendar on Train+Val (Development set)
    X_dev_cal = pd.concat([splits_cal.X_train, splits_cal.X_val])
    y_dev_cal = pd.concat([splits_cal.y_train, splits_cal.y_val])
    hybrid_cal_tuned = HybridLinearXGBoostForecaster(xgb_tuned_params.copy(), scale_invariant=True)
    t0 = time.time()
    hybrid_cal_tuned.fit(X_dev_cal, y_dev_cal, verbose=False)
    elapsed_tuned = time.time() - t0
    if save_models:
        hybrid_cal_tuned.save(Config.MODELS_DIR / Config.MODEL_FILENAMES["Hybrid Linear + XGBoost + Calendar (35 Features)"])

    y_tot, y_lin, e_xgb = hybrid_cal_tuned.predict_components(splits_cal.X_test)
    m_tuned = evaluate_forecast(splits_cal.y_test, y_tot)
    ramp_tuned = evaluate_forecast(splits_cal.y_test[ramp_mask], y_tot[ramp_mask])["RMSE"]
    err_sq_t = pd.Series((splits_cal.y_test.values - y_tot) ** 2, index=splits_cal.y_test.index)
    tuned_name = "Hybrid Linear + XGBoost + Calendar (Refit)"
    exp3_preds[tuned_name] = y_tot
    exp3_hourly_rmse[tuned_name] = [float(np.sqrt(err_sq_t[err_sq_t.index.hour == h].mean())) for h in range(24)]
    exp3_rows.append({
        "model": tuned_name,
        "features": int(splits_cal.X_train.shape[1]),
        "test_mae": m_tuned["MAE"],
        "test_rmse": m_tuned["RMSE"],
        "test_mape": m_tuned["MAPE"],
        "test_r2": m_tuned["R2"],
        "ramp_06_09_rmse": ramp_tuned,
        "train_time_s": elapsed_tuned,
    })

    hybrid_cal_components = {
        "y_true": splits_cal.y_test,
        "y_total": pd.Series(y_tot, index=splits_cal.y_test.index),
        "y_linear": pd.Series(y_lin, index=splits_cal.y_test.index),
        "e_xgb": pd.Series(e_xgb, index=splits_cal.y_test.index),
    }

    return {
        "exp1_trend_stress": exp1_res,
        "exp2_peak_stress": exp2_res,
        "exp3_full_aep": {
            "rows": exp3_rows,
            "hourly_rmse": exp3_hourly_rmse,
            "cal_xgb_importance": cal_xgb_importance,
            "hybrid_components": hybrid_cal_components,
        },
    }


