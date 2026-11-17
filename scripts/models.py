"""
Module định nghĩa các Class đóng gói mô hình học máy cho bài toán dự báo chuỗi thời gian:
- XGBoostForecaster (Mô hình chính của Lab 3)
- RandomForestForecaster (Mô hình đối chứng nâng cao)
- LinearRegressionForecaster (Mô hình cơ sở so sánh)
"""

from typing import Dict, Any, List, Optional, Tuple
import json
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
        """Lưu trọng số mô hình ra file (.json hoặc .joblib)."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if filepath.suffix.lower() == ".json":
            # Lưu trực tiếp cấu trúc cây XGBoost ra định dạng JSON chuẩn
            self.model.save_model(str(filepath))
        else:
            joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: Path) -> "XGBoostForecaster":
        """Nạp lại mô hình đã lưu."""
        filepath = Path(filepath)
        if filepath.suffix.lower() == ".json":
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            forecaster = XGBoostForecaster()
            forecaster.model = xgb.XGBRegressor()
            if isinstance(data, dict) and "trained_parameters" in data and "booster_raw" in data["trained_parameters"]:
                booster = xgb.Booster()
                raw_b = json.dumps(data["trained_parameters"]["booster_raw"]).encode("utf-8")
                booster.load_model(bytearray(raw_b))
                forecaster.model._Booster = booster
                forecaster.feature_names_ = data.get("feature_engineering", {}).get("feature_names", [])
            else:
                forecaster.model.load_model(str(filepath))
                try:
                    forecaster.feature_names_ = forecaster.model.get_booster().feature_names or []
                except Exception:
                    forecaster.feature_names_ = []
            forecaster.is_fitted = True
            return forecaster
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
        if hasattr(self.model, "estimators_") and len(self.model.estimators_) > 0:
            cols = [c for c in self.feature_names_ if c in X.columns]
            return self.model.predict(X[cols] if len(cols) == len(self.feature_names_) else X)
        raise RuntimeError("Mô hình RandomForestForecaster nạp từ JSON không chứa cây nhị phân (chỉ chứa metadata & feature importances).")

    def get_feature_importance(self) -> pd.Series:
        if hasattr(self, "feature_importances_"):
            return pd.Series(self.feature_importances_, index=self.feature_names_).sort_values(ascending=False)
        return pd.Series(
            self.model.feature_importances_, 
            index=self.feature_names_
        ).sort_values(ascending=False)

    def save(self, filepath: Path) -> None:
        """Lưu thông tin mô hình Random Forest ra disk (.json hoặc .joblib)."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if filepath.suffix.lower() == ".json":
            rf_data = {
                "model_type": "RandomForestRegressor",
                "params": self.params,
                "n_estimators": len(self.model.estimators_) if self.is_fitted else 0,
                "n_features_in": int(self.model.n_features_in_) if self.is_fitted else 0,
                "feature_names": self.feature_names_ if self.is_fitted else [],
                "feature_importances": {
                    col: float(val) for col, val in zip(self.feature_names_, self.model.feature_importances_)
                } if self.is_fitted else {}
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(rf_data, f, indent=4, ensure_ascii=False)
        else:
            joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: Path) -> "RandomForestForecaster":
        """Load Random Forest model từ disk."""
        filepath = Path(filepath)
        if filepath.suffix.lower() == ".json":
            with open(filepath, "r", encoding="utf-8") as f:
                rf_data = json.load(f)
            params = rf_data.get("hyperparameters", rf_data.get("params", {}))
            forecaster = RandomForestForecaster(params=params)
            feat_info = rf_data.get("feature_engineering", {})
            forecaster.feature_names_ = feat_info.get("feature_names", rf_data.get("feature_names", []))
            forecaster.is_fitted = True
            t_params = rf_data.get("trained_parameters", {})
            imp_dict = t_params.get("feature_importances", rf_data.get("feature_importances", {}))
            forecaster.feature_importances_ = np.array([imp_dict.get(col, 0.0) for col in forecaster.feature_names_])
            return forecaster
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
        cols = [c for c in self.feature_names_ if c in X.columns]
        X_mat = X[cols].values if len(cols) == len(self.feature_names_) else X.values
        if hasattr(self.model, "coef_"):
            return np.dot(X_mat, self.model.coef_) + self.model.intercept_
        return self.model.predict(X_mat)

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
        """Lưu mô hình Linear Regression ra disk (.json hoặc .joblib)."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if filepath.suffix.lower() == ".json":
            coef_dict = {
                col: float(coef) for col, coef in zip(self.feature_names_, self.model.coef_)
            } if self.is_fitted else {}
            lr_data = {
                "model_type": "LinearRegression",
                "intercept": float(self.model.intercept_) if self.is_fitted else 0.0,
                "coefficients": coef_dict,
                "formula_preview": (
                    f"y = {float(self.model.intercept_):.4f} + " +
                    " + ".join([f"({float(coef):.4f} * {col})" for col, coef in list(coef_dict.items())[:5]]) +
                    " ... (xem chi tiết trong coefficients)"
                ) if self.is_fitted else ""
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(lr_data, f, indent=4, ensure_ascii=False)
        else:
            joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: Path) -> "LinearRegressionForecaster":
        """Load Linear Regression model từ disk."""
        filepath = Path(filepath)
        if filepath.suffix.lower() == ".json":
            with open(filepath, "r", encoding="utf-8") as f:
                lr_data = json.load(f)
            forecaster = LinearRegressionForecaster()
            t_params = lr_data.get("trained_parameters", {})
            coef_dict = t_params.get("coefficients", lr_data.get("coefficients", {}))
            intercept_val = t_params.get("intercept", lr_data.get("intercept", 0.0))
            forecaster.feature_names_ = list(coef_dict.keys())
            forecaster.model.intercept_ = float(intercept_val)
            forecaster.model.coef_ = np.array(list(coef_dict.values()), dtype=float)
            forecaster.model.feature_names_in_ = np.array(forecaster.feature_names_)
            forecaster.model.n_features_in_ = len(forecaster.feature_names_)
            forecaster.is_fitted = True
            return forecaster
        return joblib.load(filepath)


def load_model(filepath: Path) -> Any:
    """Tự động nhận diện định dạng file (.json hoặc .joblib) và nạp đúng mô hình."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Không tìm thấy file mô hình: {filepath}")
    if filepath.suffix.lower() == ".joblib":
        return joblib.load(filepath)
    if filepath.suffix.lower() == ".json":
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return XGBoostForecaster.load(filepath)
        
        algo = data.get("algorithm", "") or data.get("model_type", "") if isinstance(data, dict) else ""
        fname = filepath.name.lower()
        if "linearregression" in algo.lower() or "linear" in fname:
            return LinearRegressionForecaster.load(filepath)
        elif "randomforest" in algo.lower() or "random_forest" in fname:
            return RandomForestForecaster.load(filepath)
        else:
            return XGBoostForecaster.load(filepath)
    raise ValueError(f"Định dạng file không được hỗ trợ: {filepath.suffix}")


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
    lags = lags or Config.BASE_LAGS
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

